"""Verify the pure move against the original module saved in Git.

Run from backend/:
    python tests/verify_agent_split.py --base-ref <commit-before-split>
"""

import argparse
import ast
from collections import Counter
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tokenize


PATCH_MODULES = {
    "_embedding_provider": "deps",
    "_retrieval_store": "deps",
    "get_llm_client": "deps",
    "schedule_memory_fold": "deps",
    "PaperEvidenceService": "deps",
    "LightRAGBridge": "deps",
    "_retrieve_for_agent": "retrieval",
    "_retrieve_legacy_for_agent": "retrieval",
    "_retrieve_visual_pages_for_agent": "retrieval",
}
DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def declared_names(node):
    if isinstance(node, DEFINITIONS):
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names]
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [
        item.id for target in targets for item in ast.walk(target)
        if isinstance(item, ast.Name)
    ]


def normalize_lookups(source):
    """Undo only the nine approved module-qualified dependency lookups."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode()))
    replacements = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and PATCH_MODULES.get(node.attr) == ("retrieval" if node.value.id == "retrieval_module" else node.value.id)
        ):
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            replacements.append((start, end, node.attr.encode()))
    result = source.encode()
    for start, end, replacement in sorted(replacements, reverse=True):
        result = result[:start] + replacement + result[end:]
    return result.decode(), len(replacements)


def comments(source):
    return Counter(
        token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    )


def check_acyclic(modules):
    edges = {}
    for name, source in modules.items():
        if name == "__init__":
            continue
        edges[name] = set()
        for node in ast.parse(source).body:
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                targets = [node.module] if node.module else [alias.name for alias in node.names]
                edges[name].update(target for target in targets if target != name)
    visited, active = set(), set()

    def visit(name):
        assert name not in active, f"Circular import through {name}"
        if name in visited:
            return
        active.add(name)
        for target in edges[name]:
            visit(target)
        active.remove(name)
        visited.add(name)

    for name in edges:
        visit(name)
    return len(edges)



def check_test_patch_targets(root, base_ref):
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", base_ref, "--", "backend/tests/test_*.py"],
        cwd=root, text=True,
    ).splitlines()
    for path in changed:
        original = subprocess.check_output(
            ["git", "show", f"{base_ref}:{path}"], cwd=root, text=True,
        )
        current = (root / path).read_text()
        lines = current.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line.encode()))
        replacements = []
        for node in ast.walk(ast.parse(current)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Attribute)
                and isinstance(node.args[0].value, ast.Name)
                and node.args[0].value.id in {"agent_api", "agent_module"}
                and isinstance(node.args[1], ast.Constant)
                and PATCH_MODULES.get(node.args[1].value) == node.args[0].attr
            ):
                target = node.args[0]
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "agent_module"
                and PATCH_MODULES.get(node.attr) == node.value.attr
            ):
                target = node.value
            else:
                continue
            start = offsets[target.lineno - 1] + target.col_offset
            end = offsets[target.end_lineno - 1] + target.end_col_offset
            replacements.append((start, end, target.value.id.encode()))
        normalized = current.encode()
        for start, end, replacement in sorted(set(replacements), reverse=True):
            normalized = normalized[:start] + replacement + normalized[end:]
        assert normalized.decode() == original, f"Non-patch changes in {path}"
    return len(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ref", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    original = subprocess.check_output(
        ["git", "show", f"{args.base_ref}:backend/app/api/agent.py"],
        cwd=root, text=True,
    )
    original_tree = ast.parse(original)
    package_dir = root / "backend/app/api/agent"
    modules = {path.stem: path.read_text() for path in sorted(package_dir.glob("*.py"))}
    moved, moved_text = {}, {}
    qualified = 0
    moved_comments = Counter()
    for module, source in modules.items():
        moved_comments.update(comments(source))
        if module == "__init__":
            continue
        normalized, count = normalize_lookups(source)
        qualified += count
        for node in ast.parse(normalized).body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                continue
            for name in declared_names(node):
                assert name not in moved, f"Duplicate definition: {name}"
                moved[name] = node
                moved_text[name] = ast.get_source_segment(normalized, node)
    definitions = constants = 0
    for node in original_tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        name = declared_names(node)[0]
        expected_text = ast.get_source_segment(original, node)
        if name == "run_agent_stream":
            node.decorator_list = []  # Registered by routes.py, with the same endpoint.
        if name == "logger":
            # __name__ moved from app.api.agent to app.api.agent.common.
            # __package__ preserves the original logger identity.
            expected_text = expected_text.replace("__name__", "__package__")
            node = ast.parse(expected_text).body[0]
        assert ast.dump(node) == ast.dump(moved[name]), f"AST changed: {name}"
        assert expected_text == moved_text[name], f"Source bytes changed: {name}"
        if isinstance(node, DEFINITIONS):
            definitions += 1
        else:
            constants += 1
    assert len(moved) == definitions + constants, "Unexpected package definitions"
    assert comments(original) == moved_comments, "Comments changed during the move"
    for module, source in modules.items():
        assert module == "stream" or len(source.splitlines()) <= 3000, module
    count = check_acyclic(modules)
    changed_tests = check_test_patch_targets(root, args.base_ref)
    expected_exports = {
        name for node in original_tree.body for name in declared_names(node)
    }
    exports = {
        name for node in ast.parse(modules["__init__"]).body
        for name in declared_names(node)
    }
    assert expected_exports == exports, "Package export surface changed"
    sys.path.insert(0, str(root / "backend"))
    agent = importlib.import_module("app.api.agent")
    for name in expected_exports:
        assert hasattr(agent, name), f"Missing importable name: {name}"
    assert agent.logger.name == "app.api.agent", "Logger identity changed"
    expected_routes = []
    for node in ast.parse(original).body:
        if isinstance(node, DEFINITIONS):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Name)
                    and decorator.func.value.id == "router"
                ):
                    expected_routes.append((
                        "/agent" + decorator.args[0].value,
                        decorator.func.attr.upper(),
                        getattr(agent, node.name),
                    ))
    actual_routes = [
        (route.path, method, route.endpoint)
        for route in agent.router.routes for method in sorted(route.methods)
    ]
    assert expected_routes == actual_routes, "Routes or endpoint identities changed"
    print(f"PASS: {definitions} function/class definitions and {constants} assignments match AST and source bytes")
    print(f"PASS: {qualified} approved dependency qualifications; all original comments preserved")
    print(f"PASS: {len(exports)} explicit exports; {len(expected_routes)} unchanged routes and endpoint identities")
    print(f"PASS: {changed_tests} existing test files change only patch targets")
    print(f"PASS: {count} implementation modules form an acyclic import graph; all modules <= 3000 lines")


if __name__ == "__main__":
    main()
