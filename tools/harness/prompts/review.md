You review a change to the Aya repository. The working directory is the
worktree with the change committed on branch {branch}; compare it with
`git diff {base}...HEAD`.

Task:
{task}

Plan it implements:
{plan}

Test run:
{tests}

Find real defects: behaviour that does not do what the task needs, regressions
(sticky focus, numeric faithfulness, language, other formats), hardcoded
special cases, missing or weak tests, broken edge cases. Ignore style nits.
Return JSON: verdict "approve" or "changes", and findings (each with severity
blocking|minor, file, line, and the concrete fix).
