You review an implementation plan for the Aya repository (working directory)
before anyone executes it. Read AGENTS.md and check the plan against the code
(open the files it cites).

Task:
{task}

Plan:
{plan}

Approve only if the plan would fix the task generally and safely. Look for:
wrong root cause or wrong files; a fix that is hardcoded to one case, paper,
figure or test; missing tests; behaviour it would break (sticky focus,
numeric faithfulness, language, other document types); steps too vague to
execute. Return JSON: verdict "approve" or "revise", and issues (each with
severity blocking|minor and a concrete fix).
