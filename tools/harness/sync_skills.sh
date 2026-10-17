#!/bin/zsh
# One source of truth for agent skills: .agents/skills/<name>/SKILL.md
# (agentskills.io format). Link each into the folders Claude Code and Codex read.
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for target in .claude/skills .codex/skills; do
  mkdir -p "$target"
  for skill in .agents/skills/*(/N); do
    name="${skill:t}"
    # graphify installs its own copies (user and project level).
    [ "$name" = graphify ] && continue
    link="$target/$name"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
      echo "skip $link (a real directory; managed elsewhere)"; continue
    fi
    ln -sfn "../../.agents/skills/$name" "$link"
  done
  # Drop links whose source skill was removed.
  for link in "$target"/*(@N); do [ -e "$link" ] || rm "$link"; done
done
echo "skills: $(ls .agents/skills | tr '\n' ' ')"
