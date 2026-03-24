#!/bin/sh
# noslop Claude Code hook: block quality-bypass attempts
INPUT=$(cat)

if command -v jq >/dev/null 2>&1; then
  COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
else
  echo '{"decision":"block","reason":"noslop: jq is required for hook security parsing. Install jq."}'
  exit 0
fi

if echo "$COMMAND" | grep -qF -- '--no-verify'; then
  echo '{"decision":"block","reason":"noslop: --no-verify bypasses pre-commit hooks."}'
  exit 0
fi

if echo "$COMMAND" | grep -qiF 'SKIP_CI' || echo "$COMMAND" | grep -qF '[skip ci]'; then
  echo '{"decision":"block","reason":"noslop: CI-skip patterns are not allowed."}'
  exit 0
fi

echo '{"decision":"allow"}'
