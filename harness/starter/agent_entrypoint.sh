#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo "AgentForge Starter Agent Entrypoint"
echo "============================================================"
echo
echo "This is intentionally a stub."
echo
echo "Contestants should replace or extend this entrypoint with"
echo "their own Agent / Harness implementation."
echo
echo "Expected arguments:"
echo
echo "  --operator <name>"
echo "  --candidate <path>"
echo "  --context <path>"
echo "  --analysis <path>"
echo "  --plan <path>"
echo "  --skills-dir <path>"
echo
echo "The Agent must:"
echo
echo "  1. inspect the provided context and candidate;"
echo "  2. write analysis.md;"
echo "  3. write plan.json;"
echo "  4. modify only the candidate workspace;"
echo "  5. return control to the Harness."
echo
exit 2
