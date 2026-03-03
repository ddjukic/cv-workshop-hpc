#!/usr/bin/env bash
# CV Workshop Setup Verification
# Run after setup_workshop_env.sh to verify everything works.
#
# Usage:
#   bash check_setup.sh          # Full check
#   bash check_setup.sh --quick  # Skip model weight download verification
#
# Or ask Claude Code: "verify my workshop setup"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

GREEN='\033[92m'
RED='\033[91m'
YELLOW='\033[93m'
NC='\033[0m'

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   CV Workshop — Setup Health Check               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Check venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}ERROR: Virtual environment not found at .venv/${NC}"
    echo "Run setup_workshop_env.sh first:"
    echo "  bash setup_workshop_env.sh"
    exit 1
fi

echo -e "${GREEN}Found venv:${NC} $VENV_PYTHON"
echo ""

# Run the Python health check
exec "$VENV_PYTHON" "$SCRIPT_DIR/check_environment.py" "$@"
