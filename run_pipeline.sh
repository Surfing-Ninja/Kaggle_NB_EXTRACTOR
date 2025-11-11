#!/bin/bash
# Kaggle Pipeline Runner - Ensures venv is activated

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Create it with: python3 -m venv venv"
    echo "Then install: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate venv
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Check if packages are installed
if ! python -c "import pandas, tqdm, yaml, nbformat" 2>/dev/null; then
    echo -e "${RED}❌ Missing packages!${NC}"
    echo "Install with: pip install -r requirements.txt"
    exit 1
fi

echo -e "${GREEN}✅ Environment ready!${NC}"
echo

# Run main.py with all arguments passed through
python main.py "$@"
