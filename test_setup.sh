#!/bin/bash
# Quick test script to validate the setup

echo "=========================================="
echo "Kaggle 5K Project - Setup Validation"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "1. Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "   $python_version"

if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo -e "   ${GREEN}✓ Python 3.8+ detected${NC}"
else
    echo -e "   ${RED}✗ Python 3.8+ required${NC}"
    exit 1
fi
echo ""

# Check if virtual environment exists
echo "2. Checking virtual environment..."
if [ -d "venv" ]; then
    echo -e "   ${GREEN}✓ Virtual environment exists${NC}"
else
    echo -e "   ${YELLOW}⚠ No virtual environment found${NC}"
    echo "   Creating virtual environment..."
    python3 -m venv venv
    echo -e "   ${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "3. Activating virtual environment..."
source venv/bin/activate
echo -e "   ${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo "4. Installing dependencies..."
pip install -q -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "   ${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

# Check Kaggle API
echo "5. Checking Kaggle API setup..."
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "   ${GREEN}✓ Kaggle API token found${NC}"
    
    # Test Kaggle API
    if kaggle competitions list --page-size 1 >/dev/null 2>&1; then
        echo -e "   ${GREEN}✓ Kaggle API working${NC}"
    else
        echo -e "   ${RED}✗ Kaggle API test failed${NC}"
        echo "   Please verify your kaggle.json token"
    fi
else
    echo -e "   ${YELLOW}⚠ Kaggle API token not found${NC}"
    echo "   Please download kaggle.json from https://www.kaggle.com/settings/account"
    echo "   and place it at ~/.kaggle/kaggle.json"
fi
echo ""

# Test imports
echo "6. Testing module imports..."
python3 -c "
from src import collector, downloader, analyzer, utils
print('   ✓ All modules imported successfully')
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✓ All modules working${NC}"
else
    echo -e "   ${RED}✗ Module import failed${NC}"
    exit 1
fi
echo ""

# Test configuration loading
echo "7. Testing configuration..."
python3 -c "
from src.utils import load_config, load_filters
config = load_config('config/config.yaml')
filters = load_filters('config/filters.json')
print('   ✓ Configuration loaded successfully')
print(f'   Target notebooks: {config[\"target_notebooks\"]}')
" 2>&1

if [ $? -eq 0 ]; then
    echo -e "   ${GREEN}✓ Configuration valid${NC}"
else
    echo -e "   ${RED}✗ Configuration loading failed${NC}"
    exit 1
fi
echo ""

# Check directory structure
echo "8. Checking directory structure..."
directories=("config" "src" "metadata" "notebooks" "notebooks_curated" "logs")
all_exist=true

for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "   ${GREEN}✓${NC} $dir/"
    else
        echo -e "   ${RED}✗${NC} $dir/ (missing)"
        all_exist=false
    fi
done

if [ "$all_exist" = true ]; then
    echo -e "   ${GREEN}✓ All directories present${NC}"
fi
echo ""

# Final summary
echo "=========================================="
echo "Setup Validation Complete"
echo "=========================================="
echo ""

if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo -e "${GREEN}Ready to run!${NC}"
    echo ""
    echo "Try a quick test:"
    echo "  python main.py --full --max-competitions 3 --download-limit 20 --target 10"
else
    echo -e "${YELLOW}Almost ready!${NC}"
    echo ""
    echo "Next step: Configure Kaggle API"
    echo "  1. Go to https://www.kaggle.com/settings/account"
    echo "  2. Click 'Create New API Token'"
    echo "  3. Move kaggle.json to ~/.kaggle/"
    echo "  4. Run: chmod 600 ~/.kaggle/kaggle.json"
fi
echo ""
