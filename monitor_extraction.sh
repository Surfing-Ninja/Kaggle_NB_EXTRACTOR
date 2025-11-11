#!/bin/bash
# Monitor the multi-run extraction progress

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        EXTRACTION PROGRESS MONITOR                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Count notebooks
NOTEBOOKS=$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
CURATED=$(find notebooks_curated -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')

echo -e "${GREEN}📊 Current Counts:${NC}"
echo "  Downloaded notebooks: $NOTEBOOKS"
echo "  Curated notebooks: $CURATED"
echo ""

# Progress bar
GOAL=5000
PERCENT=$((CURATED * 100 / GOAL))
FILLED=$((PERCENT / 2))
BAR=$(printf '%*s' "$FILLED" | tr ' ' '█')
EMPTY=$(printf '%*s' "$((50 - FILLED))" | tr ' ' '░')

echo -e "${BLUE}Progress to 5000:${NC}"
echo "[$BAR$EMPTY] $PERCENT% ($CURATED/$GOAL)"
echo ""

# Show current run
if [ -f "logs/extract_all.log" ]; then
    echo -e "${YELLOW}📝 Latest Activity:${NC}"
    tail -15 logs/extract_all.log 2>/dev/null | grep -E "Stage|Complete|kernels|notebooks" | tail -5
    echo ""
fi

# Show summary if exists
if [ -f "logs/extraction_summary.txt" ]; then
    echo -e "${BLUE}📋 Run Summary:${NC}"
    tail -10 logs/extraction_summary.txt
    echo ""
fi

# Disk usage
DISK_USAGE=$(du -sh notebooks 2>/dev/null | awk '{print $1}')
echo -e "${YELLOW}💾 Storage Used:${NC} $DISK_USAGE"
echo ""

# Estimate completion
if [ $CURATED -gt 0 ]; then
    # Estimate based on current rate
    TIME_ELAPSED=$(( $(date +%s) - $(stat -f %B logs/extract_all.log 2>/dev/null || echo $(date +%s)) ))
    if [ $TIME_ELAPSED -gt 0 ]; then
        RATE=$(echo "scale=2; $CURATED / $TIME_ELAPSED * 3600" | bc 2>/dev/null || echo "0")
        if [ $(echo "$RATE > 0" | bc 2>/dev/null || echo 0) -eq 1 ]; then
            REMAINING=$(echo "scale=0; ($GOAL - $CURATED) / $RATE" | bc 2>/dev/null || echo "?")
            echo -e "${GREEN}⏱️  Estimated time to 5000:${NC} ${REMAINING} hours"
        fi
    fi
fi

echo ""
echo -e "${BLUE}Commands:${NC}"
echo "  Monitor logs: tail -f logs/extract_all.log"
echo "  Check progress: ./monitor_extraction.sh"
echo "  Stop extraction: pkill -f extract_5k.sh"
