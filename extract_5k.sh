#!/bin/bash
# Multi-Run Extraction Script
# Goal: Extract 5000+ notebooks through multiple runs with decreasing vote thresholds

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="config/config.yaml"
STATE_FILE="logs/pipeline_state.json"
LOG_FILE="logs/extract_all.log"
SUMMARY_FILE="logs/extraction_summary.txt"

# Vote thresholds for each run
VOTE_THRESHOLDS=(10 5 3 1 0)
QUALITY_SCORES=(50 45 40 35 30)

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     KAGGLE 5K NOTEBOOK MULTI-RUN EXTRACTOR              ║${NC}"
echo -e "${BLUE}║     Goal: Extract 5000+ Notebooks                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check current notebooks
CURRENT_COUNT=$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
CURATED_COUNT=$(find notebooks_curated -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')

echo -e "${YELLOW}Current Status:${NC}"
echo "  Downloaded notebooks: $CURRENT_COUNT"
echo "  Curated notebooks: $CURATED_COUNT"
echo ""

# Calculate which run to start from
if [ $CURRENT_COUNT -lt 400 ]; then
    START_RUN=0
elif [ $CURRENT_COUNT -lt 900 ]; then
    START_RUN=1
elif [ $CURRENT_COUNT -lt 1800 ]; then
    START_RUN=2
elif [ $CURRENT_COUNT -lt 3500 ]; then
    START_RUN=3
else
    START_RUN=4
fi

echo -e "${GREEN}Starting from Run $(($START_RUN + 1))/5${NC}"
echo ""

# Function to update config
update_config() {
    local min_votes=$1
    local min_score=$2
    
    # Backup original config
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
    
    # Update config (simple sed replacement)
    sed -i.tmp "s/min_votes: [0-9]*/min_votes: $min_votes/" "$CONFIG_FILE"
    sed -i.tmp "s/min_score: [0-9]*/min_score: $min_score/" "$CONFIG_FILE"
    rm -f "${CONFIG_FILE}.tmp"
    
    echo -e "${BLUE}Updated config: min_votes=$min_votes, min_score=$min_score${NC}"
}

# Function to run extraction
run_extraction() {
    local run_num=$1
    local min_votes=$2
    local min_score=$3
    
    echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  RUN $run_num/5: min_votes=$min_votes, min_score=$min_score${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}\n"
    
    # Update configuration
    update_config $min_votes $min_score
    
    # Clear state for fresh run
    rm -f "$STATE_FILE"
    
    # Run pipeline
    echo "Y" | bash run_pipeline.sh --full --verbose >> "$LOG_FILE" 2>&1
    
    # Count results
    local new_count=$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
    local new_curated=$(find notebooks_curated -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
    
    echo -e "\n${GREEN}✅ Run $run_num Complete!${NC}"
    echo "  Total downloaded: $new_count"
    echo "  Total curated: $new_curated"
    
    # Log to summary
    echo "Run $run_num: min_votes=$min_votes | Downloaded=$new_count | Curated=$new_curated" >> "$SUMMARY_FILE"
    
    # Check if we've reached goal
    if [ $new_curated -ge 5000 ]; then
        echo -e "\n${GREEN}🎉 GOAL REACHED! ${new_curated} notebooks curated!${NC}"
        return 0
    fi
    
    return 1
}

# Initialize summary file
echo "=== Multi-Run Extraction Summary ===" > "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run extractions
for i in $(seq $START_RUN 4); do
    run_num=$(($i + 1))
    min_votes=${VOTE_THRESHOLDS[$i]}
    min_score=${QUALITY_SCORES[$i]}
    
    if run_extraction $run_num $min_votes $min_score; then
        break
    fi
    
    # Wait between runs to avoid rate limiting
    if [ $i -lt 4 ]; then
        echo -e "\n${YELLOW}⏸️  Waiting 30 seconds before next run...${NC}"
        sleep 30
    fi
done

# Final summary
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              EXTRACTION COMPLETE                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

FINAL_COUNT=$(find notebooks -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')
FINAL_CURATED=$(find notebooks_curated -name "*.ipynb" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo -e "${GREEN}📊 Final Results:${NC}"
echo "  Total downloaded: $FINAL_COUNT notebooks"
echo "  Total curated: $FINAL_CURATED notebooks"
echo ""
echo -e "${BLUE}📁 Output locations:${NC}"
echo "  Raw notebooks: notebooks/"
echo "  Curated notebooks: notebooks_curated/"
echo "  Metadata: metadata/"
echo "  Reports: reports/"
echo ""
echo -e "${YELLOW}📄 Detailed logs:${NC}"
echo "  Full log: $LOG_FILE"
echo "  Summary: $SUMMARY_FILE"
echo ""

# Restore original config
if [ -f "${CONFIG_FILE}.backup" ]; then
    mv "${CONFIG_FILE}.backup" "$CONFIG_FILE"
    echo -e "${BLUE}✅ Restored original configuration${NC}"
fi

echo "Completed: $(date)" >> "$SUMMARY_FILE"
echo ""
echo -e "${GREEN}🎉 All done! Happy analyzing!${NC}"
