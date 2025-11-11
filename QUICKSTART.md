# Quick Start Guide - Kaggle 5K Notebook Pipeline

## 🚀 Get Started Fast

### ⚠️ IMPORTANT: Always Activate Virtual Environment

```bash
cd /Users/mohit/Github_5k/kaggle_5k_project
source venv/bin/activate  # You should see (venv) in prompt
```

**All commands below assume venv is activated!**

---

## 📦 What You Have

The complete pipeline with 5 stages:
1. ✅ **Collection**: Find competitions (src/collector.py)
2. ✅ **Indexing**: Index notebooks (src/indexer.py)
3. ✅ **Download**: Download files (src/downloader.py)
4. ✅ **Analysis**: Analyze & curate (src/analyzer.py)
5. ✅ **Reporting**: Generate reports (main.py)

## ⚡ Quick Commands

### 🎮 Interactive Mode (Recommended for First Run)

**Option 1: Using helper script (easiest)**
```bash
./run_pipeline.sh --interactive
```

**Option 2: Manual activation**
```bash
source venv/bin/activate
python main.py --interactive
```

The helper script (`run_pipeline.sh`) automatically activates the venv for you!

Choose from menu:
- `1` = Run full pipeline (automated)
- `2-6` = Run individual stages
- `7` = Resume from checkpoint
- `8` = View current progress
- `9` = Check prerequisites ✅ (do this first!)
- `0` = Exit

### 🧪 Test Prerequisites

```bash
./run_pipeline.sh --interactive
# Then choose option 9
```

Should show all ✅:
- ✅ Kaggle CLI: v1.7.4.5
- ✅ API Credentials: Found
- ✅ Packages: pandas, tqdm, yaml, nbformat
- ✅ Disk Space: 15+ GB
- ✅ Network: Connected

### 🤖 Automated Mode

```bash
# Run complete pipeline (5-15 hours) - EASIEST WAY
./run_pipeline.sh --full

# Or manually activate venv first
source venv/bin/activate
python main.py --full

# With verbose output
./run_pipeline.sh --full --verbose

# Quiet mode (logs only)
./run_pipeline.sh --full --quiet

# Dry run (see what would happen)
./run_pipeline.sh --dry-run --full
```

### 🔄 Resume After Interruption

```bash
# Ctrl+C saves state automatically
# Resume with:
./run_pipeline.sh --resume
```

### 🎯 Run Individual Stages

```bash
# Stage 1: Collect competitions (~5 min)
./run_pipeline.sh --stage collection

# Stage 2: Index notebooks (~10 min)
./run_pipeline.sh --stage indexing

# Stage 3: Download notebooks (~3-4 hours)
./run_pipeline.sh --stage download

# Stage 4: Analyze & curate (~20 min)
./run_pipeline.sh --stage analysis

# Stage 5: Generate reports (~1 min)
./run_pipeline.sh --stage reporting
```

## 📊 What You'll Get

### After Stage 1 (Collection)
- `metadata/competitions.json` - 500 competitions with scores

### After Stage 2 (Indexing)
- `metadata/download_manifest.json` - 5000 kernels ready to download

### After Stage 3 (Download)
- `notebooks/{competition}/{kernel}/` - Downloaded .ipynb files
- `logs/download.jsonl` - Download log (for resume)
- `metadata/download_report.json` - Statistics

### After Stage 4 (Analysis)
- `notebooks_curated/{category}/` - Organized by technique
- `metadata/notebook_analyses.json` - Full analysis data
- `metadata/curated_catalog.json` - Final catalog
- `reports/analysis_summary.txt` - Report

### After Stage 5 (Reporting)
- `USAGE.md` - Dataset usage guide
- `PROVENANCE.json` - Legal/attribution
- `reports/summary.json` - Statistics
- `notebooks_curated/README.md` - Master README

---

## 🔧 Configuration

Edit `config/config.yaml`:

```yaml
collection:
  target_count: 500        # Competitions to collect

indexing:
  target_count: 5000       # Notebooks to index
  min_votes: 10            # Minimum votes required

download:
  batch_size: 100          # Notebooks per batch
  rate_limit: 1.5          # Seconds between requests

analysis:
  min_score: 50            # Quality threshold (0-100)
  target_count: 5000       # Final curated count
  workers: 4               # Parallel workers
```

## 🔍 Troubleshooting

### ❌ Prerequisite Check Failing?

**Problem**: Shows "Package nbformat: Not installed" but it IS installed
```bash
# Solution 1: Use the helper script (easiest!)
./run_pipeline.sh --interactive

# Solution 2: Make sure venv is activated manually
source venv/bin/activate  # Must see (venv) in prompt!
python main.py --interactive
```

**Problem**: Helper script doesn't work
```bash
# Make it executable
chmod +x run_pipeline.sh

# Or use manual activation
source venv/bin/activate
python main.py --interactive
```

**Problem**: Kaggle API credentials not found
```bash
# Download kaggle.json from: https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Test it works
kaggle competitions list
```

**Problem**: Not enough disk space
```bash
# Check available space
df -h .

# Need at least 15 GB free
# Clean up if needed or use different directory
```

### 🔄 Downloads Failing?

```bash
# Pipeline has built-in resume capability
# Just run again, it will continue where it left off
./run_pipeline.sh --resume
```

### 🐛 Other Issues?

```bash
# Check logs
tail -f logs/pipeline.log

# View full state
cat logs/pipeline_state.json | python -m json.tool

# Start fresh (removes state)
rm logs/pipeline_state.json
./run_pipeline.sh --full
```

## 📈 Monitor Progress

```bash
# Interactive progress viewer
python main.py --interactive
# Choose option 8

# Watch live logs
tail -f logs/pipeline.log

# Check state file
cat logs/pipeline_state.json
```

## ✅ Success Criteria

After running `python main.py --full`, you should see:

✓ All 5 stages completed with ✅  
✓ `metadata/curated_catalog.json` created  
✓ `notebooks_curated/` directory with organized notebooks  
✓ `reports/summary.json` with statistics  
✓ `USAGE.md` and `PROVENANCE.json` generated  
✓ Beautiful completion banner with timing stats  

## � You're Ready!

**First Time Setup:**
```bash
# 1. Activate venv
source venv/bin/activate

# 2. Check prerequisites
python main.py --interactive  # Option 9

# 3. Run test (small dataset ~30 min)
# Edit config/config.yaml:
#   - collection.target_count: 20
#   - indexing.target_count: 100
./run_pipeline.sh --full

# 4. Run production (full dataset ~12-15 hours)
# Edit config/config.yaml back to:
#   - collection.target_count: 500
#   - indexing.target_count: 5000
./run_pipeline.sh --full
```

**Production Tips:**
- ✅ **Always use** `./run_pipeline.sh` (handles venv automatically)
- Use `screen` or `tmux` for long-running pipelines
- Monitor with: `tail -f logs/pipeline.log`
- Ctrl+C saves state - resume anytime with: `./run_pipeline.sh --resume`
- Check progress: `./run_pipeline.sh --interactive` → Option 8

**Quick Reference:**
```bash
./run_pipeline.sh --interactive    # Interactive menu (recommended)
./run_pipeline.sh --full           # Run complete pipeline
./run_pipeline.sh --resume         # Resume after interruption
./run_pipeline.sh --stage <name>   # Run specific stage
./run_pipeline.sh --dry-run --full # Preview execution plan
```

Happy curating! 🚀📊

---

**Need detailed docs?** See `README.md` for full documentation.
