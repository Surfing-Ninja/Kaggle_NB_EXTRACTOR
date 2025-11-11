#!/usr/bin/env python3
"""
Helper script for common project tasks and utilities.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import load_json, load_config


def show_stats():
    """Show current project statistics."""
    print("\n" + "=" * 60)
    print("KAGGLE 5K PROJECT - STATISTICS")
    print("=" * 60 + "\n")
    
    metadata_dir = Path("metadata")
    notebooks_dir = Path("notebooks")
    curated_dir = Path("notebooks_curated")
    logs_dir = Path("logs")
    
    # Collection stats
    kernels_file = metadata_dir / "kernels" / "all_kernels_metadata.json"
    if kernels_file.exists():
        kernels = load_json(kernels_file)
        print(f"📊 Kernels Collected: {len(kernels)}")
    else:
        print("📊 Kernels Collected: 0 (not yet collected)")
    
    # Download stats
    download_progress = metadata_dir / "download_progress.json"
    if download_progress.exists():
        progress = load_json(download_progress)
        print(f"⬇️  Downloads Completed: {len(progress.get('completed', []))}")
        print(f"❌ Downloads Failed: {len(progress.get('failed', []))}")
    else:
        print("⬇️  Downloads: Not started")
    
    # Downloaded notebooks count
    notebook_files = list(notebooks_dir.rglob('*.ipynb')) if notebooks_dir.exists() else []
    total_size = sum(f.stat().st_size for f in notebook_files) / (1024**2)  # MB
    print(f"📔 Notebooks Downloaded: {len(notebook_files)} ({total_size:.1f} MB)")
    
    # Analysis stats
    analysis_file = metadata_dir / "analysis" / "all_notebooks_analysis.json"
    if analysis_file.exists():
        analyses = load_json(analysis_file)
        selected = sum(1 for a in analyses if a.get('selected', False))
        avg_quality = sum(a.get('quality_score', 0) for a in analyses) / len(analyses)
        print(f"🔍 Notebooks Analyzed: {len(analyses)}")
        print(f"✅ Passed Quality Filters: {selected}")
        print(f"📈 Average Quality Score: {avg_quality:.2f}")
    else:
        print("🔍 Analysis: Not yet run")
    
    # Curated notebooks
    curated_files = list(curated_dir.rglob('*.ipynb')) if curated_dir.exists() else []
    print(f"⭐ Curated Notebooks: {len(curated_files)}")
    
    # Logs
    log_files = list(logs_dir.glob('*.log')) if logs_dir.exists() else []
    log_size = sum(f.stat().st_size for f in log_files) / (1024**2)  # MB
    print(f"📝 Log Files: {len(log_files)} ({log_size:.1f} MB)")
    
    print("\n" + "=" * 60 + "\n")


def show_progress():
    """Show current pipeline progress."""
    print("\n" + "=" * 60)
    print("PIPELINE PROGRESS")
    print("=" * 60 + "\n")
    
    metadata_dir = Path("metadata")
    
    # Phase 1: Collection
    collection_summary = metadata_dir / "collection_summary.json"
    if collection_summary.exists():
        summary = load_json(collection_summary)
        print("✅ Phase 1: Collection - COMPLETE")
        print(f"   Competitions: {summary.get('total_competitions', 0)}")
        print(f"   Kernels: {summary.get('total_kernels', 0)}")
        print(f"   Timestamp: {summary.get('timestamp', 'N/A')}")
    else:
        print("⏳ Phase 1: Collection - NOT STARTED")
    print()
    
    # Phase 2: Download
    download_progress = metadata_dir / "download_progress.json"
    if download_progress.exists():
        progress = load_json(download_progress)
        completed = len(progress.get('completed', []))
        failed = len(progress.get('failed', []))
        total = completed + failed
        print("🔄 Phase 2: Download - IN PROGRESS / PAUSED")
        print(f"   Completed: {completed}")
        print(f"   Failed: {failed}")
        print(f"   Total Processed: {total}")
        if total > 0:
            success_rate = completed / total * 100
            print(f"   Success Rate: {success_rate:.1f}%")
    else:
        print("⏳ Phase 2: Download - NOT STARTED")
    print()
    
    # Phase 3: Analysis
    analysis_summary = metadata_dir / "analysis" / "analysis_summary.json"
    if analysis_summary.exists():
        summary = load_json(analysis_summary)
        print("✅ Phase 3: Analysis - COMPLETE")
        print(f"   Analyzed: {summary.get('total_analyzed', 0)}")
        print(f"   Selected: {summary.get('selected_count', 0)}")
        print(f"   Selection Rate: {summary.get('selection_rate', 0):.1%}")
        print(f"   Timestamp: {summary.get('timestamp', 'N/A')}")
    else:
        print("⏳ Phase 3: Analysis - NOT STARTED")
    print()
    
    # Phase 4: Selection
    curated_dir = Path("notebooks_curated")
    curated_files = list(curated_dir.rglob('*.ipynb')) if curated_dir.exists() else []
    if curated_files:
        print(f"✅ Phase 4: Selection - COMPLETE")
        print(f"   Curated Notebooks: {len(curated_files)}")
    else:
        print("⏳ Phase 4: Selection - NOT STARTED")
    
    print("\n" + "=" * 60 + "\n")


def show_top_notebooks(limit=10):
    """Show top N notebooks by quality score."""
    analysis_file = Path("metadata/analysis/all_notebooks_analysis.json")
    
    if not analysis_file.exists():
        print("❌ Analysis not yet run. Use: python main.py --analyze")
        return
    
    analyses = load_json(analysis_file)
    
    # Sort by quality score
    sorted_notebooks = sorted(
        analyses,
        key=lambda x: x.get('quality_score', 0),
        reverse=True
    )[:limit]
    
    print("\n" + "=" * 60)
    print(f"TOP {limit} NOTEBOOKS BY QUALITY SCORE")
    print("=" * 60 + "\n")
    
    for i, nb in enumerate(sorted_notebooks, 1):
        print(f"{i}. {nb.get('filename', 'Unknown')}")
        print(f"   Quality Score: {nb.get('quality_score', 0):.2f}")
        print(f"   Keyword Score: {nb.get('total_keyword_score', 0):.2f}")
        print(f"   Code Cells: {nb.get('code_cells', 0)}")
        print(f"   Has Outputs: {nb.get('has_outputs', False)}")
        print()
    
    print("=" * 60 + "\n")


def show_category_distribution():
    """Show distribution of notebooks by keyword categories."""
    analysis_file = Path("metadata/analysis/all_notebooks_analysis.json")
    
    if not analysis_file.exists():
        print("❌ Analysis not yet run. Use: python main.py --analyze")
        return
    
    analyses = load_json(analysis_file)
    
    # Count by category
    from collections import defaultdict
    category_counts = defaultdict(int)
    
    for nb in analyses:
        if not nb.get('selected', False):
            continue
        
        keyword_scores = nb.get('keyword_scores', {})
        for category, score in keyword_scores.items():
            if score > 0:
                category_counts[category] += 1
    
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION (Selected Notebooks)")
    print("=" * 60 + "\n")
    
    # Sort by count
    sorted_categories = sorted(
        category_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for category, count in sorted_categories:
        bar = "█" * (count // 50)  # Scale for display
        print(f"{category:25s}: {count:4d} {bar}")
    
    print("\n" + "=" * 60 + "\n")


def show_config():
    """Show current configuration."""
    config = load_config("config/config.yaml")
    
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60 + "\n")
    
    print("🎯 Target Settings:")
    print(f"   Target Notebooks: {config.get('target_notebooks', 'N/A')}")
    print(f"   Notebooks per Competition: {config.get('notebooks_per_competition', 'N/A')}")
    print(f"   Min Votes: {config.get('min_votes', 'N/A')}")
    print()
    
    print("⚡ Performance:")
    print(f"   Max Workers: {config.get('max_workers', 'N/A')}")
    print(f"   Rate Limit Delay: {config.get('rate_limit_delay', 'N/A')}s")
    print(f"   Max Retries: {config.get('max_retries', 'N/A')}")
    print(f"   Timeout: {config.get('timeout', 'N/A')}s")
    print()
    
    quality_filters = config.get('quality_filters', {})
    print("✅ Quality Filters:")
    print(f"   Min Code Cells: {quality_filters.get('min_code_cells', 'N/A')}")
    print(f"   Min Markdown Cells: {quality_filters.get('min_markdown_cells', 'N/A')}")
    print(f"   Min Code Length: {quality_filters.get('min_code_length', 'N/A')}")
    print(f"   Min Unique Imports: {quality_filters.get('min_unique_imports', 'N/A')}")
    print()
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Helper utilities for Kaggle 5K project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python helper.py --stats              Show project statistics
  python helper.py --progress           Show pipeline progress
  python helper.py --top 20             Show top 20 notebooks
  python helper.py --categories         Show category distribution
  python helper.py --config             Show current configuration
        """
    )
    
    parser.add_argument('--stats', action='store_true', help='Show project statistics')
    parser.add_argument('--progress', action='store_true', help='Show pipeline progress')
    parser.add_argument('--top', type=int, metavar='N', help='Show top N notebooks')
    parser.add_argument('--categories', action='store_true', help='Show category distribution')
    parser.add_argument('--config', action='store_true', help='Show current configuration')
    
    args = parser.parse_args()
    
    if not any([args.stats, args.progress, args.top, args.categories, args.config]):
        parser.print_help()
        return
    
    if args.stats:
        show_stats()
    
    if args.progress:
        show_progress()
    
    if args.top:
        show_top_notebooks(limit=args.top)
    
    if args.categories:
        show_category_distribution()
    
    if args.config:
        show_config()


if __name__ == "__main__":
    main()
