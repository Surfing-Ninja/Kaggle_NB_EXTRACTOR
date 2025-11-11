#!/usr/bin/env python3
"""
Main orchestrator for Kaggle 5K Notebook Extraction Project
Coordinates collection, downloading, and analysis pipelines.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.collector import KaggleCollector
from src.downloader import KaggleDownloader
from src.analyzer import NotebookAnalyzer
from src.utils import setup_logger, load_config, Timer


class Kaggle5KOrchestrator:
    """Main orchestrator for the entire pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "orchestrator",
            log_dir=self.config['storage']['logs_dir'],
            level=self.config['logging']['level'],
            format_type="text"  # Use text for main orchestrator
        )
        
        # Initialize components
        self.collector = KaggleCollector(config_path)
        self.downloader = KaggleDownloader(config_path)
        self.analyzer = NotebookAnalyzer(config_path)
        
        self.logger.info("Kaggle5K Orchestrator initialized")
    
    def run_collection(
        self,
        max_competitions: Optional[int] = None,
        kernels_per_comp: Optional[int] = None
    ):
        """
        Run the metadata collection phase.
        
        Args:
            max_competitions: Maximum competitions to process
            kernels_per_comp: Maximum kernels per competition
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: METADATA COLLECTION")
        self.logger.info("=" * 60)
        
        with Timer("Collection Phase", self.logger):
            summary = self.collector.collect_all_metadata(
                max_competitions=max_competitions,
                kernels_per_competition=kernels_per_comp
            )
        
        self.logger.info(f"Collected metadata for {summary['total_kernels']} kernels")
        
        # Export download list
        download_list = self.collector.export_download_list()
        self.logger.info(f"Download list saved to: {download_list}")
        
        return summary
    
    def run_download(
        self,
        limit: Optional[int] = None,
        metadata_file: Optional[str] = None
    ):
        """
        Run the download phase.
        
        Args:
            limit: Maximum notebooks to download
            metadata_file: Path to metadata file
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 2: NOTEBOOK DOWNLOAD")
        self.logger.info("=" * 60)
        
        with Timer("Download Phase", self.logger):
            summary = self.downloader.download_from_metadata(
                metadata_file=metadata_file,
                limit=limit
            )
        
        self.logger.info(
            f"Downloaded {summary['success_count']} notebooks "
            f"(failed: {summary['failed_count']}, "
            f"skipped: {summary['skipped_count']})"
        )
        
        return summary
    
    def run_analysis(self):
        """Run the analysis phase."""
        self.logger.info("=" * 60)
        self.logger.info("PHASE 3: NOTEBOOK ANALYSIS")
        self.logger.info("=" * 60)
        
        with Timer("Analysis Phase", self.logger):
            summary = self.analyzer.analyze_all_notebooks()
        
        self.logger.info(
            f"Analyzed {summary['total_analyzed']} notebooks, "
            f"selected {summary['selected_count']} "
            f"({summary['selection_rate']:.1%})"
        )
        
        return summary
    
    def run_selection(self, target_count: int = 5000):
        """
        Run the final selection phase.
        
        Args:
            target_count: Number of notebooks to select
        """
        self.logger.info("=" * 60)
        self.logger.info("PHASE 4: FINAL SELECTION")
        self.logger.info("=" * 60)
        
        with Timer("Selection Phase", self.logger):
            selection = self.analyzer.select_top_notebooks(target_count=target_count)
            
            # Copy to curated directory
            self.analyzer.copy_curated_notebooks(selection)
        
        self.logger.info(f"Selected and copied {len(selection)} top notebooks")
        
        return selection
    
    def run_full_pipeline(
        self,
        max_competitions: Optional[int] = None,
        download_limit: Optional[int] = None,
        target_count: int = 5000
    ):
        """
        Run the complete pipeline from start to finish.
        
        Args:
            max_competitions: Maximum competitions to collect from
            download_limit: Maximum notebooks to download
            target_count: Final number of notebooks to select
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("KAGGLE 5K NOTEBOOK EXTRACTION - FULL PIPELINE")
        self.logger.info("=" * 60 + "\n")
        
        try:
            # Phase 1: Collection
            collection_summary = self.run_collection(
                max_competitions=max_competitions,
                kernels_per_comp=self.config.get('notebooks_per_competition', 10)
            )
            
            # Phase 2: Download
            download_summary = self.run_download(limit=download_limit)
            
            # Phase 3: Analysis
            analysis_summary = self.run_analysis()
            
            # Phase 4: Selection
            selection = self.run_selection(target_count=target_count)
            
            # Final Summary
            self.logger.info("\n" + "=" * 60)
            self.logger.info("PIPELINE COMPLETE - FINAL SUMMARY")
            self.logger.info("=" * 60)
            self.logger.info(f"Competitions Processed: {collection_summary['total_competitions']}")
            self.logger.info(f"Kernels Collected: {collection_summary['total_kernels']}")
            self.logger.info(f"Notebooks Downloaded: {download_summary['success_count']}")
            self.logger.info(f"Notebooks Analyzed: {analysis_summary['total_analyzed']}")
            self.logger.info(f"Final Selection: {len(selection)}")
            self.logger.info(f"Selection Rate: {len(selection) / analysis_summary['total_analyzed']:.1%}")
            self.logger.info("=" * 60 + "\n")
            
            return {
                'collection': collection_summary,
                'download': download_summary,
                'analysis': analysis_summary,
                'selection': len(selection),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Kaggle 5K Notebook Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --full --target 5000
  
  # Run individual phases
  python main.py --collect --max-competitions 50
  python main.py --download --limit 1000
  python main.py --analyze
  python main.py --select --target 5000
  
  # Quick test run
  python main.py --full --max-competitions 5 --download-limit 50 --target 50
        """
    )
    
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    
    # Pipeline modes
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--collect', action='store_true', help='Run collection phase only')
    parser.add_argument('--download', action='store_true', help='Run download phase only')
    parser.add_argument('--analyze', action='store_true', help='Run analysis phase only')
    parser.add_argument('--select', action='store_true', help='Run selection phase only')
    
    # Parameters
    parser.add_argument('--max-competitions', type=int, help='Maximum competitions to process')
    parser.add_argument('--download-limit', type=int, help='Maximum notebooks to download')
    parser.add_argument('--target', type=int, default=5000, help='Target number of notebooks')
    parser.add_argument('--metadata', help='Path to metadata file for downloading')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = Kaggle5KOrchestrator(config_path=args.config)
    
    # Run appropriate phase(s)
    if args.full:
        result = orchestrator.run_full_pipeline(
            max_competitions=args.max_competitions,
            download_limit=args.download_limit,
            target_count=args.target
        )
        sys.exit(0 if result.get('success') else 1)
    
    elif args.collect:
        orchestrator.run_collection(
            max_competitions=args.max_competitions
        )
    
    elif args.download:
        orchestrator.run_download(
            limit=args.download_limit,
            metadata_file=args.metadata
        )
    
    elif args.analyze:
        orchestrator.run_analysis()
    
    elif args.select:
        orchestrator.run_selection(target_count=args.target)
    
    else:
        parser.print_help()
        print("\nError: Please specify a mode (--full, --collect, --download, --analyze, or --select)")
        sys.exit(1)


if __name__ == "__main__":
    main()
