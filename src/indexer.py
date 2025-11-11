"""
Kernel Indexer - Build master index of notebooks with intelligent filtering
Collects and scores kernels from selected competitions for download.
"""
import argparse
import subprocess
import json
import csv
import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import statistics

from .utils import (
    setup_logger,
    load_config,
    load_filters,
    load_json,
    save_json,
    retry_with_backoff,
    Timer,
    ensure_dir
)


class KernelIndexer:
    """
    Build comprehensive index of Kaggle kernels with intelligent filtering.
    
    This class fetches kernels from selected competitions, enriches them with
    semantic metadata from titles, scores them for quality, and selects the
    best candidates for downloading.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the KernelIndexer.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = load_config(config_path)
        self.filters = load_filters()
        
        # Setup directories
        self.metadata_dir = Path(self.config['storage']['metadata_dir'])
        self.kernels_dir = self.metadata_dir / "kernels"
        ensure_dir(self.kernels_dir)
        
        # Setup logging
        self.logger = setup_logger(
            name="kernel_indexer",
            log_dir=self.config['storage']['logs_dir'],
            level=self.config['logging']['level'],
            format_type=self.config['logging']['format']
        )
        
        # Statistics tracking
        self.stats = {
            'api_calls': 0,
            'api_failures': 0,
            'kernels_found': [],
            'language_counts': Counter(),
            'vote_distribution': [],
            'keyword_matches': defaultdict(int),
            'failed_competitions': []
        }
        
        self.logger.info("KernelIndexer initialized", extra={
            "config_path": config_path,
            "metadata_dir": str(self.metadata_dir)
        })
    
    @retry_with_backoff(max_tries=3)
    def fetch_kernels_for_competition(
        self, 
        competition_ref: str, 
        limit: int = 250  # Increased from 15 to get more kernels per competition
    ) -> List[Dict[str, Any]]:
        """
        Get top kernels for a single competition.
        
        Args:
            competition_ref: Competition reference/slug (e.g., 'titanic')
            limit: Maximum number of kernels to fetch
            
        Returns:
            List of kernel dictionaries with metadata
        """
        # Extract competition slug from reference
        if '/' in competition_ref:
            slug = competition_ref.split('/')[-1]
        else:
            slug = competition_ref
        
        self.logger.debug(f"Fetching kernels for competition: {slug}")
        
        try:
            # Execute Kaggle CLI command
            cmd = [
                "kaggle", "kernels", "list",
                "--competition", slug,
                "--sort-by", "voteCount",
                "--page-size", str(limit),
                "--csv"
            ]
            
            self.stats['api_calls'] += 1
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False  # Don't raise on non-zero exit
            )
            
            # Rate limiting
            time.sleep(1.5)
            
            # Debug logging
            self.logger.debug(f"Kaggle command for {slug}: returncode={result.returncode}, stdout_len={len(result.stdout)}, stderr={result.stderr[:100] if result.stderr else 'none'}")
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                self.logger.warning(f"Failed to fetch kernels for {slug}: {error_msg}")
                self.stats['api_failures'] += 1
                self.stats['failed_competitions'].append(slug)
                return []
            
            # Parse CSV output
            csv_data = result.stdout
            if not csv_data.strip() or csv_data.count('\n') <= 1:
                self.logger.info(f"No kernels found for {slug}")
                return []
            
            # Convert CSV to list of dictionaries
            csv_reader = csv.DictReader(io.StringIO(csv_data))
            kernels = []
            
            for row in csv_reader:
                # Parse votes
                try:
                    votes = int(row.get('totalVotes', '0'))
                except ValueError:
                    votes = 0
                
                # Note: Language info not available in CSV, will be 'unknown' until detailed fetch
                kernel = {
                    'kernel_id': row.get('ref', ''),
                    'title': row.get('title', ''),
                    'author': row.get('author', ''),
                    'language': 'unknown',  # Will be determined later if needed
                    'votes': votes,
                    'lastRunTime': row.get('lastRunTime', ''),
                    'competition': slug,
                    'url': f"https://www.kaggle.com/{row.get('ref', '')}"
                }
                
                kernels.append(kernel)
                self.stats['vote_distribution'].append(votes)
            
            self.stats['kernels_found'].append(len(kernels))
            
            self.logger.info(f"Found {len(kernels)} kernels for {slug}")
            
            return kernels
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout fetching kernels for {slug}")
            self.stats['api_failures'] += 1
            self.stats['failed_competitions'].append(slug)
            return []
        except Exception as e:
            self.logger.error(f"Error fetching kernels for {slug}: {str(e)}")
            self.stats['api_failures'] += 1
            self.stats['failed_competitions'].append(slug)
            return []
    
    def extract_metadata_from_title(self, title: str) -> Dict[str, Any]:
        """
        Extract semantic information from notebook title.
        
        Args:
            title: Notebook title
            
        Returns:
            Dictionary with keyword score and detected techniques
        """
        title_lower = title.lower()
        
        keyword_score = 0
        category_hits = defaultdict(int)
        detected_techniques = []
        
        # Score by category
        category_weights = {
            'feature_engineering': 3,
            'hyperparameter': 3,
            'eda': 2,
            'dimensionality': 2,
            'preprocessing': 1,
            'advanced_ml': 2,
            'ensemble': 2
        }
        
        keywords_data = self.filters.get('keywords', {})
        
        for category, weight in category_weights.items():
            category_keywords = keywords_data.get(category, {})
            terms = category_keywords.get('terms', [])
            
            for term in terms:
                term_lower = term.lower()
                if term_lower in title_lower:
                    keyword_score += weight
                    category_hits[category] += 1
                    detected_techniques.append(term)
                    self.stats['keyword_matches'][category] += 1
        
        return {
            'keyword_score': keyword_score,
            'category_hits': dict(category_hits),
            'detected_techniques': list(set(detected_techniques))  # Remove duplicates
        }
    
    def enrich_kernel_metadata(self, kernel_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add semantic scoring to kernel metadata.
        
        Args:
            kernel_dict: Basic kernel metadata
            
        Returns:
            Enriched kernel dictionary with quality scores
        """
        # Extract metadata from title
        title_metadata = self.extract_metadata_from_title(kernel_dict['title'])
        
        # Calculate quality score
        votes = kernel_dict.get('votes', 0)
        keyword_score = title_metadata['keyword_score']
        quality_score = votes + (keyword_score * 2)
        
        # Add enrichment
        enriched = kernel_dict.copy()
        enriched.update({
            'keyword_score': keyword_score,
            'category_hits': title_metadata['category_hits'],
            'detected_techniques': title_metadata['detected_techniques'],
            'quality_score': quality_score
        })
        
        return enriched
    
    def build_master_index(
        self, 
        competitions: List[Dict[str, Any]], 
        batch_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Build complete index of all kernels from all competitions.
        
        Args:
            competitions: List of competition dictionaries
            batch_size: Number of competitions to process before saving checkpoint
            
        Returns:
            Complete list of enriched kernels
        """
        self.logger.info(f"Building master index from {len(competitions)} competitions")
        
        print(f"\n{'='*70}")
        print(f"BUILDING KERNEL MASTER INDEX")
        print(f"{'='*70}")
        print(f"Competitions to process: {len(competitions)}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")
        
        master_list = []
        
        for i, comp in enumerate(competitions, 1):
            comp_ref = comp.get('ref', '')
            comp_slug = comp_ref.split('/')[-1] if '/' in comp_ref else comp_ref
            
            print(f"[{i}/{len(competitions)}] Processing: {comp_slug[:50]}", end=' ... ')
            
            # Fetch kernels (increased from 15 to 250 to get more kernels)
            kernels = self.fetch_kernels_for_competition(comp_ref, limit=250)
            
            if not kernels:
                print("❌ No kernels found")
                continue
            
            # Enrich each kernel
            enriched_kernels = []
            for kernel in kernels:
                enriched = self.enrich_kernel_metadata(kernel)
                enriched_kernels.append(enriched)
                master_list.append(enriched)
            
            # Save per-competition cache
            cache_file = self.kernels_dir / f"{comp_slug}.json"
            save_json(enriched_kernels, cache_file)
            
            print(f"✓ Found {len(kernels)} kernels")
            
            # Batch checkpoint
            if i % batch_size == 0:
                checkpoint_file = self.kernels_dir / "kernels_checkpoint.json"
                save_json(master_list, checkpoint_file)
                
                # Print batch statistics
                print(f"\n{'-'*70}")
                print(f"BATCH CHECKPOINT - {i}/{len(competitions)} competitions processed")
                print(f"{'-'*70}")
                print(f"Total kernels collected: {len(master_list)}")
                
                if self.stats['vote_distribution']:
                    print(f"Average votes per kernel: {statistics.mean(self.stats['vote_distribution']):.1f}")
                
                print(f"Language distribution:")
                for lang, count in self.stats['language_counts'].most_common():
                    print(f"  {lang}: {count}")
                
                print(f"{'-'*70}\n")
        
        # Final save
        final_file = self.metadata_dir / "kernels_master.json"
        save_json(master_list, final_file)
        
        self.logger.info(f"Master index built: {len(master_list)} kernels")
        
        print(f"\n{'='*70}")
        print(f"✅ MASTER INDEX COMPLETE")
        print(f"{'='*70}")
        print(f"Total kernels collected: {len(master_list)}")
        print(f"Saved to: {final_file}")
        print(f"{'='*70}\n")
        
        return master_list
    
    def filter_by_quality(
        self, 
        all_kernels: List[Dict[str, Any]], 
        target_count: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Select top N notebooks based on quality criteria.
        
        Args:
            all_kernels: Complete list of kernels
            target_count: Number of notebooks to select
            
        Returns:
            Filtered list of top notebooks
        """
        self.logger.info(f"Filtering {len(all_kernels)} kernels to select top {target_count}")
        
        print(f"\n{'='*70}")
        print(f"FILTERING KERNELS BY QUALITY")
        print(f"{'='*70}")
        print(f"Total candidates: {len(all_kernels)}")
        print(f"Target count: {target_count}")
        print(f"{'='*70}\n")
        
        min_votes = self.config.get('min_votes', 3)
        
        # Filter step 1: Apply basic filters
        print("🔍 Applying quality filters...")
        filtered = []
        
        for kernel in all_kernels:
            # Must have votes OR keyword score
            votes = kernel.get('votes', 0)
            keyword_score = kernel.get('keyword_score', 0)
            
            if votes < min_votes and keyword_score < 5:
                continue
            
            # Title must not be generic
            title = kernel.get('title', '').lower()
            if not title or title in ['untitled', 'copy of', 'fork of']:
                continue
            
            if any(x in title for x in ['untitled', 'copy of', 'fork of', 'test notebook']):
                continue
            
            # Check last run time (within 10 years)
            try:
                last_run = kernel.get('lastRunTime', '')
                if last_run:
                    run_date = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
                    if datetime.now() - run_date > timedelta(days=3650):
                        continue
            except:
                pass  # If parsing fails, keep the kernel
            
            filtered.append(kernel)
        
        print(f"✓ {len(filtered)} kernels passed basic filters")
        
        # Calculate enhanced quality scores with bonuses
        print("\n📊 Calculating quality scores with bonuses...")
        
        for kernel in filtered:
            votes = kernel.get('votes', 0)
            keyword_score = kernel.get('keyword_score', 0)
            title_lower = kernel.get('title', '').lower()
            
            # Base score
            quality_score = (votes * 1.0) + (keyword_score * 2.5)
            
            # Bonuses
            if 'feature' in title_lower and 'engineering' in title_lower:
                quality_score += 10
            
            if any(x in title_lower for x in ['hyperparameter', 'optuna', 'hyperopt']):
                quality_score += 8
            
            if any(x in title_lower for x in ['pca', 'tsne', 't-sne', 'umap']):
                quality_score += 5
            
            kernel['quality_score'] = round(quality_score, 2)
        
        # Sort by quality score
        filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Ensure diversity: max 250 notebooks per competition (increased from 20)
        print("\n🎯 Ensuring diversity across competitions...")
        
        selected = []
        comp_counts = defaultdict(int)
        
        for kernel in filtered:
            if len(selected) >= target_count:
                break
            
            comp = kernel.get('competition', '')
            if comp_counts[comp] >= 250:
                continue
            
            selected.append(kernel)
            comp_counts[comp] += 1
        
        # Save filtered list
        output_file = self.metadata_dir / "kernels_filtered.json"
        save_json(selected, output_file)
        
        # Print selection statistics
        print(f"\n{'='*70}")
        print(f"SELECTION STATISTICS")
        print(f"{'='*70}")
        print(f"Total candidates considered: {len(all_kernels)}")
        print(f"Passed quality filters: {len(filtered)}")
        print(f"Final selected count: {len(selected)}")
        print(f"Competitions represented: {len(comp_counts)}")
        
        if selected:
            quality_scores = [k['quality_score'] for k in selected]
            print(f"\nQuality Score Distribution:")
            print(f"  Minimum: {min(quality_scores):.1f}")
            print(f"  Maximum: {max(quality_scores):.1f}")
            print(f"  Median: {statistics.median(quality_scores):.1f}")
            print(f"  Mean: {statistics.mean(quality_scores):.1f}")
            
            votes = [k['votes'] for k in selected]
            print(f"\nVote Distribution:")
            print(f"  Minimum: {min(votes)}")
            print(f"  Maximum: {max(votes)}")
            print(f"  Median: {statistics.median(votes):.1f}")
            print(f"  Mean: {statistics.mean(votes):.1f}")
            
            # Category distribution
            all_categories = []
            for kernel in selected:
                all_categories.extend(kernel.get('category_hits', {}).keys())
            
            if all_categories:
                print(f"\nCategory Distribution:")
                cat_counts = Counter(all_categories)
                for cat, count in cat_counts.most_common(10):
                    print(f"  {cat}: {count}")
        
        print(f"\nSaved to: {output_file}")
        print(f"{'='*70}\n")
        
        self.logger.info(f"Selected {len(selected)} kernels from {len(all_kernels)} candidates")
        
        return selected
    
    def generate_download_manifest(
        self, 
        filtered_kernels: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create structured download plan.
        
        Args:
            filtered_kernels: List of filtered kernels to download
            
        Returns:
            Manifest dictionary with download plan
        """
        self.logger.info("Generating download manifest")
        
        # Group by competition
        by_competition = defaultdict(list)
        for kernel in filtered_kernels:
            comp = kernel.get('competition', 'unknown')
            by_competition[comp].append(kernel)
        
        # Sort competitions by average quality score
        comp_priorities = []
        for comp, kernels in by_competition.items():
            avg_score = statistics.mean([k['quality_score'] for k in kernels])
            comp_priorities.append((comp, avg_score, kernels))
        
        comp_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Build download order
        download_order = []
        for priority, (comp, avg_score, kernels) in enumerate(comp_priorities, 1):
            download_order.append({
                'competition': comp,
                'priority': priority,
                'kernel_count': len(kernels),
                'avg_quality_score': round(avg_score, 2),
                'kernels': [
                    {
                        'kernel_id': k['kernel_id'],
                        'title': k['title'],
                        'quality_score': k['quality_score'],
                        'votes': k['votes']
                    }
                    for k in kernels
                ]
            })
        
        # Estimate time and size
        rate_limit_delay = self.config.get('rate_limit_delay', 2)
        estimated_time_seconds = len(filtered_kernels) * rate_limit_delay
        estimated_time_hours = estimated_time_seconds / 3600
        
        # Estimate size (average notebook ~200KB, conservative estimate)
        estimated_size_gb = (len(filtered_kernels) * 0.2) / 1024
        
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'total_notebooks': len(filtered_kernels),
            'competitions': len(by_competition),
            'estimated_time_hours': round(estimated_time_hours, 2),
            'estimated_size_gb': round(estimated_size_gb, 2),
            'rate_limit_delay': rate_limit_delay,
            'download_order': download_order
        }
        
        # Save manifest
        manifest_file = self.metadata_dir / "download_manifest.json"
        save_json(manifest, manifest_file)
        
        print(f"\n{'='*70}")
        print(f"DOWNLOAD MANIFEST GENERATED")
        print(f"{'='*70}")
        print(f"Total notebooks: {manifest['total_notebooks']}")
        print(f"Competitions: {manifest['competitions']}")
        print(f"Estimated time: {manifest['estimated_time_hours']:.1f} hours")
        print(f"Estimated size: {manifest['estimated_size_gb']:.1f} GB")
        print(f"\nManifest saved to: {manifest_file}")
        print(f"{'='*70}\n")
        
        self.logger.info("Download manifest generated", extra=manifest)
        
        return manifest
    
    def run(
        self, 
        competitions_file: Optional[str] = None,
        target_count: int = 5000
    ) -> Dict[str, Any]:
        """
        Main orchestration method.
        
        Args:
            competitions_file: Path to selected competitions JSON
            target_count: Number of notebooks to select
            
        Returns:
            Summary dictionary
        """
        start_time = datetime.now()
        
        self.logger.info("Starting kernel indexing run", extra={
            "target_count": target_count,
            "timestamp": start_time.isoformat()
        })
        
        print(f"\n{'='*70}")
        print(f"KAGGLE KERNEL INDEXER")
        print(f"{'='*70}")
        print(f"Target notebooks: {target_count}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Load selected competitions
            if competitions_file is None:
                competitions_file = str(self.metadata_dir / "competitions" / "selected_competitions.json")
            
            print(f"📂 Loading competitions from: {competitions_file}")
            competitions = load_json(Path(competitions_file))
            print(f"✓ Loaded {len(competitions)} competitions\n")
            
            # Step 2: Build master index
            print("🔨 Building master index...")
            with Timer() as timer:
                all_kernels = self.build_master_index(competitions, batch_size=50)
            print(f"✓ Index built in {timer.elapsed:.1f} seconds\n")
            
            # Step 3: Filter by quality
            print("🎯 Filtering by quality...")
            with Timer() as timer:
                filtered_kernels = self.filter_by_quality(all_kernels, target_count)
            print(f"✓ Filtered in {timer.elapsed:.1f} seconds\n")
            
            # Step 4: Generate download manifest
            print("📝 Generating download manifest...")
            manifest = self.generate_download_manifest(filtered_kernels)
            
            # Step 5: Print top notebooks
            print(f"\n{'='*70}")
            print(f"TOP 20 NOTEBOOKS BY QUALITY SCORE")
            print(f"{'='*70}")
            
            for i, kernel in enumerate(filtered_kernels[:20], 1):
                title = kernel['title'][:50]
                score = kernel['quality_score']
                votes = kernel['votes']
                comp = kernel['competition'][:20]
                print(f"{i:2d}. [{score:6.1f}] {title:50s} | Votes: {votes:4d} | {comp}")
            
            print(f"{'='*70}\n")
            
            # Calculate final statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': round(duration, 2),
                'competitions_processed': len(competitions),
                'total_kernels_found': len(all_kernels),
                'kernels_selected': len(filtered_kernels),
                'api_calls': self.stats['api_calls'],
                'api_failures': self.stats['api_failures'],
                'failed_competitions': self.stats['failed_competitions'],
                'manifest': manifest
            }
            
            # Save summary
            summary_file = self.metadata_dir / "indexer_summary.json"
            save_json(summary, summary_file)
            
            # Final output
            print(f"{'='*70}")
            print(f"✅ INDEXING COMPLETE")
            print(f"{'='*70}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Competitions processed: {len(competitions)}")
            print(f"Total kernels found: {len(all_kernels)}")
            print(f"Kernels selected: {len(filtered_kernels)}")
            print(f"API calls made: {self.stats['api_calls']}")
            print(f"API failures: {self.stats['api_failures']}")
            print(f"\nReady for download!")
            print(f"Next step: python main.py --download --limit {len(filtered_kernels)}")
            print(f"{'='*70}\n")
            
            self.logger.info("Kernel indexing completed successfully", extra=summary)
            
            return summary
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Kernel indexing failed: {str(e)}", extra={
                "error_type": type(e).__name__,
                "duration_seconds": duration
            })
            
            print(f"\n❌ ERROR: {str(e)}")
            print(f"Check logs for details: logs/kernel_indexer_*.log\n")
            
            raise


def main():
    """Command-line interface for kernel indexing."""
    parser = argparse.ArgumentParser(
        description="Kaggle Kernel Indexer - Build master index with intelligent filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from selected competitions (default)
  python -m src.indexer --target-count 5000
  
  # Use custom competitions file
  python -m src.indexer --competitions metadata/my_competitions.json --target-count 3000
  
  # With custom minimum votes threshold
  python -m src.indexer --target-count 5000 --min-votes 5
  
  # Custom output directory
  python -m src.indexer --target-count 5000 --output metadata/custom
        """
    )
    
    parser.add_argument(
        '--competitions',
        help='Path to selected competitions JSON file'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=5000,
        help='Number of notebooks to select (default: 5000)'
    )
    parser.add_argument(
        '--min-votes',
        type=int,
        help='Minimum votes threshold (overrides config)'
    )
    parser.add_argument(
        '--output',
        help='Output directory for metadata files'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize indexer
        indexer = KernelIndexer(config_path=args.config)
        
        # Override config if specified
        if args.min_votes is not None:
            indexer.config['min_votes'] = args.min_votes
        
        if args.output:
            indexer.metadata_dir = Path(args.output)
            indexer.kernels_dir = indexer.metadata_dir / "kernels"
            ensure_dir(indexer.kernels_dir)
        
        # Run indexing
        summary = indexer.run(
            competitions_file=args.competitions,
            target_count=args.target_count
        )
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Indexing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nCheck logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
