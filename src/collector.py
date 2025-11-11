"""
Competition and Kernel Metadata Collector
Collects metadata about competitions and their notebooks from Kaggle API.
"""
import argparse
import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import csv
import io
import time
import sys

from .utils import (
    setup_logger,
    load_config,
    load_filters,
    save_json,
    append_jsonl,
    read_jsonl,
    retry_with_backoff,
    rate_limited,
    ProgressTracker,
    Timer,
    ensure_dir
)


class CompetitionCollector:
    """
    Robust collector for Kaggle competition metadata with intelligent scoring.
    
    This class fetches all available competitions, scores them based on their
    likelihood of containing quality notebooks, and selects the best candidates
    for notebook extraction with advanced feature engineering content.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the CompetitionCollector.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = load_config(config_path)
        self.filters = load_filters()
        
        # Setup directories
        self.metadata_dir = Path(self.config['storage']['metadata_dir'])
        self.competitions_dir = self.metadata_dir / "competitions"
        ensure_dir(self.competitions_dir)
        
        # Setup logging
        self.logger = setup_logger(
            name="competition_collector",
            log_dir=self.config['storage']['logs_dir'],
            level=self.config['logging']['level'],
            format_type=self.config['logging']['format']
        )
        
        # Cache for notebook availability checks
        self.notebook_cache = {}
        self.load_cache()
        
        self.logger.info("CompetitionCollector initialized", extra={
            "config_path": config_path,
            "metadata_dir": str(self.metadata_dir)
        })
    
    def load_cache(self):
        """Load cached notebook availability checks."""
        cache_file = self.competitions_dir / "notebook_availability_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.notebook_cache = json.load(f)
                self.logger.info(f"Loaded {len(self.notebook_cache)} cached availability checks")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self.notebook_cache = {}
    
    def save_cache(self):
        """Save notebook availability cache."""
        cache_file = self.competitions_dir / "notebook_availability_cache.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.notebook_cache, f, indent=2)
            self.logger.info(f"Saved {len(self.notebook_cache)} availability checks to cache")
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
    
    @retry_with_backoff(max_tries=3)
    def fetch_all_competitions(self) -> List[Dict[str, Any]]:
        """
        Fetch complete list of all Kaggle competitions.
        
        Returns:
            List of competition dictionaries with metadata
            
        Raises:
            RuntimeError: If fetching fails after retries
        """
        self.logger.info("Starting to fetch all competitions", extra={
            "action": "fetch_competitions",
            "status": "started"
        })
        
        try:
            # Execute Kaggle CLI command
            cmd = ["kaggle", "competitions", "list", "--csv"]
            self.logger.debug(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with code {result.returncode}: {result.stderr}")
            
            # Parse CSV output
            csv_data = result.stdout
            if not csv_data.strip():
                self.logger.warning("Empty response from Kaggle API")
                return []
            
            # Convert CSV to list of dictionaries
            csv_reader = csv.DictReader(io.StringIO(csv_data))
            competitions = []
            
            for row in csv_reader:
                # Parse team count (handle commas in numbers)
                team_count_str = row.get('teamCount', '0').replace(',', '')
                try:
                    team_count = int(team_count_str)
                except ValueError:
                    team_count = 0
                
                competition = {
                    'ref': row.get('ref', ''),
                    'deadline': row.get('deadline', ''),
                    'category': row.get('category', ''),
                    'reward': row.get('reward', ''),
                    'teamCount': team_count,
                    'userHasEntered': row.get('userHasEntered', 'False') == 'True'
                }
                competitions.append(competition)
            
            # Save raw data
            output_file = self.competitions_dir / "competitions_raw.json"
            save_json(competitions, output_file)
            
            self.logger.info("Successfully fetched competitions", extra={
                "action": "fetch_competitions",
                "status": "success",
                "count": len(competitions),
                "output_file": str(output_file)
            })
            
            print(f"✓ Fetched {len(competitions)} competitions")
            
            return competitions
            
        except subprocess.TimeoutExpired:
            self.logger.error("Command timed out after 60 seconds")
            raise RuntimeError("Kaggle API request timed out")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e.stderr}", extra={
                "returncode": e.returncode,
                "stderr": e.stderr
            })
            raise RuntimeError(f"Kaggle API error: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}", extra={
                "error_type": type(e).__name__
            })
            raise
    
    def score_competition(self, competition: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Score a competition based on likelihood of quality notebooks.
        
        Scoring factors:
        - Competition category (featured, playground, research, etc.)
        - Team participation count
        - Competition recency (year)
        - Presence of reward/prize
        
        Args:
            competition: Competition dictionary with metadata
            
        Returns:
            Tuple of (score, competition_dict)
        """
        score = 0.0
        scoring_details = {}
        
        # Category-based scoring
        category = competition.get('category', '').lower()
        category_scores = {
            'featured': 100,
            'playground': 80,
            'research': 60,
            'getting started': 40,
            'getting-started': 40
        }
        
        category_bonus = category_scores.get(category, 0)
        score += category_bonus
        scoring_details['category_bonus'] = category_bonus
        
        # Team participation scoring (capped at 50 points)
        team_count = competition.get('teamCount', 0)
        participation_bonus = min(team_count / 10.0, 50)
        score += participation_bonus
        scoring_details['participation_bonus'] = round(participation_bonus, 2)
        scoring_details['team_count'] = team_count
        
        # Recency scoring based on deadline year
        try:
            deadline_str = competition.get('deadline', '')
            if deadline_str:
                # Parse date (format: YYYY-MM-DD HH:MM:SS or similar)
                year = int(deadline_str.split('-')[0])
                
                if year >= 2024:
                    recency_bonus = 30
                elif year >= 2022:
                    recency_bonus = 20
                elif year >= 2020:
                    recency_bonus = 15
                elif year >= 2018:
                    recency_bonus = 10
                else:
                    recency_bonus = 0
                
                score += recency_bonus
                scoring_details['recency_bonus'] = recency_bonus
                scoring_details['year'] = year
        except (ValueError, IndexError):
            scoring_details['recency_bonus'] = 0
        
        # Reward presence scoring
        reward = competition.get('reward', '').lower()
        has_reward = reward and reward not in ['knowledge', 'kudos', 'swag', '']
        if has_reward:
            score += 10
            scoring_details['reward_bonus'] = 10
        else:
            scoring_details['reward_bonus'] = 0
        
        # Add scoring details to competition
        competition_scored = competition.copy()
        competition_scored['score'] = round(score, 2)
        competition_scored['scoring_details'] = scoring_details
        
        return (score, competition_scored)
    
    def select_top_competitions(
        self, 
        all_competitions: List[Dict[str, Any]], 
        target_count: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Select top N competitions by score.
        
        Args:
            all_competitions: List of all competition dictionaries
            target_count: Number of top competitions to select
            
        Returns:
            List of top-scored competitions
        """
        self.logger.info(f"Selecting top {target_count} competitions from {len(all_competitions)} total")
        
        print(f"\n⚙️  Scoring {len(all_competitions)} competitions...")
        
        # Score all competitions
        scored_competitions = []
        for i, comp in enumerate(all_competitions, 1):
            score, scored_comp = self.score_competition(comp)
            scored_competitions.append(scored_comp)
            
            # Progress indicator every 100 competitions
            if i % 100 == 0:
                print(f"   Scored {i}/{len(all_competitions)} competitions...")
        
        # Sort by score descending
        scored_competitions.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Take top N
        selected = scored_competitions[:target_count]
        
        # Save selected competitions
        output_file = self.competitions_dir / "selected_competitions.json"
        save_json(selected, output_file)
        
        # Print statistics
        print(f"\n{'='*70}")
        print(f"COMPETITION SELECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total competitions analyzed: {len(all_competitions)}")
        print(f"Selected for notebook extraction: {target_count}")
        print(f"\nTop 10 Competitions by Score:")
        print(f"{'-'*70}")
        
        for i, comp in enumerate(selected[:10], 1):
            ref = comp['ref'].split('/')[-1][:40]
            score = comp.get('score', 0)
            category = comp.get('category', 'N/A')
            teams = comp.get('teamCount', 0)
            print(f"{i:2d}. {ref:40s} | Score: {score:6.1f} | {category:15s} | Teams: {teams:5d}")
        
        # Category distribution
        print(f"\n{'-'*70}")
        print(f"Category Distribution of Selected Competitions:")
        print(f"{'-'*70}")
        
        category_counts = {}
        for comp in selected:
            cat = comp.get('category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(selected)) * 100
            print(f"{cat:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # Year distribution
        print(f"\n{'-'*70}")
        print(f"Year Distribution:")
        print(f"{'-'*70}")
        
        year_counts = {}
        for comp in selected:
            try:
                year = int(comp.get('deadline', '').split('-')[0])
                year_counts[year] = year_counts.get(year, 0) + 1
            except (ValueError, IndexError):
                pass
        
        for year in sorted(year_counts.keys(), reverse=True):
            count = year_counts[year]
            percentage = (count / len(selected)) * 100
            print(f"{year}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"{'='*70}\n")
        
        # Log results
        self.logger.info("Competition selection completed", extra={
            "action": "select_competitions",
            "status": "success",
            "total_analyzed": len(all_competitions),
            "selected_count": len(selected),
            "output_file": str(output_file)
        })
        
        return selected
    
    @retry_with_backoff(max_tries=2)
    def estimate_notebook_availability(self, competition_ref: str) -> bool:
        """
        Quick check if competition has notebooks available.
        
        Args:
            competition_ref: Competition reference/slug
            
        Returns:
            True if competition has notebooks, False otherwise
        """
        # Check cache first
        if competition_ref in self.notebook_cache:
            return self.notebook_cache[competition_ref]
        
        try:
            # Extract competition slug from ref
            slug = competition_ref.split('/')[-1]
            
            # Quick check with page size 1
            cmd = ["kaggle", "kernels", "list", "-c", slug, "--page-size", "1", "--csv"]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=False  # Don't raise on non-zero exit
            )
            
            # Check if we got any results (more than just header)
            has_notebooks = result.returncode == 0 and result.stdout.count('\n') > 1
            
            # Cache result
            self.notebook_cache[competition_ref] = has_notebooks
            
            return has_notebooks
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Timeout checking notebooks for {competition_ref}")
            return False
        except Exception as e:
            self.logger.warning(f"Error checking notebooks for {competition_ref}: {e}")
            return False
    
    def run(self, target_count: int = 500, check_availability: bool = False) -> Dict[str, Any]:
        """
        Main orchestration method to run the complete collection process.
        
        Args:
            target_count: Number of top competitions to select
            check_availability: Whether to check notebook availability (slower)
            
        Returns:
            Dictionary with summary statistics
        """
        start_time = datetime.now()
        
        self.logger.info("Starting competition collection run", extra={
            "action": "run",
            "status": "started",
            "target_count": target_count,
            "check_availability": check_availability,
            "timestamp": start_time.isoformat()
        })
        
        print(f"\n{'='*70}")
        print(f"KAGGLE COMPETITION COLLECTOR")
        print(f"{'='*70}")
        print(f"Target competitions: {target_count}")
        print(f"Check availability: {check_availability}")
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Fetch all competitions
            print("📡 Step 1: Fetching all competitions from Kaggle API...")
            all_competitions = self.fetch_all_competitions()
            
            if not all_competitions:
                raise RuntimeError("No competitions fetched")
            
            # Step 2: Optional availability check
            if check_availability:
                print(f"\n🔍 Step 2: Checking notebook availability (this may take a while)...")
                available_comps = []
                
                for i, comp in enumerate(all_competitions, 1):
                    if i % 50 == 0:
                        print(f"   Checked {i}/{len(all_competitions)} competitions...")
                        self.save_cache()  # Save cache periodically
                    
                    if self.estimate_notebook_availability(comp['ref']):
                        available_comps.append(comp)
                
                self.save_cache()  # Final cache save
                
                print(f"✓ Found {len(available_comps)}/{len(all_competitions)} competitions with notebooks")
                all_competitions = available_comps
            
            # Step 3: Score and select top competitions
            print(f"\n📊 Step {'3' if check_availability else '2'}: Scoring and selecting top competitions...")
            selected_competitions = self.select_top_competitions(all_competitions, target_count)
            
            # Calculate statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": round(duration, 2),
                "total_competitions_fetched": len(all_competitions),
                "competitions_selected": len(selected_competitions),
                "check_availability_performed": check_availability,
                "output_files": {
                    "raw_competitions": str(self.competitions_dir / "competitions_raw.json"),
                    "selected_competitions": str(self.competitions_dir / "selected_competitions.json")
                }
            }
            
            # Save summary
            summary_file = self.competitions_dir / "collection_summary.json"
            save_json(summary, summary_file)
            
            # Final output
            print(f"\n{'='*70}")
            print(f"✅ COLLECTION COMPLETE")
            print(f"{'='*70}")
            print(f"Duration: {duration:.1f} seconds")
            print(f"Competitions fetched: {len(all_competitions)}")
            print(f"Competitions selected: {len(selected_competitions)}")
            print(f"\nOutput files:")
            for key, path in summary['output_files'].items():
                print(f"  • {key}: {path}")
            print(f"{'='*70}\n")
            
            self.logger.info("Competition collection completed successfully", extra={
                "action": "run",
                "status": "success",
                "duration_seconds": duration,
                **summary
            })
            
            return summary
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Competition collection failed: {str(e)}", extra={
                "action": "run",
                "status": "failed",
                "duration_seconds": duration,
                "error_type": type(e).__name__
            })
            
            print(f"\n❌ ERROR: {str(e)}")
            print(f"Check logs for details: logs/competition_collector_*.log\n")
            
            raise


class KaggleCollector:
    """Collect competition and kernel metadata from Kaggle API (Legacy collector for kernels)."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the collector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.filters = load_filters("config/filters.json")
        self.logger = setup_logger(
            "collector",
            log_dir=self.config['storage']['logs_dir'],
            level=self.config['logging']['level'],
            format_type=self.config['logging']['format']
        )
        
        # Paths
        self.metadata_dir = Path(self.config['storage']['metadata_dir'])
        self.competitions_dir = self.metadata_dir / "competitions"
        self.kernels_dir = self.metadata_dir / "kernels"
        
        # Create directories
        ensure_dir(self.competitions_dir)
        ensure_dir(self.kernels_dir)
        
        # Progress tracking
        self.progress = ProgressTracker(self.metadata_dir / "collection_progress.json")
        
        self.logger.info("KaggleCollector initialized")
    
    @retry_with_backoff(max_tries=3)
    @rate_limited(calls=20, period=60)
    def _run_kaggle_command(self, command: List[str]) -> str:
        """
        Run a Kaggle CLI command and return output.
        
        Args:
            command: List of command arguments
        
        Returns:
            Command output as string
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config['timeout'],
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Kaggle command failed: {' '.join(command)}")
            self.logger.error(f"Error: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            self.logger.error(f"Kaggle command timed out: {' '.join(command)}")
            raise
    
    def list_competitions(self) -> List[Dict[str, Any]]:
        """
        List all available competitions.
        
        Returns:
            List of competition metadata dictionaries
        """
        self.logger.info("Fetching competitions list...")
        
        try:
            # Use kaggle CLI to list competitions
            output = self._run_kaggle_command(['kaggle', 'competitions', 'list', '--csv'])
            
            # Parse CSV output
            from io import StringIO
            df = pd.read_csv(StringIO(output))
            
            competitions = df.to_dict('records')
            self.logger.info(f"Found {len(competitions)} competitions")
            
            # Save full list
            save_json(
                competitions,
                self.competitions_dir / "all_competitions.json"
            )
            
            return competitions
            
        except Exception as e:
            self.logger.error(f"Failed to list competitions: {e}")
            raise
    
    def filter_competitions(self, competitions: List[Dict]) -> List[Dict]:
        """
        Filter competitions based on configuration criteria.
        
        Args:
            competitions: List of competition metadata
        
        Returns:
            Filtered list of competitions
        """
        self.logger.info("Filtering competitions...")
        
        filters = self.config.get('competition_filters', {})
        start_date = filters.get('start_date')
        end_date = filters.get('end_date')
        
        filtered = []
        for comp in competitions:
            # Date filtering
            if start_date or end_date:
                deadline = comp.get('deadline', '')
                if deadline:
                    try:
                        comp_date = datetime.strptime(deadline.split()[0], '%Y-%m-%d')
                        if start_date and comp_date < datetime.strptime(start_date, '%Y-%m-%d'):
                            continue
                        if end_date and comp_date > datetime.strptime(end_date, '%Y-%m-%d'):
                            continue
                    except:
                        pass
            
            filtered.append(comp)
        
        self.logger.info(f"Filtered to {len(filtered)} competitions")
        
        # Save filtered list
        save_json(
            filtered,
            self.competitions_dir / "filtered_competitions.json"
        )
        
        return filtered
    
    def get_competition_kernels(self, competition_slug: str) -> List[Dict[str, Any]]:
        """
        Get all kernels for a specific competition.
        
        Args:
            competition_slug: Competition identifier
        
        Returns:
            List of kernel metadata dictionaries
        """
        if self.progress.is_completed(f"comp_{competition_slug}"):
            self.logger.info(f"Skipping already processed competition: {competition_slug}")
            # Load from cache
            cache_file = self.kernels_dir / f"{competition_slug}_kernels.json"
            if cache_file.exists():
                from .utils import load_json
                return load_json(cache_file)
            return []
        
        self.logger.info(f"Fetching kernels for competition: {competition_slug}")
        
        try:
            # Use kaggle CLI to list competition kernels
            output = self._run_kaggle_command([
                'kaggle', 'kernels', 'list',
                '--competition', competition_slug,
                '--csv',
                '--page-size', '100'
            ])
            
            # Parse CSV output
            from io import StringIO
            df = pd.read_csv(StringIO(output))
            
            kernels = df.to_dict('records')
            self.logger.info(f"Found {len(kernels)} kernels for {competition_slug}")
            
            # Save per-competition kernels
            save_json(
                kernels,
                self.kernels_dir / f"{competition_slug}_kernels.json"
            )
            
            # Mark as completed
            self.progress.mark_completed(f"comp_{competition_slug}")
            
            return kernels
            
        except Exception as e:
            self.logger.error(f"Failed to get kernels for {competition_slug}: {e}")
            self.progress.mark_failed(f"comp_{competition_slug}")
            return []
    
    def filter_kernels(self, kernels: List[Dict]) -> List[Dict]:
        """
        Filter kernels based on configuration criteria.
        
        Args:
            kernels: List of kernel metadata
        
        Returns:
            Filtered and sorted list of kernels
        """
        kernel_filters = self.config.get('kernel_filters', {})
        min_votes = self.config.get('min_votes', 3)
        
        filtered = []
        for kernel in kernels:
            # Language filter
            languages = kernel_filters.get('languages', [])
            if languages and kernel.get('language', '').lower() not in [l.lower() for l in languages]:
                continue
            
            # Vote filter
            vote_count = kernel.get('voteCount', 0)
            if isinstance(vote_count, str):
                vote_count = int(vote_count) if vote_count.isdigit() else 0
            if vote_count < min_votes:
                continue
            
            # Kernel type filter (notebook vs script)
            kernel_type = kernel.get('kernelType', '').lower()
            if kernel_type not in ['notebook', 'script']:
                continue
            
            filtered.append(kernel)
        
        # Sort by votes
        sort_by = kernel_filters.get('sort_by', 'voteCount')
        filtered = sorted(
            filtered,
            key=lambda x: int(x.get(sort_by, 0)) if str(x.get(sort_by, 0)).isdigit() else 0,
            reverse=True
        )
        
        return filtered
    
    def collect_all_metadata(
        self,
        max_competitions: Optional[int] = None,
        kernels_per_competition: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Collect metadata for competitions and kernels.
        
        Args:
            max_competitions: Maximum number of competitions to process
            kernels_per_competition: Maximum kernels per competition
        
        Returns:
            Summary statistics dictionary
        """
        with Timer("Metadata Collection", self.logger):
            # Get competitions
            competitions = self.list_competitions()
            competitions = self.filter_competitions(competitions)
            
            if max_competitions:
                competitions = competitions[:max_competitions]
            
            # Get kernels for each competition
            all_kernels = []
            kernels_per_comp = kernels_per_competition or self.config.get('notebooks_per_competition', 10)
            
            for i, comp in enumerate(competitions, 1):
                comp_slug = comp.get('ref', comp.get('id', ''))
                if not comp_slug:
                    continue
                
                self.logger.info(f"Processing competition {i}/{len(competitions)}: {comp_slug}")
                
                kernels = self.get_competition_kernels(comp_slug)
                kernels = self.filter_kernels(kernels)
                
                # Limit kernels per competition
                kernels = kernels[:kernels_per_comp]
                
                # Add competition info to each kernel
                for kernel in kernels:
                    kernel['competition'] = comp_slug
                    kernel['competition_title'] = comp.get('title', '')
                
                all_kernels.extend(kernels)
                
                self.logger.info(f"Added {len(kernels)} kernels from {comp_slug}")
            
            # Save consolidated kernel list
            kernels_file = self.kernels_dir / "all_kernels_metadata.json"
            save_json(all_kernels, kernels_file)
            
            # Create CSV for easier analysis
            if all_kernels:
                df = pd.DataFrame(all_kernels)
                csv_file = self.kernels_dir / "all_kernels_metadata.csv"
                df.to_csv(csv_file, index=False)
            
            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_competitions': len(competitions),
                'total_kernels': len(all_kernels),
                'kernels_per_competition': kernels_per_comp,
                'output_files': {
                    'competitions': str(self.competitions_dir / "filtered_competitions.json"),
                    'kernels_json': str(kernels_file),
                    'kernels_csv': str(csv_file) if all_kernels else None
                }
            }
            
            # Save summary
            save_json(summary, self.metadata_dir / "collection_summary.json")
            
            self.logger.info(f"Collection complete: {len(all_kernels)} kernels from {len(competitions)} competitions")
            
            return summary
    
    def get_kernel_details(self, kernel_ref: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed metadata for a specific kernel.
        
        Args:
            kernel_ref: Kernel reference (username/kernel-slug)
        
        Returns:
            Kernel details dictionary or None
        """
        try:
            output = self._run_kaggle_command([
                'kaggle', 'kernels', 'metadata', kernel_ref
            ])
            
            # Parse JSON output
            details = json.loads(output)
            return details
            
        except Exception as e:
            self.logger.error(f"Failed to get details for {kernel_ref}: {e}")
            return None
    
    def export_download_list(self, output_file: Optional[str] = None) -> str:
        """
        Export list of kernel references for downloading.
        
        Args:
            output_file: Output file path (default: kernels_to_download.txt)
        
        Returns:
            Path to output file
        """
        if output_file is None:
            output_file = self.metadata_dir / "kernels_to_download.txt"
        
        # Load all kernels metadata
        kernels_file = self.kernels_dir / "all_kernels_metadata.json"
        if not kernels_file.exists():
            self.logger.error("No kernels metadata found. Run collect_all_metadata first.")
            return ""
        
        from .utils import load_json
        kernels = load_json(kernels_file)
        
        # Extract kernel refs
        kernel_refs = []
        for kernel in kernels:
            ref = kernel.get('ref')
            if ref:
                kernel_refs.append(ref)
        
        # Write to file
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            f.write('\n'.join(kernel_refs))
        
        self.logger.info(f"Exported {len(kernel_refs)} kernel references to {output_path}")
        
        return str(output_path)


def main():
    """Command-line interface for competition and kernel metadata collection."""
    parser = argparse.ArgumentParser(
        description="Kaggle Competition and Kernel Metadata Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect and score competitions (recommended first step)
  python -m src.collector --mode competitions --target-count 500
  
  # Include notebook availability check (slower but more accurate)
  python -m src.collector --mode competitions --target-count 500 --check-availability
  
  # Collect kernels from selected competitions
  python -m src.collector --mode kernels --max-competitions 50 --kernels-per-comp 10
  
  # Export kernel download list
  python -m src.collector --mode kernels --export-list
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['competitions', 'kernels'],
        default='competitions',
        help='Collection mode: competitions (score and select) or kernels (collect notebook metadata)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=500,
        help='Number of top competitions to select (competitions mode)'
    )
    parser.add_argument(
        '--check-availability',
        action='store_true',
        help='Check notebook availability for each competition (slower)'
    )
    parser.add_argument(
        '--max-competitions',
        type=int,
        help='Maximum competitions to process (kernels mode)'
    )
    parser.add_argument(
        '--kernels-per-comp',
        type=int,
        help='Maximum kernels per competition (kernels mode)'
    )
    parser.add_argument(
        '--export-list',
        action='store_true',
        help='Export download list (kernels mode)'
    )
    parser.add_argument(
        '--output-dir',
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'competitions':
            # Competition scoring and selection mode
            print("\n🎯 Running Competition Collector (Scoring Mode)")
            print("=" * 70)
            
            collector = CompetitionCollector(config_path=args.config)
            
            summary = collector.run(
                target_count=args.target_count,
                check_availability=args.check_availability
            )
            
            sys.exit(0)
            
        else:  # kernels mode
            # Kernel metadata collection mode
            print("\n📚 Running Kernel Collector (Metadata Mode)")
            print("=" * 70)
            
            collector = KaggleCollector(config_path=args.config)
            
            # Collect metadata
            summary = collector.collect_all_metadata(
                max_competitions=args.max_competitions,
                kernels_per_competition=args.kernels_per_comp
            )
            
            print(f"\n{'='*60}")
            print("COLLECTION SUMMARY")
            print(f"{'='*60}")
            print(f"Total Competitions: {summary['total_competitions']}")
            print(f"Total Kernels: {summary['total_kernels']}")
            print(f"Kernels per Competition: {summary['kernels_per_competition']}")
            print(f"\nOutput Files:")
            for key, path in summary['output_files'].items():
                if path:
                    print(f"  {key}: {path}")
            print(f"{'='*60}\n")
            
            # Export download list if requested
            if args.export_list:
                download_list = collector.export_download_list()
                print(f"Download list exported to: {download_list}")
            
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("\nCheck logs for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
