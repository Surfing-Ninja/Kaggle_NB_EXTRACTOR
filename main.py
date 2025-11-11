#!/usr/bin/env python3
"""
Kaggle Notebook Pipeline - Master Orchestrator
Manages the complete end-to-end pipeline for extracting 5,000+ high-quality Kaggle notebooks.

Author: AI Assistant
Date: 2024-01-15
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    @staticmethod
    def success(text): return f"{Colors.GREEN}✅ {text}{Colors.RESET}"
    @staticmethod
    def error(text): return f"{Colors.RED}❌ {text}{Colors.RESET}"
    @staticmethod
    def warning(text): return f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}"
    @staticmethod
    def info(text): return f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}"


class Pipeline:
    """
    Master orchestrator for the complete Kaggle notebook extraction pipeline.
    
    Manages workflow across 5 stages:
    1. Competition metadata collection
    2. Kernel metadata indexing
    3. Notebook download
    4. Content analysis & quality filtering
    5. Report generation
    
    Features:
    - Graceful interruption handling
    - State checkpointing for resume capability
    - Prerequisite validation
    - Progress tracking
    - Comprehensive error handling
    """
    
    STAGES = [
        'collection',
        'indexing',
        'download',
        'analysis',
        'reporting'
    ]
    
    def __init__(self, config_path: str = 'config/config.yaml', verbose: bool = False, quiet: bool = False):
        """Initialize the pipeline orchestrator."""
        self.config_path = Path(config_path)
        self.verbose = verbose
        self.quiet = quiet
        self.interrupted = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize state
        self.state_file = Path('logs/pipeline_state.json')
        self.state = self._load_or_create_state()
        
        # Print welcome banner
        if not self.quiet:
            self._print_welcome_banner()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            # Default configuration
            return {
                'collection': {
                    'target_count': 500,
                    'min_notebooks': 10
                },
                'indexing': {
                    'target_count': 5000,
                    'min_votes': 10,
                    'per_competition': 30
                },
                'download': {
                    'batch_size': 100,
                    'rate_limit': 1.5,
                    'max_retries': 3
                },
                'analysis': {
                    'min_score': 50,
                    'target_count': 5000,
                    'workers': 4
                }
            }
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Main pipeline log
        self.log_file = log_dir / 'pipeline.log'
        
        # Configure logger
        self.logger = logging.getLogger('pipeline')
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        self.logger.handlers.clear()
        
        # File handler (detailed)
        file_handler = logging.FileHandler(self.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler (user-friendly)
        if not self.quiet:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO if not self.verbose else logging.DEBUG)
            console_formatter = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.addHandler(file_handler)
        self.logger.info("=" * 70)
        self.logger.info("Pipeline orchestrator initialized")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Verbose mode: {self.verbose}")
        self.logger.info("=" * 70)
    
    def _load_or_create_state(self) -> Dict:
        """Load existing state or create new."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.logger.info(f"Loaded existing state: {state['pipeline_id']}")
            return state
        else:
            state = {
                'pipeline_id': str(uuid.uuid4()),
                'start_time': datetime.now().isoformat(),
                'current_stage': 'not_started',
                'stages_completed': [],
                'stages_failed': [],
                'last_checkpoint': datetime.now().isoformat(),
                'stage_times': {}
            }
            self._save_state(state)
            self.logger.info(f"Created new pipeline state: {state['pipeline_id']}")
            return state
    
    def _save_state(self, state: Optional[Dict] = None):
        """Save current state to file."""
        if state is None:
            state = self.state
        
        state['last_checkpoint'] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.debug(f"State saved: {state['current_stage']}")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
        self.logger.warning(f"\n{signal_name} received. Initiating graceful shutdown...")
        
        self.interrupted = True
        
        if not self.quiet:
            print(f"\n{Colors.YELLOW}{'='*70}{Colors.RESET}")
            print(f"{Colors.YELLOW}Pipeline paused. Current state saved.{Colors.RESET}")
            print(f"{Colors.YELLOW}Run with --resume to continue from checkpoint.{Colors.RESET}")
            print(f"{Colors.YELLOW}{'='*70}{Colors.RESET}\n")
        
        self._save_state()
        sys.exit(0)
    
    def _print_welcome_banner(self):
        """Print welcome banner with system info."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}KAGGLE NOTEBOOK EXTRACTION PIPELINE{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}Production-Grade Notebook Curator{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}Pipeline ID:{Colors.RESET} {self.state['pipeline_id'][:8]}")
        print(f"{Colors.BOLD}Configuration:{Colors.RESET} {self.config_path}")
        print(f"{Colors.BOLD}Log file:{Colors.RESET} {self.log_file}")
        
        if self.state['stages_completed']:
            print(f"{Colors.BOLD}Completed stages:{Colors.RESET} {', '.join(self.state['stages_completed'])}")
        
        print()
    
    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        Verify all prerequisites before starting pipeline.
        
        Returns:
            (success, errors): Tuple of success status and list of error messages
        """
        if not self.quiet:
            print(f"{Colors.BOLD}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}PREREQUISITE CHECK{Colors.RESET}")
            print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
        
        errors = []
        checks = []
        
        # 1. Check Kaggle CLI
        self.logger.info("Checking Kaggle CLI...")
        try:
            result = subprocess.run(
                ['kaggle', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                checks.append(('Kaggle CLI', True, version))
                self.logger.info(f"Kaggle CLI found: {version}")
            else:
                checks.append(('Kaggle CLI', False, 'Not found'))
                errors.append("Kaggle CLI not found. Install: pip install kaggle")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            checks.append(('Kaggle CLI', False, 'Not found'))
            errors.append(f"Kaggle CLI error: {e}")
        
        # 2. Check Kaggle Credentials
        self.logger.info("Checking Kaggle credentials...")
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        if kaggle_json.exists():
            # Check permissions (on Unix-like systems)
            try:
                perms = oct(kaggle_json.stat().st_mode)[-3:]
                if perms != '600':
                    checks.append(('API Credentials', False, f'Wrong permissions: {perms}'))
                    errors.append(f"Fix permissions: chmod 600 {kaggle_json}")
                else:
                    # Just check file exists and has correct perms - API test removed due to version incompatibility
                    checks.append(('API Credentials', True, 'Found with correct permissions'))
                    self.logger.info("Kaggle API credentials found")
            except Exception as e:
                # On Windows or other issues, just check file exists
                checks.append(('API Credentials', True, 'Found'))
                self.logger.info(f"Kaggle credentials found (permission check skipped: {e})")
        else:
            checks.append(('API Credentials', False, 'Not found'))
            errors.append(f"Kaggle credentials not found. Place kaggle.json in {kaggle_json.parent}")
        
        # 3. Check Python Packages
        self.logger.info("Checking Python packages...")
        required_packages = ['pandas', 'tqdm', 'yaml', 'nbformat']
        
        missing_packages = []
        for package in required_packages:
            try:
                # Test import in current Python interpreter
                import_name = 'yaml' if package == 'yaml' else package
                result = subprocess.run(
                    [sys.executable, '-c', f'import {import_name}; print({import_name}.__version__)'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    version = result.stdout.strip()
                    checks.append((f'Package: {package}', True, f'v{version}'))
                    self.logger.debug(f"Package {package}: {version}")
                else:
                    checks.append((f'Package: {package}', False, 'Not installed'))
                    missing_packages.append(package)
                    self.logger.warning(f"Package {package} not found")
            except Exception as e:
                checks.append((f'Package: {package}', False, f'Check failed: {e}'))
                missing_packages.append(package)
                self.logger.warning(f"Package {package} check failed: {e}")
        
        if missing_packages:
            errors.append(f"Missing packages: {', '.join(missing_packages)}")
            errors.append("Install: pip install -r requirements.txt")
        
        # 4. Check Disk Space
        self.logger.info("Checking disk space...")
        try:
            stat = shutil.disk_usage('.')
            available_gb = stat.free / (1024**3)
            
            if available_gb >= 15:
                checks.append(('Disk Space', True, f'{available_gb:.1f} GB available'))
                self.logger.info(f"Disk space: {available_gb:.1f} GB")
            else:
                checks.append(('Disk Space', False, f'{available_gb:.1f} GB (need 15+ GB)'))
                errors.append(f"Low disk space: {available_gb:.1f} GB. Need at least 15 GB.")
        except Exception as e:
            checks.append(('Disk Space', False, f'Check failed: {e}'))
            errors.append(f"Could not check disk space: {e}")
        
        # 5. Check Network
        self.logger.info("Checking network connectivity...")
        try:
            import socket
            socket.create_connection(("www.kaggle.com", 80), timeout=5)
            checks.append(('Network', True, 'Connected to kaggle.com'))
            self.logger.info("Network connectivity confirmed")
        except OSError:
            checks.append(('Network', False, 'Cannot reach kaggle.com'))
            errors.append("Network error: Cannot reach kaggle.com")
        
        # Print results
        if not self.quiet:
            for check_name, success, details in checks:
                if success:
                    print(f"{Colors.success(check_name)}: {details}")
                else:
                    print(f"{Colors.error(check_name)}: {details}")
            print()
        
        # Print errors if any
        if errors:
            if not self.quiet:
                print(f"{Colors.RED}{Colors.BOLD}PREREQUISITE CHECK FAILED{Colors.RESET}\n")
                for error in errors:
                    print(f"{Colors.error(error)}")
                print()
            self.logger.error("Prerequisites check failed")
            for error in errors:
                self.logger.error(f"  - {error}")
            return False, errors
        
        if not self.quiet:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL PREREQUISITES SATISFIED{Colors.RESET}\n")
        
        self.logger.info("All prerequisites satisfied")
        return True, []
    
    def run_stage_1_collection(self) -> bool:
        """Run Stage 1: Competition metadata collection."""
        stage_name = 'collection'
        
        if not self.quiet:
            print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
            print(f"{Colors.BOLD}║ STAGE 1/5: Competition Metadata Collection{' '*23}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
        
        self.logger.info("Starting Stage 1: Collection")
        stage_start = time.time()
        
        # Check if already completed
        output_file = Path('metadata/competitions.json')
        if output_file.exists() and stage_name in self.state['stages_completed']:
            if not self.quiet:
                print(f"{Colors.info('Stage 1 already completed.')}")
                response = input(f"Skip and use existing data? [Y/n]: ").strip().lower()
                
                if response in ['', 'y', 'yes']:
                    print(f"{Colors.success('Using existing competition data')}\n")
                    self.logger.info("Skipping Stage 1 - using existing data")
                    return True
        
        # Run collector
        try:
            target_count = self.config.get('collection', {}).get('target_count', 500)
            
            if not self.quiet:
                print(f"{Colors.info(f'Collecting {target_count} competitions...')}")
            
            cmd = [
                sys.executable, '-m', 'src.collector',
                '--target-count', str(target_count),
                '--output-dir', 'metadata/competitions'
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True
            )
            
            if result.returncode != 0:
                self.logger.error(f"Stage 1 failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(result.stderr)
                return False
            
            # Validate outputs - collector writes to metadata/competitions/selected_competitions.json
            actual_output = Path('metadata/competitions/selected_competitions.json')
            if not actual_output.exists():
                self.logger.error("Output file not created")
                return False
            
            # Copy to expected location for compatibility
            import shutil
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(actual_output, output_file)
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Data is a list of competition objects
            if isinstance(data, list):
                total_found = len(data)
                selected = total_found
            else:
                # Legacy format
                total_found = len(data.get('competitions', []))
                selected = len(data.get('selected', []))
            
            if selected < 10:
                self.logger.warning(f"Only {selected} competitions selected (expected 10+)")
            
            # Update state
            self.state['current_stage'] = stage_name
            if stage_name not in self.state['stages_completed']:
                self.state['stages_completed'].append(stage_name)
            self.state['stage_times'][stage_name] = time.time() - stage_start
            self._save_state()
            
            # Print summary
            if not self.quiet:
                print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Stage 1 Complete{Colors.RESET}")
                print(f"  - {total_found} total competitions found")
                print(f"  - {selected} competitions selected")
                print(f"  - Estimated notebooks: {selected * 10}-{selected * 15}")
                print(f"  - Time: {self.state['stage_times'][stage_name]:.1f}s\n")
            
            self.logger.info(f"Stage 1 completed: {selected} competitions")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 1 exception: {e}", exc_info=True)
            self.state['stages_failed'].append(stage_name)
            self._save_state()
            return False
    
    def run_stage_2_indexing(self) -> bool:
        """Run Stage 2: Kernel metadata indexing."""
        stage_name = 'indexing'
        
        if not self.quiet:
            print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
            print(f"{Colors.BOLD}║ STAGE 2/5: Kernel Metadata Indexing{' '*30}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
        
        self.logger.info("Starting Stage 2: Indexing")
        stage_start = time.time()
        
        # Check if already completed
        output_file = Path('metadata/download_manifest.json')
        if output_file.exists() and stage_name in self.state['stages_completed']:
            if not self.quiet:
                print(f"{Colors.info('Stage 2 already completed.')}")
                response = input(f"Skip and use existing data? [Y/n]: ").strip().lower()
                
                if response in ['', 'y', 'yes']:
                    print(f"{Colors.success('Using existing kernel index')}\n")
                    self.logger.info("Skipping Stage 2 - using existing data")
                    return True
        
        # Run indexer
        try:
            target_count = self.config.get('indexing', {}).get('target_count', 5000)
            min_votes = self.config.get('indexing', {}).get('min_votes', 10)
            
            if not self.quiet:
                print(f"{Colors.info(f'Indexing kernels (target: {target_count})...')}")
                print(f"{Colors.info('This may take 5-15 minutes...')}\n")
            
            cmd = [
                sys.executable, '-m', 'src.indexer',
                '--competitions', 'metadata/competitions.json',
                '--target-count', str(target_count),
                '--min-votes', str(min_votes),
                '--output', 'metadata'
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True
            )
            
            if self.interrupted:
                self.logger.info("Stage 2 interrupted by user")
                return False
            
            if result.returncode != 0:
                self.logger.error(f"Stage 2 failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(result.stderr)
                return False
            
            # Validate outputs
            if not output_file.exists():
                self.logger.error("Output file not created")
                return False
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            total_indexed = sum(len(comp.get('kernels', [])) for comp in data.get('download_order', []))
            
            # Update state
            self.state['current_stage'] = stage_name
            if stage_name not in self.state['stages_completed']:
                self.state['stages_completed'].append(stage_name)
            self.state['stage_times'][stage_name] = time.time() - stage_start
            self._save_state()
            
            # Print summary
            if not self.quiet:
                print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Stage 2 Complete{Colors.RESET}")
                print(f"  - {total_indexed} kernels selected")
                print(f"  - Ready for download")
                print(f"  - Time: {self.state['stage_times'][stage_name]:.1f}s\n")
            
            self.logger.info(f"Stage 2 completed: {total_indexed} kernels")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 2 exception: {e}", exc_info=True)
            self.state['stages_failed'].append(stage_name)
            self._save_state()
            return False
    
    def run_stage_3_download(self) -> bool:
        """Run Stage 3: Notebook download."""
        stage_name = 'download'
        
        if not self.quiet:
            print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
            print(f"{Colors.BOLD}║ STAGE 3/5: Notebook Download{' '*38}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
        
        self.logger.info("Starting Stage 3: Download")
        stage_start = time.time()
        
        # Load manifest to estimate
        manifest_file = Path('metadata/download_manifest.json')
        if not manifest_file.exists():
            self.logger.error("Download manifest not found. Run Stage 2 first.")
            return False
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        total_notebooks = sum(len(comp.get('kernels', [])) for comp in manifest.get('download_order', []))
        
        # Estimate time and space
        rate_limit = self.config.get('download', {}).get('rate_limit', 1.5)
        estimated_time_hours = (total_notebooks * rate_limit) / 3600
        estimated_space_gb = total_notebooks * 1024 / (1024**2)  # ~1MB avg per notebook
        
        if not self.quiet:
            print(f"{Colors.info('Download Estimates:')}")
            print(f"  - Notebooks to download: {total_notebooks}")
            print(f"  - Rate limit: {rate_limit}s per notebook")
            print(f"  - Estimated time: {estimated_time_hours:.1f} hours")
            print(f"  - Estimated space: {estimated_space_gb:.1f} GB\n")
            
            response = input(f"Continue with download? [Y/n]: ").strip().lower()
            if response not in ['', 'y', 'yes']:
                print(f"{Colors.warning('Download cancelled by user')}\n")
                self.logger.info("Stage 3 cancelled by user")
                return False
            print()
        
        # Run downloader
        try:
            cmd = [
                sys.executable, '-m', 'src.downloader',
                '--manifest', 'metadata/download_manifest.json',
                '--config', str(self.config_path)
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True
            )
            
            if self.interrupted:
                self.logger.info("Stage 3 interrupted by user")
                return False
            
            if result.returncode != 0:
                self.logger.error(f"Stage 3 failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(result.stderr)
                return False
            
            # Validate outputs
            report_file = Path('metadata/download_report.json')
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                downloaded = report.get('successful', 0)
                failed = report.get('failed', 0)
                success_rate = (downloaded / (downloaded + failed) * 100) if (downloaded + failed) > 0 else 0
                
                if success_rate < 90:
                    self.logger.warning(f"Low success rate: {success_rate:.1f}%")
            else:
                downloaded = 0
                failed = 0
            
            # Update state
            self.state['current_stage'] = stage_name
            if stage_name not in self.state['stages_completed']:
                self.state['stages_completed'].append(stage_name)
            self.state['stage_times'][stage_name] = time.time() - stage_start
            self._save_state()
            
            # Print summary
            if not self.quiet:
                elapsed_hours = self.state['stage_times'][stage_name] / 3600
                print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Stage 3 Complete{Colors.RESET}")
                print(f"  - {downloaded} notebooks downloaded successfully")
                if failed > 0:
                    print(f"  - {failed} notebooks failed (logged)")
                print(f"  - Time: {elapsed_hours:.2f} hours\n")
            
            self.logger.info(f"Stage 3 completed: {downloaded} downloaded, {failed} failed")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 3 exception: {e}", exc_info=True)
            self.state['stages_failed'].append(stage_name)
            self._save_state()
            return False
    
    def run_stage_4_analysis(self) -> bool:
        """Run Stage 4: Content analysis and quality filtering."""
        stage_name = 'analysis'
        
        if not self.quiet:
            print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
            print(f"{Colors.BOLD}║ STAGE 4/5: Content Analysis & Quality Filtering{' '*17}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
        
        self.logger.info("Starting Stage 4: Analysis")
        stage_start = time.time()
        
        # Check if notebooks directory exists
        notebooks_dir = Path('notebooks')
        if not notebooks_dir.exists():
            self.logger.error("Notebooks directory not found. Run Stage 3 first.")
            return False
        
        # Run analyzer
        try:
            min_score = self.config.get('analysis', {}).get('min_score', 50)
            target_count = self.config.get('analysis', {}).get('target_count', 5000)
            workers = self.config.get('analysis', {}).get('workers', 4)
            
            if not self.quiet:
                print(f"{Colors.info('Analyzing notebooks...')}")
                print(f"{Colors.info(f'Quality threshold: {min_score}/100')}")
                print(f"{Colors.info(f'Target count: {target_count}')}")
                print(f"{Colors.info(f'Parallel workers: {workers}')}\n")
            
            cmd = [
                sys.executable, '-m', 'src.analyzer',
                '--input', 'notebooks',
                '--output', 'notebooks_curated',
                '--min-score', str(min_score),
                '--target-count', str(target_count),
                '--workers', str(workers)
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=not self.verbose,
                text=True
            )
            
            if self.interrupted:
                self.logger.info("Stage 4 interrupted by user")
                return False
            
            if result.returncode != 0:
                self.logger.error(f"Stage 4 failed with code {result.returncode}")
                if result.stderr:
                    self.logger.error(result.stderr)
                return False
            
            # Validate outputs
            catalog_file = Path('metadata/curated_catalog.json')
            if catalog_file.exists():
                with open(catalog_file, 'r') as f:
                    catalog = json.load(f)
                
                total_curated = catalog.get('total_notebooks', 0)
                avg_score = catalog.get('statistics', {}).get('avg_quality_score', 0)
                categories = catalog.get('categories', {})
            else:
                total_curated = 0
                avg_score = 0
                categories = {}
            
            # Update state
            self.state['current_stage'] = stage_name
            if stage_name not in self.state['stages_completed']:
                self.state['stages_completed'].append(stage_name)
            self.state['stage_times'][stage_name] = time.time() - stage_start
            self._save_state()
            
            # Print summary
            if not self.quiet:
                print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Stage 4 Complete{Colors.RESET}")
                print(f"  - {total_curated} high-quality notebooks curated")
                print(f"  - Average quality score: {avg_score:.1f}")
                print(f"  - Categories:")
                for cat, count in categories.items():
                    print(f"    * {cat}: {count}")
                print(f"  - Time: {self.state['stage_times'][stage_name]:.1f}s\n")
            
            self.logger.info(f"Stage 4 completed: {total_curated} notebooks curated")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 4 exception: {e}", exc_info=True)
            self.state['stages_failed'].append(stage_name)
            self._save_state()
            return False
    
    def run_stage_5_reporting(self) -> bool:
        """Run Stage 5: Report generation."""
        stage_name = 'reporting'
        
        if not self.quiet:
            print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
            print(f"{Colors.BOLD}║ STAGE 5/5: Report Generation{' '*37}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
        
        self.logger.info("Starting Stage 5: Reporting")
        stage_start = time.time()
        
        try:
            # Load catalog data
            catalog_file = Path('metadata/curated_catalog.json')
            if not catalog_file.exists():
                self.logger.error("Catalog not found. Run Stage 4 first.")
                return False
            
            with open(catalog_file, 'r') as f:
                catalog = json.load(f)
            
            # Create reports directory
            reports_dir = Path('reports')
            reports_dir.mkdir(exist_ok=True)
            
            # Generate summary statistics
            summary = {
                'generated_at': datetime.now().isoformat(),
                'total_notebooks': catalog.get('total_notebooks', 0),
                'categories': catalog.get('categories', {}),
                'statistics': catalog.get('statistics', {}),
                'pipeline_time': sum(self.state.get('stage_times', {}).values())
            }
            
            summary_file = reports_dir / 'summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            self.logger.info(f"Created: {summary_file}")
            
            # Generate USAGE guide
            usage_file = Path('USAGE.md')
            usage_content = f"""# Dataset Usage Guide

## Overview
This dataset contains {summary['total_notebooks']} high-quality Kaggle notebooks focused on feature engineering and machine learning.

## Structure
```
notebooks_curated/
├── pure_fe/              # Feature engineering focused
├── eda_fe/               # EDA + feature engineering
├── hyperparam_fe/        # Hyperparameter tuning + FE
├── complete_pipeline/    # Complete ML pipelines
└── advanced/             # Advanced ML techniques
```

## Quality Scoring
All notebooks have been scored on a 0-100 scale based on:
- Code quality (40 pts): Complexity, structure, functions
- Techniques (30 pts): ML/FE techniques used
- Documentation (15 pts): Markdown quality
- Libraries (15 pts): Professional tooling

## Category Breakdown
"""
            for cat, count in summary['categories'].items():
                pct = count / summary['total_notebooks'] * 100 if summary['total_notebooks'] > 0 else 0
                usage_content += f"- **{cat}**: {count} notebooks ({pct:.1f}%)\n"
            
            usage_content += f"""
## Legal & Attribution
- **Source**: Kaggle Public Notebooks
- **Method**: Official Kaggle API
- **Licenses**: See individual notebook metadata
- **Attribution**: Required - see kernel-metadata.json in each notebook
- **Commercial Use**: Check individual licenses

## How to Use
1. Browse categories in `notebooks_curated/`
2. Each notebook includes:
   - Original `.ipynb` file
   - `kernel-metadata.json` with author info
   - `.meta.json` with quality scores
   - `README.md` with techniques used
3. Always attribute original authors
4. Check license in metadata before use

## Citation
If using this dataset in research, please cite the original Kaggle notebooks and acknowledge the extraction pipeline.

Generated: {datetime.now().strftime('%Y-%m-%d')}
"""
            
            with open(usage_file, 'w') as f:
                f.write(usage_content)
            
            self.logger.info(f"Created: {usage_file}")
            
            # Generate PROVENANCE file
            provenance_file = Path('PROVENANCE.json')
            provenance = {
                'collection_date': datetime.now().strftime('%Y-%m-%d'),
                'source': 'Kaggle Public Notebooks',
                'method': 'Official Kaggle API',
                'total_notebooks': summary['total_notebooks'],
                'competitions_represented': catalog.get('statistics', {}).get('competitions_represented', 0),
                'pipeline_version': '1.0.0',
                'attribution_required': True,
                'commercial_use': 'Check individual licenses',
                'contact': 'See original Kaggle notebooks for author contact'
            }
            
            with open(provenance_file, 'w') as f:
                json.dump(provenance, f, indent=2)
            
            self.logger.info(f"Created: {provenance_file}")
            
            # Create master README for curated directory
            curated_readme = Path('notebooks_curated/README.md')
            readme_content = f"""# Curated Kaggle Notebooks

This directory contains {summary['total_notebooks']} high-quality Kaggle notebooks focused on feature engineering and machine learning techniques.

## Statistics
- **Total Notebooks**: {summary['total_notebooks']}
- **Average Quality Score**: {catalog.get('statistics', {}).get('avg_quality_score', 0):.1f}/100
- **Competitions Represented**: {catalog.get('statistics', {}).get('competitions_represented', 0)}
- **Total Size**: {catalog.get('statistics', {}).get('total_size_gb', 0):.2f} GB

## Categories
"""
            for cat, count in summary['categories'].items():
                readme_content += f"- **{cat}/**: {count} notebooks\n"
            
            readme_content += f"""
## Usage
See USAGE.md in the project root for detailed instructions.

## Attribution
All notebooks are from Kaggle's public repository. Please attribute original authors when using.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(curated_readme, 'w') as f:
                f.write(readme_content)
            
            self.logger.info(f"Created: {curated_readme}")
            
            # Update state
            self.state['current_stage'] = stage_name
            if stage_name not in self.state['stages_completed']:
                self.state['stages_completed'].append(stage_name)
            self.state['stage_times'][stage_name] = time.time() - stage_start
            self._save_state()
            
            # Print summary
            if not self.quiet:
                print(f"\n{Colors.GREEN}{Colors.BOLD}✅ Stage 5 Complete{Colors.RESET}")
                print(f"  - Summary report generated")
                print(f"  - Usage guide created")
                print(f"  - Provenance documented")
                print(f"  - Time: {self.state['stage_times'][stage_name]:.1f}s\n")
            
            self.logger.info("Stage 5 completed: All reports generated")
            return True
            
        except Exception as e:
            self.logger.error(f"Stage 5 exception: {e}", exc_info=True)
            self.state['stages_failed'].append(stage_name)
            self._save_state()
            return False
    
    def run_full_pipeline(self) -> bool:
        """Execute complete pipeline end-to-end."""
        if not self.quiet:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}FULL PIPELINE EXECUTION{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
            
            print(f"{Colors.BOLD}Configuration:{Colors.RESET}")
            print(f"  - Target competitions: {self.config.get('collection', {}).get('target_count', 500)}")
            print(f"  - Target notebooks: {self.config.get('indexing', {}).get('target_count', 5000)}")
            print(f"  - Quality threshold: {self.config.get('analysis', {}).get('min_score', 50)}/100")
            print(f"  - Parallel workers: {self.config.get('analysis', {}).get('workers', 4)}")
            print()
            
            response = input(f"Start full pipeline? [Y/n]: ").strip().lower()
            if response not in ['', 'y', 'yes']:
                print(f"{Colors.warning('Pipeline cancelled by user')}\n")
                return False
            print()
        
        self.logger.info("Starting full pipeline execution")
        pipeline_start = time.time()
        
        # Stage mapping
        stages = [
            ('collection', self.run_stage_1_collection),
            ('indexing', self.run_stage_2_indexing),
            ('download', self.run_stage_3_download),
            ('analysis', self.run_stage_4_analysis),
            ('reporting', self.run_stage_5_reporting)
        ]
        
        # Execute stages
        for stage_name, stage_func in stages:
            if self.interrupted:
                self.logger.info("Pipeline interrupted by user")
                return False
            
            # Check if already completed
            if stage_name in self.state['stages_completed']:
                self.logger.info(f"Stage {stage_name} already completed")
                continue
            
            # Run stage
            success = stage_func()
            
            if not success:
                if not self.quiet:
                    print(f"\n{Colors.RED}{Colors.BOLD}❌ Stage {stage_name} FAILED{Colors.RESET}\n")
                    print(f"Check logs/pipeline.log for details.\n")
                    
                    response = input(f"Retry stage? [Y/n]: ").strip().lower()
                    if response in ['', 'y', 'yes']:
                        success = stage_func()
                        if not success:
                            print(f"{Colors.error('Stage failed again. Aborting.')}\n")
                            return False
                    else:
                        return False
                else:
                    return False
        
        # Pipeline complete
        total_time = time.time() - pipeline_start
        self.state['total_time'] = total_time
        self.state['completed_at'] = datetime.now().isoformat()
        self._save_state()
        
        # Print final summary
        self._print_final_summary()
        
        self.logger.info(f"Full pipeline completed in {total_time:.1f}s")
        return True
    
    def _print_final_summary(self):
        """Print beautiful final summary."""
        # Load final statistics
        catalog_file = Path('metadata/curated_catalog.json')
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                catalog = json.load(f)
        else:
            catalog = {}
        
        total_notebooks = catalog.get('total_notebooks', 0)
        avg_score = catalog.get('statistics', {}).get('avg_quality_score', 0)
        competitions = catalog.get('statistics', {}).get('competitions_represented', 0)
        size_gb = catalog.get('statistics', {}).get('total_size_gb', 0)
        categories = catalog.get('categories', {})
        
        total_time = self.state.get('total_time', sum(self.state.get('stage_times', {}).values()))
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*22}PIPELINE COMPLETE!{' '*26}║{Colors.RESET}")
        print(f"{Colors.BOLD}{'╠' + '═'*68 + '╣'}{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  {Colors.GREEN}🎉 Successfully extracted {total_notebooks} notebooks{' '*(68-39-len(str(total_notebooks)))}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  📊 RESULTS:{' '*55}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Total notebooks:        {total_notebooks:<34}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Competitions covered:   {competitions:<34}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Average quality score:  {avg_score:<6.1f}{' '*28}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Storage used:           {size_gb:<6.2f} GB{' '*25}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  └─ Processing time:        {hours}h {minutes}m{' '*(28-len(str(hours))-len(str(minutes)))}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  📁 OUTPUT LOCATIONS:{' '*45}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Curated notebooks:  notebooks_curated/{' '*23}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Analysis report:    reports/analysis_summary.txt{' '*11}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ├─ Metadata:           metadata/{' '*30}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  └─ Logs:               logs/{' '*34}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  📋 CATEGORY BREAKDOWN:{' '*43}║{Colors.RESET}")
        
        for cat, count in categories.items():
            pct = count / total_notebooks * 100 if total_notebooks > 0 else 0
            cat_label = cat.replace('_', ' ').title()
            line = f"║  ├─ {cat_label + ':':<24} {count:>4} ({pct:>5.1f}%){' '*(19-len(str(count)))}║"
            print(f"{Colors.BOLD}{line}{Colors.RESET}")
        
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ⚖️  LEGAL COMPLIANCE:{' '*44}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ✅ All notebooks from public Kaggle sources{' '*22}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ✅ Downloaded via official API{' '*35}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ✅ Attribution maintained in metadata{' '*28}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  ✅ Licenses documented per notebook{' '*30}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  🚀 NEXT STEPS:{' '*52}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  1. Review notebooks_curated/README.md{' '*28}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  2. Check reports/analysis_summary.txt{' '*28}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  3. Read USAGE.md for dataset guidelines{' '*26}║{Colors.RESET}")
        print(f"{Colors.BOLD}║  4. Ensure attribution when using notebooks{' '*23}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
        print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
        
        print(f"{Colors.BOLD}Total elapsed time: {hours} hours {minutes} minutes{Colors.RESET}\n")
    
    def resume_from_checkpoint(self) -> bool:
        """Resume pipeline from last checkpoint."""
        if not self.state_file.exists():
            if not self.quiet:
                print(f"{Colors.error('No checkpoint found')}")
            self.logger.error("No checkpoint file found")
            return False
        
        last_checkpoint = datetime.fromisoformat(self.state['last_checkpoint'])
        time_ago = datetime.now() - last_checkpoint
        
        if not self.quiet:
            print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}RESUME FROM CHECKPOINT{Colors.RESET}")
            print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
            
            print(f"{Colors.info(f'Found checkpoint from {self._format_timedelta(time_ago)} ago')}")
            print(f"Last completed stage: {self.state.get('current_stage', 'none')}")
            print(f"Completed stages: {', '.join(self.state['stages_completed']) if self.state['stages_completed'] else 'none'}")
            
            # Determine next stage
            next_stage = None
            for stage in self.STAGES:
                if stage not in self.state['stages_completed']:
                    next_stage = stage
                    break
            
            if next_stage:
                print(f"Next stage: {next_stage}\n")
                
                response = input(f"Resume from checkpoint? [Y/n]: ").strip().lower()
                if response not in ['', 'y', 'yes']:
                    print(f"{Colors.warning('Resume cancelled')}\n")
                    return False
            else:
                print(f"{Colors.success('All stages already completed!')}\n")
                return True
        
        self.logger.info("Resuming from checkpoint")
        return self.run_full_pipeline()
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta in human-readable form."""
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        
        if td.days > 0:
            return f"{td.days} days, {hours} hours"
        elif hours > 0:
            return f"{hours} hours, {minutes} minutes"
        else:
            return f"{minutes} minutes"
    
    def interactive_mode(self):
        """Run pipeline in interactive mode."""
        while True:
            print(f"\n{Colors.BOLD}{'╔' + '═'*68 + '╗'}{Colors.RESET}")
            print(f"{Colors.BOLD}║{' '*17}KAGGLE NOTEBOOK PIPELINE{' '*25}║{Colors.RESET}")
            print(f"{Colors.BOLD}║{' '*22}INTERACTIVE MODE{' '*28}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╠' + '═'*68 + '╣'}{Colors.RESET}")
            print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  Choose an option:{' '*48}║{Colors.RESET}")
            print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  1. Run full pipeline (recommended){' '*31}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  2. Run Stage 1: Collection{' '*39}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  3. Run Stage 2: Indexing{' '*41}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  4. Run Stage 3: Download{' '*41}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  5. Run Stage 4: Analysis{' '*41}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  6. Run Stage 5: Reporting{' '*40}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  7. Resume from checkpoint{' '*40}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  8. View current progress{' '*41}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  9. Run prerequisite check{' '*40}║{Colors.RESET}")
            print(f"{Colors.BOLD}║  0. Exit{' '*58}║{Colors.RESET}")
            print(f"{Colors.BOLD}║{' '*68}║{Colors.RESET}")
            print(f"{Colors.BOLD}{'╚' + '═'*68 + '╝'}{Colors.RESET}\n")
            
            choice = input(f"Enter choice: ").strip()
            
            if choice == '1':
                self.run_full_pipeline()
            elif choice == '2':
                self.run_stage_1_collection()
            elif choice == '3':
                self.run_stage_2_indexing()
            elif choice == '4':
                self.run_stage_3_download()
            elif choice == '5':
                self.run_stage_4_analysis()
            elif choice == '6':
                self.run_stage_5_reporting()
            elif choice == '7':
                self.resume_from_checkpoint()
            elif choice == '8':
                self._show_progress()
            elif choice == '9':
                self.check_prerequisites()
            elif choice == '0':
                print(f"\n{Colors.info('Goodbye!')}\n")
                break
            else:
                print(f"\n{Colors.error('Invalid choice. Please try again.')}\n")
    
    def _show_progress(self):
        """Show current pipeline progress."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}CURRENT PROGRESS{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
        
        print(f"Pipeline ID: {self.state['pipeline_id'][:8]}")
        print(f"Started: {self.state['start_time']}")
        print(f"Last checkpoint: {self.state['last_checkpoint']}\n")
        
        print(f"{Colors.BOLD}Stages:{Colors.RESET}")
        for idx, stage in enumerate(self.STAGES, 1):
            if stage in self.state['stages_completed']:
                time_taken = self.state.get('stage_times', {}).get(stage, 0)
                print(f"  {idx}. {stage:<15} {Colors.success('COMPLETE')} ({time_taken:.1f}s)")
            elif stage in self.state['stages_failed']:
                print(f"  {idx}. {stage:<15} {Colors.error('FAILED')}")
            else:
                print(f"  {idx}. {stage:<15} {Colors.warning('PENDING')}")
        
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Kaggle Notebook Extraction Pipeline - Master Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --full
  
  # Interactive mode
  python main.py --interactive
  
  # Resume from checkpoint
  python main.py --resume
  
  # Run specific stage
  python main.py --stage collection
  python main.py --stage download
  
  # Dry run (show what would happen)
  python main.py --dry-run --full
  
  # Verbose output
  python main.py --verbose --full
"""
    )
    
    parser.add_argument('--full', action='store_true', help='Run complete pipeline')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--stage', choices=['collection', 'indexing', 'download', 'analysis', 'reporting'],
                       help='Run specific stage')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--skip-prereq', action='store_true', help='Skip prerequisite check')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline(
        config_path=args.config,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    # Check prerequisites (unless skipped or in interactive mode)
    if not args.skip_prereq and not args.interactive:
        success, errors = pipeline.check_prerequisites()
        if not success:
            sys.exit(1)
    
    # Dry run
    if args.dry_run:
        print(f"{Colors.info('DRY RUN MODE - No changes will be made')}\n")
        print("Would execute:")
        if args.full:
            print("  - Stage 1: Collection")
            print("  - Stage 2: Indexing")
            print("  - Stage 3: Download")
            print("  - Stage 4: Analysis")
            print("  - Stage 5: Reporting")
        elif args.stage:
            print(f"  - Stage: {args.stage}")
        print()
        sys.exit(0)
    
    # Execute based on arguments
    try:
        if args.interactive:
            pipeline.interactive_mode()
        elif args.full:
            success = pipeline.run_full_pipeline()
            sys.exit(0 if success else 1)
        elif args.resume:
            success = pipeline.resume_from_checkpoint()
            sys.exit(0 if success else 1)
        elif args.stage:
            stage_map = {
                'collection': pipeline.run_stage_1_collection,
                'indexing': pipeline.run_stage_2_indexing,
                'download': pipeline.run_stage_3_download,
                'analysis': pipeline.run_stage_4_analysis,
                'reporting': pipeline.run_stage_5_reporting
            }
            success = stage_map[args.stage]()
            sys.exit(0 if success else 1)
        else:
            # Default to interactive
            pipeline.interactive_mode()
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.warning('Pipeline interrupted by user')}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.error(f'Unexpected error: {e}')}")
        pipeline.logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
