#!/usr/bin/env python3
"""
Kaggle Notebook Downloader - Production Grade
Downloads 5000+ Kaggle notebooks safely with full resume capability.

Author: AI Assistant
Date: 2024-01-15
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import yaml
try:
    import nbformat
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Progress bars disabled.")

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


def setup_logger(log_file: Path) -> logging.Logger:
    """Setup structured JSON logging."""
    logger = logging.getLogger('notebook_downloader')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # File handler for JSON logs (append mode for resume)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    
    # Console handler for user feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class NotebookDownloader:
    """
    Production-grade Kaggle notebook downloader with resume capability.
    
    Features:
    - Resume from interruption
    - Retry logic with exponential backoff
    - Rate limiting
    - Graceful shutdown
    - Progress tracking
    - Detailed logging
    """
    
    def __init__(self, manifest_path: str, config_path: str, dry_run: bool = False):
        """
        Initialize the downloader.
        
        Args:
            manifest_path: Path to download_manifest.json
            config_path: Path to config.yaml
            dry_run: If True, only simulate downloads
        """
        self.dry_run = dry_run
        self.interrupted = False
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Load manifest
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Load config
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        self.log_file = log_dir / 'download.jsonl'
        self.logger = setup_logger(self.log_file)
        
        # Initialize state tracking
        self.downloaded_kernels: Set[str] = set()
        self.failed_kernels: Dict[str, str] = {}
        self.skipped_kernels: Set[str] = set()
        
        # Initialize statistics
        total_notebooks = self.manifest.get('total_notebooks', 5000)
        self.stats = {
            "total_target": total_notebooks,
            "downloaded": 0,
            "failed": 0,
            "skipped": 0,
            "in_progress": 0,
            "remaining": total_notebooks,
            "start_time": None,
            "estimated_completion": None,
            "total_bytes": 0,
            "avg_time_per_notebook": 0.0
        }
        
        # Load previous progress if exists
        self.load_progress()
        
        # Create output directories
        self.output_base = Path('notebooks')
        self.output_base.mkdir(exist_ok=True)
        
        # Checkpoint path
        self.checkpoint_path = Path('metadata/download_checkpoint.json')
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "event": "downloader_initialized",
            "manifest": str(manifest_path),
            "total_target": total_notebooks,
            "dry_run": dry_run
        }))
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print(f"\n\n{Colors.YELLOW}⚠️  Interrupt received. Finishing current download...{Colors.RESET}")
        self.interrupted = True
    
    def load_progress(self) -> Tuple[Set[str], Dict[str, str]]:
        """
        Resume from previous incomplete run.
        
        Returns:
            Tuple of (downloaded_kernels, failed_kernels)
        """
        if not self.log_file.exists():
            print(f"{Colors.CYAN}Starting fresh download...{Colors.RESET}")
            return set(), {}
        
        print(f"{Colors.CYAN}Loading previous progress...{Colors.RESET}")
        
        downloaded = set()
        failed = {}
        
        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        kernel_id = entry.get('kernel_id')
                        status = entry.get('status')
                        
                        if status == 'success' and kernel_id:
                            downloaded.add(kernel_id)
                            self.stats['downloaded'] += 1
                            # Remove from failed if it was previously failed
                            if kernel_id in failed:
                                del failed[kernel_id]
                        elif status == 'failed' and kernel_id:
                            error = entry.get('error', 'Unknown error')
                            failed[kernel_id] = error
                            # Remove from downloaded if it was previously successful
                            if kernel_id in downloaded:
                                downloaded.remove(kernel_id)
                                self.stats['downloaded'] -= 1
                            if kernel_id not in self.failed_kernels:
                                self.stats['failed'] += 1
                        elif status == 'skipped' and kernel_id:
                            self.skipped_kernels.add(kernel_id)
                            self.stats['skipped'] += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"{Colors.RED}Error loading progress: {e}{Colors.RESET}")
            return set(), {}
        
        self.downloaded_kernels = downloaded
        self.failed_kernels = failed
        self.stats['remaining'] = self.stats['total_target'] - self.stats['downloaded'] - self.stats['skipped']
        
        if downloaded or failed or self.skipped_kernels:
            print(f"{Colors.GREEN}Resuming: {len(downloaded)} already downloaded, "
                  f"{len(failed)} failed, {len(self.skipped_kernels)} skipped, "
                  f"{self.stats['remaining']} remaining{Colors.RESET}")
        
        return downloaded, failed
    
    def sanitize_path(self, kernel_id: str) -> str:
        """
        Convert kernel_id to safe filesystem path.
        
        Args:
            kernel_id: e.g., "username/kernel-title"
            
        Returns:
            Sanitized string safe for filesystem
        """
        # Replace / with __
        sanitized = kernel_id.replace('/', '__')
        
        # Replace spaces with _
        sanitized = sanitized.replace(' ', '_')
        
        # Remove special characters
        sanitized = re.sub(r'[<>:"|?*]', '', sanitized)
        
        # Remove any path traversal attempts
        sanitized = sanitized.replace('..', '')
        
        # Truncate to 200 characters
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        
        return sanitized
    
    def verify_download(self, output_dir: Path) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify that download succeeded and contains valid notebook.
        
        Args:
            output_dir: Directory where notebook was downloaded
            
        Returns:
            Tuple of (is_valid, file_path, error_message)
        """
        if not output_dir.exists():
            return False, None, "Output directory does not exist"
        
        # Find .ipynb files
        ipynb_files = list(output_dir.glob('*.ipynb'))
        
        if not ipynb_files:
            return False, None, "No .ipynb file found"
        
        # Check the first notebook file
        notebook_file = ipynb_files[0]
        
        # Check file size
        file_size = notebook_file.stat().st_size
        if file_size < 1024:  # Less than 1KB
            return False, str(notebook_file), f"File too small ({file_size} bytes)"
        
        # Try parsing with nbformat
        try:
            with open(notebook_file, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)
            
            # Check minimum structure
            if 'cells' not in nb:
                return False, str(notebook_file), "No cells key in notebook"
            
            if len(nb['cells']) < 1:
                return False, str(notebook_file), "Notebook has no cells"
            
            return True, str(notebook_file), None
            
        except Exception as e:
            return False, str(notebook_file), f"Invalid notebook format: {str(e)}"
    
    def download_single_notebook(
        self, 
        kernel_id: str, 
        competition: str, 
        retry_count: int = 0
    ) -> Dict:
        """
        Download one notebook with retry logic.
        
        Args:
            kernel_id: "username/kernel-title"
            competition: "titanic"
            retry_count: Current retry attempt (0-based)
            
        Returns:
            Dict with download result
        """
        start_time = time.time()
        
        # Create output directory
        sanitized_id = self.sanitize_path(kernel_id)
        output_dir = self.output_base / competition / sanitized_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists and is valid
        if retry_count == 0:  # Only check on first attempt
            is_valid, file_path, _ = self.verify_download(output_dir)
            if is_valid:
                elapsed = time.time() - start_time
                result = {
                    "success": True,
                    "kernel_id": kernel_id,
                    "error": None,
                    "output_path": str(output_dir),
                    "retries": 0,
                    "skipped": True
                }
                
                # Log skipped
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "kernel_id": kernel_id,
                    "competition": competition,
                    "status": "skipped",
                    "attempt": 1,
                    "error": None,
                    "output_dir": str(output_dir),
                    "file_size": Path(file_path).stat().st_size if file_path else 0,
                    "elapsed_seconds": elapsed
                }
                self.logger.info(json.dumps(log_entry))
                
                return result
        
        if self.dry_run:
            # Simulate download
            time.sleep(0.1)
            elapsed = time.time() - start_time
            result = {
                "success": True,
                "kernel_id": kernel_id,
                "error": None,
                "output_path": str(output_dir),
                "retries": retry_count,
                "skipped": False
            }
            return result
        
        # Build command
        cmd = [
            "kaggle", "kernels", "pull",
            kernel_id,
            "-p", str(output_dir),
            "--metadata"
        ]
        
        try:
            # Execute with timeout
            result_proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            elapsed = time.time() - start_time
            
            # Check return code
            if result_proc.returncode != 0:
                error_msg = result_proc.stderr.strip() if result_proc.stderr else "Unknown error"
                
                # Determine if retryable
                retryable = self._is_retryable_error(error_msg, result_proc.returncode)
                
                if retryable and retry_count < self.config.get('max_retries', 3):
                    # Log retry attempt
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "kernel_id": kernel_id,
                        "competition": competition,
                        "status": "retry",
                        "attempt": retry_count + 1,
                        "error": error_msg,
                        "output_dir": str(output_dir),
                        "elapsed_seconds": elapsed
                    }
                    self.logger.info(json.dumps(log_entry))
                    
                    # Exponential backoff
                    delays = [5, 15, 45]
                    delay = delays[min(retry_count, len(delays) - 1)]
                    
                    # Special handling for rate limits
                    if '429' in error_msg or 'rate limit' in error_msg.lower():
                        delay = 30
                    
                    time.sleep(delay)
                    
                    # Retry
                    return self.download_single_notebook(kernel_id, competition, retry_count + 1)
                else:
                    # Max retries exceeded or non-retryable error
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "kernel_id": kernel_id,
                        "competition": competition,
                        "status": "failed",
                        "attempt": retry_count + 1,
                        "error": error_msg,
                        "output_dir": str(output_dir),
                        "elapsed_seconds": elapsed
                    }
                    self.logger.info(json.dumps(log_entry))
                    
                    return {
                        "success": False,
                        "kernel_id": kernel_id,
                        "error": error_msg,
                        "output_path": None,
                        "retries": retry_count,
                        "skipped": False
                    }
            
            # Verify download
            is_valid, file_path, verify_error = self.verify_download(output_dir)
            
            if not is_valid:
                if retry_count < self.config.get('max_retries', 3):
                    # Retry
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "kernel_id": kernel_id,
                        "competition": competition,
                        "status": "retry",
                        "attempt": retry_count + 1,
                        "error": f"Verification failed: {verify_error}",
                        "output_dir": str(output_dir),
                        "elapsed_seconds": elapsed
                    }
                    self.logger.info(json.dumps(log_entry))
                    
                    time.sleep(5)
                    return self.download_single_notebook(kernel_id, competition, retry_count + 1)
                else:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "kernel_id": kernel_id,
                        "competition": competition,
                        "status": "failed",
                        "attempt": retry_count + 1,
                        "error": f"Verification failed: {verify_error}",
                        "output_dir": str(output_dir),
                        "elapsed_seconds": elapsed
                    }
                    self.logger.info(json.dumps(log_entry))
                    
                    return {
                        "success": False,
                        "kernel_id": kernel_id,
                        "error": f"Verification failed: {verify_error}",
                        "output_path": None,
                        "retries": retry_count,
                        "skipped": False
                    }
            
            # Success!
            file_size = Path(file_path).stat().st_size if file_path else 0
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "kernel_id": kernel_id,
                "competition": competition,
                "status": "success",
                "attempt": retry_count + 1,
                "error": None,
                "output_dir": str(output_dir),
                "file_size": file_size,
                "elapsed_seconds": elapsed
            }
            self.logger.info(json.dumps(log_entry))
            
            return {
                "success": True,
                "kernel_id": kernel_id,
                "error": None,
                "output_path": str(output_dir),
                "retries": retry_count,
                "skipped": False,
                "file_size": file_size
            }
            
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            
            if retry_count < self.config.get('max_retries', 3):
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "kernel_id": kernel_id,
                    "competition": competition,
                    "status": "retry",
                    "attempt": retry_count + 1,
                    "error": "Timeout",
                    "output_dir": str(output_dir),
                    "elapsed_seconds": elapsed
                }
                self.logger.info(json.dumps(log_entry))
                
                time.sleep(10)
                return self.download_single_notebook(kernel_id, competition, retry_count + 1)
            else:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "kernel_id": kernel_id,
                    "competition": competition,
                    "status": "failed",
                    "attempt": retry_count + 1,
                    "error": "Timeout after retries",
                    "output_dir": str(output_dir),
                    "elapsed_seconds": elapsed
                }
                self.logger.info(json.dumps(log_entry))
                
                return {
                    "success": False,
                    "kernel_id": kernel_id,
                    "error": "Timeout",
                    "output_path": None,
                    "retries": retry_count,
                    "skipped": False
                }
                
        except Exception as e:
            elapsed = time.time() - start_time
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "kernel_id": kernel_id,
                "competition": competition,
                "status": "failed",
                "attempt": retry_count + 1,
                "error": str(e),
                "output_dir": str(output_dir),
                "elapsed_seconds": elapsed
            }
            self.logger.info(json.dumps(log_entry))
            
            return {
                "success": False,
                "kernel_id": kernel_id,
                "error": str(e),
                "output_path": None,
                "retries": retry_count,
                "skipped": False
            }
    
    def _is_retryable_error(self, error_msg: str, return_code: int) -> bool:
        """Determine if error is retryable."""
        error_lower = error_msg.lower()
        
        # Non-retryable errors
        if '404' in error_msg or 'not found' in error_lower:
            return False
        if 'unauthorized' in error_lower or '401' in error_msg:
            return False
        if 'forbidden' in error_lower or '403' in error_msg:
            return False
        
        # Retryable errors
        if 'timeout' in error_lower:
            return True
        if 'connection' in error_lower:
            return True
        if '429' in error_msg or 'rate limit' in error_lower:
            return True
        if '503' in error_msg or 'service unavailable' in error_lower:
            return True
        if '502' in error_msg or 'bad gateway' in error_lower:
            return True
        
        # Default: retry on non-zero exit code
        return return_code != 0
    
    def download_batch(
        self, 
        kernel_list: List[Dict], 
        batch_num: int, 
        total_batches: int
    ) -> Dict:
        """
        Download a batch of notebooks with progress tracking.
        
        Args:
            kernel_list: List of kernel dicts to download
            batch_num: Current batch number (1-based)
            total_batches: Total number of batches
            
        Returns:
            Dict with batch statistics
        """
        batch_success = 0
        batch_failed = 0
        batch_skipped = 0
        
        print(f"\n{Colors.BOLD}╔════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}║ BATCH {batch_num}/{total_batches} - {len(kernel_list)} notebooks{' ' * (38 - len(str(batch_num)) - len(str(total_batches)) - len(str(len(kernel_list))))}║{Colors.RESET}")
        print(f"{Colors.BOLD}╚════════════════════════════════════════════════════════╝{Colors.RESET}\n")
        
        batch_start_time = time.time()
        
        # Use tqdm if available
        iterator = tqdm(kernel_list, desc=f"Batch {batch_num}", unit="notebook") if TQDM_AVAILABLE else kernel_list
        
        for idx, kernel in enumerate(iterator, 1):
            if self.interrupted:
                print(f"\n{Colors.YELLOW}Batch interrupted. Saving progress...{Colors.RESET}")
                break
            
            kernel_id = kernel['kernel_id']
            competition = kernel['competition']
            
            # Check if already downloaded
            if kernel_id in self.downloaded_kernels:
                batch_skipped += 1
                self.stats['skipped'] += 1
                if not TQDM_AVAILABLE:
                    print(f"[{idx}/{len(kernel_list)}] {Colors.YELLOW}⏭️  Skipped: {kernel_id}{Colors.RESET}")
                continue
            
            if not TQDM_AVAILABLE:
                print(f"[{idx}/{len(kernel_list)}] Downloading: {kernel_id}")
            
            # Download
            result = self.download_single_notebook(kernel_id, competition)
            
            # Update stats
            if result['success']:
                if result.get('skipped'):
                    batch_skipped += 1
                    self.skipped_kernels.add(kernel_id)
                    self.stats['skipped'] += 1
                else:
                    batch_success += 1
                    self.downloaded_kernels.add(kernel_id)
                    self.stats['downloaded'] += 1
                    if 'file_size' in result:
                        self.stats['total_bytes'] += result['file_size']
                
                if not TQDM_AVAILABLE:
                    print(f"{Colors.GREEN}✅ Success{Colors.RESET}")
            else:
                batch_failed += 1
                self.failed_kernels[kernel_id] = result['error']
                self.stats['failed'] += 1
                
                if not TQDM_AVAILABLE:
                    print(f"{Colors.RED}❌ Failed: {result['error'][:50]}{Colors.RESET}")
            
            self.stats['remaining'] = self.stats['total_target'] - self.stats['downloaded'] - self.stats['skipped']
            
            # Rate limiting
            rate_limit = self.config.get('rate_limit', {}).get('delay', 1.5)
            if not self.dry_run:
                time.sleep(rate_limit)
            
            # Print mini-statistics every 10 downloads
            if idx % 10 == 0 and not TQDM_AVAILABLE:
                eta = self.calculate_eta()
                print(f"{Colors.CYAN}Progress: {batch_success} ✅ | {batch_failed} ❌ | {batch_skipped} ⏭️  | ETA: {eta}{Colors.RESET}")
        
        batch_elapsed = time.time() - batch_start_time
        
        # Update average time per notebook
        total_processed = self.stats['downloaded'] + self.stats['skipped']
        if total_processed > 0 and self.stats['start_time']:
            total_elapsed = time.time() - self.stats['start_time']
            self.stats['avg_time_per_notebook'] = total_elapsed / total_processed
        
        # Print batch summary
        print(f"\n{Colors.BOLD}Batch {batch_num} Complete:{Colors.RESET}")
        print(f"  ✅ Success: {batch_success}")
        print(f"  ❌ Failed: {batch_failed}")
        print(f"  ⏭️  Skipped: {batch_skipped}")
        print(f"  ⏱️  Time: {batch_elapsed:.1f}s")
        
        # Save checkpoint
        self.save_checkpoint(batch_num)
        
        return {
            "batch_success": batch_success,
            "batch_failed": batch_failed,
            "batch_skipped": batch_skipped
        }
    
    def save_checkpoint(self, batch_num: int):
        """Save current progress to checkpoint file."""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "batch_num": batch_num,
            "stats": self.stats,
            "downloaded_count": len(self.downloaded_kernels),
            "failed_count": len(self.failed_kernels),
            "skipped_count": len(self.skipped_kernels)
        }
        
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def calculate_eta(self) -> str:
        """
        Estimate time remaining.
        
        Returns:
            Human-readable string like "2h 35m remaining"
        """
        if not self.stats['start_time']:
            return "Calculating..."
        
        elapsed = time.time() - self.stats['start_time']
        processed = self.stats['downloaded'] + self.stats['skipped']
        
        if processed == 0:
            return "Calculating..."
        
        avg_time = elapsed / processed
        remaining = self.stats['remaining']
        
        eta_seconds = remaining * avg_time
        
        if eta_seconds < 60:
            return f"{int(eta_seconds)}s remaining"
        elif eta_seconds < 3600:
            minutes = int(eta_seconds / 60)
            return f"{minutes}m remaining"
        else:
            hours = int(eta_seconds / 3600)
            minutes = int((eta_seconds % 3600) / 60)
            return f"{hours}h {minutes}m remaining"
    
    def print_progress_summary(self):
        """Print current download progress."""
        total = self.stats['total_target']
        downloaded = self.stats['downloaded']
        failed = self.stats['failed']
        skipped = self.stats['skipped']
        remaining = self.stats['remaining']
        
        downloaded_pct = (downloaded / total * 100) if total > 0 else 0
        failed_pct = (failed / total * 100) if total > 0 else 0
        skipped_pct = (skipped / total * 100) if total > 0 else 0
        remaining_pct = (remaining / total * 100) if total > 0 else 0
        
        # Calculate rate
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            processed = downloaded + skipped
            rate = (processed / elapsed * 60) if elapsed > 0 else 0
        else:
            elapsed = 0
            rate = 0
        
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        eta_str = self.calculate_eta()
        
        # Storage used
        storage_gb = self.stats['total_bytes'] / (1024**3)
        
        print(f"\n{Colors.BOLD}╔══════════════════════════════════════════════════════════════╗{Colors.RESET}")
        print(f"{Colors.BOLD}║              DOWNLOAD PROGRESS                               ║{Colors.RESET}")
        print(f"{Colors.BOLD}╠══════════════════════════════════════════════════════════════╣{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} Total Target:        {total:<40} {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.GREEN}✅ Downloaded:{Colors.RESET}       {downloaded:<6} ({downloaded_pct:>5.1f}%)                    {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.RED}❌ Failed:{Colors.RESET}           {failed:<6} ({failed_pct:>5.1f}%)                    {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.YELLOW}⏭️  Skipped:{Colors.RESET}          {skipped:<6} ({skipped_pct:>5.1f}%)                    {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} {Colors.CYAN}⏳ Remaining:{Colors.RESET}        {remaining:<6} ({remaining_pct:>5.1f}%)                    {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET}                                                              {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} Rate:                {rate:<.1f} notebooks/min                   {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} Elapsed:             {elapsed_str:<40} {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} ETA:                 {eta_str:<40} {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET}                                                              {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}║{Colors.RESET} Storage Used:        {storage_gb:<.2f} GB                             {Colors.BOLD}║{Colors.RESET}")
        print(f"{Colors.BOLD}╚══════════════════════════════════════════════════════════════╝{Colors.RESET}\n")
    
    def run(self):
        """Main download orchestration."""
        try:
            # Print welcome banner
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}KAGGLE NOTEBOOK DOWNLOADER - PRODUCTION MODE{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
            
            print(f"{Colors.BOLD}Configuration:{Colors.RESET}")
            print(f"  Manifest: {self.manifest_path}")
            print(f"  Target: {self.stats['total_target']} notebooks")
            print(f"  Batch size: {self.config.get('batch_size', 100)}")
            print(f"  Rate limit: {self.config.get('rate_limit', {}).get('delay', 1.5)}s between downloads")
            print(f"  Max retries: {self.config.get('max_retries', 3)}")
            print(f"  Dry run: {self.dry_run}")
            print()
            
            # Check for resume
            if self.downloaded_kernels or self.failed_kernels or self.skipped_kernels:
                print(f"{Colors.GREEN}Resuming from previous run...{Colors.RESET}\n")
            
            # Get list of kernels to download
            kernels_to_download = []
            
            # Extract from manifest
            if 'download_order' in self.manifest:
                for comp_data in self.manifest['download_order']:
                    competition = comp_data['competition']
                    for kernel in comp_data.get('kernels', []):
                        kernels_to_download.append({
                            'kernel_id': kernel['kernel_id'],
                            'competition': competition,
                            'votes': kernel.get('votes', 0)
                        })
            
            print(f"Total kernels in manifest: {len(kernels_to_download)}")
            
            # Filter out already downloaded
            kernels_remaining = [
                k for k in kernels_to_download 
                if k['kernel_id'] not in self.downloaded_kernels
            ]
            
            print(f"Kernels remaining to download: {len(kernels_remaining)}\n")
            
            if not kernels_remaining:
                print(f"{Colors.GREEN}All notebooks already downloaded!{Colors.RESET}\n")
                self.print_progress_summary()
                return
            
            # Group into batches
            batch_size = self.config.get('batch_size', 100)
            batches = [
                kernels_remaining[i:i + batch_size]
                for i in range(0, len(kernels_remaining), batch_size)
            ]
            total_batches = len(batches)
            
            print(f"Processing {total_batches} batches...\n")
            
            # Start timer
            self.stats['start_time'] = time.time()
            
            # Download batches
            for batch_num, batch in enumerate(batches, 1):
                if self.interrupted:
                    print(f"\n{Colors.YELLOW}Download paused. Run again to resume from batch {batch_num}.{Colors.RESET}")
                    break
                
                self.download_batch(batch, batch_num, total_batches)
                
                # Print progress summary every 5 batches
                if batch_num % 5 == 0:
                    self.print_progress_summary()
            
            # Print final summary
            print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.GREEN}DOWNLOAD COMPLETE!{Colors.RESET}")
            print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.RESET}\n")
            
            self.print_progress_summary()
            
            # List failed notebooks if any
            if self.failed_kernels:
                print(f"\n{Colors.RED}Failed notebooks ({len(self.failed_kernels)}):{Colors.RESET}")
                for kernel_id, error in list(self.failed_kernels.items())[:20]:
                    print(f"  - {kernel_id}: {error[:60]}")
                if len(self.failed_kernels) > 20:
                    print(f"  ... and {len(self.failed_kernels) - 20} more")
            
            # Save final report
            report = {
                "timestamp": datetime.now().isoformat(),
                "stats": self.stats,
                "failed_kernels": self.failed_kernels,
                "total_elapsed": time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            }
            
            report_path = Path('metadata/download_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n{Colors.CYAN}Final report saved to: {report_path}{Colors.RESET}")
            print(f"{Colors.CYAN}Notebooks saved to: {self.output_base}/{Colors.RESET}\n")
            
            # Next steps
            success_rate = (self.stats['downloaded'] / self.stats['total_target'] * 100) if self.stats['total_target'] > 0 else 0
            
            if success_rate > 95:
                print(f"{Colors.GREEN}✨ Excellent! Ready for analysis.{Colors.RESET}\n")
            elif self.failed_kernels:
                print(f"{Colors.YELLOW}💡 Next step: Review failed notebooks and retry if needed.{Colors.RESET}\n")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Download interrupted by user.{Colors.RESET}")
            self.save_checkpoint(-1)
            print(f"{Colors.CYAN}Progress saved. Run again to resume.{Colors.RESET}\n")
        except Exception as e:
            print(f"\n{Colors.RED}Critical error: {e}{Colors.RESET}")
            self.save_checkpoint(-1)
            print(f"{Colors.CYAN}Progress saved before error.{Colors.RESET}\n")
            raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Download Kaggle notebooks with resume capability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh download
  python src/downloader.py
  
  # Resume from interruption (auto-detected)
  python src/downloader.py
  
  # Dry run to see what would be downloaded
  python src/downloader.py --dry-run
  
  # Custom configuration
  python src/downloader.py --batch-size 50 --rate-limit 2.0
        """
    )
    
    parser.add_argument(
        '--manifest',
        default='metadata/download_manifest.json',
        help='Path to download manifest (default: metadata/download_manifest.json)'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--rate-limit',
        type=float,
        help='Override rate limit delay in seconds'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Force resume from checkpoint (default: auto-detect)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be downloaded without actually downloading'
    )
    
    args = parser.parse_args()
    
    # Check if manifest exists
    if not Path(args.manifest).exists():
        print(f"{Colors.RED}Error: Manifest file not found: {args.manifest}{Colors.RESET}")
        print(f"{Colors.YELLOW}Run the indexer first to generate the manifest:{Colors.RESET}")
        print(f"  python src/indexer.py")
        sys.exit(1)
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"{Colors.RED}Error: Config file not found: {args.config}{Colors.RESET}")
        sys.exit(1)
    
    # Load config and apply overrides
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.batch_size:
        config['batch_size'] = args.batch_size
    
    if args.rate_limit:
        if 'rate_limit' not in config:
            config['rate_limit'] = {}
        config['rate_limit']['delay'] = args.rate_limit
    
    # Save modified config to temp file
    temp_config_path = Path('config/config_temp.yaml')
    temp_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Create downloader and run
        downloader = NotebookDownloader(
            manifest_path=args.manifest,
            config_path=str(temp_config_path),
            dry_run=args.dry_run
        )
        
        downloader.run()
        
    finally:
        # Cleanup temp config
        if temp_config_path.exists():
            temp_config_path.unlink()


if __name__ == '__main__':
    main()
