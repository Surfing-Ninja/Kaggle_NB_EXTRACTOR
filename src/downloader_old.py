"""
Notebook Downloader
Downloads Kaggle kernels/notebooks with rate limiting, retry logic, and resume capability.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

from .utils import (
    setup_logger,
    load_config,
    load_json,
    save_json,
    retry_with_backoff,
    rate_limited,
    ProgressTracker,
    Timer,
    ensure_dir,
    validate_notebook,
    sanitize_filename,
    append_jsonl
)


class KaggleDownloader:
    """Download Kaggle notebooks with rate limiting and error handling."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the downloader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.logger = setup_logger(
            "downloader",
            log_dir=self.config['storage']['logs_dir'],
            level=self.config['logging']['level'],
            format_type=self.config['logging']['format']
        )
        
        # Paths
        self.notebooks_dir = Path(self.config['storage']['notebooks_dir'])
        self.metadata_dir = Path(self.config['storage']['metadata_dir'])
        
        ensure_dir(self.notebooks_dir)
        
        # Progress tracking
        self.progress = ProgressTracker(self.metadata_dir / "download_progress.json")
        
        # Download log
        self.download_log = self.metadata_dir / "downloads.jsonl"
        
        # Rate limiting
        self.rate_limit_delay = self.config.get('rate_limit_delay', 2)
        self.last_request_time = 0
        
        self.logger.info("KaggleDownloader initialized")
    
    def _apply_rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    @retry_with_backoff(max_tries=3)
    def download_kernel(self, kernel_ref: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Download a single kernel.
        
        Args:
            kernel_ref: Kernel reference (username/kernel-slug)
            output_dir: Output directory (default: notebooks/)
        
        Returns:
            Download result dictionary with status and metadata
        """
        if self.progress.is_completed(kernel_ref):
            self.logger.info(f"Skipping already downloaded: {kernel_ref}")
            return {
                'kernel_ref': kernel_ref,
                'status': 'skipped',
                'reason': 'already_downloaded',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        if output_dir is None:
            output_dir = self.notebooks_dir
        
        # Create subdirectory for this kernel
        kernel_slug = kernel_ref.replace('/', '_')
        kernel_dir = output_dir / kernel_slug
        ensure_dir(kernel_dir)
        
        self.logger.info(f"Downloading kernel: {kernel_ref}")
        
        result = {
            'kernel_ref': kernel_ref,
            'download_dir': str(kernel_dir),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Download using Kaggle CLI
            command = [
                'kaggle', 'kernels', 'pull',
                kernel_ref,
                '--path', str(kernel_dir),
                '--metadata'
            ]
            
            start_time = time.time()
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config['timeout'],
                check=True
            )
            
            download_time = time.time() - start_time
            
            # Find the downloaded notebook file
            notebook_files = list(kernel_dir.glob('*.ipynb'))
            
            if not notebook_files:
                raise FileNotFoundError(f"No .ipynb file found in {kernel_dir}")
            
            notebook_path = notebook_files[0]
            
            # Validate notebook
            if not validate_notebook(notebook_path):
                raise ValueError(f"Invalid notebook file: {notebook_path}")
            
            # Get file size
            file_size = notebook_path.stat().st_size
            
            result.update({
                'status': 'success',
                'notebook_path': str(notebook_path),
                'file_size_bytes': file_size,
                'download_time_seconds': download_time,
                'stdout': process.stdout[:500] if process.stdout else '',
            })
            
            # Mark as completed
            self.progress.mark_completed(kernel_ref)
            
            # Log download
            append_jsonl(result, self.download_log)
            
            self.logger.info(f"Successfully downloaded: {kernel_ref} ({file_size} bytes)")
            
        except subprocess.TimeoutExpired:
            result['status'] = 'failed'
            result['error'] = 'timeout'
            self.progress.mark_failed(kernel_ref)
            self.logger.error(f"Timeout downloading {kernel_ref}")
            append_jsonl(result, self.download_log)
            
        except subprocess.CalledProcessError as e:
            result['status'] = 'failed'
            result['error'] = 'api_error'
            result['stderr'] = e.stderr[:500] if e.stderr else ''
            self.progress.mark_failed(kernel_ref)
            self.logger.error(f"API error downloading {kernel_ref}: {e.stderr}")
            append_jsonl(result, self.download_log)
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            self.progress.mark_failed(kernel_ref)
            self.logger.error(f"Failed to download {kernel_ref}: {e}")
            append_jsonl(result, self.download_log)
        
        return result
    
    def download_batch(
        self,
        kernel_refs: List[str],
        max_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download multiple kernels in parallel.
        
        Args:
            kernel_refs: List of kernel references
            max_workers: Number of parallel workers (default: from config)
        
        Returns:
            Summary statistics dictionary
        """
        if max_workers is None:
            max_workers = self.config.get('max_workers', 4)
        
        enable_parallel = self.config.get('enable_parallel', True)
        if not enable_parallel:
            max_workers = 1
        
        self.logger.info(f"Starting batch download of {len(kernel_refs)} kernels with {max_workers} workers")
        
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        
        with Timer("Batch Download", self.logger):
            if max_workers == 1:
                # Sequential download
                for kernel_ref in kernel_refs:
                    result = self.download_kernel(kernel_ref)
                    status = result['status']
                    results[status].append(result)
            else:
                # Parallel download
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_ref = {
                        executor.submit(self.download_kernel, ref): ref
                        for ref in kernel_refs
                    }
                    
                    for future in as_completed(future_to_ref):
                        ref = future_to_ref[future]
                        try:
                            result = future.result()
                            status = result['status']
                            results[status].append(result)
                            
                            # Progress update
                            total_processed = sum(len(v) for v in results.values())
                            if total_processed % 10 == 0:
                                self.logger.info(
                                    f"Progress: {total_processed}/{len(kernel_refs)} "
                                    f"(Success: {len(results['success'])}, "
                                    f"Failed: {len(results['failed'])}, "
                                    f"Skipped: {len(results['skipped'])})"
                                )
                        except Exception as e:
                            self.logger.error(f"Exception processing {ref}: {e}")
                            results['failed'].append({
                                'kernel_ref': ref,
                                'status': 'failed',
                                'error': str(e)
                            })
        
        # Save summary
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_requested': len(kernel_refs),
            'success_count': len(results['success']),
            'failed_count': len(results['failed']),
            'skipped_count': len(results['skipped']),
            'success_rate': len(results['success']) / len(kernel_refs) if kernel_refs else 0,
            'results': results
        }
        
        summary_file = self.metadata_dir / f"download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(summary, summary_file)
        
        self.logger.info(
            f"Batch download complete: "
            f"{summary['success_count']} success, "
            f"{summary['failed_count']} failed, "
            f"{summary['skipped_count']} skipped"
        )
        
        return summary
    
    def download_from_metadata(
        self,
        metadata_file: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Download kernels from metadata file.
        
        Args:
            metadata_file: Path to kernels metadata JSON file
            limit: Maximum number of kernels to download
        
        Returns:
            Download summary
        """
        if metadata_file is None:
            metadata_file = self.metadata_dir / "kernels" / "all_kernels_metadata.json"
        
        metadata_file = Path(metadata_file)
        
        if not metadata_file.exists():
            self.logger.error(f"Metadata file not found: {metadata_file}")
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load kernel metadata
        kernels = load_json(metadata_file)
        
        if not kernels:
            self.logger.warning("No kernels found in metadata file")
            return {}
        
        self.logger.info(f"Loaded {len(kernels)} kernels from metadata")
        
        # Extract kernel refs
        kernel_refs = []
        for kernel in kernels:
            ref = kernel.get('ref')
            if ref:
                kernel_refs.append(ref)
        
        # Apply limit
        if limit:
            kernel_refs = kernel_refs[:limit]
            self.logger.info(f"Limited to {limit} kernels")
        
        # Download
        return self.download_batch(kernel_refs)
    
    def download_from_list(self, list_file: str) -> Dict[str, Any]:
        """
        Download kernels from a text file (one ref per line).
        
        Args:
            list_file: Path to text file with kernel refs
        
        Returns:
            Download summary
        """
        list_path = Path(list_file)
        
        if not list_path.exists():
            self.logger.error(f"List file not found: {list_file}")
            raise FileNotFoundError(f"List file not found: {list_file}")
        
        # Read kernel refs
        with open(list_path, 'r') as f:
            kernel_refs = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"Loaded {len(kernel_refs)} kernel refs from {list_file}")
        
        # Download
        return self.download_batch(kernel_refs)
    
    def cleanup_failed_downloads(self):
        """Remove directories for failed downloads."""
        self.logger.info("Cleaning up failed downloads...")
        
        failed_refs = list(self.progress.failed)
        cleaned = 0
        
        for kernel_ref in failed_refs:
            kernel_slug = kernel_ref.replace('/', '_')
            kernel_dir = self.notebooks_dir / kernel_slug
            
            if kernel_dir.exists():
                try:
                    shutil.rmtree(kernel_dir)
                    cleaned += 1
                    self.logger.info(f"Removed failed download: {kernel_dir}")
                except Exception as e:
                    self.logger.error(f"Failed to remove {kernel_dir}: {e}")
        
        self.logger.info(f"Cleaned up {cleaned} failed download directories")
        
        return cleaned
    
    def get_download_stats(self) -> Dict[str, Any]:
        """Get download statistics."""
        stats = self.progress.get_stats()
        
        # Count actual files
        notebook_files = list(self.notebooks_dir.rglob('*.ipynb'))
        
        stats.update({
            'downloaded_files': len(notebook_files),
            'notebooks_directory': str(self.notebooks_dir),
            'total_size_bytes': sum(f.stat().st_size for f in notebook_files)
        })
        
        return stats


def main():
    """Command-line interface for downloading notebooks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Kaggle notebooks")
    parser.add_argument('--config', default='config/config.yaml', help='Config file path')
    parser.add_argument('--metadata', help='Path to kernels metadata JSON file')
    parser.add_argument('--list', help='Path to text file with kernel refs')
    parser.add_argument('--limit', type=int, help='Maximum kernels to download')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    parser.add_argument('--cleanup', action='store_true', help='Clean up failed downloads')
    parser.add_argument('--stats', action='store_true', help='Show download statistics')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = KaggleDownloader(config_path=args.config)
    
    # Update workers if specified
    if args.workers:
        downloader.config['max_workers'] = args.workers
    
    # Show stats and exit
    if args.stats:
        stats = downloader.get_download_stats()
        print(f"\n{'='*60}")
        print("DOWNLOAD STATISTICS")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print(f"{'='*60}\n")
        return
    
    # Cleanup and exit
    if args.cleanup:
        cleaned = downloader.cleanup_failed_downloads()
        print(f"Cleaned up {cleaned} failed download directories")
        return
    
    # Download from list or metadata
    if args.list:
        summary = downloader.download_from_list(args.list)
    else:
        summary = downloader.download_from_metadata(
            metadata_file=args.metadata,
            limit=args.limit
        )
    
    # Print summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Total Requested: {summary['total_requested']}")
    print(f"Success: {summary['success_count']}")
    print(f"Failed: {summary['failed_count']}")
    print(f"Skipped: {summary['skipped_count']}")
    print(f"Success Rate: {summary['success_rate']:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
