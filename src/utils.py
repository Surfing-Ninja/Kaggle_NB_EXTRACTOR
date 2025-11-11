"""
Utility functions and helpers for the Kaggle notebook extraction project.
Includes logging, retry logic, rate limiting, file operations, and more.
"""

import json
import logging
import time
import hashlib
import yaml
from pathlib import Path
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional, Callable
import colorlog
import backoff
from ratelimit import limits, sleep_and_retry


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")


def load_filters(filters_path: str = "config/filters.json") -> Dict[str, Any]:
    """Load JSON filters file."""
    try:
        with open(filters_path, 'r') as f:
            filters = json.load(f)
        return filters
    except Exception as e:
        raise RuntimeError(f"Failed to load filters from {filters_path}: {e}")


# =============================================================================
# LOGGING SETUP
# =============================================================================

class JSONLFormatter(logging.Formatter):
    """Custom formatter that outputs JSON lines."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)
        
        return json.dumps(log_obj)


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    format_type: str = "jsonl",
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Set up logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_type: 'jsonl' or 'text'
        console_output: Enable console logging
        file_output: Enable file logging
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # Remove existing handlers
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Console handler with color
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        if format_type == "text":
            console_format = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        else:
            console_format = JSONLFormatter()
        
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        if format_type == "jsonl":
            file_formatter = JSONLFormatter()
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_with_extra(logger: logging.Logger, level: str, message: str, **kwargs):
    """Log message with extra data fields."""
    extra = {"extra_data": kwargs}
    getattr(logger, level.lower())(message, extra=extra)


# =============================================================================
# RETRY & RATE LIMITING DECORATORS
# =============================================================================

def retry_with_backoff(
    max_tries: int = 3,
    max_time: int = 300,
    expo_base: int = 2,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_tries: Maximum number of retry attempts
        max_time: Maximum time to spend retrying (seconds)
        expo_base: Base for exponential backoff
        logger: Logger instance for logging retries
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            max_time=max_time,
            base=expo_base,
            on_backoff=lambda details: logger.warning(
                f"Retry attempt {details['tries']} for {func.__name__} after {details['wait']:.2f}s"
            ) if logger else None
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limited(calls: int = 30, period: int = 60):
    """
    Decorator for rate limiting function calls.
    
    Args:
        calls: Number of calls allowed
        period: Time period in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @sleep_and_retry
        @limits(calls=calls, period=period)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# FILE OPERATIONS
# =============================================================================

def ensure_dir(directory: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Any, filepath: Path, indent: int = 2):
    """Save data to JSON file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def append_jsonl(data: Dict, filepath: Path):
    """Append a JSON object as a new line to a JSONL file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data) + '\n')


def read_jsonl(filepath: Path) -> List[Dict]:
    """Read all JSON objects from a JSONL file."""
    filepath = Path(filepath)
    if not filepath.exists():
        return []
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# =============================================================================
# HASHING & DEDUPLICATION
# =============================================================================

def compute_sha256(content: str) -> str:
    """Compute SHA256 hash of string content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def compute_file_hash(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Compute hash of file contents.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256)
    
    Returns:
        Hex digest of file hash
    """
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def extract_code_cells(notebook_path: Path) -> str:
    """
    Extract all code cells from a notebook for comparison.
    
    Args:
        notebook_path: Path to .ipynb file
    
    Returns:
        Concatenated code cell content
    """
    import nbformat
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        code_cells = []
        for cell in nb.cells:
            if cell.cell_type == 'code':
                code_cells.append(cell.source)
        
        return '\n'.join(code_cells)
    except Exception as e:
        return ""


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ProgressTracker:
    """Track and persist progress for resume capability."""
    
    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)
        self.completed = set()
        self.failed = set()
        self.load()
    
    def load(self):
        """Load progress from file."""
        if self.filepath.exists():
            data = load_json(self.filepath)
            self.completed = set(data.get('completed', []))
            self.failed = set(data.get('failed', []))
    
    def save(self):
        """Save progress to file."""
        data = {
            'completed': list(self.completed),
            'failed': list(self.failed),
            'last_updated': datetime.utcnow().isoformat()
        }
        save_json(data, self.filepath)
    
    def mark_completed(self, item_id: str):
        """Mark an item as completed."""
        self.completed.add(item_id)
        if item_id in self.failed:
            self.failed.remove(item_id)
        self.save()
    
    def mark_failed(self, item_id: str):
        """Mark an item as failed."""
        self.failed.add(item_id)
        self.save()
    
    def is_completed(self, item_id: str) -> bool:
        """Check if item is already completed."""
        return item_id in self.completed
    
    def is_failed(self, item_id: str) -> bool:
        """Check if item has failed."""
        return item_id in self.failed
    
    def get_stats(self) -> Dict[str, int]:
        """Get progress statistics."""
        return {
            'completed': len(self.completed),
            'failed': len(self.failed),
            'total_processed': len(self.completed) + len(self.failed)
        }


# =============================================================================
# TIMING & PERFORMANCE
# =============================================================================

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.logger:
            self.logger.info(f"Completed {self.name} in {elapsed:.2f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time if self.start_time else 0


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_notebook(notebook_path: Path) -> bool:
    """
    Validate that a notebook file is well-formed.
    
    Args:
        notebook_path: Path to .ipynb file
    
    Returns:
        True if valid, False otherwise
    """
    import nbformat
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nbformat.read(f, as_version=4)
        return True
    except Exception:
        return False


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitize filename to remove invalid characters.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
    
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Truncate if too long
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        name = name[:max_length - len(ext) - 1]
        filename = f"{name}.{ext}" if ext else name
    
    return filename


# =============================================================================
# STATISTICS
# =============================================================================

def compute_statistics(data: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of numbers."""
    import numpy as np
    
    if not data:
        return {}
    
    return {
        'count': len(data),
        'mean': float(np.mean(data)),
        'median': float(np.median(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'sum': float(np.sum(data))
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'load_config',
    'load_filters',
    'setup_logger',
    'log_with_extra',
    'retry_with_backoff',
    'rate_limited',
    'ensure_dir',
    'save_json',
    'load_json',
    'append_jsonl',
    'read_jsonl',
    'compute_sha256',
    'compute_file_hash',
    'extract_code_cells',
    'ProgressTracker',
    'Timer',
    'format_duration',
    'validate_notebook',
    'sanitize_filename',
    'compute_statistics',
]
