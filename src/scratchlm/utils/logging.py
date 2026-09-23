"""
Logging configuration for ScratchLM.

Provides standardized logging setup across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
import datetime


# Default format for console logging
CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Date format
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColorFormatter(logging.Formatter):
    """Formatter that adds colors to log messages."""
    
    # ANSI color codes
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    # Color mapping by level
    COLORS = {
        logging.DEBUG: CYAN,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }
    
    def format(self, record):
        # Add color based on level
        color = self.COLORS.get(record.levelno, self.WHITE)
        levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Format the message
        message = super().format(record)
        
        # Replace the levelname with colored version
        return message.replace(record.levelname, levelname)


def setup_logging(
    name: str = "scratchlm",
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    color: bool = True,
    propagate: bool = False,
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional path to log file
        console: Whether to log to console
        color: Whether to use colored output (console only)
        propagate: Whether to propagate to root logger
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = ColorFormatter(CONSOLE_FORMAT, DATE_FORMAT) if color else \
                logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT)
    file_formatter = logging.Formatter(FILE_FORMAT, DATE_FORMAT)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "scratchlm") -> logging.Logger:
    """
    Get a logger with default configuration.
    
    This is a convenience function that returns a logger with sensible defaults.
    For more control, use setup_logging() directly.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class TrainingLogger:
    """Specialized logger for training runs."""
    
    def __init__(
        self,
        experiment_dir: Path,
        name: str = "training",
        level: int = logging.INFO,
        console: bool = True,
        color: bool = True,
    ):
        """
        Initialize training logger.
        
        Args:
            experiment_dir: Directory for this experiment
            name: Logger name
            level: Logging level
            console: Log to console
            color: Use colored output
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file path
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.experiment_dir / f"{timestamp}_train.log"
        
        # Set up logger
        self.logger = setup_logging(
            name=name,
            level=level,
            log_file=self.log_file,
            console=console,
            color=color,
        )
        
        # Also create a metrics log file (CSV format)
        self.metrics_file = self.experiment_dir / f"{timestamp}_metrics.csv"
        self.metrics_header_written = False
    
    def log_metrics(self, metrics: dict, step: int, epoch: Optional[int] = None):
        """
        Log training metrics to CSV file.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step number
            epoch: Optional epoch number
        """
        # Write header if not already written
        if not self.metrics_header_written:
            with open(self.metrics_file, 'w') as f:
                header = ["step", "epoch"] + list(metrics.keys())
                f.write(",".join(header) + "\n")
            self.metrics_header_written = True
        
        # Write metrics
        with open(self.metrics_file, 'a') as f:
            row = [str(step), str(epoch) if epoch is not None else ""] + \
                  [str(metrics.get(k, "")) for k in metrics.keys()]
            f.write(",".join(row) + "\n")
        
        # Also log to main logger
        metrics_str = " | ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        if epoch is not None:
            self.logger.info(f"Step {step} | Epoch {epoch} | {metrics_str}")
        else:
            self.logger.info(f"Step {step} | {metrics_str}")
    
    def log_checkpoint(self, checkpoint_path: Path, metrics: dict, step: int, epoch: int):
        """Log checkpoint creation."""
        metrics_str = " | ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
        self.logger.info(f"Checkpoint saved: {checkpoint_path} | Step {step} | Epoch {epoch} | {metrics_str}")
    
    def log_evaluation(self, results: dict, step: int, epoch: int, split: str = "val"):
        """Log evaluation results."""
        results_str = " | ".join(f"{k}: {v:.6f}" for k, v in results.items())
        self.logger.info(f"Evaluation ({split}) | Step {step} | Epoch {epoch} | {results_str}")


if __name__ == "__main__":
    # Test logging
    logger = setup_logging("test", level=logging.DEBUG, console=True, color=True)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
