"""
Metrics Logger for PPO Training
--------------------------------
Records training metrics to JSON without affecting training output

Usage:
    from metrics_logger import MetricsLogger
    
    logger = MetricsLogger("runs/pico64/metrics.json")
    logger.log_iteration(
        iteration=100,
        level="C0",
        mean_return=45.3,
        std_return=5.2,
        policy_loss=0.123,
        value_loss=0.456,
        elapsed_time=300.5
    )
    logger.save()  # Automatically called periodically
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class MetricEntry:
    """Single iteration metric record"""
    iteration: int
    level: str
    mean_return: float
    std_return: float
    policy_loss: float
    value_loss: float
    elapsed_time: float  # seconds


class MetricsLogger:
    def __init__(self, filepath: str):
        """
        Initialize metrics logger.
        
        Args:
            filepath: Path to save metrics JSON file (e.g., "runs/pico64/metrics.json")
        """
        self.filepath = filepath
        self.metrics: List[MetricEntry] = []
        self._load_existing()
    
    def _load_existing(self):
        """Try to load existing metrics from file"""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    self.metrics = [MetricEntry(**d) for d in data]
            except Exception:
                self.metrics = []
    
    def log_iteration(self,
                      iteration: int,
                      level: str,
                      mean_return: float,
                      std_return: float,
                      policy_loss: float,
                      value_loss: float,
                      elapsed_time: float):
        """
        Log a single training iteration.
        
        Args:
            iteration: Global iteration number
            level: Curriculum level name (e.g., "EASY", "C0", "C1")
            mean_return: Mean evaluation return
            std_return: Std dev of evaluation return
            policy_loss: Policy loss value
            value_loss: Value function loss value
            elapsed_time: Elapsed time in seconds since training start
        """
        entry = MetricEntry(
            iteration=iteration,
            level=level,
            mean_return=mean_return,
            std_return=std_return,
            policy_loss=policy_loss,
            value_loss=value_loss,
            elapsed_time=elapsed_time
        )
        self.metrics.append(entry)
        self.save()
    
    def save(self):
        """Save metrics to JSON file"""
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
