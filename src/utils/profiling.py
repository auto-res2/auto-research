"""
Utility functions for profiling memory usage and performance
"""
import time
import numpy as np
from memory_profiler import memory_usage
import torch

class PerformanceTracker:
    """Track runtime, memory usage, and other performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.peak_memory = 0
        self.step_count = 0
        self.metrics = {}
        
    def start(self):
        """Start tracking performance"""
        self.start_time = time.time()
        return self
        
    def end(self):
        """End tracking and calculate elapsed time"""
        if self.start_time is None:
            raise ValueError("Tracker was not started")
        elapsed = time.time() - self.start_time
        self.metrics["runtime"] = elapsed
        return elapsed
        
    def update_step_count(self, steps):
        """Update the number of steps taken"""
        self.step_count = steps
        self.metrics["steps"] = steps
        
    def update_peak_memory(self, peak_mb):
        """Update the peak memory usage"""
        self.peak_memory = peak_mb
        self.metrics["peak_memory_mb"] = peak_mb
        
    def get_metrics(self):
        """Get all tracked metrics"""
        return self.metrics

def measure_peak_memory(func, *args, **kwargs):
    """
    Measure peak memory usage of a function
    
    Args:
        func: Function to measure
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        (result, peak_memory_mb)
    """
    def wrapper():
        return func(*args, **kwargs)
        
    mem_usage = memory_usage((wrapper, ), interval=0.1, timeout=None, 
                             max_iterations=1, include_children=True)
    peak_memory = max(mem_usage) if mem_usage else 0
    
    # Run again to get the result
    result = func(*args, **kwargs)
    
    return result, peak_memory

def track_gpu_memory():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0
