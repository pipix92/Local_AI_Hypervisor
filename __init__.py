"""
ai_tui - Local AI & Model Management Terminal User Interface
"""

from .backend import (
    HardwareMonitor,
    HardwareSnapshot,
    GPUStats,
    CPUStats,
    RAMStats,
    LlamaServerManager,
    KVCacheOptimizer,
    AINewsAggregator,
    NewsItem,
    TokenItem,
    SlotInfo,
    ModelMetadata,
    HFModelManager,
    HFModelInfo,
    QuantFileInfo,
    DownloadTask,
)

__version__ = "1.1.0"
__all__ = [
    "HardwareMonitor",
    "HardwareSnapshot",
    "GPUStats",
    "CPUStats",
    "RAMStats",
    "LlamaServerManager",
    "KVCacheOptimizer",
    "AINewsAggregator",
    "NewsItem",
    "TokenItem",
    "SlotInfo",
    "ModelMetadata",
    "HFModelManager",
    "HFModelInfo",
    "QuantFileInfo",
    "DownloadTask",
]
