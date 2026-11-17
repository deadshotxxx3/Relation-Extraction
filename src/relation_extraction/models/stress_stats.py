from dataclasses import dataclass, field
from collections import Counter


@dataclass
class StressStats:
    original_no_path: int = 0
    noisy_no_path: int = 0
    original_failed: int = 0
    noisy_failed: int = 0
    comparable: int = 0
    unchanged: int = 0
    insertion_skipped: int = 0
    template_usage: Counter = field(default_factory=Counter)
