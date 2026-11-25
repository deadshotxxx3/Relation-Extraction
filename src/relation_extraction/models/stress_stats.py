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
    unchanged_words: int = 0
    changed_through_insert: int = 0
    template_comparable: Counter = field(default_factory=Counter)
    template_unchanged: Counter = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)