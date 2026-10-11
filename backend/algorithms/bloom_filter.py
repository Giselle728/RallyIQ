"""
RallyIQ Bloom Filter & Stream Integrity Monitor
Implements:
1. Configurable Bit-Array Bloom Filter with double-hashing (Kirsch-Mitzenmacher optimization).
2. Theoretical false-positive estimation: p = (1 - e^(-kn/m))^k
3. Exact verification against ground-truth HashSet to compute empirical false positives.
4. Stream processing pipeline for incremental tennis point events with MongoDB logging.
"""

import math
import hashlib
from typing import List, Set, Dict, Any, Optional

class BloomFilter:
    """
    Space-efficient probabilistic data structure for set membership queries.
    Parameters:
      m: bit array size (in bits)
      k: number of hash functions
    """
    def __init__(self, m: int = 8192, k: int = 5):
        if m <= 0 or k <= 0:
            raise ValueError("m and k must be positive integers.")
        self.m = m
        self.k = k
        self.bit_array = bytearray((m + 7) // 8)  # Byte array representing m bits
        self.n = 0  # Number of inserted items
        self.exact_set: Set[str] = set()  # Ground-truth hash set for exact verification
        self.total_queries = 0
        self.false_positive_count = 0
        self.true_positive_count = 0
        self.true_negative_count = 0
        self.duplicate_reject_count = 0

    def _get_hashes(self, item: str) -> List[int]:
        """
        Generate k hash positions using Kirsch-Mitzenmacher double hashing:
        h_i(x) = (h1(x) + i * h2(x)) mod m
        """
        data = item.encode("utf-8")
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        h2 = int(hashlib.sha256(data).hexdigest(), 16)
        
        hashes = []
        for i in range(self.k):
            pos = (h1 + i * h2) % self.m
            hashes.append(pos)
        return hashes

    def is_exact_member(self, item: str) -> bool:
        """Ground-truth exact check against the set of inserted items."""
        return item in self.exact_set

    def record_duplicate(self, item: str) -> None:
        """
        Record a confirmed duplicate rejection without modifying n or exact_set.
        Increments duplicate_reject_count and true_positive_count.
        """
        self.duplicate_reject_count += 1
        self.true_positive_count += 1
        self.total_queries += 1

    def commit_new_item(self, item: str) -> bool:
        """
        Safely commit a verified new unique item into the Bloom filter and exact set.
        Increments n exactly once. Sets bits and adds to exact_set.
        Returns True if item was a false positive prior to insertion.
        """
        if item in self.exact_set:
            self.record_duplicate(item)
            return False

        is_fp = self.contains(item)
        hashes = self._get_hashes(item)
        for bit_idx in hashes:
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            self.bit_array[byte_idx] |= (1 << bit_offset)

        self.exact_set.add(item)
        self.n += 1
        return is_fp

    def add(self, item: str) -> bool:
        """
        Insert an item into the Bloom filter with exact ground-truth verification.
        Returns True if item was already a confirmed duplicate (rejected).
        Returns False if item was a new item (even if false positive) and was inserted.
        """
        if item in self.exact_set:
            self.record_duplicate(item)
            return True

        # Check if absent item triggers Bloom filter false positive
        if self.contains(item):
            self.false_positive_count += 1
            self.total_queries += 1

        self.commit_new_item(item)
        return False

    def contains(self, item: str) -> bool:
        """Query if item is possibly in the set."""
        hashes = self._get_hashes(item)
        for bit_idx in hashes:
            byte_idx = bit_idx // 8
            bit_offset = bit_idx % 8
            if not (self.bit_array[byte_idx] & (1 << bit_offset)):
                return False
        return True

    def query_with_verification(self, item: str) -> Dict[str, Any]:
        """
        Query item with ground-truth verification to detect empirical false positives
        without altering the Bloom filter state or inserting the item.
        Tracks query statistics correctly:
        - If item is absent: negative query (True Negative or False Positive).
        - If item is present: positive query (True Positive / Duplicate).
        """
        self.total_queries += 1
        bloom_result = self.contains(item)
        exact_result = item in self.exact_set
        
        is_false_positive = bloom_result and not exact_result
        if is_false_positive:
            self.false_positive_count += 1
        elif bloom_result and exact_result:
            self.true_positive_count += 1
        elif not bloom_result and not exact_result:
            self.true_negative_count += 1

        return {
            "item": item,
            "bloom_result": bloom_result,
            "exact_result": exact_result,
            "is_false_positive": is_false_positive,
            "is_duplicate": exact_result,
            "theoretical_fp_rate": self.theoretical_fp_rate(),
            "empirical_fp_rate": self.empirical_fp_rate()
        }

    def bits_set_count(self) -> int:
        """Count number of bits set to 1."""
        count = 0
        for byte in self.bit_array:
            count += bin(byte).count("1")
        return count

    def saturation(self) -> float:
        """Percentage of bit array set to 1."""
        return (self.bits_set_count() / self.m) * 100.0 if self.m > 0 else 0.0

    def theoretical_fp_rate(self) -> float:
        """
        Calculate theoretical false positive probability: p = (1 - e^(-kn/m))^k
        Returns raw float with full precision (not rounded to 6 decimal places).
        Handles n=0 and edge cases safely.
        """
        if self.m <= 0 or self.k <= 0 or self.n <= 0:
            return 0.0
        exponent = - (float(self.k) * float(self.n)) / float(self.m)
        p = math.pow(1.0 - math.exp(exponent), self.k)
        return float(min(1.0, max(0.0, p)))

    def empirical_fp_rate(self) -> float:
        """
        Empirical false positive rate observed across all negative queries (queries on absent items).
        Formula: FP / (FP + TN)
        Duplicate queries (where exact_result is True) are strictly excluded from the denominator.
        """
        negative_queries = self.false_positive_count + self.true_negative_count
        if negative_queries == 0:
            return 0.0
        return float(self.false_positive_count / negative_queries)

    def get_metrics(self) -> Dict[str, Any]:
        """Return diagnostic metrics of the Bloom filter."""
        bits_on = self.bits_set_count()
        neg_queries = self.false_positive_count + self.true_negative_count
        return {
            "bit_array_size_m": self.m,
            "hash_functions_k": self.k,
            "inserted_items_n": self.n,
            "bits_set": bits_on,
            "bit_density_pct": round((bits_on / self.m) * 100.0, 2) if self.m > 0 else 0.0,
            "theoretical_fp_prob": self.theoretical_fp_rate(),
            "empirical_fp_rate": self.empirical_fp_rate(),
            "false_positives_detected": self.false_positive_count,
            "duplicate_rejections": self.duplicate_reject_count,
            "exact_set_size": len(self.exact_set),
            "negative_queries_tested": neg_queries,
            "true_negatives": self.true_negative_count,
            "true_positives": self.true_positive_count,
            "total_queries": self.total_queries,
            "memory_bytes": len(self.bit_array)
        }

    def reset(self, m: Optional[int] = None, k: Optional[int] = None) -> None:
        """Reset the Bloom filter with optional new parameters."""
        if m is not None:
            if m <= 0:
                raise ValueError("m must be a positive integer.")
            self.m = m
        if k is not None:
            if k <= 0:
                raise ValueError("k must be a positive integer.")
            self.k = k
        self.bit_array = bytearray((self.m + 7) // 8)
        self.n = 0
        self.exact_set.clear()
        self.total_queries = 0
        self.false_positive_count = 0
        self.true_positive_count = 0
        self.true_negative_count = 0
        self.duplicate_reject_count = 0
