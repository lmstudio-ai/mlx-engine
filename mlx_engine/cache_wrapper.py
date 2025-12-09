import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream, maybe_quantize_kv_cache
from mlx_lm.models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)

PROMPT_PROCESSING_CHUNK_SIZE = 512

logger = logging.getLogger(__name__)

# Import backward compatibility wrapper for progress callbacks
try:
    from mlx_engine.utils.progress_decorators import backward_compatible
except ImportError:
    # Fallback for cases where progress_decorators might not be available
    def backward_compatible(callback):
        if callback is None:
            return None

        def inner_callback(percentage: float) -> bool:
            try:
                result = callback(percentage)
                if result is None:
                    return True
                return bool(result)
            except Exception:
                return True

        return inner_callback


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    trims: int = 0
    size_bytes: int = 0
    utilization_ratio: float = 0.0
    max_size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size_gb(self) -> float:
        """Get cache size in GB."""
        return self.size_bytes / (1024**3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "trims": self.trims,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "utilization_ratio": self.utilization_ratio,
            "hit_rate": self.hit_rate,
            "max_size_bytes": self.max_size_bytes,
        }


class StopPromptProcessing(Exception):
    """
    Exception to signal that the user aborted generation during prompt processing.
    """


class CacheWrapper:
    """
    Wrapper class for the MLX LM cache to maintain an in-memory cache
    """

    def __init__(
        self,
        model: nn.Module,
        max_kv_size: Optional[int],
        *,
        verbose: bool = False,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
        chunk_size: int = PROMPT_PROCESSING_CHUNK_SIZE,
    ):
        """
        Initialize the CacheWrapper.

        Args:
            model (nn.Module): The model to be cached.
            max_kv_size (Optional[int]): Maximum size of the key-value cache.
        """
        # utilize a simple ordered list of tokens processed so far for cache invalidation checking
        self.tokens: Optional[mx.array] = None
        self.cache: List[Any] = make_prompt_cache(model, max_kv_size)
        self.model = model
        self.draft_model: Optional[nn.Module] = None
        self.max_kv_size = max_kv_size
        self.verbose = verbose
        self.kv_cache_qtn_params = dict(
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self.chunk_size = chunk_size

        # Cache statistics tracking
        self.stats = CacheStats()
        self.stats.max_size_bytes = (
            max_kv_size * 4 if max_kv_size else 0
        )  # Rough estimate

    def _get_num_tokens_in_cache(self) -> int | None:
        """
        Get the number of tokens in the cache.

        Returns:
            int | None: The number of tokens in the cache, or None if the size cannot be determined.
        """
        for c in self.cache:
            if hasattr(c, "offset"):
                return c.offset
        return None

    def _update_cache_stats(self) -> None:
        """Update cache statistics."""
        try:
            # Update cache size estimate
            num_tokens = self._get_num_tokens_in_cache()
            if num_tokens is not None:
                # Rough estimate: 2KB per token for KV cache
                self.stats.size_bytes = num_tokens * 2048

            # Update utilization ratio
            if self.stats.max_size_bytes > 0:
                self.stats.utilization_ratio = (
                    self.stats.size_bytes / self.stats.max_size_bytes
                )
        except Exception as e:
            logger.debug(f"Failed to update cache stats: {e}")

    def get_cache_stats(self) -> CacheStats:
        """Get current cache statistics."""
        self._update_cache_stats()
        return self.stats

    def reset_cache_stats(self) -> None:
        """Reset cache statistics."""
        self.stats = CacheStats()
        self.stats.max_size_bytes = self.max_kv_size * 4 if self.max_kv_size else 0

    @staticmethod
    def _find_common_prefix(
        current_tokens: mx.array, prompt_tokens: mx.array, num_tokens_to_exclude: int
    ) -> int:
        """
        Determine the common prefix length between the current tokens and the prompt tokens.

        Args:
            current_tokens (mx.array): The cached tokens (self.tokens).
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The minimum length of the remaining prompt tokens array.

        Returns:
            int: The length of the common prefix.
        """
        prompt_tokens = prompt_tokens
        current_tokens = current_tokens
        # Find the minimum length between the two arrays
        min_length = min(len(current_tokens), len(prompt_tokens))

        # Compare elements up to the minimum length
        mask = prompt_tokens[:min_length] == current_tokens[:min_length]

        # Find the index where the first mismatch occurs
        if mx.any(mask == False):  # noqa E712
            common_length = int(mx.argmax(mask == False))  # noqa E712
        else:
            common_length = int(min_length)

        # Adjust for num_tokens_to_exclude
        common_length = max(common_length - num_tokens_to_exclude, 0)
        return common_length

    def _get_unprocessed_tokens(
        self, prompt_tokens: mx.array, num_tokens_to_exclude: int
    ):
        """
        Get the unprocessed tokens from the prompt.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The unprocessed tokens.
        """
        if self.tokens is None:
            self.tokens = prompt_tokens
            return self.tokens

        # Find common KV between the last generation and the current prompt
        common_prefix = self._find_common_prefix(
            self.tokens, prompt_tokens, num_tokens_to_exclude
        )

        # Trim the cache if the common prefix is shorter than the current cache
        num_tokens_in_cache = self._get_num_tokens_in_cache()
        if num_tokens_in_cache is None:
            logger.warning(
                "Could not determine the number of tokens in the cache, clearing the cache."
            )
            self.stats.misses += 1
            self.stats.evictions += 1
            self.cache = make_prompt_cache(self.model, self.max_kv_size)
            self.tokens = prompt_tokens
            return self.tokens
        num_tokens_to_trim = num_tokens_in_cache - common_prefix
        if num_tokens_to_trim > 0:
            if not can_trim_prompt_cache(self.cache):
                logger.warning(
                    f"Tried to trim '{num_tokens_to_trim}' tokens from the prompt cache, but could not: Cache is not trimmable. Clearing the cache instead."
                )
                self.stats.misses += 1
                self.stats.evictions += 1
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                return self.tokens
            tokens_trimmed = trim_prompt_cache(self.cache, num_tokens_to_trim)
            if tokens_trimmed != num_tokens_to_trim:
                # If we trimmed fewer tokens than expected, the cache is invalid
                logger.error(
                    f"Tokens trimmed from cache ({tokens_trimmed}) is less than expected ({num_tokens_to_trim}). Clearing the cache."
                )
                self.stats.misses += 1
                self.stats.evictions += 1
                self.cache = make_prompt_cache(self.model, self.max_kv_size)
                self.tokens = prompt_tokens
                return self.tokens
            logger.info(f"Trimmed {num_tokens_to_trim} tokens from the prompt cache")
            self.stats.trims += 1
        else:
            # Cache hit - no trimming needed
            self.stats.hits += 1

        # Keep track of the prompt tokens
        self.tokens = prompt_tokens

        if self.verbose:
            print(f"Common prefix length: {common_prefix}", file=sys.stderr)
            print(f"Trimmed tokens: {num_tokens_to_trim}", file=sys.stderr)

        # All of the common tokens are now in the cache, so we can return the remaining tokens that still need to be processed
        return prompt_tokens[common_prefix:]

    def _prefill(
        self,
        model,
        cache,
        tokens,
        progress_callback: Callable[[float], bool],
        start_progress: float,
        end_progress: float,
    ):
        """
        Fill a KV cache for a specific model

        Args:
            model: The model to use for cache filling
            cache: The cache to fill
            tokens: Tokens to process
            progress_callback: Callback for reporting progress
            start_progress: Starting progress percentage
            end_progress: Ending progress percentage
        """
        remaining_tokens = tokens
        num_processed = 0
        total_tokens = len(tokens)

        while remaining_tokens.size > 0:
            current_chunk_size = min(self.chunk_size, remaining_tokens.size)
            current_chunk = remaining_tokens[:current_chunk_size]

            model(current_chunk[None], cache=cache)
            maybe_quantize_kv_cache(prompt_cache=cache, **self.kv_cache_qtn_params)
            mx.eval([c.state for c in cache])

            remaining_tokens = remaining_tokens[current_chunk_size:]
            num_processed += current_chunk_size

            # Scale progress to fit between start_progress and end_progress
            progress = start_progress + (end_progress - start_progress) * (
                num_processed / total_tokens
            )
            mx.clear_cache()
            should_continue = progress_callback(progress)
            if should_continue is False:  # If it's None, assume continue generation
                logger.info("Prompt processing was cancelled by the user.")
                num_tokens_in_cache = self._get_num_tokens_in_cache()
                if num_tokens_in_cache is not None and num_tokens_in_cache > len(
                    self.tokens
                ):
                    logger.warning(
                        "The number of tokens in the cache is greater than the number of prompt tokens. This is unexpected. Clearing the cache."
                    )
                    num_tokens_in_cache = None
                if num_tokens_in_cache is None:
                    self.cache = make_prompt_cache(self.model, self.max_kv_size)
                    self.tokens = None
                else:
                    # Remember which tokens were processed so far, so that we can continue processing at a later point
                    self.tokens = self.tokens[:num_tokens_in_cache]
                raise StopPromptProcessing

    def set_draft_model(self, draft_model: nn.Module):
        """
        Sets or updates the draft model to use in the cache.

        If the provided draft_model is already set, returns without changes.
        Otherwise, clears existing cache and rebuilds it by combining caches
        from the main model and draft model. Requires a main model to be set first.
        Args:
            draft_model: The draft model to cache. Pass None to remove draft model.

        Raises:
            ValueError: If main model hasn't been set yet.
        """
        if self.model is None:
            raise ValueError("Cannot add a draft model to cache without a main model")
        if self.max_kv_size is not None:
            logger.info("Disabling max_kv_size when setting a draft model for cache")
            self.max_kv_size = None

        if self.draft_model is draft_model:
            # Skip if the exact same draft model instance is already in cache
            return

        # clear the current cache, append draft model cache to the end of the main model cache as per
        # https://github.com/ml-explore/mlx-examples/blob/514502da22f0dc4c1ac439bdf78c07d5ec41acf7/llms/mlx_lm/utils.py#L381-L382
        logger.info("Clearing current prompt cache and adding draft model to the cache")
        self.tokens = None
        self.cache: List[Any] = make_prompt_cache(self.model)
        if draft_model is not None:
            self.cache += make_prompt_cache(draft_model)
        self.draft_model = draft_model

    def unset_draft_model(self):
        """Removes the draft model from the cache if one exists."""
        if self.draft_model is None:
            return
        self.draft_model = None
        self.cache = self.cache[: len(self.model.layers)]

    def update_cache(
        self,
        prompt_tokens: mx.array,
        prompt_progress_callback: Optional[Callable[[float], Union[bool, None]]],
        *,
        num_tokens_to_exclude: int = 1,
    ) -> mx.array:
        """
        Set up the KV cache for the next generation.
        Re-use as much of the KV cache from the previous generation as possible.

        Args:
            prompt_tokens (mx.array): The prompt tokens.
            prompt_progress_callback (Optional[Callable[[float], Union[bool, None]]]): A callback function to report prompt processing progress.
            For backward compatibility, accepts both new-style callbacks that return True/False and old-style
            callbacks that return None or have no explicit return. All callbacks are treated as continuing
            processing unless they explicitly return False.
            num_tokens_to_exclude (int): The number of tokens that should not be added to the cache.

        Returns:
            mx.array: The prompt tokens to be used for the next generation.
        """
        if prompt_progress_callback is None:

            def prompt_progress_callback(_) -> bool:
                return True
        else:
            # Apply backward compatibility wrapper to handle both old (None return) and new (bool return) callback patterns
            prompt_progress_callback = backward_compatible(prompt_progress_callback)

        num_tokens_to_exclude = max(num_tokens_to_exclude, 1)
        prompt_tokens = self._get_unprocessed_tokens(
            prompt_tokens, num_tokens_to_exclude
        )

        # Prefill the cache with the non-excluded prompt tokens
        num_tokens_to_exclude = min(num_tokens_to_exclude, len(prompt_tokens))
        prefill_tokens = prompt_tokens[:-num_tokens_to_exclude]
        prompt_progress_callback(0)
        with mx.stream(generation_stream):
            if self.draft_model is not None:
                # Fill draft model cache (0% to 50% progress)
                draft_cache = self.cache[len(self.model.layers) :]
                self._prefill(
                    model=self.draft_model,
                    cache=draft_cache,
                    tokens=prefill_tokens,
                    progress_callback=prompt_progress_callback,
                    start_progress=0,
                    end_progress=50,
                )
            # Fill main model cache (50% to 100% progress for draft model, 0% to 100% otherwise)
            main_cache = self.cache[: len(self.model.layers)]
            self._prefill(
                model=self.model,
                cache=main_cache,
                tokens=prefill_tokens,
                progress_callback=prompt_progress_callback,
                start_progress=50 if self.draft_model is not None else 0,
                end_progress=100,
            )

        # Return the tokens that must still be processed outside of the cache
        non_prefill_tokens = prompt_tokens[-num_tokens_to_exclude:]
        return non_prefill_tokens

    def record_generated_token(self, token):
        """
        Add the generated token to the token list, so that we can map the token to the KV cache.
        """
        self.tokens = mx.concat([self.tokens, mx.array([token])])


@dataclass
class CacheSlot:
    """Metadata for a cache slot in the multi-slot cache system."""

    cache: Any
    last_used: float = field(default_factory=time.time)
    size: int = 0
    branch_id: str = ""
    pinned: bool = False
    access_count: int = 0


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_accesses: int = 0

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like access for backward compatibility."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Dictionary-like membership test."""
        return hasattr(self, key)


class BranchingCacheWrapper:
    """
    Multi-slot KV cache with LRU eviction for high-bandwidth Apple Silicon support.

    Provides branch management APIs with configurable cache slots and LRU eviction
    while protecting active and pinned branches from eviction.
    """

    def __init__(
        self,
        max_slots: int = 4,
        eviction_policy: str = "lru",
        memory_headroom_ratio: float = 0.1,
    ):
        """
        Initialize the BranchingCacheWrapper.

        Args:
            max_slots: Maximum number of cache slots to maintain
            eviction_policy: Eviction policy ('lru' currently supported)
            memory_headroom_ratio: Ratio of memory to keep free (0.0-1.0)
        """
        if max_slots < 1:
            raise ValueError("max_slots must be at least 1")
        if eviction_policy not in ["lru"]:
            raise ValueError("Only 'lru' eviction policy is currently supported")
        if not 0.0 <= memory_headroom_ratio <= 1.0:
            raise ValueError("memory_headroom_ratio must be between 0.0 and 1.0")

        self.max_slots = max_slots
        self.eviction_policy = eviction_policy
        self.memory_headroom_ratio = memory_headroom_ratio

        # Multi-slot storage: Dict[prompt_hash, CacheSlot]
        self.branches: Dict[str, CacheSlot] = {}

        # LRU tracking using OrderedDict (most recent first)
        self.lru_order: OrderedDict[str, None] = OrderedDict()

        # Active branch tracking
        self.active_branch_id: Optional[str] = None

        # Statistics tracking
        self.stats = CacheStats()

        logger.info(f"Initialized BranchingCacheWrapper with max_slots={max_slots}")

    def _update_lru(self, branch_id: str) -> None:
        """Update LRU order when a branch is accessed."""
        if branch_id in self.lru_order:
            self.lru_order.move_to_end(
                branch_id, last=False
            )  # Move to front (most recent)
        else:
            self.lru_order[branch_id] = None
            self.lru_order.move_to_end(branch_id, last=False)

    def _estimate_cache_size(self, cache: Any) -> int:
        """Estimate the memory size of a cache in bytes."""
        try:
            # Try to get size from cache if available
            if hasattr(cache, "__sizeof__"):
                return cache.__sizeof__()
            # Rough estimate for MLX caches
            if hasattr(cache, "__len__"):
                return len(cache) * 1024  # Rough estimate
            return 1024 * 1024  # 1MB default estimate
        except Exception:
            return 1024  # 1KB fallback

    def _check_memory_headroom(self) -> bool:
        """Check if there's sufficient memory headroom for new cache."""
        try:
            import mlx.core as mx

            current_memory = (
                mx.metal.get_active_memory()
                if hasattr(mx.metal, "get_active_memory")
                else 0
            )
            total_memory = (
                mx.metal.get_cache_memory()
                if hasattr(mx.metal, "get_cache_memory")
                else 8 * 1024**3
            )  # 8GB default
            available_ratio = (total_memory - current_memory) / total_memory
            return available_ratio >= self.memory_headroom_ratio
        except Exception:
            return True  # Assume OK if we can't check

    def _evict_lru_branch(
        self, exclude_active: bool = True, exclude_pinned: bool = True
    ) -> Optional[str]:
        """
        Evict the least recently used branch according to policy.

        Args:
            exclude_active: Whether to exclude the active branch from eviction
            exclude_pinned: Whether to exclude pinned branches from eviction

        Returns:
            The branch_id that was evicted, or None if no branch could be evicted
        """
        if not self.lru_order:
            return None

        # Find the oldest branch that can be evicted
        for branch_id in reversed(list(self.lru_order.keys())):  # Start from oldest
            if exclude_active and branch_id == self.active_branch_id:
                continue
            if (
                exclude_pinned
                and branch_id in self.branches
                and self.branches[branch_id].pinned
            ):
                continue

            # Evict this branch
            del self.lru_order[branch_id]
            if branch_id in self.branches:
                del self.branches[branch_id]
            self.stats.evictions += 1
            logger.info(f"Evicted branch {branch_id} due to LRU policy")
            return branch_id

        return None

    def checkpoint_branch(
        self,
        branch_id: str,
        cache: Any,
        prompt_hash: Optional[str] = None,
        pin: bool = False,
    ) -> None:
        """
        Checkpoint a cache state for a branch.

        Args:
            branch_id: Unique identifier for the branch
            cache: The KV cache to store
            prompt_hash: Optional hash of the prompt for identification
            pin: Whether to pin this branch to prevent eviction
        """
        if not branch_id:
            raise ValueError("branch_id cannot be empty")

        # Use branch_id as prompt_hash if not provided
        if prompt_hash is None:
            prompt_hash = branch_id

        # Check if we need to evict branches
        if len(self.branches) >= self.max_slots and prompt_hash not in self.branches:
            if not self._check_memory_headroom():
                logger.warning("Low memory headroom, attempting eviction")

            evicted = self._evict_lru_branch()
            if evicted is None and len(self.branches) >= self.max_slots:
                # Force eviction if we're at capacity and couldn't evict normally
                logger.warning("Forcing eviction of oldest branch")
                oldest_branch = next(reversed(self.lru_order.keys()), None)
                if oldest_branch and oldest_branch != self.active_branch_id:
                    self._evict_lru_branch(exclude_active=False, exclude_pinned=True)

        # Create or update the cache slot
        cache_size = self._estimate_cache_size(cache)
        slot = CacheSlot(
            cache=cache,
            last_used=time.time(),
            size=cache_size,
            branch_id=branch_id,
            pinned=pin,
            access_count=0,
        )

        self.branches[prompt_hash] = slot
        self._update_lru(prompt_hash)

        logger.info(f"Checkpointed branch {branch_id} (pinned: {pin})")

    def restore_branch(self, branch_id: str) -> Any:
        """
        Restore a cached branch state.

        Args:
            branch_id: The branch identifier to restore

        Returns:
            The cached KV cache

        Raises:
            KeyError: If the branch is not found
        """
        if branch_id not in self.branches:
            self.stats.misses += 1
            self.stats.total_accesses += 1
            raise KeyError(f"Branch {branch_id} not found in cache")

        # Update access statistics
        slot = self.branches[branch_id]
        slot.last_used = time.time()
        slot.access_count += 1

        # Update LRU order
        self._update_lru(branch_id)

        # Set as active branch
        self.active_branch_id = branch_id

        # Update statistics
        self.stats.hits += 1
        self.stats.total_accesses += 1

        logger.info(f"Restored branch {branch_id} (access_count: {slot.access_count})")
        return slot.cache

    def release_branch(self, branch_id: str) -> None:
        """
        Release a branch from the cache without counting as eviction.

        Args:
            branch_id: The branch identifier to release
        """
        if branch_id not in self.branches:
            return

        # Remove from branches and LRU tracking
        del self.branches[branch_id]
        if branch_id in self.lru_order:
            del self.lru_order[branch_id]

        # Clear active branch if this was it
        if self.active_branch_id == branch_id:
            self.active_branch_id = None

        logger.info(f"Released branch {branch_id}")

    def pin_branch(self, branch_id: str) -> None:
        """
        Pin a branch to prevent eviction.

        Args:
            branch_id: The branch identifier to pin

        Raises:
            KeyError: If the branch is not found
        """
        if branch_id not in self.branches:
            raise KeyError(f"Branch {branch_id} not found in cache")

        self.branches[branch_id].pinned = True
        logger.info(f"Pinned branch {branch_id}")

    def unpin_branch(self, branch_id: str) -> None:
        """
        Unpin a branch to allow eviction.

        Args:
            branch_id: The branch identifier to unpin

        Raises:
            KeyError: If the branch is not found
        """
        if branch_id not in self.branches:
            raise KeyError(f"Branch {branch_id} not found in cache")

        self.branches[branch_id].pinned = False
        logger.info(f"Unpinned branch {branch_id}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary containing cache performance metrics
        """
        total_size = sum(slot.size for slot in self.branches.values())
        pinned_count = sum(1 for slot in self.branches.values() if slot.pinned)

        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "evictions": self.stats.evictions,
            "total_accesses": self.stats.total_accesses,
            "hit_rate": self.stats.hits / max(self.stats.total_accesses, 1),
            "total_branches": len(self.branches),
            "max_slots": self.max_slots,
            "active_branch": self.active_branch_id,
            "pinned_branches": pinned_count,
            "total_cache_size_bytes": total_size,
            "utilization": len(self.branches) / self.max_slots,
        }

    def clear_cache(self) -> None:
        """Clear all cached branches and reset statistics."""
        self.branches.clear()
        self.lru_order.clear()
        self.active_branch_id = None
        self.stats = CacheStats()
        logger.info("Cleared all cache branches")

    def list_branches(self) -> List[str]:
        """
        Get a list of all cached branch IDs.

        Returns:
            List of branch identifiers
        """
        return list(self.branches.keys())

    def get_branch_info(self, branch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific branch.

        Args:
            branch_id: The branch identifier

        Returns:
            Dictionary with branch information, or None if not found
        """
        if branch_id not in self.branches:
            return None

        slot = self.branches[branch_id]
        return {
            "branch_id": slot.branch_id,
            "last_used": slot.last_used,
            "size_bytes": slot.size,
            "pinned": slot.pinned,
            "access_count": slot.access_count,
            "is_active": branch_id == self.active_branch_id,
        }


# Global branching cache instance for public API
_global_branching_cache: Optional[BranchingCacheWrapper] = None


def initialize_branching_cache(
    max_slots: int = 4,
    eviction_policy: str = "lru",
    memory_headroom_ratio: float = 0.1,
) -> BranchingCacheWrapper:
    """
    Initialize the global branching cache instance.

    Args:
        max_slots: Maximum number of cache slots to maintain
        eviction_policy: Eviction policy ('lru' currently supported)
        memory_headroom_ratio: Ratio of memory to keep free (0.0-1.0)

    Returns:
        The initialized BranchingCacheWrapper instance
    """
    global _global_branching_cache
    _global_branching_cache = BranchingCacheWrapper(
        max_slots=max_slots,
        eviction_policy=eviction_policy,
        memory_headroom_ratio=memory_headroom_ratio,
    )
    return _global_branching_cache


def get_branching_cache() -> Optional[BranchingCacheWrapper]:
    """
    Get the global branching cache instance.

    Returns:
        The global BranchingCacheWrapper instance, or None if not initialized
    """
    return _global_branching_cache


def checkpoint_branch(
    branch_id: str,
    cache: Any,
    prompt_hash: Optional[str] = None,
    pin: bool = False,
) -> None:
    """
    Checkpoint a cache state for a branch using the global cache instance.

    Args:
        branch_id: Unique identifier for the branch
        cache: The KV cache to store
        prompt_hash: Optional hash of the prompt for identification
        pin: Whether to pin this branch to prevent eviction

    Raises:
        RuntimeError: If branching cache is not initialized
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    _global_branching_cache.checkpoint_branch(branch_id, cache, prompt_hash, pin)


def restore_branch(branch_id: str) -> Any:
    """
    Restore a cached branch state using the global cache instance.

    Args:
        branch_id: The branch identifier to restore

    Returns:
        The cached KV cache

    Raises:
        RuntimeError: If branching cache is not initialized
        KeyError: If the branch is not found
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    return _global_branching_cache.restore_branch(branch_id)


def release_branch(branch_id: str) -> None:
    """
    Release a branch from the cache using the global cache instance.

    Args:
        branch_id: The branch identifier to release

    Raises:
        RuntimeError: If branching cache is not initialized
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    _global_branching_cache.release_branch(branch_id)


def pin_branch(branch_id: str) -> None:
    """
    Pin a branch to prevent eviction using the global cache instance.

    Args:
        branch_id: The branch identifier to pin

    Raises:
        RuntimeError: If branching cache is not initialized
        KeyError: If the branch is not found
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    _global_branching_cache.pin_branch(branch_id)


def unpin_branch(branch_id: str) -> None:
    """
    Unpin a branch to allow eviction using the global cache instance.

    Args:
        branch_id: The branch identifier to unpin

    Raises:
        RuntimeError: If branching cache is not initialized
        KeyError: If the branch is not found
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    _global_branching_cache.unpin_branch(branch_id)


def get_cache_stats() -> Dict[str, Any]:
    """
    Get comprehensive cache statistics from the global cache instance.

    Returns:
        Dictionary containing cache performance metrics

    Raises:
        RuntimeError: If branching cache is not initialized
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    return _global_branching_cache.get_cache_stats()


def list_branches() -> List[str]:
    """
    Get a list of all cached branch IDs from the global cache instance.

    Returns:
        List of branch identifiers

    Raises:
        RuntimeError: If branching cache is not initialized
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    return _global_branching_cache.list_branches()


def get_branch_info(branch_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific branch from the global cache instance.

    Args:
        branch_id: The branch identifier

    Returns:
        Dictionary with branch information, or None if not found

    Raises:
        RuntimeError: If branching cache is not initialized
    """
    global _global_branching_cache
    if _global_branching_cache is None:
        raise RuntimeError(
            "Branching cache not initialized. Call initialize_branching_cache() first."
        )
    return _global_branching_cache.get_branch_info(branch_id)
