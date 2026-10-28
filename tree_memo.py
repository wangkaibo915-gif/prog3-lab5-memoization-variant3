"""Recursive and iterative binary-tree builders with memoization."""

from collections import deque
from functools import lru_cache, wraps
from typing import Callable, Optional


LEFT_3 = lambda value: value + 2
RIGHT_3 = lambda value: value * 3
LEFT_13 = lambda value: value + 1
RIGHT_13 = lambda value: value - 1


def memoize(function: Callable) -> Callable:
    """Cache results by positional and named arguments and count cache uses."""
    cache = {}

    @wraps(function)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key in cache:
            wrapper.cache_hits += 1
            return cache[key]
        wrapper.cache_misses += 1
        result = function(*args, **kwargs)
        cache[key] = result
        return result

    def cache_clear() -> None:
        """Remove saved results and reset the counters."""
        cache.clear()
        wrapper.cache_hits = 0
        wrapper.cache_misses = 0

    wrapper.cache_clear = cache_clear
    wrapper.cache_hits = 0
    wrapper.cache_misses = 0
    return wrapper


def _rules(left_leaf, right_leaf):
    """Use variant 3 rules for functions that were not supplied."""
    return (LEFT_3 if left_leaf is None else left_leaf,
            RIGHT_3 if right_leaf is None else right_leaf)


def _check_height(height: int) -> None:
    """Reject a tree without a root level."""
    if height < 1:
        raise ValueError("height must be at least 1")


def gen_bin_tree_recursive(
    height: int = 4,
    root: int = 3,
    left_leaf: Optional[Callable[[int], int]] = None,
    right_leaf: Optional[Callable[[int], int]] = None,
) -> dict:
    """Build a full binary tree recursively without caching."""
    _check_height(height)
    left_leaf, right_leaf = _rules(left_leaf, right_leaf)
    node = {"value": root, "left": None, "right": None}
    if height > 1:
        node["left"] = gen_bin_tree_recursive(height - 1, left_leaf(root), left_leaf, right_leaf)
        node["right"] = gen_bin_tree_recursive(height - 1, right_leaf(root), left_leaf, right_leaf)
    return node


def gen_bin_tree_iterative(
    height: int = 4,
    root: int = 3,
    left_leaf: Optional[Callable[[int], int]] = None,
    right_leaf: Optional[Callable[[int], int]] = None,
) -> dict:
    """Build a full binary tree level by level without caching."""
    _check_height(height)
    left_leaf, right_leaf = _rules(left_leaf, right_leaf)
    tree = {"value": root, "left": None, "right": None}
    queue = deque([(tree, 1)])
    while queue:
        node, level = queue.popleft()
        if level < height:
            node["left"] = {"value": left_leaf(node["value"]), "left": None, "right": None}
            node["right"] = {"value": right_leaf(node["value"]), "left": None, "right": None}
            queue.append((node["left"], level + 1))
            queue.append((node["right"], level + 1))
    return tree


@memoize
def gen_bin_tree_recursive_memo(
    height: int = 4,
    root: int = 3,
    left_leaf: Optional[Callable[[int], int]] = None,
    right_leaf: Optional[Callable[[int], int]] = None,
) -> dict:
    """Build a tree recursively, reusing equal subtrees from the cache."""
    _check_height(height)
    left_leaf, right_leaf = _rules(left_leaf, right_leaf)
    node = {"value": root, "left": None, "right": None}
    if height > 1:
        node["left"] = gen_bin_tree_recursive_memo(
            height - 1, left_leaf(root), left_leaf, right_leaf)
        node["right"] = gen_bin_tree_recursive_memo(
            height - 1, right_leaf(root), left_leaf, right_leaf)
    return node


def gen_bin_tree_iterative_memo(
    height: int = 4,
    root: int = 3,
    left_leaf: Optional[Callable[[int], int]] = None,
    right_leaf: Optional[Callable[[int], int]] = None,
) -> dict:
    """Build each different (value, remaining height) subtree only once."""
    _check_height(height)
    left_leaf, right_leaf = _rules(left_leaf, right_leaf)
    cache = {}
    queue = deque()

    def get_node(value: int, remaining: int) -> dict:
        """Return an existing node or schedule a new one for construction."""
        key = (value, remaining)
        if key not in cache:
            cache[key] = {"value": value, "left": None, "right": None}
            queue.append((cache[key], remaining))
        return cache[key]

    tree = get_node(root, height)
    while queue:
        node, remaining = queue.popleft()
        if remaining > 1:
            node["left"] = get_node(left_leaf(node["value"]), remaining - 1)
            node["right"] = get_node(right_leaf(node["value"]), remaining - 1)
    return tree


@lru_cache(maxsize=None)
def gen_bin_tree_recursive_lru(
    height: int = 4,
    root: int = 3,
    left_leaf: Optional[Callable[[int], int]] = None,
    right_leaf: Optional[Callable[[int], int]] = None,
) -> dict:
    """Build a tree recursively using functools.lru_cache."""
    _check_height(height)
    left_leaf, right_leaf = _rules(left_leaf, right_leaf)
    node = {"value": root, "left": None, "right": None}
    if height > 1:
        node["left"] = gen_bin_tree_recursive_lru(
            height - 1, left_leaf(root), left_leaf, right_leaf)
        node["right"] = gen_bin_tree_recursive_lru(
            height - 1, right_leaf(root), left_leaf, right_leaf)
    return node


if __name__ == "__main__":
    gen_bin_tree_recursive_lru.cache_clear()
    gen_bin_tree_recursive_lru()
    print("lru_cache:", gen_bin_tree_recursive_lru.cache_info())
