"""Tests for the binary-tree builders and the memoize decorator."""

import unittest
from functools import wraps

from tree_memo import (
    LEFT_13, RIGHT_13, gen_bin_tree_recursive,
    gen_bin_tree_iterative, gen_bin_tree_recursive_memo,
    gen_bin_tree_iterative_memo, gen_bin_tree_recursive_lru, memoize,
)


BUILDERS = (
    gen_bin_tree_recursive, gen_bin_tree_iterative,
    gen_bin_tree_recursive_memo, gen_bin_tree_iterative_memo,
    gen_bin_tree_recursive_lru,
)


class TestTrees(unittest.TestCase):
    """Check tree equality, caching and shared objects."""

    def setUp(self):
        gen_bin_tree_recursive_memo.cache_clear()
        gen_bin_tree_recursive_lru.cache_clear()

    def test_default_trees_are_equal(self):
        expected = gen_bin_tree_recursive()
        for builder in BUILDERS:
            self.assertEqual(builder(), expected)
        self.assertEqual(expected["value"], 3)
        self.assertEqual(expected["left"]["value"], 5)
        self.assertEqual(expected["right"]["value"], 9)

    def test_custom_arguments_are_equal(self):
        options = dict(height=5, root=13, left_leaf=LEFT_13, right_leaf=RIGHT_13)
        expected = gen_bin_tree_recursive(**options)
        for builder in BUILDERS:
            self.assertEqual(builder(**options), expected)

    def test_height_one_and_invalid_height(self):
        for builder in BUILDERS:
            self.assertEqual(builder(height=1), {"value": 3, "left": None, "right": None})
            with self.assertRaises(ValueError):
                builder(height=0)

    def test_memoize_uses_all_arguments_and_preserves_metadata(self):
        calls = []

        @memoize
        def add(a: int, b: int = 0) -> int:
            """Add two integers."""
            calls.append((a, b))
            return a + b

        self.assertEqual(add(2, b=3), 5)
        self.assertEqual(add(2, b=3), 5)
        self.assertEqual(add(2, b=4), 6)
        self.assertEqual(add(3, b=3), 6)
        self.assertEqual(calls, [(2, 3), (2, 4), (3, 3)])
        self.assertEqual((add.cache_hits, add.cache_misses), (1, 3))
        self.assertEqual(add.__name__, "add")
        self.assertEqual(add.__doc__, "Add two integers.")
        add.cache_clear()
        self.assertEqual((add.cache_hits, add.cache_misses), (0, 0))
        add(2, b=3)
        self.assertEqual(len(calls), 4)

    def test_recursive_cache_hits_inside_construction(self):
        gen_bin_tree_recursive_memo(height=5, root=13, left_leaf=LEFT_13, right_leaf=RIGHT_13)
        self.assertGreater(gen_bin_tree_recursive_memo.cache_hits, 0)
        self.assertGreater(gen_bin_tree_recursive_memo.cache_misses, 0)

    def test_equal_subtrees_are_shared(self):
        options = dict(height=3, root=13, left_leaf=LEFT_13, right_leaf=RIGHT_13)
        for builder in (gen_bin_tree_recursive_memo, gen_bin_tree_iterative_memo,
                        gen_bin_tree_recursive_lru):
            tree = builder(**options)
            self.assertIs(tree["left"]["right"], tree["right"]["left"])
            tree["left"]["right"]["value"] = 999
            self.assertEqual(tree["right"]["left"]["value"], 999)
        first = gen_bin_tree_recursive_memo(**options)
        self.assertIs(first, gen_bin_tree_recursive_memo(**options))
        first["value"] = 888
        self.assertEqual(gen_bin_tree_recursive_memo(**options)["value"], 888)

    def test_unmemoized_subtrees_are_separate(self):
        tree = gen_bin_tree_recursive(height=3, root=13,
                                      left_leaf=LEFT_13, right_leaf=RIGHT_13)
        self.assertIsNot(tree["left"]["right"], tree["right"]["left"])

    def test_lru_cache_info(self):
        gen_bin_tree_recursive_lru(height=5, root=13,
                                   left_leaf=LEFT_13, right_leaf=RIGHT_13)
        info = gen_bin_tree_recursive_lru.cache_info()
        self.assertGreater(info.hits, 0)
        self.assertGreater(info.misses, 0)

    def test_new_lambda_prevents_cross_call_hit(self):
        for _ in range(2):
            gen_bin_tree_recursive_memo(
                height=2, root=3,
                left_leaf=lambda value: value + 2,
                right_leaf=lambda value: value * 3,
            )
        self.assertEqual(gen_bin_tree_recursive_memo.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()
