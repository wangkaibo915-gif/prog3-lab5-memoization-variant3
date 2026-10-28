"""Measure tree builders and create tables, charts and a short report."""

import csv
import statistics
import timeit
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tree_memo import (
    LEFT_3, RIGHT_3, LEFT_13, RIGHT_13,
    gen_bin_tree_recursive, gen_bin_tree_iterative,
    gen_bin_tree_recursive_memo, gen_bin_tree_iterative_memo,
    gen_bin_tree_recursive_lru,
)

HEIGHTS = list(range(2, 21, 2))
RUNS = 5
BUILDERS = {
    "recursive": gen_bin_tree_recursive,
    "iterative": gen_bin_tree_iterative,
    "recursive_memo": gen_bin_tree_recursive_memo,
    "iterative_memo": gen_bin_tree_iterative_memo,
    "recursive_lru": gen_bin_tree_recursive_lru,
}
VARIANTS = {
    "3": (3, LEFT_3, RIGHT_3),
    "13": (13, LEFT_13, RIGHT_13),
}
FIGURES = Path("figures")


def measure(builder, options, repeated=False):
    """Return average seconds for one build or one warmed repeat call."""
    times = []
    for _ in range(RUNS):
        if hasattr(builder, "cache_clear"):
            builder.cache_clear()
        if repeated:
            builder(**options)
        timer = timeit.Timer(lambda: builder(**options))
        calls = 1000 if repeated else 1
        times.append(timer.timeit(number=calls) / calls)
    return statistics.mean(times)


def count_operations(builder, root, left_rule, right_rule, height):
    """Count calls to both child-value functions during construction."""
    counts = {"left": 0, "right": 0}

    def left(value):
        counts["left"] += 1
        return left_rule(value)

    def right(value):
        counts["right"] += 1
        return right_rule(value)

    if hasattr(builder, "cache_clear"):
        builder.cache_clear()
    builder(height=height, root=root, left_leaf=left, right_leaf=right)
    hits = getattr(builder, "cache_hits", None)
    misses = getattr(builder, "cache_misses", None)
    if builder is gen_bin_tree_recursive_lru:
        info = builder.cache_info()
        hits, misses = info.hits, info.misses
    return counts["left"], counts["right"], hits, misses


def make_chart(rows, variant, scenario):
    """Save one chart for a variant and measurement scenario."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = BUILDERS if scenario == "first" else {
        "recursive_memo": gen_bin_tree_recursive_memo,
        "recursive_lru": gen_bin_tree_recursive_lru,
    }
    for name in names:
        points = [row for row in rows if row["variant"] == variant
                  and row["scenario"] == scenario and row["builder"] == name]
        ax.plot([row["height"] for row in points],
                [row["seconds"] * 1000 for row in points],
                marker="o", label=name)
    ax.set(title=f"Variant {variant}: {scenario} call", xlabel="Tree height",
           ylabel="Average time (ms)", xticks=HEIGHTS)
    if scenario == "first":
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"variant_{variant}_{scenario}.png", dpi=160)
    plt.close(fig)


def main():
    """Run the full reproducible comparison."""
    rows = []
    for variant, (root, left, right) in VARIANTS.items():
        for height in HEIGHTS:
            options = dict(height=height, root=root,
                           left_leaf=left, right_leaf=right)
            for name, builder in BUILDERS.items():
                rows.append(dict(variant=variant, scenario="first", height=height,
                                 builder=name, seconds=measure(builder, options)))
            for name in ("recursive_memo", "recursive_lru"):
                rows.append(dict(variant=variant, scenario="repeat", height=height,
                                 builder=name, seconds=measure(BUILDERS[name], options, True)))
        for scenario in ("first", "repeat"):
            make_chart(rows, variant, scenario)

    with open("results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "scenario", "height",
                                                  "builder", "seconds"))
        writer.writeheader()
        writer.writerows(rows)

    operation_rows = []
    for variant, (root, left, right) in VARIANTS.items():
        for name, builder in BUILDERS.items():
            left_count, right_count, hits, misses = count_operations(
                builder, root, left, right, 16)
            operation_rows.append(dict(variant=variant, height=16, builder=name,
                                       left_calls=left_count, right_calls=right_count,
                                       cache_hits=hits, cache_misses=misses))
    with open("operation_counts.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("variant", "height", "builder",
                                                  "left_calls", "right_calls",
                                                  "cache_hits", "cache_misses"))
        writer.writeheader()
        writer.writerows(operation_rows)

    gen_bin_tree_recursive_lru.cache_clear()
    gen_bin_tree_recursive_lru(height=16, root=13,
                               left_leaf=LEFT_13, right_leaf=RIGHT_13)
    print("lru_cache variant 13, height 16:", gen_bin_tree_recursive_lru.cache_info())
    print("Saved results.csv, operation_counts.csv and four figures")


if __name__ == "__main__":
    main()
