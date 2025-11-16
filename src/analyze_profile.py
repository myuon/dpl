"""
cProfileの結果を分析するスクリプト

Usage:
    python analyze_profile.py [profile_file]
"""

import pstats
import sys
from pathlib import Path


def analyze_profile(profile_file: str = "main.prof") -> None:
    """プロファイル結果を分析して表示する

    Args:
        profile_file: プロファイルファイルのパス
    """
    if not Path(profile_file).exists():
        print(f"Error: Profile file '{profile_file}' not found")
        sys.exit(1)

    # プロファイル結果を読み込む
    stats = pstats.Stats(profile_file)

    # 累積時間でソートして表示
    print("=" * 80)
    print("Top 20 functions by cumulative time")
    print("=" * 80)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)

    print("\n" + "=" * 80)
    print("Top 20 functions by total time (excluding subcalls)")
    print("=" * 80)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats(20)

    print("\n" + "=" * 80)
    print("Top 20 functions by number of calls")
    print("=" * 80)
    stats.sort_stats(pstats.SortKey.CALLS)
    stats.print_stats(20)

    # 特定のモジュールに絞って分析
    print("\n" + "=" * 80)
    print("Functions in 'layers' module (sorted by cumulative time)")
    print("=" * 80)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats("layers")

    print("\n" + "=" * 80)
    print("Functions in 'optimizers' module (sorted by cumulative time)")
    print("=" * 80)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats("optimizers")

    # 詳細な統計情報を取得
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    # 全体の統計を取得 (type: ignore for pstats internal attributes)
    total_calls = stats.total_calls  # type: ignore[attr-defined]
    prim_calls = stats.prim_calls  # type: ignore[attr-defined]
    total_time = stats.total_tt  # type: ignore[attr-defined]

    print(f"Total function calls: {total_calls:,}")
    print(f"Primitive calls: {prim_calls:,}")
    print(f"Total time: {total_time:.3f} seconds")

    # ボトルネックを特定
    print("\n" + "=" * 80)
    print("Potential Bottlenecks (functions taking > 1% of total time)")
    print("=" * 80)

    stats.sort_stats(pstats.SortKey.TIME)

    # 統計情報を辞書として取得 (type: ignore for pstats internal attributes)
    func_stats = stats.stats  # type: ignore[attr-defined]

    bottlenecks = []
    for func, (cc, nc, tt, ct, callers) in func_stats.items():
        if total_time > 0:
            percentage = (tt / total_time) * 100
            if percentage > 1.0:
                bottlenecks.append((func, tt, ct, nc, percentage))

    # パーセンテージでソート
    bottlenecks.sort(key=lambda x: x[4], reverse=True)

    print(f"{'Function':<60} {'Time':<10} {'Calls':<10} {'%':<8}")
    print("-" * 88)
    for func, tt, ct, nc, pct in bottlenecks[:20]:
        # 関数名を整形
        if isinstance(func, tuple) and len(func) == 3:
            filename, line, name = func
            # ファイル名を短縮
            if filename.startswith("/"):
                filename = Path(filename).name
            func_str = f"{filename}:{line}({name})"
        else:
            func_str = str(func)

        print(f"{func_str:<60} {tt:>8.3f}s {nc:>9,} {pct:>6.1f}%")


if __name__ == "__main__":
    # コマンドライン引数からファイル名を取得
    if len(sys.argv) > 1:
        profile_file = sys.argv[1]
    else:
        profile_file = "main.prof"

    analyze_profile(profile_file)
