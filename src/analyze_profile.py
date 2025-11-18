"""
cProfileの結果を分析するスクリプト

Usage:
    python analyze_profile.py [profile_file] [--flamegraph]

Options:
    --flamegraph: Generate flamegraph visualization (requires flameprof or snakeviz)
"""

import pstats
import sys
from pathlib import Path
import subprocess


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


def generate_flamegraph(profile_file: str = "main.prof") -> None:
    """Flamegraphを生成する

    Args:
        profile_file: プロファイルファイルのパス
    """
    print("\n" + "=" * 80)
    print("Generating Flamegraph")
    print("=" * 80)

    # snakevizを試す（インタラクティブなブラウザベース可視化）
    try:
        print("\nTrying snakeviz (interactive browser-based visualization)...")
        print("This will open a browser window with the profile visualization.")
        print("Press Ctrl+C to stop the server when done.")
        subprocess.run(["snakeviz", profile_file], check=True)
        return
    except FileNotFoundError:
        print("snakeviz not found. Install with: uv add snakeviz")
    except subprocess.CalledProcessError:
        print("snakeviz failed to run")
    except KeyboardInterrupt:
        print("\nsnakeviz server stopped")
        return

    # flameprofを試す（SVGのflamegraph生成）
    try:
        print("\nTrying flameprof (generates SVG flamegraph)...")
        output_file = profile_file.replace('.prof', '_flamegraph.svg')
        subprocess.run(
            ["flameprof", profile_file, "-o", output_file],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Flamegraph generated: {output_file}")
        print("Open this file in a web browser to view the flamegraph.")
        return
    except FileNotFoundError:
        print("flameprof not found. Install with: pip install flameprof")
    except subprocess.CalledProcessError as e:
        print(f"flameprof failed: {e}")

    # gprof2dotとgraphvizを試す（コールグラフ生成）
    try:
        print("\nTrying gprof2dot + graphviz (generates call graph)...")
        dot_output = profile_file.replace('.prof', '_callgraph.dot')
        png_output = profile_file.replace('.prof', '_callgraph.png')

        # pstatsをdot形式に変換
        with open(dot_output, 'w') as f:
            subprocess.run(
                ["gprof2dot", "-f", "pstats", profile_file],
                stdout=f,
                check=True
            )

        # dotをPNGに変換
        subprocess.run(
            ["dot", "-Tpng", dot_output, "-o", png_output],
            check=True
        )
        print(f"Call graph generated: {png_output}")
        return
    except FileNotFoundError:
        print("gprof2dot or graphviz not found.")
        print("Install with: pip install gprof2dot && brew install graphviz (macOS)")
    except subprocess.CalledProcessError as e:
        print(f"gprof2dot/graphviz failed: {e}")

    print("\nNo visualization tools available. Please install one of:")
    print("  - snakeviz: uv add snakeviz (recommended for interactive exploration)")
    print("  - flameprof: pip install flameprof (generates flamegraph SVG)")
    print("  - gprof2dot: pip install gprof2dot && brew install graphviz")


if __name__ == "__main__":
    # コマンドライン引数を解析
    args = sys.argv[1:]
    profile_file = "main.prof"
    show_flamegraph = False

    for arg in args:
        if arg == "--flamegraph":
            show_flamegraph = True
        elif not arg.startswith("--"):
            profile_file = arg

    if show_flamegraph:
        generate_flamegraph(profile_file)
    else:
        analyze_profile(profile_file)
