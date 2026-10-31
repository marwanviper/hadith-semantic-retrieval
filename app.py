"""
Hadith Semantic Search — Main CLI Entry Point

Usage examples:

  # Prepare dataset (creates sample data if none exists)
  python app.py prepare

  # Generate and cache embeddings
  python app.py embeddings
  python app.py embeddings --force

  # Build a search index
  python app.py build --algorithm hnsw
  python app.py build --algorithm all

  # Search
  python app.py search --algorithm hnsw --query "حديث عن الأعمال بالنية"
  python app.py search --algorithm brute_force --query "فضل العلم" --top-k 5

  # Run benchmarks
  python app.py benchmark --algorithm hnsw
  python app.py benchmark --algorithm all

  # Show index statistics
  python app.py stats --algorithm hnsw

Configuration overrides:
  Any config key can be overridden with --set:
  python app.py build --algorithm hnsw --set hnsw.M=32 --set hnsw.ef_construction=400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---- project root setup ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.config import load_config, Config
from src.logging.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_set_overrides(set_args: Optional[List[str]]) -> dict:
    """Parse '--set key=value' arguments into a dict."""
    overrides = {}
    if not set_args:
        return overrides
    for item in set_args:
        if "=" not in item:
            print(f"WARNING: --set argument '{item}' ignored (expected key=value format)")
            continue
        key, _, value = item.partition("=")
        # Attempt numeric conversion
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string
        overrides[key.strip()] = value
    return overrides


def print_search_results(result, verbose: bool = True) -> None:
    """Pretty-print search results."""
    print(f"\n{'='*60}")
    print(f"Query: {result.query}")
    print(f"Algorithm: {result.algorithm.upper()}")
    print(f"{'='*60}")
    if not result.results:
        print("No results found.")
        return
    for r in result.results:
        print(f"\nRank {r.rank:>2} | Score: {r.score:.4f} | ID: {r.hadith_id}")
        if r.text:
            text_display = r.text[:200] + ("..." if len(r.text) > 200 else "")
            print(f"  Text: {text_display}")
        if verbose:
            if r.book:
                print(f"  Book: {r.book}", end="")
            if r.chapter:
                print(f"  | Chapter: {r.chapter}", end="")
            if r.narrator:
                print(f"  | Narrator: {r.narrator}", end="")
            if r.book or r.chapter or r.narrator:
                print()
    print(f"\n{'─'*60}")
    metrics = result.metrics
    print(f"  Embed: {metrics.get('query_embed_ms', 0):.2f}ms | "
          f"Search: {metrics.get('search_time_ms', 0):.2f}ms | "
          f"Total: {metrics.get('total_time_ms', 0):.2f}ms")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_prepare(args, config: Config) -> None:
    """Prepare the dataset."""
    from scripts.prepare_dataset import main as prepare_main
    print("\n▶ Preparing dataset...")
    prepare_main()


def cmd_embeddings(args, config: Config) -> None:
    """Generate and cache embeddings."""
    from scripts.generate_embeddings import main as embed_main
    print("\n▶ Generating embeddings...")
    embed_main(force=args.force)


def cmd_build(args, config: Config) -> None:
    """Build one or all search indexes."""
    from src.embeddings.cache import EmbeddingCache
    from src.indexes.factory import VectorSearchFactory, ALL_ALGORITHMS

    algorithms = ALL_ALGORITHMS if args.algorithm == "all" else [args.algorithm]

    # Load embeddings once
    cache = EmbeddingCache(
        npz_path=config.embeddings_file(),
        meta_path=config.embeddings_meta_file(),
    )
    try:
        embeddings, hadith_ids, meta = cache.load(
            expected_model=config.embedding.model
        )
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"\n▶ Building indexes for: {', '.join(algorithms)}")
    print(f"  Dataset: {len(hadith_ids)} Hadiths, dim={embeddings.shape[1]}\n")

    for algo in algorithms:
        try:
            t0 = time.perf_counter()
            index = VectorSearchFactory.create(algo, config)
            index.build(embeddings, hadith_ids)
            index_path = config.index_path(algo)
            index.save(index_path)
            elapsed = time.perf_counter() - t0
            stats = index.get_stats()
            print(f"  ✓ {algo:<15} built in {elapsed:.2f}s | "
                  f"size={stats.index_size_mb():.1f} MB | path={index_path}")
        except Exception as e:
            print(f"  ✗ {algo:<15} FAILED: {e}")
            logger.error("Build failed for %s", algo, exc_info=True)


def cmd_search(args, config: Config) -> None:
    """Search with a specified algorithm."""
    from src.search.engine import HadithSearchEngine

    engine = HadithSearchEngine(config)
    try:
        engine.load_or_build(algorithm=args.algorithm)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print(f"Run: python app.py build --algorithm {args.algorithm}")
        sys.exit(1)

    top_k = args.top_k or config.search.top_k
    result = engine.search(args.query, top_k=top_k)
    print_search_results(result, verbose=not args.brief)


def cmd_benchmark(args, config: Config) -> None:
    """Run benchmarks."""
    from src.benchmarking.benchmark import BenchmarkRunner

    algorithms = None if args.algorithm == "all" else [args.algorithm]

    print(f"\n▶ Benchmarking: {args.algorithm.upper()}")
    print(f"  Model: {config.embedding.model}")
    print(f"  Queries: {config.benchmark.n_queries} | K={config.benchmark.top_k}\n")

    runner = BenchmarkRunner(config)
    try:
        runner.run(algorithms=algorithms, rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Run 'python app.py embeddings' first.")
        sys.exit(1)

    runner.print_summary()
    output_path = runner.save_results()
    print(f"Results saved to: {output_path}")


def cmd_stats(args, config: Config) -> None:
    """Print index statistics."""
    from src.indexes.factory import VectorSearchFactory, ALL_ALGORITHMS

    algorithms = ALL_ALGORITHMS if args.algorithm == "all" else [args.algorithm]

    for algo in algorithms:
        index_path = config.index_path(algo)
        if not index_path.exists():
            print(f"\n{algo}: Index not found at {index_path}")
            continue
        try:
            index = VectorSearchFactory.create(algo, config)
            index.load(index_path)
            stats = index.get_stats()
            print(f"\n{'='*50}")
            print(f"Algorithm: {stats.algorithm.upper()}")
            print(f"{'='*50}")
            print(f"  Vectors:    {stats.num_vectors:,}")
            print(f"  Dimension:  {stats.embedding_dim}")
            print(f"  Build time: {stats.build_time_s:.2f}s")
            print(f"  Index size: {stats.index_size_mb():.2f} MB")
            print(f"  Parameters: {stats.params}")

            # Also show disk size
            disk_size = sum(
                f.stat().st_size for f in index_path.rglob("*") if f.is_file()
            )
            print(f"  Disk size:  {disk_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"\n{algo}: Failed to load — {e}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Hadith Semantic Search Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    parser.add_argument(
        "--set", dest="set_overrides", action="append", metavar="KEY=VALUE",
        help="Override config values, e.g. --set hnsw.M=32"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare
    subparsers.add_parser("prepare", help="Prepare the Hadith dataset")

    # embeddings
    embed_parser = subparsers.add_parser("embeddings", help="Generate and cache embeddings")
    embed_parser.add_argument("--force", action="store_true",
                              help="Regenerate even if cache exists")

    # build
    build_parser_sub = subparsers.add_parser("build", help="Build a search index")
    build_parser_sub.add_argument(
        "--algorithm", required=True,
        help="Algorithm name: brute_force | ivf | hnsw | pq | ivf_pq | all"
    )

    # search
    search_parser = subparsers.add_parser("search", help="Run a search query")
    search_parser.add_argument("--algorithm", default="hnsw",
                               help="Algorithm to search with")
    search_parser.add_argument("--query", "-q", required=True,
                               help="Arabic search query")
    search_parser.add_argument("--top-k", type=int, default=None,
                               help="Number of results (default from config)")
    search_parser.add_argument("--brief", action="store_true",
                               help="Minimal output (no metadata)")

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument(
        "--algorithm", default="all",
        help="Algorithm to benchmark, or 'all'"
    )
    bench_parser.add_argument("--rebuild", action="store_true",
                              help="Rebuild indexes even if they exist")

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show index statistics")
    stats_parser.add_argument(
        "--algorithm", default="all",
        help="Algorithm name, or 'all'"
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(level=args.log_level)

    overrides = parse_set_overrides(args.set_overrides)
    config = load_config(
        config_path=args.config,
        overrides=overrides,
        project_root=str(PROJECT_ROOT),
    )

    logger.debug("Command: %s | Config: %s", args.command, config.project_root)

    dispatch = {
        "prepare": cmd_prepare,
        "embeddings": cmd_embeddings,
        "build": cmd_build,
        "search": cmd_search,
        "benchmark": cmd_benchmark,
        "stats": cmd_stats,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        sys.exit(1)

    handler(args, config)


if __name__ == "__main__":
    main()
