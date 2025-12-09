#!/usr/bin/env python3
"""
Benchmark harness for mlx-engine high-bandwidth Apple Silicon support.

This script provides comprehensive benchmarking capabilities for prefill performance,
branch restore operations, and profile comparisons.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add mlx_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mlx_engine.generate import load_model, tokenize, create_generator
    from mlx_engine.utils.hardware import (
        get_optimal_profile,
        detect_apple_silicon_hardware,
    )
    from mlx_engine.utils.metrics import (
        create_session_collector,
        MetalThroughputMonitor,
    )
    from mlx_engine.utils.logger import get_structured_logger
    from mlx_engine.utils.prompt_processing import plan_prefill_strategy

    MLX_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Error importing mlx_engine: {e}")
    print("Make sure mlx_engine is properly installed")
    MLX_ENGINE_AVAILABLE = False


class BenchmarkHarness:
    """Main benchmark harness for mlx-engine performance testing."""

    def __init__(self, model_path: str, verbose: bool = False):
        """Initialize benchmark harness."""
        if not MLX_ENGINE_AVAILABLE:
            raise RuntimeError("mlx_engine not available")

        self.model_path = Path(model_path)
        self.verbose = verbose
        self.model_kit = None
        self.structured_logger = get_structured_logger("benchmark", verbose)
        self.metal_monitor = MetalThroughputMonitor()

        # Setup logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def setup_model(self) -> None:
        """Load the model for benchmarking."""
        print(f"Loading model from {self.model_path}...")
        start_time = time.time()

        try:
            self.model_kit = load_model(self.model_path)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.2f}s")

            if self.verbose:
                self.structured_logger.log_performance(
                    operation="model_load",
                    duration_s=load_time,
                    additional_data={"model_path": str(self.model_path)},
                )
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def benchmark_prefill_performance(
        self, prompts: List[str], profiles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark prefill performance across different prompts and profiles.

        Args:
            prompts: List of prompt strings to test
            profiles: List of profile names to test (None for auto)

        Returns:
            Dictionary with benchmark results
        """
        if not self.model_kit:
            self.setup_model()

        results = {
            "model_path": str(self.model_path),
            "timestamp": time.time(),
            "prompts_tested": len(prompts),
            "profiles_tested": profiles or ["auto"],
            "results": [],
        }

        # Get available profiles if none specified
        if profiles is None:
            try:
                hardware = detect_apple_silicon_hardware()
                optimal_profile, _, _ = get_optimal_profile()
                profiles = [optimal_profile.value]
            except Exception:
                profiles = ["default_safe"]

        for profile_name in profiles:
            profile_results = {"profile": profile_name, "prompt_results": []}

            print(f"\n=== Benchmarking profile: {profile_name} ===")

            for i, prompt in enumerate(prompts):
                print(f"\nPrompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

                # Tokenize prompt
                tokens = tokenize(self.model_kit, prompt)
                token_count = len(tokens)

                # Create metrics collector for this run
                session_id = f"prefill_{profile_name}_{i}"
                metrics_collector = create_session_collector(session_id)

                # Plan prefill strategy
                try:
                    # Get profile config
                    from mlx_engine.utils.hardware import (
                        PROFILE_CONFIGS,
                        PerformanceProfileEnum,
                    )

                    if profile_name in [p.value for p in PerformanceProfileEnum]:
                        profile_enum = PerformanceProfileEnum(profile_name)
                        profile_config = PROFILE_CONFIGS[profile_enum]

                        # Create compatible profile object
                        from mlx_engine.utils.prompt_processing import (
                            PerformanceProfileCompat,
                        )

                        compat_profile = PerformanceProfileCompat(
                            name=profile_name,
                            prefill_mode="unbounded"
                            if profile_name.startswith("m3_")
                            else "chunked",
                            unbounded_allowed=profile_name.startswith("m3_"),
                            cache_slots=profile_config.parallel_threads // 4,
                            chunk_size_min=256,
                            chunk_size_max=4096,
                            kv_bytes_per_token_estimate=2048,
                            max_prefill_tokens_per_pass=8192,
                        )
                    else:
                        # Fallback profile
                        from mlx_engine.utils.prompt_processing import (
                            PerformanceProfileCompat,
                        )

                        compat_profile = PerformanceProfileCompat(
                            name=profile_name,
                            prefill_mode="chunked",
                            unbounded_allowed=False,
                            cache_slots=1,
                            chunk_size_min=256,
                            chunk_size_max=2048,
                            kv_bytes_per_token_estimate=2048,
                            max_prefill_tokens_per_pass=8192,
                        )

                    # Plan strategy
                    available_mem_gb = 32  # Conservative estimate
                    plan = plan_prefill_strategy(
                        prompt_tokens=token_count,
                        profile=compat_profile,
                        kv_bytes_per_token=compat_profile.kv_bytes_per_token_estimate,
                        available_mem_bytes=int(available_mem_gb * 1024**3),
                        requested_mode=None,
                        speculative_required=False,
                    )

                    print(f"  Strategy: {plan.mode}, chunks: {plan.total_chunks}")
                    if plan.chunk_size:
                        print(f"  Chunk size: {plan.chunk_size}")
                    print(f"  Reason: {plan.reason}")

                except Exception as e:
                    print(f"  Strategy planning failed: {e}")
                    plan = None

                # Benchmark prefill timing
                try:
                    # Start Metal throughput monitoring
                    self.metal_monitor.start_measurement()

                    # Time the prefill operation
                    start_time = time.time()

                    # Create generator (this triggers prefill)
                    generator = create_generator(
                        self.model_kit,
                        tokens,
                        max_tokens=1,  # Only prefill, no generation
                        performance_profile=compat_profile
                        if "compat_profile" in locals()
                        else None,
                        available_mem_gb=available_mem_gb,
                    )

                    # Consume first result to complete prefill
                    result = next(generator)
                    prefill_time = time.time() - start_time

                    # End Metal throughput monitoring
                    metal_throughput = self.metal_monitor.end_measurement()

                    # Calculate metrics
                    tokens_per_second = (
                        token_count / prefill_time if prefill_time > 0 else 0
                    )

                    prompt_result = {
                        "prompt_length": token_count,
                        "prefill_time_s": prefill_time,
                        "tokens_per_second": tokens_per_second,
                        "strategy": {
                            "mode": plan.mode if plan else "unknown",
                            "chunks": plan.total_chunks if plan else 1,
                            "chunk_size": plan.chunk_size if plan else None,
                            "reason": plan.reason if plan else "failed",
                        }
                        if plan
                        else None,
                        "metal_throughput_gb_s": metal_throughput,
                    }

                    profile_results["prompt_results"].append(prompt_result)

                    print(f"  Prefill time: {prefill_time:.3f}s")
                    print(f"  Tokens/sec: {tokens_per_second:.1f}")
                    if metal_throughput:
                        print(f"  Metal throughput: {metal_throughput:.1f} GB/s")

                except Exception as e:
                    print(f"  Prefill benchmark failed: {e}")
                    profile_results["prompt_results"].append(
                        {"prompt_length": token_count, "error": str(e)}
                    )

            results["results"].append(profile_results)

        return results

    def benchmark_branch_restore(
        self, branch_ids: List[str], prompt: str
    ) -> Dict[str, Any]:
        """
        Benchmark branch restore operations.

        Args:
            branch_ids: List of branch IDs to create and restore
            prompt: Base prompt for generation

        Returns:
            Dictionary with benchmark results
        """
        if not self.model_kit:
            self.setup_model()

        # Check if branching is supported
        if not hasattr(self.model_kit, "enable_branching_cache"):
            return {"error": "Branching cache not supported for this model"}

        results = {
            "model_path": str(self.model_path),
            "timestamp": time.time(),
            "prompt": prompt,
            "branch_ids": branch_ids,
            "operations": [],
        }

        print(f"\n=== Benchmarking Branch Restore Operations ===")

        try:
            # Enable branching cache
            self.model_kit.enable_branching_cache(max_slots=len(branch_ids) + 1)
            print(f"Enabled branching cache with {len(branch_ids) + 1} slots")

            # Tokenize prompt
            tokens = tokenize(self.model_kit, prompt)

            # Create base generation and checkpoint branches
            print("Creating base generation and checkpoints...")
            base_generator = create_generator(
                self.model_kit, tokens, max_tokens=50, enable_branching=True
            )

            generated_text = ""
            for result in base_generator:
                generated_text += result.text
                if len(generated_text.split()) > 10:  # Checkpoint after some words
                    break

            # Create checkpoints
            for branch_id in branch_ids:
                try:
                    start_time = time.time()
                    self.model_kit.checkpoint_branch(branch_id)
                    checkpoint_time = time.time() - start_time

                    results["operations"].append(
                        {
                            "operation": "checkpoint",
                            "branch_id": branch_id,
                            "time_s": checkpoint_time,
                        }
                    )

                    print(
                        f"  Checkpointed branch '{branch_id}' in {checkpoint_time:.3f}s"
                    )

                except Exception as e:
                    results["operations"].append(
                        {
                            "operation": "checkpoint",
                            "branch_id": branch_id,
                            "error": str(e),
                        }
                    )
                    print(f"  Failed to checkpoint branch '{branch_id}': {e}")

            # Benchmark restore operations
            print("Benchmarking restore operations...")
            for branch_id in branch_ids:
                try:
                    start_time = time.time()
                    self.model_kit.restore_branch(branch_id)
                    restore_time = time.time() - start_time

                    results["operations"].append(
                        {
                            "operation": "restore",
                            "branch_id": branch_id,
                            "time_s": restore_time,
                        }
                    )

                    print(f"  Restored branch '{branch_id}' in {restore_time:.3f}s")

                except Exception as e:
                    results["operations"].append(
                        {
                            "operation": "restore",
                            "branch_id": branch_id,
                            "error": str(e),
                        }
                    )
                    print(f"  Failed to restore branch '{branch_id}': {e}")

        except Exception as e:
            results["error"] = str(e)
            print(f"Branch restore benchmark failed: {e}")

        return results

    def compare_profiles(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Compare performance across different profiles.

        Args:
            prompts: List of prompts to test

        Returns:
            Dictionary with comparison results
        """
        profiles = [
            "default_safe",
            "m3_pro_64",
            "m3_max_128",
            "m3_ultra_256",
            "m3_ultra_512",
        ]

        # Filter profiles that might work
        available_profiles = []
        try:
            hardware = detect_apple_silicon_hardware()
            memory_gb = hardware.total_memory_gb

            # Only test profiles that could potentially work
            if memory_gb >= 64:
                available_profiles.extend(["default_safe", "m3_pro_64"])
            if memory_gb >= 128:
                available_profiles.append("m3_max_128")
            if memory_gb >= 256:
                available_profiles.append("m3_ultra_256")
            if memory_gb >= 512:
                available_profiles.append("m3_ultra_512")
        except Exception:
            available_profiles = ["default_safe"]

        print(f"Comparing profiles: {available_profiles}")

        comparison_results = self.benchmark_prefill_performance(
            prompts, available_profiles
        )

        # Add summary statistics
        summary = {"profiles": {}}
        for profile_result in comparison_results["results"]:
            profile_name = profile_result["profile"]
            prompt_results = [
                r for r in profile_result["prompt_results"] if "tokens_per_second" in r
            ]

            if prompt_results:
                avg_tokens_per_sec = sum(
                    r["tokens_per_second"] for r in prompt_results
                ) / len(prompt_results)
                avg_prefill_time = sum(
                    r["prefill_time_s"] for r in prompt_results
                ) / len(prompt_results)

                summary["profiles"][profile_name] = {
                    "avg_tokens_per_second": avg_tokens_per_sec,
                    "avg_prefill_time_s": avg_prefill_time,
                    "successful_runs": len(prompt_results),
                    "total_runs": len(profile_result["prompt_results"]),
                }

        comparison_results["summary"] = summary
        return comparison_results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run a comprehensive benchmark suite."""
        print("=== Running Full Benchmark Suite ===")

        # Test prompts of different lengths
        prompts = [
            "What is the capital of France?",  # Short
            "Explain the principles of machine learning in detail, including supervised learning, unsupervised learning, and reinforcement learning. Provide examples for each.",  # Medium
            "Write a comprehensive guide to climate change, covering its causes, effects, potential solutions, and the current state of international efforts to address it. Include scientific evidence and policy recommendations.",  # Long
        ]

        full_results = {
            "timestamp": time.time(),
            "model_path": str(self.model_path),
            "hardware_info": {},
            "prefill_benchmark": None,
            "branch_benchmark": None,
            "profile_comparison": None,
        }

        # Get hardware info
        try:
            hardware = detect_apple_silicon_hardware()
            full_results["hardware_info"] = {
                "chip": hardware.chip,
                "memory_gb": hardware.total_memory_gb,
                "cpu_cores": hardware.cpu_cores,
                "gpu_cores": hardware.gpu_cores,
                "memory_bandwidth_gb_s": hardware.memory_bandwidth_gb_s,
            }
        except Exception as e:
            full_results["hardware_info"] = {"error": str(e)}

        # Run prefill benchmarks
        try:
            print("\n--- Running Prefill Benchmarks ---")
            full_results["prefill_benchmark"] = self.benchmark_prefill_performance(
                prompts
            )
        except Exception as e:
            full_results["prefill_benchmark"] = {"error": str(e)}

        # Run branch restore benchmarks
        try:
            print("\n--- Running Branch Restore Benchmarks ---")
            full_results["branch_benchmark"] = self.benchmark_branch_restore(
                ["branch1", "branch2", "branch3"],
                prompts[1],  # Use medium prompt
            )
        except Exception as e:
            full_results["branch_benchmark"] = {"error": str(e)}

        # Run profile comparison
        try:
            print("\n--- Running Profile Comparison ---")
            full_results["profile_comparison"] = self.compare_profiles(
                prompts[:2]
            )  # Use first two prompts
        except Exception as e:
            full_results["profile_comparison"] = {"error": str(e)}

        return full_results


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(
        description="mlx-engine performance benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python bench_prefill.py --model /path/to/model --full-benchmark
  
  # Benchmark specific prompts
  python bench_prefill.py --model /path/to/model --prompts "short prompt" "long prompt with more text"
  
  # Compare profiles
  python bench_prefill.py --model /path/to/model --compare-profiles
  
  # Test branch restore
  python bench_prefill.py --model /path/to/model --test-branches
        """,
    )

    parser.add_argument("--model", required=True, help="Path to the model directory")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--output", help="Output JSON file for results")

    parser.add_argument("--prompts", nargs="+", help="Custom prompts to benchmark")

    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run comprehensive benchmark suite",
    )

    parser.add_argument(
        "--compare-profiles",
        action="store_true",
        help="Compare performance across profiles",
    )

    parser.add_argument(
        "--test-branches", action="store_true", help="Test branch restore performance"
    )

    args = parser.parse_args()

    if not MLX_ENGINE_AVAILABLE:
        print("Error: mlx_engine not available")
        sys.exit(1)

    # Create benchmark harness
    harness = BenchmarkHarness(args.model, args.verbose)

    # Run benchmarks based on arguments
    results = None

    if args.full_benchmark:
        results = harness.run_full_benchmark()
    elif args.compare_profiles:
        prompts = args.prompts or [
            "What is machine learning?",
            "Explain the history of artificial intelligence from its beginnings to the present day, including major breakthroughs, key researchers, and the evolution of different approaches such as symbolic AI, connectionism, and modern deep learning.",
        ]
        results = harness.compare_profiles(prompts)
    elif args.test_branches:
        prompt = (
            args.prompts[0]
            if args.prompts
            else "Explain the concept of quantum computing in detail."
        )
        results = harness.benchmark_branch_restore(
            ["branch1", "branch2", "branch3"], prompt
        )
    else:
        # Default: prefill benchmark
        prompts = args.prompts or [
            "Short prompt",
            "This is a medium length prompt that contains more content to test how the system handles inputs of moderate size.",
            "This is a much longer prompt designed to test the performance characteristics of the system when processing substantial amounts of text. It includes multiple sentences and various types of content to provide a realistic test scenario for benchmarking purposes.",
        ]
        results = harness.benchmark_prefill_performance(prompts)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n=== Benchmark Results ===")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
