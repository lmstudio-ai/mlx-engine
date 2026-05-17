import argparse
import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

distributed_init_started_at_env = "MLX_ENGINE_DISTRIBUTED_WORKER_INIT_STARTED_AT"
source_checkout_runtime_path_segments = (
    "/electron/vendor/llm-engine/build/",
    "/electron/vendor/llm-engine/src/",
)


def normalized_runtime_path(path_value: str) -> str:
    return path_value.replace("\\", "/")


def assert_not_source_checkout_runtime() -> None:
    python_executable = normalized_runtime_path(sys.executable)
    for source_path_segment in source_checkout_runtime_path_segments:
        if source_path_segment in python_executable:
            raise RuntimeError(
                "MLX distributed packaged rank is running from source-checkout "
                f"Python {sys.executable}. Rebuild or stage the packaged MLX "
                "runtime so ranks use the installed Amphibian Python."
            )


def init_distributed_with_retry(timeout_seconds: float):
    started_at = float(os.environ.get(distributed_init_started_at_env, time.monotonic()))
    os.environ[distributed_init_started_at_env] = str(started_at)

    try:
        logger.info("Importing mlx.core before distributed worker init")
        import mlx.core as mx

        logger.info("Calling distributed worker init")
        group = mx.distributed.init()
        logger.info(
            "Distributed worker init completed rank %s/%s",
            group.rank(),
            group.size(),
        )
        return group
    except RuntimeError as error:
        elapsed_seconds = time.monotonic() - started_at
        if elapsed_seconds >= timeout_seconds:
            raise
        logger.info(
            "Distributed init failed while waiting for rank 0 after %.1fs: %s",
            elapsed_seconds,
            error,
        )
        time.sleep(min(2.0, max(0.0, timeout_seconds - elapsed_seconds)))
        os.execv(
            sys.executable,
            [
                sys.executable,
                "-I",
                "-m",
                "mlx_engine.distributed_worker",
                *sys.argv[1:],
            ],
        )
        raise


def run_collective_smoke(rank: int, size: int) -> None:
    import mlx.core as mx

    logger.info("Running worker CPU collective smoke rank %s/%s", rank, size)
    cpu_result = mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu)
    mx.eval(cpu_result)
    logger.info(
        "Worker CPU collective smoke completed rank %s/%s result=%s",
        rank,
        size,
        cpu_result.item(),
    )

    logger.info(
        "Running worker default-stream collective smoke rank %s/%s",
        rank,
        size,
    )
    default_result = mx.distributed.all_sum(mx.array(1.0))
    mx.eval(default_result)
    logger.info(
        "Worker default-stream collective smoke completed rank %s/%s result=%s",
        rank,
        size,
        default_result.item(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Packaged mlx-engine distributed worker rank loop."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--distributed-init-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--init-smoke-only", action="store_true")
    parser.add_argument("--max-seq-nums", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    assert_not_source_checkout_runtime()

    group = init_distributed_with_retry(args.distributed_init_timeout_seconds)
    rank = group.rank()
    size = group.size()
    if rank == 0:
        raise RuntimeError(
            "Packaged distributed worker loop can only run on non-coordinator ranks."
        )
    if args.init_smoke_only:
        run_collective_smoke(rank, size)
        logger.info("Packaged distributed worker init smoke completed rank %s/%s", rank, size)
        return

    from mlx_engine import load_model, unload
    from mlx_engine.distributed_rank import run_worker_loop

    logger.info(
        "Starting packaged distributed worker rank %s/%s model=%s max_kv_size=%s prefill_step_size=%s",
        rank,
        size,
        args.model,
        args.max_kv_size,
        args.prefill_step_size,
    )
    logger.info("Worker rank %s calling mlx_engine.load_model(distributed=True)", rank)
    model_kit = load_model(
        args.model,
        max_kv_size=args.max_kv_size,
        max_seq_nums=args.max_seq_nums,
        trust_remote_code=args.trust_remote_code,
        prefill_step_size=args.prefill_step_size,
        distributed=True,
        distributed_group=group,
    )
    logger.info("Worker rank %s mlx_engine.load_model returned", rank)
    try:
        logger.info("Worker rank %s entering native worker loop", rank)
        run_worker_loop(rank, model_kit)
    finally:
        logger.info("Worker rank %s unloading model kit", rank)
        unload(model_kit)


if __name__ == "__main__":
    main()
