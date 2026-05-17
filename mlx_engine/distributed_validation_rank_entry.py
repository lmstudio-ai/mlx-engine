import os
import sys


def main() -> None:
    raw_rank = os.environ.get("MLX_RANK")
    if raw_rank is None:
        raise RuntimeError("MLX_RANK is not set. Run this module under mlx.launch.")

    try:
        rank = int(raw_rank)
    except ValueError as caught_error:
        raise RuntimeError(f"Invalid MLX_RANK value: {raw_rank!r}") from caught_error

    module_name = (
        "mlx_engine.distributed_coordinator"
        if rank == 0
        else "mlx_engine.distributed_worker"
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            module_name,
            *sys.argv[1:],
        ],
    )


if __name__ == "__main__":
    main()
