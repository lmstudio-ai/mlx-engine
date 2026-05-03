import argparse
import logging

import mlx.core as mx

from mlx_engine import load_model, unload
from mlx_engine.distributed_rank import assert_not_source_checkout_runtime, run_worker_loop


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Packaged mlx-engine distributed worker rank loop."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-kv-size", type=int, default=4096)
    parser.add_argument("--prefill-step-size", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    assert_not_source_checkout_runtime()

    group = mx.distributed.init()
    rank = group.rank()
    size = group.size()
    if rank == 0:
        raise RuntimeError(
            "Packaged distributed worker loop can only run on non-coordinator ranks."
        )

    logger.info("Starting packaged distributed worker rank %s/%s", rank, size)
    model_kit = load_model(
        args.model,
        max_kv_size=args.max_kv_size,
        max_seq_nums=1,
        trust_remote_code=args.trust_remote_code,
        prefill_step_size=args.prefill_step_size,
        distributed=True,
        distributed_group=group,
    )
    try:
        run_worker_loop(rank, model_kit)
    finally:
        unload(model_kit)


if __name__ == "__main__":
    main()
