import argparse
import logging
import signal
import threading

from mlx_engine import load_model

from .http import EngineRuntime, MlxEngineHttpServer


logger = logging.getLogger(__name__)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the private mlx-engine server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--context-length", required=True, type=int)
    parser.add_argument("--parallel-sessions", required=True, type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = _create_parser().parse_args()

    logger.info("Loading MLX model from %s", args.model)
    model_kit = load_model(
        args.model,
        max_kv_size=args.context_length,
        max_seq_nums=args.parallel_sessions,
        seed=args.seed,
        trust_remote_code=False,
    )
    runtime = EngineRuntime(model_kit)
    server = None

    try:
        server = MlxEngineHttpServer(
            (args.host, args.port),
            api_key=args.api_key,
            runtime=runtime,
        )

        def request_shutdown(_signal_number: int, _frame: object) -> None:
            logger.info("Stopping MLX server")
            server.cancel_active_sessions()
            threading.Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)
        logger.info("MLX server listening on %s:%d", args.host, args.port)
        server.serve_forever()
    finally:
        try:
            if server is not None:
                server.cancel_active_sessions()
                server.server_close()
        finally:
            runtime.unload()


if __name__ == "__main__":
    main()
