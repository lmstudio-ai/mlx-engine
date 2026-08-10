import sys

import pytest

from mlx_engine.server import __main__ as server_main


_REQUIRED_ARGS = [
    "--model",
    "model-path",
    "--host",
    "127.0.0.1",
    "--port",
    "1234",
    "--context-length",
    "4096",
    "--parallel-sessions",
    "2",
]


def test_api_key_is_not_a_command_line_argument():
    args = server_main._create_parser().parse_args(_REQUIRED_ARGS)

    assert not hasattr(args, "api_key")


def test_api_key_environment_variable_is_required(monkeypatch, capsys):
    monkeypatch.delenv("MLX_ENGINE_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["mlx-engine-server", *_REQUIRED_ARGS])

    with pytest.raises(SystemExit) as error:
        server_main.main()

    assert error.value.code == 2
    assert (
        "MLX_ENGINE_API_KEY environment variable is required" in capsys.readouterr().err
    )
