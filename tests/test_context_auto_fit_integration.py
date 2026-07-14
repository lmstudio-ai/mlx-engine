from types import SimpleNamespace

import mlx_engine.generate as generate_module


def test_runtime_load_info_reports_only_fitted_context():
    fitted_kit = SimpleNamespace(effective_context_length=65_536)
    ordinary_kit = SimpleNamespace()

    assert generate_module.get_runtime_load_info(fitted_kit) == {
        "context_length": 65_536
    }
    assert generate_module.get_runtime_load_info(ordinary_kit) == {}
