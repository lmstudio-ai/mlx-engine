import os
import sys
import types
from unittest import mock

import pytest

# Ensure repository root is on sys.path for imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Avoid heavy initialization paths during tests
os.environ.setdefault("MLX_ENGINE_SKIP_INIT", "1")

# Check if we need real tokenizers (for stop string processor tests)
USE_REAL_TOKENIZERS = (
    os.environ.get("MLX_ENGINE_USE_REAL_TOKENIZERS", "false").lower() == "true"
)

# Provide lightweight mocks for heavy optional dependencies so tests can run
_mock_modules = {
    "mlx": mock.MagicMock(),
    "mlx.core": mock.MagicMock(),
    "mlx.nn": mock.MagicMock(),
    "sentencepiece": mock.MagicMock(),
}


# Create a mock MLX array class that supports basic operations
class MockMLXArray:
    def __init__(self, data):
        if isinstance(data, MockMLXArray):
            self.data = data.data
        else:
            self.data = list(data)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MockMLXArray(self.data[key])
        return self.data[key]

    def __eq__(self, other):
        # For MLX array comparison, we need to return a boolean array-like object
        # But Python's __eq__ must return bool, so we'll handle this differently
        if hasattr(other, "data"):
            return all(a == b for a, b in zip(self.data, other.data)) and len(
                self.data
            ) == len(other.data)
        return False

    def elementwise_eq(self, other):
        """Helper method for element-wise comparison that returns MockMLXArray"""
        if hasattr(other, "data"):
            return MockMLXArray([a == b for a, b in zip(self.data, other.data)])
        return MockMLXArray([a == other for a in self.data])

    def __bool__(self):
        return any(self.data) if self.data else False

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"MockMLXArray({self.data})"


# Make mlx.core.array return MockMLXArray for testing
def mock_array(x):
    return MockMLXArray(x)


_mock_modules["mlx.core"].array = mock.Mock(side_effect=mock_array)
_mock_modules["mlx.core"].any = mock.Mock(side_effect=lambda x: any(x))
_mock_modules["mlx.core"].argmax = mock.Mock(
    side_effect=lambda x: x.index(True) if True in x else len(x)
)
_mock_modules["mlx.core"].concat = mock.Mock(
    side_effect=lambda arrays, **kwargs: MockMLXArray(
        sum([list(arr) for arr in arrays], [])
    )
)
_mock_modules["mlx.core"].concat = mock.Mock(
    side_effect=lambda arrays, **kwargs: MockMLXArray(
        sum([list(arr) for arr in arrays], [])
    )
)
_mock_modules["mlx.core"].concat = mock.Mock(
    side_effect=lambda arrays, **kwargs: MockMLXArray(
        sum([list(arr) for arr in arrays], [])
    )
)
_mock_modules["mlx.core"].concat = mock.Mock(
    side_effect=lambda arrays, **kwargs: MockMLXArray(
        sum([list(arr) for arr in arrays], [])
    )
)
for name, module in _mock_modules.items():
    sys.modules.setdefault(name, module)


# Mock mlx_lm submodules used by cache_wrapper
_cache_module = types.SimpleNamespace(
    make_prompt_cache=lambda *args, **kwargs: mock.MagicMock(),
    trim_prompt_cache=lambda cache, **kwargs: cache,
    can_trim_prompt_cache=lambda cache: False,
)
_generation_module = types.SimpleNamespace(
    generation_stream=lambda *args, **kwargs: iter(()),
    stream_generate=lambda *args, **kwargs: iter(()),
    maybe_quantize_kv_cache=lambda cache, **kwargs: cache,
)
# Always use real tokenizer_utils to avoid mocking conflicts
# The real tokenizer_utils module should be available for all tests
try:
    import mlx_lm.tokenizer_utils

    _tokenizer_utils_module = mlx_lm.tokenizer_utils
except ImportError:
    # If real tokenizer_utils is not available, create a mock
    _tokenizer_utils_module = types.SimpleNamespace(
        TokenizerWrapper=type("TokenizerWrapper", (), {}),
        StreamingDetokenizer=type("StreamingDetokenizer", (), {}),
        load=mock.MagicMock(
            return_value=mock.MagicMock(
                encode=mock.MagicMock(return_value=[1, 2, 3]),
                decode=mock.MagicMock(return_value="test"),
            )
        ),
    )

# Only mock mlx_lm if we're not using real tokenizers
if not USE_REAL_TOKENIZERS:
    # Create mlx_lm module with proper structure
    _mlx_lm_module = types.SimpleNamespace(
        models=types.SimpleNamespace(cache=_cache_module),
        generate=_generation_module,
        tokenizer_utils=_tokenizer_utils_module,
        utils=mock.MagicMock(),
        load=mock.MagicMock(),
        convert=mock.MagicMock(),
        stream_generate=mock.MagicMock(),
        batch_generate=mock.MagicMock(),
    )

    # Set up mlx_lm modules
    sys.modules.setdefault("mlx_lm", _mlx_lm_module)
    sys.modules.setdefault("mlx_lm.models", _mlx_lm_module.models)
    sys.modules.setdefault("mlx_lm.models.cache", _cache_module)
    sys.modules.setdefault("mlx_lm.generate", _generation_module)
    sys.modules.setdefault("mlx_lm.sample_utils", mock.MagicMock())
    sys.modules.setdefault("mlx_lm.utils", _mlx_lm_module.utils)
    sys.modules.setdefault("mlx_lm.load", _mlx_lm_module.load)
    sys.modules.setdefault("mlx_lm.convert", _mlx_lm_module.convert)
    sys.modules.setdefault("mlx_lm.stream_generate", _mlx_lm_module.stream_generate)
    sys.modules.setdefault("mlx_lm.batch_generate", _mlx_lm_module.batch_generate)
    sys.modules.setdefault("mlx_lm.tokenizer_utils", _tokenizer_utils_module)

# Mock mlx_vlm vision modules
_mlx_vlm_modules = [
    "mlx_vlm",
    "mlx_vlm.utils",
    "mlx_vlm.models",
    "mlx_vlm.models.gemma3",
    "mlx_vlm.models.gemma3.gemma3",
    "mlx_vlm.models.pixtral",
    "mlx_vlm.models.pixtral.pixtral",
    "mlx_vlm.models.qwen3_vl",
    "mlx_vlm.models.qwen3_vl_moe",
    "mlx_vlm.models.qwen2_vl",
    "mlx_vlm.models.qwen2_5_vl",
    "mlx_vlm.models.mistral3",
    "mlx_vlm.models.mistral3.mistral3",
    "mlx_vlm.models.gemma3n",
    "mlx_vlm.models.gemma3n.gemma3n",
    "mlx_vlm.models.lfm2_vl",
    "mlx_vlm.models.lfm2_vl.lfm2_vl",
    "mlx_vlm.models.cache",
]
for name in _mlx_vlm_modules:
    sys.modules.setdefault(name, mock.MagicMock())

# Mock vision add-on modules to avoid heavy dependencies during tests
_vision_add_on_modules = [
    "mlx_engine.model_kit.vision_add_ons.gemma3",
    "mlx_engine.model_kit.vision_add_ons.gemma3n",
    "mlx_engine.model_kit.vision_add_ons.pixtral",
    "mlx_engine.model_kit.vision_add_ons.mistral3",
    "mlx_engine.model_kit.vision_add_ons.lfm2_vl",
    "mlx_engine.model_kit.vision_add_ons.qwen2_vl",
    "mlx_engine.model_kit.vision_add_ons.qwen3_vl",
    "mlx_engine.model_kit.vision_add_ons.qwen3_vl_moe",
    "mlx_engine.model_kit.vision_add_ons.process_prompt_with_images",
    "mlx_engine.model_kit.vision_add_ons.qwen_vl_utils",
    "mlx_engine.model_kit.vision_add_ons.load_utils",
]
for name in _vision_add_on_modules:
    sys.modules.setdefault(name, mock.MagicMock())

# Mock transformers utilities used by external tokenizers
_transformers_utils = types.SimpleNamespace(logging=mock.MagicMock())
_tokenization_utils = types.SimpleNamespace(PreTrainedTokenizer=object)
_transformers_models = types.SimpleNamespace()
_transformers_models_auto = types.SimpleNamespace(
    processing_auto=types.SimpleNamespace()
)
sys.modules.setdefault("transformers", mock.MagicMock())
sys.modules.setdefault("transformers.models", _transformers_models)
sys.modules.setdefault("transformers.models.auto", _transformers_models_auto)
sys.modules.setdefault(
    "transformers.models.auto.processing_auto",
    _transformers_models_auto.processing_auto,
)
sys.modules.setdefault("transformers.utils", _transformers_utils)
sys.modules.setdefault("transformers.tokenization_utils", _tokenization_utils)
sys.modules.setdefault(
    "transformers.tokenization_utils_base", types.SimpleNamespace(AddedToken=object)
)

# Mock outlines to avoid optional dependency imports
_mock_outlines = mock.MagicMock()
_mock_outlines.processors = mock.MagicMock()
_mock_outlines.processors.structured = mock.MagicMock()
_mock_outlines.processors.structured.JSONLogitsProcessor = mock.MagicMock()
_mock_outlines.models = mock.MagicMock()
_mock_outlines.models.transformers = mock.MagicMock()
_mock_outlines.models.transformers.TransformerTokenizer = mock.MagicMock()
sys.modules.setdefault("outlines", _mock_outlines)
sys.modules.setdefault("outlines.processors", _mock_outlines.processors)
sys.modules.setdefault(
    "outlines.processors.structured", _mock_outlines.processors.structured
)
sys.modules.setdefault("outlines.models", _mock_outlines.models)
sys.modules.setdefault(
    "outlines.models.transformers", _mock_outlines.models.transformers
)


def pytest_addoption(parser):
    """Add command line option for heavy tests."""
    parser.addoption(
        "--heavy",
        action="store_true",
        default=False,
        help="run heavy tests (e.g., tests that require large models or long execution time)",
    )


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "heavy: mark test as heavy (requires --heavy option to run)"
    )
    config.addinivalue_line(
        "markers",
        "real_tokenizers: mark test as requiring real tokenizers (requires --real-tokenizers option to run)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip heavy tests unless --heavy option is provided."""
    if config.getoption("--heavy"):
        # --heavy given in cli: do not skip heavy tests
        pass
    else:
        skip_heavy = pytest.mark.skip(reason="need --heavy option to run")
        for item in items:
            if "heavy" in item.keywords:
                item.add_marker(skip_heavy)
