"""Register local model-specific code to bypass enabling `trust_remote_code`."""

from transformers import AutoTokenizer, AutoProcessor
from mlx_engine.external.models.ernie4_5.configuration_ernie4_5 import Ernie4_5_Config
from mlx_engine.external.models.ernie4_5_moe.configuration_ernie4_5_moe import (
    Ernie4_5_MoeConfig,
)
from mlx_engine.external.models.ernie4_5.tokenization_ernie4_5 import Ernie4_5_Tokenizer
from mlx_engine.external.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor
from mlx_engine.external.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig


def register_models():
    AutoTokenizer.register(Ernie4_5_Config, Ernie4_5_Tokenizer, exist_ok=True)
    AutoTokenizer.register(Ernie4_5_MoeConfig, Ernie4_5_Tokenizer, exist_ok=True)
    AutoProcessor.register(Lfm2VlConfig, Lfm2VlProcessor, exist_ok=True)
