"""Register local model-specific code to bypass enabling `trust_remote_code`."""

from transformers import AutoTokenizer
# from mlx_engine.external.models.ernie4_5.configuration_ernie4_5 import Ernie4_5_Config
# from mlx_engine.external.models.ernie4_5_moe.configuration_ernie4_5_moe import (
#     Ernie4_5_MoeConfig,
# )
# from mlx_engine.external.models.ernie4_5.tokenization_ernie4_5 import Ernie4_5_Tokenizer


# breaks after updating transformers to 4.54.0 which brings in native ernie support
def register_models():
    pass
    # AutoTokenizer.register(Ernie4_5_Config, Ernie4_5_Tokenizer)
    # AutoTokenizer.register(Ernie4_5_MoeConfig, Ernie4_5_Tokenizer)
