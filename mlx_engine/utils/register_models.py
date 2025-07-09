"""Register local model-specific code to bypass enabling `trust_remote_code`."""

from transformers import AutoTokenizer
from mlx_engine.external.models.ernie4_5.configuration_ernie4_5 import Ernie4_5_Config
from mlx_engine.external.models.ernie4_5.tokenization_ernie4_5 import Ernie4_5_Tokenizer

def register_models():
    AutoTokenizer.register(Ernie4_5_Config, Ernie4_5_Tokenizer)