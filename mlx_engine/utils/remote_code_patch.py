from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from mlx_engine.models.ernie4_5.configuration_ernie4_5 import Ernie4_5_Config
from mlx_engine.models.ernie4_5.modeling_ernie4_5 import Ernie4_5_ForCausalLM
from mlx_engine.models.ernie4_5.tokenization_ernie4_5 import Ernie4_5_Tokenizer

def register_models():
    AutoConfig.register("ernie4_5", Ernie4_5_Config)
    AutoModelForCausalLM.register(Ernie4_5_Config, Ernie4_5_ForCausalLM)
    AutoTokenizer.register(Ernie4_5_Config, Ernie4_5_Tokenizer)