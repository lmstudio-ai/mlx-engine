"""Register local model-specific code to bypass enabling `trust_remote_code`."""

from transformers import AutoTokenizer, AutoProcessor
import transformers.models.auto.processing_auto as processing_auto
from mlx_engine.external.models.ernie4_5.configuration_ernie4_5 import Ernie4_5_Config
from mlx_engine.external.models.ernie4_5_moe.configuration_ernie4_5_moe import (
    Ernie4_5_MoeConfig,
)
from mlx_engine.external.models.ernie4_5.tokenization_ernie4_5 import Ernie4_5_Tokenizer
from mlx_engine.external.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
from mlx_engine.external.models.lfm2_vl.router_lfm2_vl_processor import (
    Lfm2VlProcessor as RouterLfm2VlProcessor,
)
from transformers.models.lfm2_vl.configuration_lfm2_vl import (
    Lfm2VlConfig as HFLfm2VlConfig,
)


def register_models():
    # exist_ok=True should be an indication that we should remove external code
    # ref https://github.com/lmstudio-ai/mlx-engine/issues/211
    AutoTokenizer.register(Ernie4_5_Config, Ernie4_5_Tokenizer, exist_ok=True)
    AutoTokenizer.register(Ernie4_5_MoeConfig, Ernie4_5_Tokenizer, exist_ok=True)

    # There are two versions of the Lfm2VlProcessor. The first "legacy" processor was released
    # as remote code in the model artifact [1], but was removed [2] when the processor was merged
    # into transformers [3]. The merged implementation of the processor differs from the legacy
    # version, so we need to keep both and route models with the legacy config to the legacy
    # processor and models with the new config to the new processor.
    # 
    # [1] https://huggingface.co/LiquidAI/LFM2-VL-1.6B/commit/5c24786dc2d7eb472899e4e3333f92f938233d4f
    # [2] https://huggingface.co/LiquidAI/LFM2-VL-1.6B/commit/125f2f31caac7328be8dae2e9204a06d6cf5b51c
    # [3] https://github.com/huggingface/transformers/blob/c7e5b749a6392ea2f42fea983af41f825b0bc78d/src/transformers/models/lfm2_vl/processing_lfm2_vl.py#L52
    del processing_auto.PROCESSOR_MAPPING_NAMES["lfm2_vl"]
    AutoProcessor.register(HFLfm2VlConfig, RouterLfm2VlProcessor, exist_ok=False)
