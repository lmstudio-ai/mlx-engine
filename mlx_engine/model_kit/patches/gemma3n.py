from mlx_lm.models.gemma3n import Model, TextConfig
from mlx.utils import tree_flatten, tree_unflatten
import inspect

APPLIED = False


def _patch_weight_ordering():
    def mlx_vlm_compatible_sanitize(self, weights):
        weights = tree_unflatten(list(weights.items()))
        if weights.get("language_model", {}).get("model", None) is not None:
            weights = {"model": {"language_model": weights["language_model"]["model"]}}
        weights = dict(tree_flatten(weights))
        return self.original_sanitize(weights)

    Model.original_sanitize = Model.sanitize
    Model.sanitize = mlx_vlm_compatible_sanitize


def _patch_intermediate_size_value():
    @classmethod
    def fix_intermediate_size_from_dict(cls, params):
        config_dict = {
            k: v for k, v in params.items() if k in inspect.signature(cls).parameters
        }
        if "intermediate_size" in config_dict and isinstance(
            config_dict["intermediate_size"], list
        ):
            config_dict["intermediate_size"] = config_dict["intermediate_size"][0]
        return cls(**config_dict)

    TextConfig.original_from_dict = TextConfig.from_dict
    TextConfig.from_dict = fix_intermediate_size_from_dict


def do_patch():
    """
    mlx-vlm's conversion changes the weight keys from the original huggingface weights.
    For example, "model.language_model.embed_tokens.weight" becomes "language_model.model.embed_tokens.weight"
    mlx-lm expects the weight keys to be in the original huggingface order.
    Here, we use some magic to minimally patch the mlx-lm gemma3n Model class's sanitize to be able to read mlx-vlm converted weights.
    """
    global APPLIED
    if APPLIED:
        return
    _patch_weight_ordering()
    _patch_intermediate_size_value()

    APPLIED = True
