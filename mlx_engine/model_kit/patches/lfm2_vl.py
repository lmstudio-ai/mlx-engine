import mlx.nn as nn
import mlx_vlm.models.lfm2_vl
import mlx_vlm.models.lfm2_vl.lfm2_vl as lfm2_vl


class CompatibleModel(lfm2_vl.Model):
    def __init__(self, config):
        super().__init__(config)
        if not getattr(self.config, "projector_use_layernorm", True):
            # Older LFM2.5-VL MLX conversions omit this disabled projector
            # LayerNorm. Do not keep an unused parameter-owning module around,
            # because MLX-format weights bypass sanitize() and strict load would
            # otherwise expect neutral layer_norm.{weight,bias} tensors.
            self.multi_modal_projector.layer_norm = nn.Identity()


def apply_patches():
    mlx_vlm.models.lfm2_vl.Model = CompatibleModel
    lfm2_vl.Model = CompatibleModel
