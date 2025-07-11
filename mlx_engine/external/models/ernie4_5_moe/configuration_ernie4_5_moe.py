# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PretrainedConfig



class Ernie4_5_MoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Ernie4_5_Model`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens)
        hidden_size (int): Dimensionality of the encoder layers and the pooler layer
        intermediate_size (int): Dimensionality of the "intermediate" (feed-forward) layer
        max_position_embeddings (int): Maximum sequence length the model can handle
        num_hidden_layers (int): Number of hidden layers in the Transformer encoder
        num_attention_heads (int): Number of attention heads for each attention layer
        rms_norm_eps (float): The epsilon used by the RMS normalization layers
        use_cache (bool): Whether to use caching for faster generation (decoding)
        use_flash_attention (bool): Whether to use FlashAttention for optimized attention computation
        pad_token_id (int): Token ID used for padding sequences
        bos_token_id (int): Token ID used for beginning-of-sequence
        eos_token_id (int): Token ID used for end-of-sequence
        use_bias (bool): Whether to use bias terms in linear layers
        rope_theta (float): The base period of the RoPE embeddings
        weight_share_add_bias (bool): Whether to share bias weights in certain layers
        ignored_index (int): Target value that is ignored during loss computation
        attention_probs_dropout_prob (float): Dropout probability for attention weights
        hidden_dropout_prob (float): Dropout probability for hidden layers
        num_key_value_heads (int): Number of key/value heads (for Grouped Query Attention)
        max_sequence_length (int): Maximum sequence length for positional embeddings
        moe_num_experts: Number of experts in MoE layers
        moe_capacity: Capacity configuration for MoE layers
        moe_layer_interval: Interval between MoE layers
        moe_layer_start_index: Starting layer index for MoE
        moe_layer_end_index: Ending layer index for MoE (-1 means last layer)
        sinkhorn_2gate: Whether to use sinkhorn 2-gate routing
        sinkhorn_temp: Temperature for sinkhorn routing
        moe_dropout_prob: Dropout probability for MoE layers
        moe_gate: Type of gating mechanism ('top2', etc.)
        moe_intermediate_size: Intermediate size for MoE layers
        moe_gate_act: Activation function for gating
        moe_k: Number of experts to route to
        num_nextn_predict_layers: Number of mtp predict layers, if use mtp, set `num_nextn_predict_layers > 0`
        multi_token_pred_lambda: The weight of multi token prediction loss
        **kwargs: Additional base model configuration parameters
    """

    model_type = "ernie4_5_moe"
    use_keep_in_fp32_modules = True
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "n_positions": "max_position_embeddings",
        "n_embd": "hidden_size",
        "n_layer": "num_hidden_layers",
        "n_head": "num_attention_heads",
        "n_inner": "intermediate_size",
        "activation_function": "hidden_act",
    }

    # Default tensor parallel plan for base model `ernie_4_5_moe`
    base_model_tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise_rep",
        "model.layers.*.self_attn.k_proj": "colwise_rep",
        "model.layers.*.self_attn.v_proj": "colwise_rep",
        "model.layers.*.self_attn.o_proj": "rowwise_rep",
        "model.layers.*.mlp.experts.*.gate_proj": "colwise",
        "model.layers.*.mlp.experts.*.up_proj": "colwise",
        "model.layers.*.mlp.experts.*.down_proj": "rowwise",
        "model.layers.*.mlp.gate_proj": "colwise",
        "model.layers.*.mlp.up_proj": "colwise",
        "model.layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        intermediate_size=11008,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=None,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        rope_theta=10000.0,
        use_flash_attention=False,
        use_rmsnorm=True,
        use_bias=False,
        weight_share_add_bias=True,
        max_sequence_length=None,
        ignored_index=-100,
        use_moe=True,
        moe_num_experts=64,
        moe_capacity=(64, 64, 64),
        moe_layer_interval=2,
        moe_layer_start_index=0,
        moe_layer_end_index=-1,
        sinkhorn_2gate=True,
        sinkhorn_temp=3e-2,
        moe_dropout_prob=0.0,
        moe_gate="top2",
        moe_intermediate_size=3584,
        moe_k=2,
        moe_gate_act: str = "softmax",
        moe_use_aux_free=False,
        num_nextn_predict_layers=0,
        multi_token_pred_lambda=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.use_rmsnorm = use_rmsnorm
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_sequence_length = max_sequence_length
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.ignored_index = ignored_index
        self.use_cache = use_cache
        self.use_bias = use_bias
        self.weight_share_add_bias = weight_share_add_bias
        self.use_flash_attention = use_flash_attention
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        self.use_moe = moe_num_experts > 0 and use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_capacity = moe_capacity
        self.sinkhorn_2gate = sinkhorn_2gate
        self.sinkhorn_temp = sinkhorn_temp
        self.moe_layer_interval = moe_layer_interval
        self.moe_dropout_prob = moe_dropout_prob
        self.moe_gate = moe_gate
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_k = moe_k
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_layer_end_index = (
            self.num_hidden_layers - 1
            if moe_layer_end_index == -1
            else moe_layer_end_index
        )
        self.moe_gate_act = moe_gate_act
        self.moe_use_aux_free = moe_use_aux_free
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.multi_token_pred_lambda = multi_token_pred_lambda

        # Set default for tied embeddings if not specified.
        if "tie_word_embeddings" not in kwargs:
            kwargs["tie_word_embeddings"] = False
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )