import json
from queue import Queue
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import mlx_engine.generate as generate_module
import mlx_engine.model_kit.distributed_model_kit as distributed_module
from mlx_engine.model_kit.distributed_model_kit import (
    DistributedModelKit,
    DistributedSchedulerGenerationRequest,
)
from mlx_engine.utils.generation_helpers import setup_repetition_logits_processors
from mlx_engine.utils.sampling import create_sampler


@pytest.mark.parametrize(
    "stop_strings, expected_text, stop_reason",
    [(["END"], "X", "stop_string"), ([], "XEND", "eos_token")],
)
def test_distributed_output_trims_complete_stop_strings(
    monkeypatch, tmp_path, stop_strings, expected_text, stop_reason
):
    vocabulary = {"<eos>": 0, "P": 1, "X": 2, "E": 3, "N": 4, "D": 5, "<unk>": 6}
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = TokenizerWrapper(
        PreTrainedTokenizerFast(
            tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>"
        )
    )

    class ScriptedModel(Model):
        def __call__(self, inputs, cache=None):
            # Exercise real attention/cache and batch generation with deterministic output.
            super().__call__(inputs, cache=cache)
            next_tokens = mx.array([0, 2, 3, 4, 5, 0, 0])[inputs]
            return mx.where(
                mx.arange(len(vocabulary)) == next_tokens[..., None], 100.0, -100.0
            )

    model = ScriptedModel(
        ModelArgs(
            model_type="llama",
            hidden_size=16,
            num_hidden_layers=1,
            intermediate_size=32,
            num_attention_heads=1,
            rms_norm_eps=1e-5,
            vocab_size=len(vocabulary),
        )
    )
    (tmp_path / "config.json").write_text('{"model_type":"llama"}')
    monkeypatch.setattr(
        distributed_module,
        "_instrumented_tensor_sharded_load",
        lambda *args, **kwargs: (model, tokenizer),
    )
    # Replace only cross-device communication; retain the scheduler and public generator.
    monkeypatch.setattr(
        DistributedModelKit, "_share_scheduler_message", lambda self, message: message
    )
    kit = DistributedModelKit(
        tmp_path,
        prefill_step_size=32,
        distributed_group=SimpleNamespace(rank=lambda: 0, size=lambda: 2),
    )
    try:
        kit.start()
        results = list(
            generate_module.create_generator(
                kit,
                [vocabulary["P"]],
                request_id="stop-regression",
                temp=0,
                max_tokens=10,
                stop_strings=stop_strings,
            )
        )
        assert "".join(result.text for result in results) == expected_text
        assert results[-1].stop_condition.stop_reason == stop_reason
    finally:
        kit.shutdown()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("trust_remote_code", [None, False, True])
def test_cluster_tokenizer_requires_explicit_trust(
    monkeypatch, tmp_path, rank, trust_remote_code
):
    marker = tmp_path / "custom-code-executed"
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"auto_map": {"AutoTokenizer": ["custom_tokenizer.CustomTokenizer", None]}}
        )
    )
    (tmp_path / "custom_tokenizer.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise RuntimeError('custom tokenizer code executed')\n"
    )
    # Avoid model weights and RDMA while retaining the real tokenizer loader.
    monkeypatch.setattr(
        distributed_module,
        "load_model",
        lambda *args, **kwargs: (SimpleNamespace(shard=lambda group: None), {}),
    )
    group = SimpleNamespace(rank=lambda: rank, size=lambda: 2)
    options = (
        {} if trust_remote_code is None else {"trust_remote_code": trust_remote_code}
    )
    kit = DistributedModelKit.__new__(DistributedModelKit)
    try:
        expected_error = RuntimeError if trust_remote_code is True else ValueError
        expected_message = (
            "custom tokenizer code executed"
            if trust_remote_code is True
            else "trust_remote_code"
        )
        with pytest.raises(expected_error, match=expected_message):
            kit.__init__(
                tmp_path, prefill_step_size=512, distributed_group=group, **options
            )
        assert marker.exists() is (trust_remote_code is True)
    finally:
        kit.shutdown()


def test_distributed_load_preserves_app_interface(monkeypatch, tmp_path):
    calls = []
    group = object()

    class FakeDistributedKit:
        def __init__(self, path, **options):
            calls.append((path, options))

        def start(self):
            calls.append("start")

    monkeypatch.setattr(generate_module, "DistributedModelKit", FakeDistributedKit)
    monkeypatch.setattr(generate_module, "sanitize_eos_tokens", lambda kit: None)
    kit = generate_module.load_model(
        tmp_path,
        distributed=True,
        distributed_group=group,
        max_kv_size=4096,
        max_seq_nums=1,
    )
    assert isinstance(kit, FakeDistributedKit)
    assert calls[0][1]["distributed_group"] is group
    assert calls[0][1]["max_kv_size"] == 4096
    assert calls[0][1]["max_seq_nums"] == 1
    assert calls[-1] == "start"


@pytest.mark.parametrize(
    "options",
    [
        {"vocab_only": True},
        {"kv_bits": 4},
        {"kv_group_size": 64},
        {"quantized_kv_start": 0},
    ],
)
def test_distributed_load_rejects_unsupported_options_before_rank_init(options):
    with pytest.raises(ValueError):
        generate_module.load_model("unused", distributed=True, **options)


def test_distributed_generator_and_cancel_route_to_scheduler(monkeypatch):
    kit = object.__new__(DistributedModelKit)
    kit.uses_distributed_batching = lambda: True
    sentinel = iter(())
    monkeypatch.setattr(
        generate_module, "_batched_generation", lambda *args, **kwargs: sentinel
    )
    assert (
        generate_module.create_generator(kit, [1, 2], request_id="request") is sentinel
    )
    removed = []
    kit.remove = removed.append
    generate_module.stop_generation(kit, "request")
    assert removed == ["request"]


@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("penalty", [None, 1.1])
def test_scheduler_wire_roundtrip_preserves_sampling(temperature, penalty):
    prompt = [1, 3, 1]
    sampling = {
        "temperature": temperature,
        "topP": 0.9,
        "topK": 4,
        "minP": 0.05,
        "seed": None,
    }
    master = DistributedSchedulerGenerationRequest(
        response_queue=Queue(),
        prompt_tokens=prompt,
        prompt_segments=None,
        segment_types=None,
        request_id="roundtrip",
        sampler=create_sampler(temperature, 0.9, 0.05, 1, 4),
        logits_processors=setup_repetition_logits_processors(
            penalty, 20, prompt, prompt
        ),
        top_logprobs=0,
        max_tokens=32,
        stop_strings=["stop"],
        sampling=sampling,
        repetition_penalty=penalty,
        repetition_context_size=20,
        min_tokens_to_keep=1,
    )
    kit = object.__new__(DistributedModelKit)
    message = json.loads(json.dumps(kit._scheduler_item_to_message(master)))
    worker = kit._scheduler_message_to_worker_item(message)
    assert worker.request_id == master.request_id
    assert worker.prompt_tokens == prompt
    assert worker.max_tokens == 32
    assert worker.stop_strings == ["stop"]
    logits = mx.array([[0.0, 3.0, 1.0, 2.0, -1.0]])
    master_logits = logits
    worker_logits = logits
    for processor in master.logits_processors:
        master_logits = processor(mx.array(prompt), master_logits)
    for processor in worker.logits_processors:
        worker_logits = processor(mx.array(prompt), worker_logits)
    assert master_logits.tolist() == worker_logits.tolist()
    mx.random.seed(42)
    expected = master.sampler(master_logits).item()
    mx.random.seed(42)
    actual = worker.sampler(worker_logits).item()
    assert actual == expected


@pytest.mark.parametrize(
    "options",
    [
        {"images_b64": ["image"]},
        {"speculative_decoding_toggle": True},
        {"num_draft_tokens": 2},
        {"seed": 0},
    ],
)
def test_unsupported_distributed_requests_fail_before_scheduler_enqueue(options):
    kit = object.__new__(DistributedModelKit)
    kit.uses_distributed_batching = lambda: True
    kit.supports_request_level_seed = lambda: False

    def unexpected_enqueue(**kwargs):
        pytest.fail("Unsupported request reached the distributed scheduler")

    kit.generate = unexpected_enqueue
    with pytest.raises(ValueError):
        list(generate_module.create_generator(kit, [1, 2], **options))


@pytest.mark.parametrize("request_id", [None, ""])
def test_distributed_request_options_remain_wire_serializable(request_id):
    kit = object.__new__(DistributedModelKit)
    kit.uses_distributed_batching = lambda: True
    kit.supports_request_level_seed = lambda: True
    kit.tokenizer = object()
    requests = []

    def capture_request(**options):
        requests.append(options)
        return iter(())

    kit.generate = capture_request
    messages = [{"role": "user", "content": "hello"}]
    template_options = {"enable_thinking": False}
    list(
        generate_module.create_generator(
            kit,
            [1, 2],
            request_id=request_id,
            temp=0.0,
            seed=0,
            chat_messages=messages,
            chat_template_kwargs=template_options,
        )
    )
    assert len(requests) == 1
    request = requests[0]
    assert isinstance(request["request_id"], str)
    assert len(request["request_id"]) > 0
    assert request["sampling"]["seed"] == 0
    assert request["chat_messages"] == messages
    assert request["chat_template_kwargs"] == template_options
    json.dumps(
        {
            "requestId": request["request_id"],
            "sampling": request["sampling"],
            "messages": request["chat_messages"],
            "template": request["chat_template_kwargs"],
        }
    )


def test_scheduler_roundtrip_reconstructs_json_constraint(monkeypatch):
    from types import SimpleNamespace
    import outlines.processors.structured as structured
    import mlx_engine.utils.outlines_transformer_tokenizer as tokenizer_module

    schema = json.dumps({"type": "object", "properties": {"title": {"type": "string"}}})
    tokenizer = object()
    captured = []
    def make_processor(received_schema, received_tokenizer, *, tensor_library_name):
        captured.append((received_schema, received_tokenizer, tensor_library_name))
        return "json-constraint"

    monkeypatch.setattr(structured, "JSONLogitsProcessor", make_processor)
    monkeypatch.setattr(tokenizer_module, "OutlinesTransformerTokenizer", lambda value: value)
    kit = object.__new__(DistributedModelKit)
    kit.tokenizer = SimpleNamespace(_tokenizer=tokenizer)
    request = DistributedSchedulerGenerationRequest(
        response_queue=Queue(), prompt_tokens=[1], prompt_segments=None,
        segment_types=None, request_id="json-title", sampler=None,
        logits_processors=[], top_logprobs=0, max_tokens=32, stop_strings=[],
        sampling={"temperature": 0, "topP": 1, "topK": 0, "minP": 0, "seed": None},
        repetition_penalty=None, repetition_context_size=20, min_tokens_to_keep=1,
        json_schema=schema,
    )
    message = json.loads(json.dumps(kit._scheduler_item_to_message(request)))
    worker = kit._scheduler_message_to_worker_item(message)
    assert worker.json_schema == schema
    assert worker.logits_processors[-1] == "json-constraint"
    assert captured == [(schema, tokenizer, "mlx")]
