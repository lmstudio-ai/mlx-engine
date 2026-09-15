import json
from queue import Queue

import mlx.core as mx
import pytest

import mlx_engine.generate as generate_module
from mlx_engine.model_kit.distributed_model_kit import (
    DistributedModelKit,
    DistributedSchedulerGenerationRequest,
)
from mlx_engine.utils.generation_helpers import setup_repetition_logits_processors
from mlx_engine.utils.sampling import create_sampler


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
