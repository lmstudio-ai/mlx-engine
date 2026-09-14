# Distributed text inference

This branch ports the distributed engine from `c86c23a` onto engine main
`08f0c0758a804168e258c4cfb80891a8fc9eccb1`. It keeps main's dependency pins and
local inference implementations. The application/native bridge is a separate
integration boundary; installing this engine alone does not add cluster UI or
peer orchestration to an application.

## Runtime contract

The coordinator calls the existing public API with `distributed=True` and an
initialized MLX group:

```python
model = load_model(
    model_path,
    distributed=True,
    distributed_group=group,
    max_kv_size=4096,
    max_seq_nums=1,
)
```

The group must contain more than one rank. Each rank needs matching engine and
dependency versions, matching model files, and the same load configuration.
Only rank zero accepts application generation requests. Other ranks run
`python -I -m mlx_engine.distributed_worker` with the corresponding model and
load arguments; the host application supplies the MLX rank/network environment.
The worker imports from its installed Python environment.

Models must implement MLX-LM's tensor-parallel `shard(group)` interface. The
loader creates the model lazily, applies the tensor shard, evaluates the local
parameters, and synchronizes the ranks before returning. Qwen3-32B-MLX-4bit is
the validated model for this port; this is not a claim that every architecture
with a shard method has been validated.

## Execution and ordering

`DistributedModelKit` owns a model thread. Loading, collective operations,
batch scheduling, and generation run on that thread and its default GPU stream.
Caller threads enqueue work instead of running GPU operations independently.
The explicit stream preparation is retained from the distributed baseline.

Distributed inference uses the batch scheduler even with `max_seq_nums=1`.
Rank zero broadcasts one ordered stream of generation, cancellation, and shutdown
commands. Workers rebuild sampling and repetition processors using the same
helpers as rank zero. Prompt tokens, cache segments, sampling values, and stop
strings travel in that scheduler protocol. Worker responses stay internal; only
rank zero yields application output.

The scheduler uses MLX-LM's `BatchGenerator` and in-memory prompt cache.
Cancellation must be coordinated across the ranks. Do not independently cancel
a worker's iteration or start a replacement rank inside a live group.

## Supported scope and limitations

- Text generation, concurrent requests, prompt reuse, token limits, stop strings,
  and coordinated cancellation.
- Request-level seeds when concurrency is one. They are rejected when concurrency
  is greater than one, since the ranks share the batch scheduler's RNG ordering.
- No distributed images, structured JSON output, draft models, speculative
  decoding, vocabulary-only loads, or KV-cache quantization.
- Distributed loads use the configured context limit and in-memory caching.
  `auto_fit_context` and `enable_disk_cache` apply to local batched loads.
- Ordinary local inference continues through main's `ModelKit` and
  `BatchedVisionModelKit`; the old branch's cache/vision implementations are not
  restored.

**Worker-loss supervision remains an application responsibility.** A dead rank
can leave an MLX collective blocked without a Python exception. This was
reproduced with both the old MLX 0.31.2 runtime and this MLX 0.32.0 port. A
supervisor must terminate the entire group before reloading it. In the tested
application, manual unload/reload recovers; automatic detection and error
reporting are not established by this port. Python exception handling alone
does not supply a collective deadline.

## Entry points

- `distributed_rank.py`: native rank initialization and worker-loop bridge.
- `distributed_worker.py`: installed-runtime worker process.
- `distributed_coordinator.py`: standalone JSON-lines coordinator for validation.
- `distributed_validation_runner.py`, `distributed_validation_rank_entry.py`,
  and `distributed_validation_harness.py`: launch and request-validation tools.
- `distributed_server.py`: retained compatibility entry for the existing
  application's diagnostic path.

Use `python -m mlx_engine.distributed_validation_runner --help` for launcher
options. This harness does not configure Thunderbolt networking or install model
files.

## Port validation

On four Mac Studios, using the same Qwen3-32B-MLX-4bit files on every rank:

- Existing application/native bridge: distributed load, arithmetic, multi-turn
  generation, user cancellation followed by another request, unload, and reload.
- Direct coordinator: three cycles of three submitted requests with two active
  sequences; independent arithmetic outputs and token-limit completion.
- Stop-string completion followed by coordinated shutdown; all four processes
  exited with code zero.
- Local sequential and local batched paths: arithmetic, cancellation, follow-up
  generation, and unload with the same 32B model.
- Worker-loss comparison: old and new runtimes stall after a worker exits;
  the existing app's unload/reload restores the new engine.

`tests/test_distributed_engine.py` covers the application load contract,
unsupported-option rejection before enqueue, scheduler routing, serializable
request IDs/options, and rank-zero/worker sampling parity. The broader available
suite passed; tests requiring absent external model fixtures and the full
vision/model integration suites were not completed on the cluster.
