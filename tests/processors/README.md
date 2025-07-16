# tests/processors README

This directory contains processors that are useful during testing, but do not have a prod use-case. They can easily be inserted into the generate flow during development by modifying `mlx_engine/generate.py`.

For example, we can add a `DumpLogitsProcessor` that writes the logits on each generated token as a CSV to the `logits-dump` directory:

```diff
--- a/mlx_engine/generate.py
+++ b/mlx_engine/generate.py
@@ -12,6 +12,9 @@ from mlx_engine.processors.outlines_logits_processor import OutlinesLogitsProces
 from mlx_engine.processors.repetition_penalty_processor import (
     RepetitionPenaltyProcessor,
 )
+from tests.processors.dump_logits_processor import (
+    DumpLogitsProcessor,
+)
 from mlx_engine.utils.token import Token
 from mlx_engine.utils.eot_tokens import get_eot_token_ids
 from mlx_engine.utils.top_logprobs import summarize_top_logprobs
@@ -236,6 +239,9 @@ def create_generator(
                 token_history=cached_tokens, **repetition_penalty_kwargs
             )
         )
+    generate_args["logits_processors"].append(
+        DumpLogitsProcessor(model_kit.tokenizer.vocab, Path("logits-dump"))
+    )

     # Set up sampler
     generate_args["sampler"] = make_sampler(
```
