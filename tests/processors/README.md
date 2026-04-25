This directory contains processors that are useful during testing, but do not have a prod use-case. They can easily be inserted into the generate flow during development by modifying `mlx_engine/generate.py`.

For example, we can add a `DumpLogitsProcessor` that writes the logits on each generated token as a CSV to the `logits-dump` directory:

```diff
--- a/mlx_engine/generate.py
+++ b/mlx_engine/generate.py
@@ -51,6 +51,9 @@ from mlx_engine.utils.generation_helpers import (
+from tests.processors.dump_logits_processor import (
+    DumpLogitsProcessor,
+)
 from mlx_engine.utils.generation_helpers import (
     setup_logits_processors,
     create_sampler,
@@ -480,6 +480,7 @@ def _sequential_generation(
         )

+        logits_processors.append(DumpLogitsProcessor(model_kit.tokenizer.vocab, Path("logits-dump")))
         # Set up sampler
         generate_args["sampler"] = create_sampler(
             temp, top_p, min_p, min_tokens_to_keep, top_k
```
