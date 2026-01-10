from functools import wraps
import sys
import huggingface_hub

# Store the original function before we patch anything
_original_snapshot_download = huggingface_hub.snapshot_download


@wraps(_original_snapshot_download)
def snapshot_download(*args, **kwargs):
    pass


def patch_huggingface_hub():
    pass
