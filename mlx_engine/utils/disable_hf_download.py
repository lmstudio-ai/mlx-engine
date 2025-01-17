from functools import wraps
import sys
import huggingface_hub

# Store the original function before we patch anything
_original_snapshot_download = huggingface_hub.snapshot_download


@wraps(_original_snapshot_download)
def snapshot_download(*args, **kwargs):
    """
    Wrapper around huggingface_hub.snapshot_download that disables it
    """
    raise RuntimeError(
        "Internal error: Cannot proceed without downloading from huggingface. Please report this error to the LM Studio team."
    )


def patch_huggingface_hub():
    """
    Patch the huggingface_hub module to use our local-only snapshot_download.
    This ensures that any import of snapshot_download from huggingface_hub
    will use our wrapped version.
    """
    huggingface_hub.snapshot_download = snapshot_download
    # Also patch the module in sys.modules to ensure any other imports get our version
    sys.modules["huggingface_hub"].snapshot_download = snapshot_download
