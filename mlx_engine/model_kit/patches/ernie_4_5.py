import re

import outlines_core.fsm.regex


def apply_patches():
    """
    Apply patches to the outlines_core module.
    """
    # Update the replacement regex to fix the ernie tokenizer
    outlines_core.fsm.regex.re_replacement_seq = re.compile(r"^▁*\.*>*�+\.*s*@*(�@)*$")
