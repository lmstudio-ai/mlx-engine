""" "
Patch outlines_core to handler ERNIE tokenizer.
Specifically, fix the handling of these tokens:
- `>�`
- `�@`
- `�@�@`

An issue is opened in outlines_core tracking this issue:
https://github.com/dottxt-ai/outlines-core/issues/222
"""

import re

import outlines_core.fsm.regex


def apply_patches():
    """
    Apply patches to the outlines_core module.
    """
    # Update the replacement regex to fix the ernie tokenizer
    # Patching this line https://github.com/dottxt-ai/outlines-core/blob/0.1.26/python/outlines_core/fsm/regex.py#L349
    outlines_core.fsm.regex.re_replacement_seq = re.compile(r"^▁*\.*>*�+\.*s*@*(�@)*$")
