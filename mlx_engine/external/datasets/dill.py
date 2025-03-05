# copied from https://github.com/huggingface/datasets/blob/1e1d313/src/datasets/utils/_dill.py

# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Extends `dill` to support pickling more types and produce more consistent dumps."""

import sys
from io import BytesIO
from types import FunctionType
from typing import Any, Dict, List, Union

import dill
import xxhash


class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: Dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def hash(cls, value: Any) -> str:
        return cls.hash_bytes(dumps(value))

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


class Pickler(dill.Pickler):
    dispatch = dill._dill.MetaCatchingDict(dill.Pickler.dispatch.copy())
    _legacy_no_dict_keys_sorting = False

    def save(self, obj, save_persistent_id=True):
        obj_type = type(obj)
        if obj_type not in self.dispatch:
            if "regex" in sys.modules:
                import regex  # type: ignore

                if obj_type is regex.Pattern:
                    pklregister(obj_type)(_save_regexPattern)
            if "spacy" in sys.modules:
                import spacy  # type: ignore

                if issubclass(obj_type, spacy.Language):
                    pklregister(obj_type)(_save_spacyLanguage)
            if "tiktoken" in sys.modules:
                import tiktoken  # type: ignore

                if obj_type is tiktoken.Encoding:
                    pklregister(obj_type)(_save_tiktokenEncoding)
            if "torch" in sys.modules:
                import torch  # type: ignore

                if issubclass(obj_type, torch.Tensor):
                    pklregister(obj_type)(_save_torchTensor)

                if obj_type is torch.Generator:
                    pklregister(obj_type)(_save_torchGenerator)

                # Unwrap `torch.compile`-ed modules
                if issubclass(obj_type, torch.nn.Module):
                    obj = getattr(obj, "_orig_mod", obj)
            if "transformers" in sys.modules:
                import transformers  # type: ignore

                if issubclass(obj_type, transformers.PreTrainedTokenizerBase):
                    pklregister(obj_type)(_save_transformersPreTrainedTokenizerBase)

        # Unwrap `torch.compile`-ed functions
        if obj_type is FunctionType:
            obj = getattr(obj, "_torchdynamo_orig_callable", obj)
        dill.Pickler.save(self, obj, save_persistent_id=save_persistent_id)

    def _batch_setitems(self, items):
        if self._legacy_no_dict_keys_sorting:
            return super()._batch_setitems(items)
        # Ignore the order of keys in a dict
        try:
            # Faster, but fails for unorderable elements
            items = sorted(items)
        except Exception:  # TypeError, decimal.InvalidOperation, etc.
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        dill.Pickler._batch_setitems(self, items)

    def memoize(self, obj):
        # Don't memoize strings since two identical strings can have different Python ids
        if type(obj) is not str:  # noqa: E721
            dill.Pickler.memoize(self, obj)


def pklregister(t):
    """Register a custom reducer for the type."""

    def proxy(func):
        Pickler.dispatch[t] = func
        return func

    return proxy


def dump(obj, file):
    """Pickle an object to a file."""
    Pickler(file, recurse=True).dump(obj)


def dumps(obj):
    """Pickle an object to a string."""
    file = BytesIO()
    dump(obj, file)
    return file.getvalue()


def log(pickler, msg):
    pass


def _save_regexPattern(pickler, obj):
    import regex  # type: ignore

    log(pickler, f"Re: {obj}")
    args = (obj.pattern, obj.flags)
    pickler.save_reduce(regex.compile, args, obj=obj)
    log(pickler, "# Re")


def _save_tiktokenEncoding(pickler, obj):
    import tiktoken  # type: ignore

    log(pickler, f"Enc: {obj}")
    args = (obj.name, obj._pat_str, obj._mergeable_ranks, obj._special_tokens)
    pickler.save_reduce(tiktoken.Encoding, args, obj=obj)
    log(pickler, "# Enc")


def _save_torchTensor(pickler, obj):
    import torch  # type: ignore

    # `torch.from_numpy` is not picklable in `torch>=1.11.0`
    def create_torchTensor(np_array, dtype=None):
        tensor = torch.from_numpy(np_array)
        if dtype:
            tensor = tensor.type(dtype)
        return tensor

    log(pickler, f"To: {obj}")
    if obj.dtype == torch.bfloat16:
        args = (obj.detach().to(torch.float).cpu().numpy(), torch.bfloat16)
    else:
        args = (obj.detach().cpu().numpy(),)
    pickler.save_reduce(create_torchTensor, args, obj=obj)
    log(pickler, "# To")


def _save_torchGenerator(pickler, obj):
    import torch  # type: ignore

    def create_torchGenerator(state):
        generator = torch.Generator()
        generator.set_state(state)
        return generator

    log(pickler, f"Ge: {obj}")
    args = (obj.get_state(),)
    pickler.save_reduce(create_torchGenerator, args, obj=obj)
    log(pickler, "# Ge")


def _save_spacyLanguage(pickler, obj):
    import spacy  # type: ignore

    def create_spacyLanguage(config, bytes):
        lang_cls = spacy.util.get_lang_class(config["nlp"]["lang"])
        lang_inst = lang_cls.from_config(config)
        return lang_inst.from_bytes(bytes)

    log(pickler, f"Sp: {obj}")
    args = (obj.config, obj.to_bytes())
    pickler.save_reduce(create_spacyLanguage, args, obj=obj)
    log(pickler, "# Sp")


def _save_transformersPreTrainedTokenizerBase(pickler, obj):
    log(pickler, f"Tok: {obj}")
    # Ignore the `cache` attribute
    state = obj.__dict__
    if "cache" in state and isinstance(state["cache"], dict):
        state["cache"] = {}
    pickler.save_reduce(type(obj), (), state=state, obj=obj)
    log(pickler, "# Tok")
