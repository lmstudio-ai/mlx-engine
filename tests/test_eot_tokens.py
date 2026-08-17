from types import SimpleNamespace

from mlx_engine.utils.eot_tokens import sanitize_eos_tokens


class _MuseGlimmerTokenizer:
    def __init__(self):
        self.eos_token_ids = {200001}
        self.eos_token_id = 200001
        self._tokenizer = SimpleNamespace(eos_token_id=200001)

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "<|eot|>":
            return [200008]
        if text == "<|eom|>":
            return [200007]
        return [1, 2]

    def decode(self, token_id):
        return f"token-{token_id}"


def test_muse_glimmer_sanitization_adds_eot_but_not_eom():
    tokenizer = _MuseGlimmerTokenizer()
    model_kit = SimpleNamespace(tokenizer=tokenizer, model_type="muse_glimmer")

    sanitize_eos_tokens(model_kit)

    assert tokenizer.eos_token_ids == {200001, 200008}
    assert 200007 not in tokenizer.eos_token_ids
