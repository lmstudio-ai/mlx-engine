from outlines.models.transformers import TransformerTokenizer
from mlx_engine.external.datasets.dill import Hasher


class OutlinesTransformerTokenizer(TransformerTokenizer):
    """
    Update the outlines TransformerTokenizer to use our own Hasher class, so that we don't need the datasets dependency

    This class and the external dependency can be removed when the following import is deleted
    https://github.com/dottxt-ai/outlines/blob/69418d/outlines/models/transformers.py#L117
    """

    def __hash__(self):
        return hash(Hasher.hash(self.tokenizer))
