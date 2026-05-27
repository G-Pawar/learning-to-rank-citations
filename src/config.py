"""
Contains configuration settings for the Bibliography Ranker system.
"""

class BibliographyRankerConfig:
    """
    Configuration class for the Bibliography Ranker system.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        use_gpu: bool = True,
        batch_size: int = 8,
        chunk_size: int = 50,
        sleep_seconds: float = 0.2,
        pooling_method: str = "mean"
    ):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.sleep_seconds = sleep_seconds
        self.pooling_method = pooling_method

# Base OpenAlex API URL
OPENALEX_BASE_URL = "https://api.openalex.org/works"
