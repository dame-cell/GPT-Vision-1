from torch import nn
import transformers
from .modeling_gpt2 import GPT2LMHeadModel
from .configuration_gptvision import GPT2Config

transformers.logging.set_verbosity_error()


class TextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if type(config.gpt2_config) == dict:
            gpt2_config = GPT2Config(**config.gpt2_config)
        else:
            gpt2_config = config.gpt2_config

        self.model = GPT2LMHeadModel(gpt2_config)
        self.text_emb = self.model.get_input_embeddings()