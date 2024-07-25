import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoTokenizer
from .configuration_gpt2vision import GPT2VisionConfig
from .vision_encoder import VisionEncoder
from .modeling_gpt2 import GPT2LMHeadModel

IMAGE_TOKEN = "<image>"
ANSWER_EOS = "<|endoftext|>"

def resize_token_embeds(model_name="openai-community/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    new_tokens = {
        "additional_special_tokens": [IMAGE_TOKEN]
    }
    tokenizer.add_special_tokens(new_tokens)
    return tokenizer

tokenizer = resize_token_embeds()

class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GPT2Vision(PreTrainedModel):
    config_class = GPT2VisionConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = VisionEncoder()
        self.mlp = MLP(in_features=768, hidden_features=768 * 4, out_features=768)
        self.language_model = GPT2LMHeadModel(config.gpt2_config)
        self.language_model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    def tokenize_encode(self, batch, device):
        text = batch['text']
        images = batch['image']
        if isinstance(text, str):
            text = [text]
        input_texts = [f"{IMAGE_TOKEN}{t}" for t in text]
        text_inputs = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=768,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)
        pixel_values = self.vision_encoder(images, device)
        return {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "pixel_values": pixel_values
        }

    def preprocess_inputs(self, batch):
        pixel_values = batch['pixel_values'].squeeze(1)
        input_ids = batch['input_ids'].squeeze(1)
        attention_mask = batch['attention_mask'].squeeze(1)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        pixel_values = pixel_values.to(self.device)
        img_embs = self.mlp(pixel_values)
        tok_embs = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
        img_attention = torch.ones((img_embs.size(0), img_embs.size(1)), dtype=torch.long, device=self.device)
        attention_mask = torch.cat((attention_mask[:, 0:1], img_attention, attention_mask[:, 1:]), dim=1)
        return inputs_embeds, attention_mask, input_ids

    def generate(self, question, image, max_new_tokens=30, **kwargs):
        prompt = f"Question: {question}\nAnswer:"
        batch = {"image": [image], "text": prompt}
        encoded_batch = self.tokenize_encode(batch, self.device)
        inputs_embeds, attention_mask, input_ids = self.preprocess_inputs(encoded_batch)
        output_sequences = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
            **kwargs
        )
        output = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return output
