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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    @property
    def device(self):
        return next(self.language_model.parameters()).device

    def preprocess_inputs(self, batch):
        img_embs = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        img_embs = img_embs.to(self.device)
        
        tok_embs = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)
        img_attention = torch.ones((img_embs.size(0), img_embs.size(1)), dtype=torch.long, device=self.device)
        attention_mask = torch.cat((attention_mask[:, 0:1], img_attention, attention_mask[:, 1:]), dim=1)
        return inputs_embeds, attention_mask, input_ids

    def generate(self, question, image, max_new_tokens=30, **kwargs):
        # Process the image
        # Convert the image to a tensor and add a batch dimension
        with torch.no_grad():
            img_features = self.vision_encoder(image,device=self.device)
        img_embs = self.mlp(img_features)
        
        # Tokenize the question
        prompt = f"{IMAGE_TOKEN}Question: {question}\nAnswer:"
        encoded_input = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,max_length=720)
        
        batch = {
            "pixel_values": img_embs,
            "input_ids": encoded_input.input_ids.to(self.device),
            "attention_mask": encoded_input.attention_mask.to(self.device)
        }
        
        inputs_embeds, attention_mask, input_ids = self.preprocess_inputs(batch)

        print("inputs_embeds",inputs_embeds.size())
        print("attention_mask",attention_mask.size())
        
        output_sequences = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        output = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return output
