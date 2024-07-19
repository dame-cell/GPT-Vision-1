import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoTokenizer
from .configuration_gpt2vision import GPT2VisionConfig ,GPT2Config
from .modeling_gpt2 import GPT2LMHeadModel
from .vision_encoder import VisionEncoder

IMAGE_TOKEN = "<image>"
ANSWER_EOS = "<|endoftext|>"

def resize_token_embeds(model_name="openai-community/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
    return tokenizer

tokenizer = resize_token_embeds()


def create_labels(input_ids, tokenizer, attention_mask):
    labels = input_ids.clone()
    
    labels[attention_mask == 0] = -100

    answer_start_tokens = tokenizer.encode("Answer:", add_special_tokens=False)

    for i, seq in enumerate(input_ids):
        # Find the start of the answer
        answer_start = (seq == answer_start_tokens[0]).nonzero(as_tuple=True)[0]
        if len(answer_start) > 0:
            answer_start = answer_start[0]
            if seq[answer_start:answer_start+len(answer_start_tokens)].tolist() == answer_start_tokens:
                # Mask out everything before the answer
                labels[i, :answer_start] = -100
                
                # Find the end of the sequence (last non-padding token)
                sequence_end = attention_mask[i].nonzero(as_tuple=True)[0][-1]
                
                # Keep the last token (EOS) as part of the label
                labels[i, sequence_end+1:] = -100

    return labels
    
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=0.1)

        # Initialize weights
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

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

    def freeze_model_components(self, freeze_vision=True, freeze_language=True,freeze_mlp=True):
        for param in self.vision_encoder.parameters():
            param.requires_grad = not freeze_vision
        for param in self.language_model.parameters():
            param.requires_grad = not freeze_language
        for param in self.mlp.parameters():
            param.requires_grad = not freeze_mlp

    def tokenize_encode(self, batch, device):
        text = batch['text']
        images = batch['image']

        if isinstance(text, str):
            text = [text]

        input_texts = [f"{IMAGE_TOKEN}{self.tokenizer.bos_token}{t}" for t in text]
        text_inputs = self.tokenizer(
            input_texts,
            padding='max_length',
            truncation=True,
            max_length=768,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ).to(device)

        pixel_values = self.vision_encoder(images,device)

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

        labels = create_labels(input_ids, self.tokenizer, attention_mask)
        labels = labels.to(self.device)
        
        img_embs = self.mlp(pixel_values)
        tok_embs = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

        img_attention = torch.ones((img_embs.size(0), img_embs.size(1)), dtype=torch.long, device=self.device)
        attention_mask = torch.cat((attention_mask[:, 0:1], img_attention, attention_mask[:, 1:]), dim=1)

        img_labels = torch.full((labels.size(0), img_embs.size(1)), fill_value=-100, dtype=torch.long, device=self.device)
        labels = torch.cat((labels[:, 0:1], img_labels, labels[:, 1:]), dim=1)
        return inputs_embeds, attention_mask, input_ids, labels

    def forward(self, batch, **kwargs):
        inputs_embeds, attention_mask, input_ids, labels = self.preprocess_inputs(batch)
        outputs = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        
        
        return outputs
        
    def generate(self, prompt, image, max_new_tokens=30, **kwargs):
        batch = {"image": [image], "text": f"Question: {prompt}\nAnswer:"}
        encoded_batch = self.tokenize_encode(batch, self.device)
        inputs_embeds, attention_mask, input_ids, _ = self.preprocess_inputs(encoded_batch)
 



        output_sequences = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            **kwargs
        )
        output = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        return output