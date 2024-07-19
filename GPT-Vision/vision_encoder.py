import torch.nn as nn 
from transformers import  ViTModel
from torchvision import transforms
import torch 
        
class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images,device):
        processed_images = torch.stack([self.image_transform(image) for image in images]).to(device)
        with torch.no_grad():
            pixel_values = self.vision_model(processed_images)
            image_features = pixel_values.last_hidden_state
            image_features = image_features
        return image_features