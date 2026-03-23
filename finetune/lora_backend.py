import torch
import sys
import os
from peft import PeftModel

sys.path.append(os.getcwd())
from zero_shot.openclip_backend import OpenCLIPBackend

class LoRAOpenCLIPBackend(OpenCLIPBackend):
    def __init__(self, model_name, pretrained, checkpoint_path, device):
        super().__init__(model_name, pretrained, device)
        print(f"Loading LoRA Adapter from: {checkpoint_path}")
        
        self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, image_paths, batch_size):
        # Temporarily expose the inner OpenCLIP model
        original_wrapper = self.model
        self.model = self.model.base_model.model 
        
        try:
            return super().encode_images(image_paths, batch_size)
        finally:
            self.model = original_wrapper

    @torch.no_grad()
    def encode_texts(self, texts, batch_size):
        original_wrapper = self.model
        self.model = self.model.base_model.model 
        
        try:
            return super().encode_texts(texts, batch_size)
        finally:
            self.model = original_wrapper