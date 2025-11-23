# src/models/multimodal_embeddings.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from typing import List, Optional
from tqdm import tqdm
import time
import faiss

from utils.config import (
    PROCESSED_RECIPES,
    EMBEDDINGS_DIR,
    setup_directories
)
from utils.device import get_device

class MultimodalModelFactory:
    """Factory for creating different multimodal embedding models"""
    
    SUPPORTED_MODELS = {
        "clip-vit-base-32": {
            "type": "clip",
            "model_name": "openai/clip-vit-base-patch32",
            "dim": 512
        },
        "clip-vit-large-14": {
            "type": "clip",
            "model_name": "openai/clip-vit-large-patch14",
            "dim": 768
        },
        "siglip-base": {
            "type": "siglip",
            "model_name": "google/siglip-base-patch16-224",
            "dim": 768
        },
        "blip-base": {
            "type": "blip",
            "model_name": "Salesforce/blip-itm-base-coco",
            "dim": 256
        },
    }
    
    @classmethod
    def get_model(cls, model_key: str):
        """Get a multimodal model by key"""
        if model_key not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_key} not supported. Choose from: {list(cls.SUPPORTED_MODELS.keys())}")
        
        config = cls.SUPPORTED_MODELS[model_key]
        model_type = config["type"]
        
        if model_type == "clip":
            return CLIPWrapper(config["model_name"], config["dim"])
        elif model_type == "siglip":
            return SigLIPWrapper(config["model_name"], config["dim"])
        elif model_type == "blip":
            return BLIPWrapper(config["model_name"], config["dim"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class CLIPWrapper:
    """Wrapper for CLIP models"""
    
    def __init__(self, model_name: str, dim: int):
        from transformers import CLIPProcessor, CLIPModel
        
        self.model_name = model_name
        self.dim = dim
        self.device = get_device()
        
        print(f"Loading CLIP: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode text to embeddings"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                embeddings = self.model.get_text_features(**inputs)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to embedding"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()


class SigLIPWrapper:
    """Wrapper for SigLIP models"""
    
    def __init__(self, model_name: str, dim: int):
        from transformers import AutoProcessor, AutoModel
        
        self.model_name = model_name
        self.dim = dim
        self.device = get_device()
        
        print(f"Loading SigLIP: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode text to embeddings"""
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                ).to(self.device)
                
                outputs = self.model.get_text_features(**inputs)
                embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to embedding"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            embeddings = self.model.get_image_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()


class BLIPWrapper:
    """Wrapper for BLIP models"""
    
    def __init__(self, model_name: str, dim: int):
        from transformers import BlipProcessor, BlipForImageTextRetrieval
        
        self.model_name = model_name
        self.dim = dim
        self.device = get_device()
        
        print(f"Loading BLIP: {model_name}")
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def encode_text(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode text to embeddings using BLIP's text encoder"""
        all_embeddings = []
        
        # BLIP needs a dummy image for text encoding in ITM mode
        dummy_image = Image.new('RGB', (224, 224), color='white')
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Process with dummy image
                inputs = self.processor(
                    images=[dummy_image] * len(batch),
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # Get text features
                outputs = self.model.text_encoder(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                )
                
                # Use CLS token as embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image to embedding"""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.vision_model(inputs.pixel_values)
            embeddings = outputs.pooler_output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()