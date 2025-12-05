# src/models/clip_embeddings.py
"""
CLIP Text Embedding Generation (Multimodal)

Purpose:
    Creates CLIP text embeddings for recipes to enable image-to-recipe search.
    Unlike sentence-transformers, CLIP embeddings live in the same space as
    image embeddings, allowing visual search capabilities.
    
    Default model: openai/clip-vit-base-patch32 (512-dim)
    
Usage:
    # Default CLIP model
    python src/models/clip_embeddings.py
    
    # Large CLIP model (better quality, slower)
    python src/models/clip_embeddings.py --model openai/clip-vit-large-patch14
    
    # Testing with limited recipes
    python src/models/clip_embeddings.py --max-recipes 10000
    
    # Without MLflow logging
    python src/models/clip_embeddings.py --no-mlflow

Input:
    - data/processed/recipes_processed.parquet
    - Uses 'recipe_text' field

Output:
    - data/embeddings/clip_recipe_embeddings.npy
    - Shares recipe_ids.npy with text embeddings

Model Specifications:
    CLIP-ViT-B/32 (default):
        - Embedding dimension: 512
        - Speed: ~4,785 recipes/sec (RTX 5090)
        - Quality: 90% text accuracy, 31.8% image similarity
        - Size: 29.8 MB for 231K recipes
        - Max text length: 77 tokens
    
    CLIP-ViT-L/14:
        - Embedding dimension: 768
        - Speed: ~1,420 recipes/sec (RTX 5090)
        - Quality: 88% text accuracy, 29.5% image similarity
        - Size: 44.7 MB for 231K recipes
        - Higher quality but slower

Key Differences from embeddings.py:
    - Multimodal: Text embeddings compatible with image embeddings
    - Shorter context: 77 tokens vs 512 tokens
    - Different model family: CLIP vs sentence-transformers
    - Use case: Image search vs pure text search

Performance (RTX 5090):
    - 231,637 recipes in ~48 seconds (base model)
    - 231,637 recipes in ~163 seconds (large model)
    - GPU memory: ~4.2 GB peak (base), ~7.5 GB (large)
    - Batch size 128 recommended

Processing Pipeline:
    1. Load CLIPModel and CLIPProcessor
    2. Tokenize recipe_text (max 77 tokens)
    3. Extract text features with get_text_features()
    4. L2 normalize for cosine similarity
    5. Save as float32 numpy array

MLflow Tracking:
    - embedding_type: "multimodal_text"
    - model_type: "CLIP"
    - Logs same metrics as embeddings.py

Example Output:
    Loading recipes from data/processed/recipes_processed.parquet...
    Processing 231637 recipes
    Loading CLIP model: openai/clip-vit-base-patch32
    Creating CLIP text embeddings...
    100%|████████████████████| 1810/1810 [00:48<00:00, 37.68it/s]
    CLIP embeddings shape: (231637, 512)
    Time taken: 48.05 seconds (4820 recipes/sec)
    Embedding file size: 29.8 MB
    ✓ Done!

When to Use:
    - Use CLIP embeddings when you need image-to-recipe search
    - Use regular embeddings (embeddings.py) for pure text search
    - Can use both simultaneously for hybrid search

"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import torch

from utils.mlflow_logger import MLflowLogger



from utils.config import (
    PROCESSED_RECIPES,
    CLIP_RECIPE_EMBEDDINGS,
    CLIP_MODEL,
    setup_directories
)
from utils.device import get_device

def create_clip_recipe_embeddings(
    model_name: str = CLIP_MODEL,
    batch_size: int = 128,
    max_recipes: int = None,
    log_to_mlflow: bool = True
):
    """
    Create CLIP text embeddings for all recipes
    These will be used to match against image queries
    """
    import time
    import os
    
    start_time = time.time()  # Add this
    
    setup_directories()
    device = get_device()
    
    # Load processed recipes
    print(f"Loading recipes from {PROCESSED_RECIPES}...")
    df = pd.read_parquet(PROCESSED_RECIPES)
    
    if max_recipes:
        df = df.head(max_recipes)
        print(f"Limited to {max_recipes} recipes for testing")
    
    num_recipes = len(df)  # Add this
    print(f"Processing {num_recipes} recipes")
    
     # Load CLIP model
    print(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Get recipe texts
    recipe_texts = df['recipe_text'].tolist()
    
    # Create embeddings in batches
    print("Creating CLIP text embeddings...")
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(recipe_texts), batch_size)):
            batch_texts = recipe_texts[i:i + batch_size]
            
            # Process batch
            inputs = processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            # Get text embeddings
            text_embeddings = model.get_text_features(**inputs)
            
            # Normalize embeddings (for cosine similarity)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(text_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    
    end_time = time.time()  # Add this
    total_time = end_time - start_time  # Add this
    
    print(f"CLIP embeddings shape: {embeddings.shape}")
    print(f"Time taken: {total_time:.2f} seconds ({num_recipes/total_time:.0f} recipes/sec)")  # Add this
    
    # Save embeddings
    print(f"Saving CLIP embeddings to {CLIP_RECIPE_EMBEDDINGS}...")
    np.save(CLIP_RECIPE_EMBEDDINGS, embeddings)
    
    # Calculate embedding size
    embedding_size_mb = os.path.getsize(CLIP_RECIPE_EMBEDDINGS) / (1024 * 1024)  # Add this
    
    # Recipe IDs already saved from previous embedding step
    print("✓ Done!")
    print(f"CLIP embeddings saved to: {CLIP_RECIPE_EMBEDDINGS}")
    print(f"Embedding file size: {embedding_size_mb:.1f} MB")  # Add this
    
    # Log to MLflow (now all variables are defined)
    if log_to_mlflow:
        print("\nLogging to MLflow...")
        logger = MLflowLogger("recipe-search-pipeline")
        logger.log_embedding_creation(model_name, {
            "embedding_type": "multimodal_text",  # Changed from "text"
            "model_type": "CLIP",  # Changed from "sentence-transformers"
            "batch_size": batch_size,
            "device": str(device),
            "num_recipes": num_recipes,
            "embedding_dim": embeddings.shape[1],
            "total_time": total_time,
            "recipes_per_second": num_recipes / total_time,
            "embedding_size_mb": embedding_size_mb,
            "embedding_path": str(CLIP_RECIPE_EMBEDDINGS)  # Changed to CLIP path
        })
        print("✓ Logged to MLflow")
    
    return embeddings, df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CLIP_MODEL, help="CLIP model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--max-recipes", type=int, default=None, help="Limit recipes (for testing)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")  # Add this line
    
    args = parser.parse_args()
    
    create_clip_recipe_embeddings(
        model_name=args.model,
        batch_size=args.batch_size,
        max_recipes=args.max_recipes,
        log_to_mlflow=not args.no_mlflow
    )