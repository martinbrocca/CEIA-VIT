"""
Generate All Vision Model Embeddings

Purpose:
    Pre-generates embeddings for all vision-language models to accelerate
    evaluation on machines without powerful GPUs.
    
    Run this ONCE on a machine with good GPU (e.g., RTX 5090).
    Share the .npy files with team members for fast evaluation.

Models Generated:
    - CLIP-ViT-B/32 (512 dim) → clip_base_embeddings.npy
    - CLIP-ViT-L/14 (768 dim) → clip_large_embeddings.npy
    - SigLIP-Base (768 dim) → siglip_base_embeddings.npy
    - SigLIP-SO400M (1152 dim) → siglip_so400m_embeddings.npy

Usage:
    # Generate for all 231K recipes (full dataset) - RECOMMENDED
    python src/models/generate_all_vision_embeddings.py
    
    # Generate for subset (faster, for testing only)
    python src/models/generate_all_vision_embeddings.py --max-recipes 10000
    
    # Skip specific models
    python src/models/generate_all_vision_embeddings.py --skip clip_large,siglip_so400m

Output:
    data/embeddings/
    ├── clip_base_embeddings.npy         (~475 MB)
    ├── clip_large_embeddings.npy        (~713 MB)
    ├── siglip_base_embeddings.npy       (~713 MB)
    └── siglip_so400m_embeddings.npy     (~1,069 MB)
    
    Total: ~2.97 GB for full dataset

Time Estimates (RTX 5090, 231K recipes):
    - CLIP-ViT-B/32: ~45 seconds
    - CLIP-ViT-L/14: ~90 seconds
    - SigLIP-Base: ~97 seconds
    - SigLIP-SO400M: ~290 seconds
    Total: ~8 minutes

Benefits:
    - Team members without GPU can evaluate instantly
    - vision_model_comparison.py becomes 20-120x faster
    - Consistent embeddings across team
    - Share via Git LFS or cloud storage



Author: Martin Brocca
Created: 2025-11-30
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from typing import Dict, List
from tqdm import tqdm
import time

from utils.config import PROCESSED_RECIPES, EMBEDDINGS_DIR, PROJECT_ROOT
from utils.device import get_device
from utils.mlflow_logger import MLflowLogger
import mlflow


# Model configurations
MODELS_CONFIG = {
    "clip_base": {
        "name": "CLIP-ViT-B/32",
        "model_id": "openai/clip-vit-base-patch32",
        "type": "clip",
        "dim": 512,
        "output_file": "clip_base_embeddings.npy",
        "description": "Best overall - fast and accurate"
    },
    "clip_large": {
        "name": "CLIP-ViT-L/14",
        "model_id": "openai/clip-vit-large-patch14",
        "type": "clip",
        "dim": 768,
        "output_file": "clip_large_embeddings.npy",
        "description": "Larger model - slower but similar quality"
    },
    "siglip_base": {
        "name": "SigLIP-Base",
        "model_id": "google/siglip-base-patch16-224",
        "type": "siglip",
        "dim": 768,
        "output_file": "siglip_base_embeddings.npy",
        "description": "Good for text, needs fine-tuning for images"
    },
    "siglip_so400m": {
        "name": "SigLIP-SO400M (v2)",
        "model_id": "google/siglip-so400m-patch14-384",
        "type": "siglip",
        "dim": 1152,
        "output_file": "siglip_so400m_embeddings.npy",
        "description": "Largest model - 10x bigger but not better for food"
    }
}


def load_model(config: Dict, device):
    """Load a vision-language model"""
    model_type = config["type"]
    model_id = config["model_id"]
    
    print(f"\nLoading {config['name']} ({model_id})...")
    
    if model_type == "clip":
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
    elif model_type == "siglip":
        from transformers import AutoProcessor, AutoModel
        model = AutoModel.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, processor


def generate_embeddings(config: Dict, recipes_df: pd.DataFrame, device, batch_size: int = 64) -> np.ndarray:
    """Generate embeddings for all recipes with a given model"""
    
    print(f"\n{'='*70}")
    print(f"Generating embeddings: {config['name']}")
    print(f"{'='*70}")
    print(f"Model: {config['model_id']}")
    print(f"Recipes: {len(recipes_df):,}")
    print(f"Dimension: {config['dim']}")
    print(f"Output: {config['output_file']}")
    
    # Load model
    start_time = time.time()
    model, processor = load_model(config, device)
    load_time = time.time() - start_time
    print(f"Model load time: {load_time:.2f}s")
    
    # Get recipe texts
    recipe_texts = recipes_df['recipe_text'].tolist()
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    start_time = time.time()
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(recipe_texts), batch_size), desc="Batches"):
            batch = recipe_texts[i:i + batch_size]
            
            if config["type"] == "clip":
                inputs = processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                embeddings = model.get_text_features(**inputs)
            elif config["type"] == "siglip":
                inputs = processor(
                    text=batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                ).to(device)
                embeddings = model.get_text_features(**inputs)
            
            # L2 normalize
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    generation_time = time.time() - start_time
    
    # Calculate throughput
    recipes_per_sec = len(recipes_df) / generation_time
    
    print(f"  Embeddings generated:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Time: {generation_time:.2f}s")
    print(f"  Throughput: {recipes_per_sec:.0f} recipes/sec")
    
    # Save to file
    output_path = EMBEDDINGS_DIR / config["output_file"]
    np.save(output_path, embeddings)
    
    # Calculate file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Clean up GPU memory
    del model, processor, inputs, embeddings
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        "model_name": config["name"],
        "model_id": config["model_id"],
        "num_recipes": len(recipes_df),
        "embedding_dim": config["dim"],
        "generation_time": generation_time,
        "recipes_per_second": recipes_per_sec,
        "file_size_mb": file_size_mb,
        "output_file": str(output_path)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for all vision models")
    parser.add_argument("--max-recipes", type=int, default=None, 
                       help="Limit number of recipes (default: all 231K)")
    parser.add_argument("--skip", type=str, default="",
                       help="Comma-separated models to skip (e.g., 'clip_large,siglip_so400m')")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing files without asking")
    parser.add_argument("--no-mlflow", action="store_true",
                       help="Skip MLflow logging")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Ensure embeddings directory exists
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load recipes
    print(f"\nLoading recipes from {PROCESSED_RECIPES}...")
    recipes_df = pd.read_parquet(PROCESSED_RECIPES)
    
    if args.max_recipes:
        recipes_df = recipes_df.head(args.max_recipes)
        print(f"Limited to {len(recipes_df):,} recipes (for testing)")
    else:
        print(f"Loaded {len(recipes_df):,} recipes (full dataset)")
    
    # Parse skip list
    skip_models = set(args.skip.split(",")) if args.skip else set()
    
    # Filter models to process
    models_to_process = {
        k: v for k, v in MODELS_CONFIG.items() 
        if k not in skip_models
    }
    
    if not models_to_process:
        print("No models to process. Exiting.")
        return
    
    print(f"\nModels to generate: {len(models_to_process)}")
    for key, config in models_to_process.items():
        status = "SKIP (exists)" if (EMBEDDINGS_DIR / config["output_file"]).exists() else "GENERATE"
        print(f"  - {config['name']}: {status}")
    
    # Check for existing files
    if not args.force:
        existing = [
            config["output_file"] 
            for config in models_to_process.values() 
            if (EMBEDDINGS_DIR / config["output_file"]).exists()
        ]
        if existing:
            print(f"\n  Warning: {len(existing)} file(s) already exist:")
            for filename in existing:
                print(f"  - {filename}")
            
            response = input("\nOverwrite existing files? [y/N]: ").strip().lower()
            if response != 'y':
                print("Aborted. Use --force to skip this prompt.")
                return
    
    # Start MLflow run
    if not args.no_mlflow:
        logger = MLflowLogger("recipe-search-pipeline")
        mlflow.start_run(run_name="generate_all_vision_embeddings")
        mlflow.log_param("num_models", len(models_to_process))
        mlflow.log_param("num_recipes", len(recipes_df))
        mlflow.log_param("device", str(device))
    
    # Generate embeddings for each model
    results = {}
    total_start = time.time()
    
    for model_key, config in models_to_process.items():
        try:
            result = generate_embeddings(config, recipes_df, device)
            results[model_key] = result
            
            # Log to MLflow
            if not args.no_mlflow:
                prefix = model_key
                mlflow.log_metric(f"{prefix}_recipes_per_sec", result["recipes_per_second"])
                mlflow.log_metric(f"{prefix}_generation_time", result["generation_time"])
                mlflow.log_metric(f"{prefix}_file_size_mb", result["file_size_mb"])
                mlflow.log_metric(f"{prefix}_embedding_dim", result["embedding_dim"])
        
        except Exception as e:
            print(f" Error generating {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start
    
    # Print summary
    print(f"\n{'='*70}")
    print("GENERATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"Models processed: {len(results)}/{len(models_to_process)}")
    print(f"Recipes: {len(recipes_df):,}")
    print(f"\nResults:")
    
    total_size = 0
    for model_key, result in results.items():
        print(f"\n{result['model_name']}:")
        print(f"  File: {MODELS_CONFIG[model_key]['output_file']}")
        print(f"  Size: {result['file_size_mb']:.1f} MB")
        print(f"  Dimension: {result['embedding_dim']}")
        print(f"  Speed: {result['recipes_per_second']:.0f} recipes/sec")
        print(f"  Time: {result['generation_time']:.1f}s")
        total_size += result['file_size_mb']
    
    print(f"\nTotal disk space: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print(f"Location: {EMBEDDINGS_DIR}")
    
    
    
    # Log summary to MLflow
    if not args.no_mlflow:
        mlflow.log_metric("total_time_seconds", total_time)
        mlflow.log_metric("total_size_mb", total_size)
        mlflow.log_param("models_generated", list(results.keys()))
        mlflow.set_tag("purpose", "pre_compute_vision_embeddings")
        mlflow.end_run()
        print("\n✓ Logged to MLflow")


if __name__ == "__main__":
    main()