import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import time
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO

from models.retrieval import RecipeRetriever
from utils.config import PROJECT_ROOT, PROCESSED_RECIPES, EMBEDDINGS_DIR
from utils.mlflow_logger import MLflowLogger
from utils.device import get_device
import mlflow
import faiss

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class VisionModelComparator:
    """Compare different vision-language models for recipe retrieval"""
    
    MODELS_TO_COMPARE = [
        {
            "name": "CLIP-ViT-B/32",
            "type": "clip",
            "model_id": "openai/clip-vit-base-patch32",
        },
        {
            "name": "CLIP-ViT-L/14",
            "type": "clip", 
            "model_id": "openai/clip-vit-large-patch14",
        },
        {
            "name": "SigLIP-Base",
            "type": "siglip",
            "model_id": "google/siglip-base-patch16-224",
        },
    ]
    
    def __init__(self, max_recipes: int = 10000, image_folder: Path = None):
        """
        Args:
            max_recipes: Limit recipes for faster comparison (use 10K for testing)
            image_folder: Path to folder with test food images
        """
        self.max_recipes = max_recipes
        self.device = get_device()
        self.recipes_df = None
        self.results = {}
        self.image_folder = image_folder
        
    def load_recipes(self):
        """Load recipe data"""
        print(f"Loading recipes from {PROCESSED_RECIPES}...")
        self.recipes_df = pd.read_parquet(PROCESSED_RECIPES)
        
        if self.max_recipes:
            self.recipes_df = self.recipes_df.head(self.max_recipes)
        
        print(f"Loaded {len(self.recipes_df)} recipes")
    
    def load_test_images(self) -> List[Dict]:
        """Load test images from folder"""
        if not self.image_folder or not self.image_folder.exists():
            print("⚠️  No image folder provided, skipping vision evaluation")
            return []
        
        image_files = list(self.image_folder.glob("*.jpg")) + \
                     list(self.image_folder.glob("*.png")) + \
                     list(self.image_folder.glob("*.jpeg"))
        
        test_images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                test_images.append({
                    "path": img_path,
                    "name": img_path.stem,
                    "image": img
                })
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Loaded {len(test_images)} test images")
        return test_images
    
    def load_model(self, model_config: Dict):
        """Load a specific model"""
        model_type = model_config["type"]
        model_id = model_config["model_id"]
        
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
        
        model = model.to(self.device)
        model.eval()
        
        return model, processor
    
    def create_embeddings(self, model, processor, model_type: str, batch_size: int = 64) -> np.ndarray:
        """Create recipe text embeddings with a given model"""
        recipe_texts = self.recipes_df['recipe_text'].tolist()
        all_embeddings = []
        
        import torch
        with torch.no_grad():
            for i in tqdm(range(0, len(recipe_texts), batch_size), desc="Creating embeddings"):
                batch = recipe_texts[i:i + batch_size]
                
                if model_type == "clip":
                    inputs = processor(
                        text=batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(self.device)
                    embeddings = model.get_text_features(**inputs)
                elif model_type == "siglip":
                    inputs = processor(
                        text=batch,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=64
                    ).to(self.device)
                    embeddings = model.get_text_features(**inputs)
                
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Build FAISS index"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index
    
    def search_text(self, query: str, model, processor, model_type: str, index: faiss.IndexFlatIP, top_k: int = 10) -> pd.DataFrame:
        """Search recipes by text query"""
        import torch
        
        with torch.no_grad():
            if model_type == "clip":
                inputs = processor(
                    text=[query],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                query_embedding = model.get_text_features(**inputs)
            elif model_type == "siglip":
                inputs = processor(
                    text=[query],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                ).to(self.device)
                query_embedding = model.get_text_features(**inputs)
            
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy()
        
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        results = self.recipes_df.iloc[indices[0]].copy()
        results['similarity_score'] = distances[0]
        
        return results
    
    def search_image(self, image: Image.Image, model, processor, model_type: str, index: faiss.IndexFlatIP, top_k: int = 10) -> pd.DataFrame:
        """Search recipes by image query - THIS IS THE NEW METHOD!"""
        import torch
        
        with torch.no_grad():
            if model_type == "clip":
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                query_embedding = model.get_image_features(**inputs)
            elif model_type == "siglip":
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                query_embedding = model.get_vision_features(**inputs)
            
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy()
        
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        results = self.recipes_df.iloc[indices[0]].copy()
        results['similarity_score'] = distances[0]
        
        return results
    
    def create_evaluation_queries(self) -> List[Dict]:
        """Create test queries"""
        return [
            {"query": "chocolate cake", "expected": ["chocolate", "cake"]},
            {"query": "pasta carbonara", "expected": ["pasta", "egg", "bacon"]},
            {"query": "chicken soup", "expected": ["chicken", "broth"]},
            {"query": "vegetable stir fry", "expected": ["vegetable", "soy"]},
            {"query": "banana bread", "expected": ["banana", "flour"]},
            {"query": "grilled salmon", "expected": ["salmon"]},
            {"query": "caesar salad", "expected": ["lettuce", "parmesan"]},
            {"query": "mushroom risotto", "expected": ["mushroom", "rice"]},
            {"query": "apple pie", "expected": ["apple", "cinnamon"]},
            {"query": "beef tacos", "expected": ["beef", "tortilla"]},
        ]
    
    def evaluate_model(self, model_config: Dict, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate a single model"""
        model_name = model_config["name"]
        model_type = model_config["type"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        print("Loading model...")
        start_load = time.time()
        model, processor = self.load_model(model_config)
        load_time = time.time() - start_load
        print(f"Model loaded in {load_time:.2f}s")
        
        # Create embeddings
        print("Creating text embeddings...")
        start_embed = time.time()
        embeddings = self.create_embeddings(model, processor, model_type)
        embed_time = time.time() - start_embed
        print(f"Embeddings created in {embed_time:.2f}s ({len(self.recipes_df)/embed_time:.0f} recipes/sec)")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Build index
        print("Building index...")
        index = self.build_index(embeddings)
        
        # Initialize results
        results = {
            "model_name": model_name,
            "model_type": model_type,
            "embedding_dim": embeddings.shape[1],
            "load_time_s": load_time,
            "embed_time_s": embed_time,
            "recipes_per_second": len(self.recipes_df) / embed_time,
            "text_accuracy_at_k": {k: [] for k in k_values},
            "text_similarity_at_k": {k: [] for k in k_values},
            "text_search_times": [],
            "vision_accuracy_at_k": {k: [] for k in k_values},
            "vision_similarity_at_k": {k: [] for k in k_values},
            "vision_search_times": [],
        }
        
        # Evaluate TEXT queries
        queries = self.create_evaluation_queries()
        print(f"\nRunning {len(queries)} text evaluation queries...")
        
        for query_data in tqdm(queries, desc="Text queries"):
            query = query_data["query"]
            expected = query_data["expected"]
            
            start_search = time.time()
            search_results = self.search_text(query, model, processor, model_type, index, top_k=max(k_values))
            search_time = (time.time() - start_search) * 1000
            
            results["text_search_times"].append(search_time)
            
            for k in k_values:
                top_k_results = search_results.head(k)
                
                # Check if expected ingredients found
                hits = 0
                for _, row in top_k_results.iterrows():
                    ingredients_str = ' '.join(row['ingredients_parsed']).lower()
                    name_str = row['name'].lower()
                    combined = ingredients_str + ' ' + name_str
                    
                    if any(exp.lower() in combined for exp in expected):
                        hits += 1
                
                accuracy = hits / k if k > 0 else 0
                avg_sim = top_k_results['similarity_score'].mean()
                
                results["text_accuracy_at_k"][k].append(accuracy)
                results["text_similarity_at_k"][k].append(avg_sim)
        
        # Evaluate VISION queries (NEW!)
        test_images = self.load_test_images()
        if test_images:
            print(f"\nRunning {len(test_images)} vision evaluation queries...")
            
            for img_data in tqdm(test_images, desc="Vision queries"):
                image = img_data["image"]
                img_name = img_data["name"]
                
                start_search = time.time()
                search_results = self.search_image(image, model, processor, model_type, index, top_k=max(k_values))
                search_time = (time.time() - start_search) * 1000
                
                results["vision_search_times"].append(search_time)
                
                # For vision, we'll just track similarity scores
                # (you'd need labeled data to compute accuracy)
                for k in k_values:
                    top_k_results = search_results.head(k)
                    avg_sim = top_k_results['similarity_score'].mean()
                    results["vision_similarity_at_k"][k].append(avg_sim)
                    
                    # Placeholder accuracy (manual inspection needed)
                    results["vision_accuracy_at_k"][k].append(0.0)
                
                # Print top result for manual inspection
                if k_values:
                    top_recipe = search_results.iloc[0]
                    print(f"\n  Image: {img_name}")
                    print(f"  Top match: {top_recipe['name']} (score: {top_recipe['similarity_score']:.3f})")
        
        # Aggregate results
        results["summary"] = {
            "avg_text_search_time_ms": np.mean(results["text_search_times"]),
        }
        
        for k in k_values:
            results["summary"][f"text_accuracy_at_{k}"] = np.mean(results["text_accuracy_at_k"][k])
            results["summary"][f"text_similarity_at_{k}"] = np.mean(results["text_similarity_at_k"][k])
        
        if test_images:
            results["summary"]["avg_vision_search_time_ms"] = np.mean(results["vision_search_times"])
            for k in k_values:
                results["summary"][f"vision_similarity_at_{k}"] = np.mean(results["vision_similarity_at_k"][k])
        
        # Cleanup GPU memory
        del model, processor, embeddings, index
        import torch
        torch.cuda.empty_cache()
        
        return results
    
    def run_comparison(self):
        """Run full comparison"""
        self.load_recipes()
        
        for model_config in self.MODELS_TO_COMPARE:
            try:
                self.results[model_config["name"]] = self.evaluate_model(model_config)
            except Exception as e:
                print(f"Error evaluating {model_config['name']}: {e}")
                continue
        
        return self.results
    
    def generate_charts(self, output_dir: Path = None):
        """Generate comparison charts"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "experiments" / "vision_comparison"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_names = list(self.results.keys())
        k_values = [1, 3, 5, 10]
        
        # Check if we have vision results
        has_vision = any("vision_similarity_at_k" in r and len(r["vision_similarity_at_k"][5]) > 0 
                        for r in self.results.values())
        
        # 1. Text vs Vision Similarity Comparison (NEW!)
        if has_vision:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            x = np.arange(len(k_values))
            width = 0.25
            
            # Text similarities
            for i, model_name in enumerate(model_names):
                sims = [self.results[model_name]["summary"][f"text_similarity_at_{k}"] for k in k_values]
                offset = width * (i - len(model_names)/2 + 0.5)
                ax1.bar(x + offset, sims, width, label=model_name)
            
            ax1.set_xlabel('K (Top-K Results)', fontsize=12)
            ax1.set_ylabel('Average Similarity', fontsize=12)
            ax1.set_title('Text Query Performance', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'@{k}' for k in k_values])
            ax1.legend()
            ax1.set_ylim(0, 1.0)
            
            # Vision similarities
            for i, model_name in enumerate(model_names):
                if f"vision_similarity_at_5" in self.results[model_name]["summary"]:
                    sims = [self.results[model_name]["summary"][f"vision_similarity_at_{k}"] for k in k_values]
                    offset = width * (i - len(model_names)/2 + 0.5)
                    ax2.bar(x + offset, sims, width, label=model_name)
            
            ax2.set_xlabel('K (Top-K Results)', fontsize=12)
            ax2.set_ylabel('Average Similarity', fontsize=12)
            ax2.set_title('Vision Query Performance (Image→Recipe)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'@{k}' for k in k_values])
            ax2.legend()
            ax2.set_ylim(0, 1.0)
            
            plt.tight_layout()
            plt.savefig(output_dir / "text_vs_vision_performance.png", dpi=150)
            plt.close()
            print(f"✓ Saved: text_vs_vision_performance.png")
        
        # 2. Original accuracy chart (text only)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(k_values))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            accuracies = [self.results[model_name]["summary"][f"text_accuracy_at_{k}"] for k in k_values]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, accuracies, width, label=model_name)
            
            for bar, acc in zip(bars, accuracies):
                ax.annotate(f'{acc:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('K (Top-K Results)', fontsize=12)
        ax.set_ylabel('Text Accuracy@K', fontsize=12)
        ax.set_title('Vision-Language Models: Text Query Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "text_accuracy_at_k.png", dpi=150)
        plt.close()
        print(f"✓ Saved: text_accuracy_at_k.png")
        
        # 3. Embedding Speed Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        speeds = [self.results[m]["recipes_per_second"] for m in model_names]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        
        bars = ax.bar(model_names, speeds, color=colors)
        
        for bar, speed in zip(bars, speeds):
            ax.annotate(f'{speed:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Recipes per Second', fontsize=12)
        ax.set_title('Embedding Speed Comparison', fontsize=14, fontweight='bold')
        plt.xticks(rotation=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / "embedding_speed.png", dpi=150)
        plt.close()
        print(f"✓ Saved: embedding_speed.png")
        
        print(f"\n✓ All charts saved to: {output_dir}")
        return output_dir
    
    def print_summary(self):
        """Print comparison summary"""
        print(f"\n{'='*80}")
        print("VISION MODEL COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        for model_name, results in self.results.items():
            s = results["summary"]
            print(f"{model_name}:")
            print(f"  Dimension: {results['embedding_dim']}")
            print(f"  Text Accuracy@5: {s['text_accuracy_at_5']:.3f}")
            if "vision_similarity_at_5" in s:
                print(f"  Vision Similarity@5: {s['vision_similarity_at_5']:.3f}")
            print(f"  Speed: {results['recipes_per_second']:.0f} recipes/sec")
            print(f"  Text search: {s['avg_text_search_time_ms']:.1f}ms")
            if "avg_vision_search_time_ms" in s:
                print(f"  Vision search: {s['avg_vision_search_time_ms']:.1f}ms")
            print()


def log_to_mlflow(comparator: VisionModelComparator, chart_dir: Path):
    """Log to MLflow"""
    logger = MLflowLogger("recipe-search-pipeline")
    
    with mlflow.start_run(run_name="vision_model_comparison"):
        mlflow.log_param("num_models", len(comparator.results))
        mlflow.log_param("num_recipes", comparator.max_recipes)
        
        for model_name, results in comparator.results.items():
            s = results["summary"]
            prefix = model_name.replace("/", "_").replace("-", "_")
            
            mlflow.log_metric(f"{prefix}_text_accuracy_at_5", s["text_accuracy_at_5"])
            if "vision_similarity_at_5" in s:
                mlflow.log_metric(f"{prefix}_vision_similarity_at_5", s["vision_similarity_at_5"])
            mlflow.log_metric(f"{prefix}_embedding_dim", results["embedding_dim"])
            mlflow.log_metric(f"{prefix}_recipes_per_sec", results["recipes_per_second"])
            mlflow.log_metric(f"{prefix}_text_search_time_ms", s["avg_text_search_time_ms"])
        
        for chart_file in chart_dir.glob("*.png"):
            mlflow.log_artifact(str(chart_file), "vision_charts")
        
        mlflow.set_tag("eval_type", "vision_model_comparison")
        
        print("✓ Logged to MLflow")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-recipes", type=int, default=10000, help="Max recipes for testing")
    parser.add_argument("--image-folder", type=str, help="Path to folder with test food images")
    parser.add_argument("--no-mlflow", action="store_true")
    
    args = parser.parse_args()
    
    image_folder = Path(args.image_folder) if args.image_folder else None
    
    comparator = VisionModelComparator(max_recipes=args.max_recipes, image_folder=image_folder)
    comparator.run_comparison()
    comparator.print_summary()
    
    chart_dir = comparator.generate_charts()
    
    if not args.no_mlflow:
        log_to_mlflow(comparator, chart_dir)