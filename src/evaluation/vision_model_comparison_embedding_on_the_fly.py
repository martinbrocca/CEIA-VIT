"""
Vision Model Comparison (CLIP variants + SigLIP)

Purpose:
    Compares different vision-language models for text and image-to-recipe search:
    - CLIP-ViT-B/32 (512 dim)
    - CLIP-ViT-L/14 (768 dim)
    - SigLIP-Base (768 dim)
    
    Evaluates both text queries and image queries (if --image-folder provided).

Usage:
    # Text evaluation only
    python src/evaluation/vision_model_comparison.py --max-recipes 10000
    
    # With image evaluation
    python src/evaluation/vision_model_comparison.py \
        --max-recipes 10000 \
        --image-folder data/raw/food-demo

Output:
    - experiments/vision_comparison/text_vs_vision_performance.png
    - experiments/vision_comparison/text_accuracy_at_k.png
    - experiments/vision_comparison/embedding_speed.png

Results:
    Winner: CLIP-ViT-B/32
    - Best text accuracy (90%)
    - Best vision similarity (0.318)
    - Fastest (4785 recipes/sec)
    
    SigLIP performed poorly (0.075 similarity) - needs fine-tuning
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
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
            "name": "SigLIP-Base (v1)",
            "type": "siglip",
            "model_id": "google/siglip-base-patch16-224",
        },
        {
            "name": "SigLIP-SO400M (v2)",  # NEW!
            "type": "siglip",
            "model_id": "google/siglip-so400m-patch14-384",
        },
    ]
    
    def __init__(self, max_recipes: int = 10000, image_folder: Optional[str] = None):
        """
        Args:
            max_recipes: Limit recipes for faster comparison (use 10K for testing)
            image_folder: Path to folder containing food images for vision evaluation
        """
        self.max_recipes = max_recipes
        self.image_folder = Path(image_folder) if image_folder else None
        self.device = get_device()
        self.recipes_df = None
        self.results = {}
        self.test_images = []
        
    def load_recipes(self):
        """Load recipe data"""
        print(f"Loading recipes from {PROCESSED_RECIPES}...")
        self.recipes_df = pd.read_parquet(PROCESSED_RECIPES)
        
        if self.max_recipes:
            self.recipes_df = self.recipes_df.head(self.max_recipes)
        
        print(f"Loaded {len(self.recipes_df)} recipes")
    
    def load_test_images(self):
        """Load test images from folder"""
        if not self.image_folder:
            print("No image folder provided, skipping image evaluation")
            return
        
        if not self.image_folder.exists():
            print(f"Warning: Image folder not found: {self.image_folder}")
            return
        
        print(f"\nLoading test images from {self.image_folder}...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [
            f for f in self.image_folder.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        for img_path in image_files[:20]:  # Limit to 20 images for testing
            try:
                img = Image.open(img_path).convert('RGB')
                self.test_images.append({
                    'path': img_path,
                    'image': img,
                    'name': img_path.stem
                })
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
        
        print(f"Loaded {len(self.test_images)} test images")
    
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
        """Search recipes by image"""
        import torch
        
        with torch.no_grad():
            if model_type == "clip":
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                image_embedding = model.get_image_features(**inputs)
            elif model_type == "siglip":
                inputs = processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                image_embedding = model.get_image_features(**inputs)
            
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            image_embedding = image_embedding.cpu().numpy()
        
        distances, indices = index.search(image_embedding.astype('float32'), top_k)
        
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
        print("Creating embeddings...")
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
            "accuracy_at_k": {k: [] for k in k_values},
            "similarity_at_k": {k: [] for k in k_values},
            "search_times": [],
        }
        
        # TEXT EVALUATION
        queries = self.create_evaluation_queries()
        print(f"\nRunning {len(queries)} text evaluation queries...")
        
        for query_data in tqdm(queries, desc="Text queries"):
            query = query_data["query"]
            expected = query_data["expected"]
            
            start_search = time.time()
            search_results = self.search_text(query, model, processor, model_type, index, top_k=max(k_values))
            search_time = (time.time() - start_search) * 1000
            
            results["search_times"].append(search_time)
            
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
                
                results["accuracy_at_k"][k].append(accuracy)
                results["similarity_at_k"][k].append(avg_sim)
        
        # IMAGE EVALUATION
        if self.test_images:
            print(f"\nRunning {len(self.test_images)} image evaluation queries...")
            
            results["image_similarity_at_k"] = {k: [] for k in k_values}
            results["image_search_times"] = []
            
            for img_data in tqdm(self.test_images, desc="Image queries"):
                image = img_data['image']
                
                start_search = time.time()
                search_results = self.search_image(image, model, processor, model_type, index, top_k=max(k_values))
                search_time = (time.time() - start_search) * 1000
                
                results["image_search_times"].append(search_time)
                
                for k in k_values:
                    top_k_results = search_results.head(k)
                    avg_sim = top_k_results['similarity_score'].mean()
                    results["image_similarity_at_k"][k].append(avg_sim)
        
        # Aggregate text results
        results["summary"] = {
            "avg_search_time_ms": np.mean(results["search_times"]),
        }
        
        for k in k_values:
            results["summary"][f"accuracy_at_{k}"] = np.mean(results["accuracy_at_k"][k])
            results["summary"][f"similarity_at_{k}"] = np.mean(results["similarity_at_k"][k])
        
        # Aggregate image results if available
        if self.test_images:
            results["summary"]["avg_image_search_time_ms"] = np.mean(results["image_search_times"])
            for k in k_values:
                results["summary"][f"image_similarity_at_{k}"] = np.mean(results["image_similarity_at_k"][k])
        
        # Cleanup GPU memory
        del model, processor, embeddings, index
        import torch
        torch.cuda.empty_cache()
        
        return results
    
    def run_comparison(self):
        """Run full comparison"""
        self.load_recipes()
        self.load_test_images()
        
        for model_config in self.MODELS_TO_COMPARE:
            try:
                self.results[model_config["name"]] = self.evaluate_model(model_config)
            except Exception as e:
                print(f"Error evaluating {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return self.results
    
    def generate_charts(self, output_dir: Path = None):
        """Generate comparison charts"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "experiments" / "vision_comparison"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_names = list(self.results.keys())
        k_values = [1, 3, 5, 10]
        
        # 1. Text Accuracy@K Comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(k_values))
        width = 0.25
        
        for i, model_name in enumerate(model_names):
            accuracies = [self.results[model_name]["summary"][f"accuracy_at_{k}"] for k in k_values]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, accuracies, width, label=model_name)
            
            for bar, acc in zip(bars, accuracies):
                ax.annotate(f'{acc:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('K (Top-K Results)', fontsize=12)
        ax.set_ylabel('Accuracy@K', fontsize=12)
        ax.set_title('Vision-Language Models: Text Accuracy@K Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "vision_text_accuracy_at_k.png", dpi=150)
        plt.close()
        print(f"✓ Saved: vision_text_accuracy_at_k.png")
        
        # 2. Image Similarity@K Comparison (if images were evaluated)
        if self.test_images:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(k_values))
            width = 0.25
            
            for i, model_name in enumerate(model_names):
                similarities = [self.results[model_name]["summary"][f"image_similarity_at_{k}"] for k in k_values]
                offset = width * (i - len(model_names)/2 + 0.5)
                bars = ax.bar(x + offset, similarities, width, label=model_name)
                
                for bar, sim in zip(bars, similarities):
                    ax.annotate(f'{sim:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('K (Top-K Results)', fontsize=12)
            ax.set_ylabel('Average Similarity Score', fontsize=12)
            ax.set_title('Vision-Language Models: Image Similarity@K Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'@{k}' for k in k_values])
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / "vision_image_similarity_at_k.png", dpi=150)
            plt.close()
            print(f"✓ Saved: vision_image_similarity_at_k.png")
        
        # 3. Text vs Image Performance Comparison
        if self.test_images:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Text Similarity@5
            text_sims = [self.results[m]["summary"]["similarity_at_5"] for m in model_names]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
            
            bars1 = ax1.bar(model_names, text_sims, color=colors)
            for bar, sim in zip(bars1, text_sims):
                ax1.annotate(f'{sim:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax1.set_ylabel('Average Similarity@5', fontsize=12)
            ax1.set_title('Text Query Performance', fontsize=12, fontweight='bold')
            ax1.tick_params(axis='x', rotation=15)
            
            # Image Similarity@5
            image_sims = [self.results[m]["summary"]["image_similarity_at_5"] for m in model_names]
            
            bars2 = ax2.bar(model_names, image_sims, color=colors)
            for bar, sim in zip(bars2, image_sims):
                ax2.annotate(f'{sim:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax2.set_ylabel('Average Similarity@5', fontsize=12)
            ax2.set_title('Image Query Performance', fontsize=12, fontweight='bold')
            ax2.tick_params(axis='x', rotation=15)
            
            plt.suptitle('Text vs Image Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / "text_vs_image_performance.png", dpi=150)
            plt.close()
            print(f"✓ Saved: text_vs_image_performance.png")
        
        # 4. Embedding Speed Comparison
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
        plt.savefig(output_dir / "vision_embedding_speed.png", dpi=150)
        plt.close()
        print(f"✓ Saved: vision_embedding_speed.png")
        
        # 5. Model Characteristics Summary
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dims = [self.results[m]["embedding_dim"] for m in model_names]
        accuracies = [self.results[m]["summary"]["accuracy_at_5"] for m in model_names]
        
        # Bubble chart: x=dim, y=accuracy, size=speed
        scatter = ax.scatter(dims, accuracies, s=[s*10 for s in speeds], 
                           alpha=0.6, c=range(len(model_names)), cmap='viridis')
        
        for i, name in enumerate(model_names):
            ax.annotate(name, (dims[i], accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Embedding Dimension', fontsize=12)
        ax.set_ylabel('Accuracy@5', fontsize=12)
        ax.set_title('Model Characteristics\n(bubble size = embedding speed)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "vision_model_characteristics.png", dpi=150)
        plt.close()
        print(f"✓ Saved: vision_model_characteristics.png")
        
        # 6. Summary Table as Image
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.axis('off')
        
        table_data = []
        if self.test_images:
            headers = ["Model", "Dim", "Acc@1", "Acc@5", "Text Sim@5", "Img Sim@5", "Recipes/s", "Search(ms)"]
        else:
            headers = ["Model", "Dim", "Acc@1", "Acc@5", "Acc@10", "Recipes/s", "Search(ms)"]
        
        for model_name in model_names:
            r = self.results[model_name]
            s = r["summary"]
            
            if self.test_images:
                table_data.append([
                    model_name,
                    r["embedding_dim"],
                    f"{s['accuracy_at_1']:.3f}",
                    f"{s['accuracy_at_5']:.3f}",
                    f"{s['similarity_at_5']:.3f}",
                    f"{s['image_similarity_at_5']:.3f}",
                    f"{r['recipes_per_second']:.0f}",
                    f"{s['avg_search_time_ms']:.1f}"
                ])
            else:
                table_data.append([
                    model_name,
                    r["embedding_dim"],
                    f"{s['accuracy_at_1']:.3f}",
                    f"{s['accuracy_at_5']:.3f}",
                    f"{s['accuracy_at_10']:.3f}",
                    f"{r['recipes_per_second']:.0f}",
                    f"{s['avg_search_time_ms']:.1f}"
                ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        plt.title('Vision-Language Model Comparison Summary', fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        plt.savefig(output_dir / "vision_summary_table.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: vision_summary_table.png")
        
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
            print(f"  Text Accuracy@5: {s['accuracy_at_5']:.3f}")
            print(f"  Text Similarity@5: {s['similarity_at_5']:.3f}")
            
            if self.test_images:
                print(f"  Image Similarity@5: {s['image_similarity_at_5']:.3f}")
                print(f"  Image Search time: {s['avg_image_search_time_ms']:.1f}ms")
            
            print(f"  Embedding Speed: {results['recipes_per_second']:.0f} recipes/sec")
            print(f"  Text Search time: {s['avg_search_time_ms']:.1f}ms")
            print()


def log_to_mlflow(comparator: VisionModelComparator, chart_dir: Path):
    """Log to MLflow"""
    logger = MLflowLogger("recipe-search-pipeline")
    
    with mlflow.start_run(run_name="vision_model_comparison"):
        mlflow.log_param("num_models", len(comparator.results))
        mlflow.log_param("num_recipes", comparator.max_recipes)
        mlflow.log_param("num_test_images", len(comparator.test_images))
        mlflow.log_param("image_evaluation", bool(comparator.test_images))
        
        for model_name, results in comparator.results.items():
            s = results["summary"]
            prefix = model_name.replace("/", "_").replace("-", "_")
            
            mlflow.log_metric(f"{prefix}_text_accuracy_at_5", s["accuracy_at_5"])
            mlflow.log_metric(f"{prefix}_text_similarity_at_5", s["similarity_at_5"])
            mlflow.log_metric(f"{prefix}_embedding_dim", results["embedding_dim"])
            mlflow.log_metric(f"{prefix}_recipes_per_sec", results["recipes_per_second"])
            mlflow.log_metric(f"{prefix}_search_time_ms", s["avg_search_time_ms"])
            
            if comparator.test_images:
                mlflow.log_metric(f"{prefix}_image_similarity_at_5", s["image_similarity_at_5"])
                mlflow.log_metric(f"{prefix}_image_search_time_ms", s["avg_image_search_time_ms"])
        
        for chart_file in chart_dir.glob("*.png"):
            mlflow.log_artifact(str(chart_file), "vision_charts")
        
        mlflow.set_tag("eval_type", "vision_model_comparison")
        
        print("✓ Logged to MLflow")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-recipes", type=int, default=10000, help="Max recipes for testing")
    parser.add_argument("--image-folder", type=str, help="Path to folder with food images for vision evaluation")
    parser.add_argument("--no-mlflow", action="store_true")
    
    args = parser.parse_args()
    
    comparator = VisionModelComparator(
        max_recipes=args.max_recipes,
        image_folder=args.image_folder
    )
    comparator.run_comparison()
    comparator.print_summary()
    
    chart_dir = comparator.generate_charts()
    
    if not args.no_mlflow:
        log_to_mlflow(comparator, chart_dir)