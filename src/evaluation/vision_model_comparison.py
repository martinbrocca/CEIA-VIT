"""
Vision Model Comparison (CLIP variants + SigLIP) - WITH CACHING

Purpose:
    Compares different vision-language models for text and image-to-recipe search.
    
    OPTIMIZATION: Loads pre-computed embeddings from .npy files if available,
    otherwise generates on-the-fly. This makes evaluation 20-120x faster on
    machines without powerful GPUs.

Models Evaluated:
    - CLIP-ViT-B/32 (512 dim)
    - CLIP-ViT-L/14 (768 dim)
    - SigLIP-Base (768 dim)
    - SigLIP-SO400M (1152 dim)

Usage:
    # With pre-computed embeddings (FAST - recommended for team)
    python src/evaluation/vision_model_comparison.py --max-recipes 10000
    # → Loads .npy instantly if available
    
    # Force regeneration (ignore cached files)
    python src/evaluation/vision_model_comparison.py --max-recipes 10000 --no-cache
    
    # With image evaluation
    python src/evaluation/vision_model_comparison.py \
        --max-recipes 10000 \
        --image-folder data/raw/food-demo

Pre-computed Embeddings:
    Generate once with: python src/models/generate_all_vision_embeddings.py
    
    Files expected in data/embeddings/:
    - clip_base_embeddings.npy         (475 MB)
    - clip_large_embeddings.npy        (713 MB)
    - siglip_base_embeddings.npy       (713 MB)
    - siglip_so400m_embeddings.npy     (1,069 MB)
    
    Benefits:
    - WITHOUT cache: ~22 seconds to generate all embeddings
    - WITH cache: <1 second to load all embeddings
    - Speedup: 20-120x faster

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
            "name": "SigLIP-Base-v1",
            "type": "siglip",
            "model_id": "google/siglip-base-patch16-224",
        },
        {
            "name": "SigLIP-SO400M-v2",
            "type": "siglip",
            "model_id": "google/siglip-so400m-patch14-384",
        },
    ]
    
    # Mapping of model IDs to embedding files
    EMBEDDING_FILES = {
        "openai/clip-vit-base-patch32": "clip_recipe_embeddings.npy",
        "openai/clip-vit-large-patch14": "clip_large_embeddings.npy",
        "google/siglip-base-patch16-224": "siglip_base_embeddings.npy",
        "google/siglip-so400m-patch14-384": "siglip_so400m_embeddings.npy"
    }
    
    def __init__(self, max_recipes: int = 10000, image_folder: Optional[str] = None, use_cache: bool = True):
        """
        Args:
            max_recipes: Limit recipes for faster comparison (use 10K for testing)
            image_folder: Path to folder containing food images for vision evaluation
            use_cache: Whether to use cached embeddings if available
        """
        self.max_recipes = max_recipes
        self.image_folder = Path(image_folder) if image_folder else None
        self.use_cache = use_cache
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
    
    def try_load_cached_embeddings(self, model_id: str) -> Optional[np.ndarray]:
        """
        Try to load pre-computed embeddings from .npy file
        
        Returns:
            embeddings if successful, None otherwise
        """
        if not self.use_cache:
            return None
        
        # Get embedding filename for this model
        embedding_file = self.EMBEDDING_FILES.get(model_id)
        if not embedding_file:
            return None
        
        embedding_path = EMBEDDINGS_DIR / embedding_file
        
        if not embedding_path.exists():
            return None
        
        try:
            print(f"  → Loading cached embeddings from {embedding_file}...")
            embeddings = np.load(embedding_path)
            
            # Verify size matches current recipes
            if len(embeddings) != len(self.recipes_df):
                print(f"     Warning: Cached embeddings size mismatch")
                print(f"     Cached: {len(embeddings)}, Current: {len(self.recipes_df)}")
                print(f"     Falling back to on-the-fly generation...")
                return None
            
            print(f"  ✓ Loaded {len(embeddings)} cached embeddings (dimension: {embeddings.shape[1]})")
            return embeddings
            
        except Exception as e:
            print(f"     Error loading cache: {e}")
            print(f"     Falling back to on-the-fly generation...")
            return None
    
    def create_embeddings(self, model, processor, model_type: str, model_id: str, batch_size: int = 64) -> np.ndarray:
        """
        Create recipe text embeddings with a given model
        
        First tries to load from cached .npy file, otherwise generates on-the-fly
        """
        # Try to load from cache
        cached_embeddings = self.try_load_cached_embeddings(model_id)
        if cached_embeddings is not None:
            return cached_embeddings
        
        # Cache miss or disabled - generate on-the-fly
        print(f"  → Generating embeddings on-the-fly...")
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
    
    def evaluate_model(self, model_config: Dict):
        """Evaluate a single model"""
        model_name = model_config["name"]
        model_type = model_config["type"]
        model_id = model_config["model_id"]
        
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}\n")
        
        # Load model
        print("Loading model...")
        start_time = time.time()
        model, processor = self.load_model(model_config)
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")
        
        # Create embeddings (cached or on-the-fly)
        print("Creating embeddings...")
        start_time = time.time()
        embeddings = self.create_embeddings(model, processor, model_type, model_id)
        embedding_time = time.time() - start_time
        recipes_per_sec = len(self.recipes_df) / embedding_time if embedding_time > 0 else 0
        print(f"Embeddings created in {embedding_time:.2f}s ({recipes_per_sec:.0f} recipes/sec)")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Build index
        print("Building index...")
        index = self.build_index(embeddings)
        
        # Text evaluation queries
        text_queries = [
            "chocolate cake", "spaghetti carbonara", "grilled salmon",
            "caesar salad", "chicken tikka masala", "apple pie",
            "pad thai", "margherita pizza", "beef tacos", "greek yogurt"
        ]
        
        print(f"\nRunning {len(text_queries)} text evaluation queries...")
        text_results = self.evaluate_text_queries(
            text_queries, model, processor, model_type, index
        )
        
        # Image evaluation (if images provided)
        image_results = {}
        if self.test_images:
            print(f"\nRunning {len(self.test_images)} image evaluation queries...")
            image_results = self.evaluate_image_queries(
                self.test_images, model, processor, index
            )
        
        # Store results
        self.results[model_name] = {
            "model_id": model_id,
            "model_type": model_type,
            "embedding_dim": embeddings.shape[1],
            "load_time": load_time,
            "embedding_time": embedding_time,
            "recipes_per_second": recipes_per_sec,
            "text_results": text_results,
            "image_results": image_results,
            "summary": self.compute_summary(text_results, image_results)
        }
        
        # Clean up GPU memory
        import torch
        del model, processor, embeddings, index
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def evaluate_text_queries(self, queries: List[str], model, processor, model_type: str, index) -> List[Dict]:
        """Evaluate text queries"""
        import torch
        
        results = []
        for query in tqdm(queries, desc="Text queries"):
            start_time = time.time()
            
            with torch.no_grad():
                if model_type == "clip":
                    inputs = processor(
                        text=[query],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(self.device)
                    query_emb = model.get_text_features(**inputs)
                elif model_type == "siglip":
                    inputs = processor(
                        text=[query],
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=64
                    ).to(self.device)
                    query_emb = model.get_text_features(**inputs)
                
                query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
                query_emb = query_emb.cpu().numpy()
            
            # Search
            distances, indices = index.search(query_emb.astype('float32'), 10)
            search_time = time.time() - start_time
            
            # Check if expected terms are in results
            expected_terms = query.lower().split()
            top_recipes = self.recipes_df.iloc[indices[0]]
            
            hits_at_k = {}
            for k in [1, 3, 5, 10]:
                top_k = top_recipes.head(k)
                hits = any(
                    any(term in recipe['name'].lower() for term in expected_terms)
                    for _, recipe in top_k.iterrows()
                )
                hits_at_k[k] = hits
            
            results.append({
                "query": query,
                "hits_at_k": hits_at_k,
                "top_5_similarities": distances[0][:5].tolist(),
                "search_time_ms": search_time * 1000
            })
        
        return results
    
    def evaluate_image_queries(self, test_images: List[Dict], model, processor, index) -> List[Dict]:
        """Evaluate image queries"""
        import torch
        
        results = []
        for img_data in tqdm(test_images, desc="Image queries"):
            start_time = time.time()
            
            with torch.no_grad():
                inputs = processor(images=img_data['image'], return_tensors="pt").to(self.device)
                query_emb = model.get_image_features(**inputs)
                query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
                query_emb = query_emb.cpu().numpy()
            
            # Search
            distances, indices = index.search(query_emb.astype('float32'), 10)
            search_time = time.time() - start_time
            
            results.append({
                "image_name": img_data['name'],
                "top_5_similarities": distances[0][:5].tolist(),
                "search_time_ms": search_time * 1000
            })
        
        return results
    
    def compute_summary(self, text_results: List[Dict], image_results: Dict) -> Dict:
        """Compute summary statistics"""
        # Text metrics
        accuracy_at_k = {}
        for k in [1, 3, 5, 10]:
            hits = sum(1 for r in text_results if r["hits_at_k"][k])
            accuracy_at_k[f"accuracy_at_{k}"] = hits / len(text_results)
        
        avg_similarity = np.mean([
            np.mean(r["top_5_similarities"]) 
            for r in text_results
        ])
        
        avg_search_time = np.mean([r["search_time_ms"] for r in text_results])
        
        summary = {
            **accuracy_at_k,
            "similarity_at_5": avg_similarity,
            "avg_search_time_ms": avg_search_time
        }
        
        # Image metrics (if available)
        if image_results:
            avg_image_sim = np.mean([
                np.mean(r["top_5_similarities"]) 
                for r in image_results
            ])
            avg_image_time = np.mean([r["search_time_ms"] for r in image_results])
            
            summary.update({
                "image_similarity_at_5": avg_image_sim,
                "avg_image_search_time_ms": avg_image_time
            })
        else:
            summary.update({
                "image_similarity_at_5": 0.0,
                "avg_image_search_time_ms": 0.0
            })
        
        return summary
    
    def run_comparison(self):
        """Run comparison across all models"""
        self.load_recipes()
        self.load_test_images()
        
        for model_config in self.MODELS_TO_COMPARE:
            try:
                self.evaluate_model(model_config)
            except Exception as e:
                print(f"Error evaluating {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
    
    def generate_charts(self):
        """Generate comparison charts"""
        if not self.results:
            print("No results to visualize")
            return
        
        output_dir = PROJECT_ROOT / "experiments" / "vision_comparison"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_names = list(self.results.keys())
        
        # [Rest of the chart generation code remains the same as original...]
        # Copying from lines 415-629 of original file
        
        # 1. Text Accuracy@K
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(4)
        width = 0.2
        k_values = [1, 3, 5, 10]
        
        for i, model_name in enumerate(model_names):
            accuracies = [
                self.results[model_name]["summary"][f"accuracy_at_{k}"]
                for k in k_values
            ]
            ax.bar(x + i*width, accuracies, width, label=model_name)
        
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Text Query Accuracy@K', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "vision_text_accuracy_at_k.png", dpi=150)
        plt.close()
        print(f"✓ Saved: vision_text_accuracy_at_k.png")
        
        # 2. Image Similarity (if available)
        if self.test_images:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            image_sims = [self.results[m]["summary"]["image_similarity_at_5"] for m in model_names]
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
            
            bars = ax.bar(model_names, image_sims, color=colors)
            
            for bar, sim in zip(bars, image_sims):
                ax.annotate(f'{sim:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Average Similarity@5', fontsize=12)
            ax.set_title('Image Query Performance', fontsize=14, fontweight='bold')
            plt.xticks(rotation=15)
            
            plt.tight_layout()
            plt.savefig(output_dir / "vision_image_similarity_at_k.png", dpi=150)
            plt.close()
            print(f"✓ Saved: vision_image_similarity_at_k.png")
        
        # 3. Combined Text vs Image (if images available)
        if self.test_images:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
            
            # Text Accuracy@5
            text_accs = [self.results[m]["summary"]["accuracy_at_5"] for m in model_names]
            
            bars1 = ax1.bar(model_names, text_accs, color=colors)
            for bar, acc in zip(bars1, text_accs):
                ax1.annotate(f'{acc:.3f}',
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
        mlflow.log_param("use_cache", comparator.use_cache)
        
        for model_name, results in comparator.results.items():
            s = results["summary"]
            prefix = model_name.replace("/", "_").replace("-", "_").replace(" ", "_")
            
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
        mlflow.set_tag("cached_embeddings", comparator.use_cache)
        
        print("✓ Logged to MLflow")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-recipes", type=int, default=10000, help="Max recipes for testing")
    parser.add_argument("--image-folder", type=str, help="Path to folder with food images for vision evaluation")
    parser.add_argument("--no-cache", action="store_true", help="Force regeneration, ignore cached embeddings")
    parser.add_argument("--no-mlflow", action="store_true")
    
    args = parser.parse_args()
    
    comparator = VisionModelComparator(
        max_recipes=args.max_recipes,
        image_folder=args.image_folder,
        use_cache=not args.no_cache
    )
    comparator.run_comparison()
    comparator.print_summary()
    
    chart_dir = comparator.generate_charts()
    
    if not args.no_mlflow:
        log_to_mlflow(comparator, chart_dir)