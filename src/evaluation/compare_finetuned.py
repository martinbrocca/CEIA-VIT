# src/evaluation/compare_finetuned.py
"""
Baseline vs Fine-tuned Model Comparison

Purpose:
    Evaluates baseline SigLIP and fine-tuned SigLIP on Food-101 ground truth test set.
    Measures image-to-recipe search accuracy using Food-101 images as queries.
    
    Generates comparison charts showing:
    - Accuracy@K improvement (K=1,3,5,10)
    - Per-category performance
    - Fine-tuning impact delta

Methodology:
    1. Load test images from Food-101 (10 per category, 15 categories)
    2. Search recipe database using image embeddings
    3. Check if correct dish appears in top-K results
    4. Compare baseline vs fine-tuned performance

Usage:
    # Default comparison
    python src/evaluation/compare_finetuned.py
    
    # Custom models
    python src/evaluation/compare_finetuned.py \
        --baseline google/siglip-base-patch16-224 \
        --finetuned models/siglip-food-finetuned
    
    # More test images per category
    python src/evaluation/compare_finetuned.py --images-per-category 20

Output:
    - experiments/finetuning_comparison/accuracy_comparison.png
    - experiments/finetuning_comparison/per_category_comparison.png
    - experiments/finetuning_comparison/improvement_delta.png
    - experiments/finetuning_comparison/comparison_results.json
    - MLflow run with all metrics and charts

Expected Results:
    - Accuracy@5: Baseline ~12% → Fine-tuned ~25-35%
    - 2-3x improvement in top-5 retrieval accuracy
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import random
import time
import json

from transformers import AutoModel, AutoProcessor
from utils.config import PROJECT_ROOT, PROCESSED_RECIPES
from utils.device import get_device
from utils.mlflow_logger import MLflowLogger
import mlflow
import faiss

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


class ModelComparator:
    """Compare baseline vs fine-tuned SigLIP on Food-101 ground truth"""
    
    FOOD101_PATH = PROJECT_ROOT / "data" / "raw" / "food-101" / "images"
    
    # Category mappings (subset for evaluation)
    EVAL_CATEGORIES = {
        "apple_pie": ["apple pie", "apple tart"],
        "caesar_salad": ["caesar salad"],
        "cheesecake": ["cheesecake"],
        "chocolate_cake": ["chocolate cake"],
        "grilled_salmon": ["grilled salmon", "salmon"],
        "hamburger": ["hamburger", "burger"],
        "ice_cream": ["ice cream"],
        "pizza": ["pizza"],
        "sushi": ["sushi"],
        "tacos": ["taco", "tacos"],
        "pancakes": ["pancake", "pancakes"],
        "spaghetti_carbonara": ["spaghetti carbonara", "carbonara"],
        "chicken_wings": ["chicken wings", "buffalo wings"],
        "donuts": ["donut", "doughnut"],
        "french_fries": ["french fries", "fries"],
    }
    
    def __init__(self, max_recipes: int = 50000, images_per_category: int = 10):
        self.max_recipes = max_recipes
        self.images_per_category = images_per_category
        self.device = get_device()
        self.recipes_df = None
        
    def load_recipes(self):
        """Load recipe database"""
        print(f"Loading recipes from {PROCESSED_RECIPES}...")
        self.recipes_df = pd.read_parquet(PROCESSED_RECIPES)
        
        if self.max_recipes:
            self.recipes_df = self.recipes_df.head(self.max_recipes)
        
        print(f"Loaded {len(self.recipes_df)} recipes")
    
    def get_test_images(self):
        """Get test images from Food-101"""
        test_images = []
        
        for category, search_terms in self.EVAL_CATEGORIES.items():
            category_path = self.FOOD101_PATH / category
            
            if not category_path.exists():
                print(f"⚠️  Category not found: {category}")
                continue
            
            # Get random images
            image_files = list(category_path.glob("*.jpg"))
            random.shuffle(image_files)
            sampled = image_files[:self.images_per_category]
            
            for img_path in sampled:
                try:
                    img = Image.open(img_path).convert("RGB")
                    test_images.append({
                        "image": img,
                        "path": img_path,
                        "category": category,
                        "expected_terms": search_terms
                    })
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        print(f"Loaded {len(test_images)} test images from {len(self.EVAL_CATEGORIES)} categories")
        return test_images
    
    def load_model(self, model_path: str):
        """Load model and processor"""
        print(f"\nLoading model: {model_path}")
        
        model = AutoModel.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Model loaded")
        return model, processor
    
    def create_embeddings(self, model, processor, batch_size: int = 64):
        """Create recipe embeddings"""
        recipe_texts = self.recipes_df['recipe_text'].tolist()
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(recipe_texts), batch_size), desc="Creating embeddings"):
                batch = recipe_texts[i:i + batch_size]
                
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
    
    def build_index(self, embeddings):
        """Build FAISS index"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        return index
    
    def search_image(self, image, model, processor, index, top_k=10):
        """Search recipes by image"""
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            query_embedding = model.get_image_features(**inputs)
            query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
            query_embedding = query_embedding.cpu().numpy()
        
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        results = self.recipes_df.iloc[indices[0]].copy()
        results['similarity_score'] = distances[0]
        
        return results
    
    def check_match(self, results, expected_terms, k):
        """Check if top-K results contain expected terms"""
        top_k = results.head(k)
        
        for _, row in top_k.iterrows():
            name_lower = row['name'].lower()
            if any(term.lower() in name_lower for term in expected_terms):
                return True
        
        return False
    
    def evaluate_model(self, model_name: str, model_path: str, test_images, k_values=[1, 3, 5, 10]):
        """Evaluate a model"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        model, processor = self.load_model(model_path)
        
        # Create embeddings
        print("Creating recipe embeddings...")
        start = time.time()
        embeddings = self.create_embeddings(model, processor)
        embed_time = time.time() - start
        print(f"Embeddings created in {embed_time:.2f}s ({len(self.recipes_df)/embed_time:.0f} recipes/sec)")
        
        # Build index
        index = self.build_index(embeddings)
        
        # Evaluate
        results = {
            "model_name": model_name,
            "accuracy_at_k": {k: [] for k in k_values},
            "similarity_at_k": {k: [] for k in k_values},
            "search_times": [],
            "per_category": {}
        }
        
        print(f"\nEvaluating {len(test_images)} images...")
        
        for test_data in tqdm(test_images):
            image = test_data["image"]
            category = test_data["category"]
            expected = test_data["expected_terms"]
            
            # Search
            start = time.time()
            search_results = self.search_image(image, model, processor, index, top_k=max(k_values))
            search_time = (time.time() - start) * 1000
            results["search_times"].append(search_time)
            
            # Check accuracy at different K
            for k in k_values:
                match = self.check_match(search_results, expected, k)
                results["accuracy_at_k"][k].append(1 if match else 0)
                
                avg_sim = search_results.head(k)['similarity_score'].mean()
                results["similarity_at_k"][k].append(avg_sim)
            
            # Per-category tracking
            if category not in results["per_category"]:
                results["per_category"][category] = []
            
            match_at_5 = self.check_match(search_results, expected, 5)
            results["per_category"][category].append(1 if match_at_5 else 0)
        
        # Aggregate
        results["summary"] = {
            "avg_search_time_ms": np.mean(results["search_times"]),
        }
        
        for k in k_values:
            results["summary"][f"accuracy_at_{k}"] = np.mean(results["accuracy_at_k"][k])
            results["summary"][f"similarity_at_{k}"] = np.mean(results["similarity_at_k"][k])
        
        results["summary"]["per_category_accuracy"] = {
            cat: np.mean(accs) for cat, accs in results["per_category"].items()
        }
        
        # Cleanup
        del model, processor, embeddings, index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return results
    
    def compare_models(self, baseline_path: str, finetuned_path: str):
        """Compare baseline vs fine-tuned"""
        self.load_recipes()
        
        # Get test images (same set for both models)
        random.seed(42)
        test_images = self.get_test_images()
        
        # Evaluate both models
        baseline_results = self.evaluate_model("Baseline SigLIP", baseline_path, test_images)
        finetuned_results = self.evaluate_model("Fine-tuned SigLIP", finetuned_path, test_images)
        
        return {
            "baseline": baseline_results,
            "finetuned": finetuned_results
        }
    
    def print_comparison(self, comparison):
        """Print comparison summary"""
        baseline = comparison["baseline"]["summary"]
        finetuned = comparison["finetuned"]["summary"]
        
        print(f"\n{'='*80}")
        print("BASELINE vs FINE-TUNED COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Metric':<30} {'Baseline':<15} {'Fine-tuned':<15} {'Δ':<15}")
        print("-" * 80)
        
        for k in [1, 3, 5, 10]:
            metric = f"Accuracy@{k}"
            base_val = baseline[f"accuracy_at_{k}"]
            ft_val = finetuned[f"accuracy_at_{k}"]
            delta = ft_val - base_val
            
            print(f"{metric:<30} {base_val:>6.1%}{'':<9} {ft_val:>6.1%}{'':<9} {delta:>+6.1%}{'':<9}")
        
        print()
        
        for k in [5]:
            metric = f"Similarity@{k}"
            base_val = baseline[f"similarity_at_{k}"]
            ft_val = finetuned[f"similarity_at_{k}"]
            delta = ft_val - base_val
            
            print(f"{metric:<30} {base_val:>6.3f}{'':<9} {ft_val:>6.3f}{'':<9} {delta:>+6.3f}{'':<9}")
        
        print()
        print(f"{'Search time (ms)':<30} {baseline['avg_search_time_ms']:>6.1f}{'':<9} {finetuned['avg_search_time_ms']:>6.1f}{'':<9}")
        print()
    
    def generate_comparison_charts(self, comparison, output_dir: Path = None):
        """Generate comparison charts"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "experiments" / "finetuning_comparison"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        baseline = comparison["baseline"]
        finetuned = comparison["finetuned"]
        
        k_values = [1, 3, 5, 10]
        
        # 1. Accuracy comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(k_values))
        width = 0.35
        
        baseline_accs = [baseline["summary"][f"accuracy_at_{k}"] * 100 for k in k_values]
        finetuned_accs = [finetuned["summary"][f"accuracy_at_{k}"] * 100 for k in k_values]
        
        bars1 = ax.bar(x - width/2, baseline_accs, width, label='Baseline', color='skyblue')
        bars2 = ax.bar(x + width/2, finetuned_accs, width, label='Fine-tuned', color='coral')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('K (Top-K Results)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Baseline vs Fine-tuned: Image→Recipe Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.legend()
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
        plt.close()
        print(f"✓ Saved: accuracy_comparison.png")
        
        # 2. Per-category improvement
        fig, ax = plt.subplots(figsize=(12, 8))
        
        categories = list(baseline["per_category"].keys())
        baseline_cat = [np.mean(baseline["per_category"][cat]) * 100 for cat in categories]
        finetuned_cat = [np.mean(finetuned["per_category"][cat]) * 100 for cat in categories]
        improvements = [ft - base for base, ft in zip(baseline_cat, finetuned_cat)]
        
        # Sort by improvement
        sorted_data = sorted(zip(categories, baseline_cat, finetuned_cat, improvements), 
                           key=lambda x: x[3], reverse=True)
        categories, baseline_cat, finetuned_cat, improvements = zip(*sorted_data)
        
        categories = [c.replace('_', ' ').title() for c in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.barh(x - width/2, baseline_cat, width, label='Baseline', color='skyblue')
        ax.barh(x + width/2, finetuned_cat, width, label='Fine-tuned', color='coral')
        
        ax.set_yticks(x)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Accuracy@5 (%)', fontsize=12)
        ax.set_title('Per-Category Performance: Baseline vs Fine-tuned', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 100)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "per_category_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: per_category_comparison.png")
        
        # 3. Improvement delta
        fig, ax = plt.subplots(figsize=(10, 6))
        
        improvements_k = [(finetuned["summary"][f"accuracy_at_{k}"] - 
                          baseline["summary"][f"accuracy_at_{k}"]) * 100 
                         for k in k_values]
        
        colors = ['green' if x > 0 else 'red' for x in improvements_k]
        bars = ax.bar(range(len(k_values)), improvements_k, color=colors, alpha=0.7)
        
        for i, (bar, val) in enumerate(zip(bars, improvements_k)):
            ax.annotate(f'{val:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3 if val > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if val > 0 else 'top',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('K (Top-K Results)', fontsize=12)
        ax.set_ylabel('Improvement (percentage points)', fontsize=12)
        ax.set_title('Fine-tuning Impact: Accuracy Improvement', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(k_values)))
        ax.set_xticklabels([f'Top-{k}' for k in k_values])
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "improvement_delta.png", dpi=150)
        plt.close()
        print(f"✓ Saved: improvement_delta.png")
        
        print(f"\n✓ All charts saved to: {output_dir}")
        return output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default="google/siglip-base-patch16-224")
    parser.add_argument("--finetuned", type=str, default="models/siglip-food-finetuned")
    parser.add_argument("--max-recipes", type=int, default=50000)
    parser.add_argument("--images-per-category", type=int, default=10)
    parser.add_argument("--no-mlflow", action="store_true")
    
    args = parser.parse_args()
    
    comparator = ModelComparator(
        max_recipes=args.max_recipes,
        images_per_category=args.images_per_category
    )
    
    comparison = comparator.compare_models(args.baseline, args.finetuned)
    comparator.print_comparison(comparison)
    chart_dir = comparator.generate_comparison_charts(comparison)
    
    # Save results
    results_path = PROJECT_ROOT / "experiments" / "finetuning_comparison" / "comparison_results.json"
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable format
        def convert_to_native(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json_comparison = {
            "baseline": {
                "summary": convert_to_native(comparison["baseline"]["summary"]),
                "model_name": comparison["baseline"]["model_name"]
            },
            "finetuned": {
                "summary": convert_to_native(comparison["finetuned"]["summary"]),
                "model_name": comparison["finetuned"]["model_name"]
            }
        }
        json.dump(json_comparison, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    if not args.no_mlflow:
        logger = MLflowLogger("recipe-search-pipeline")
        with mlflow.start_run(run_name="finetuning_comparison"):
            # Log metrics
            for model_type in ["baseline", "finetuned"]:
                prefix = model_type
                summary = comparison[model_type]["summary"]
                
                for k in [1, 3, 5, 10]:
                    mlflow.log_metric(f"{prefix}_accuracy_at_{k}", summary[f"accuracy_at_{k}"])
                    mlflow.log_metric(f"{prefix}_similarity_at_{k}", summary[f"similarity_at_{k}"])
            
            # Log improvement
            for k in [1, 3, 5, 10]:
                improvement = (comparison["finetuned"]["summary"][f"accuracy_at_{k}"] - 
                             comparison["baseline"]["summary"][f"accuracy_at_{k}"])
                mlflow.log_metric(f"improvement_accuracy_at_{k}", improvement)
            
            # Log charts
            for chart_file in chart_dir.glob("*.png"):
                mlflow.log_artifact(str(chart_file), "comparison_charts")
            
            # Log results JSON
            mlflow.log_artifact(str(results_path), "results")
            
            mlflow.set_tag("eval_type", "finetuning_comparison")
            
            print("✓ Logged to MLflow")


if __name__ == "__main__":
    main()