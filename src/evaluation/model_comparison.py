# src/evaluation/model_comparison.py
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

from models.retrieval import RecipeRetriever
from models.clip_retrieval import CLIPRecipeRetriever
from utils.mlflow_logger import MLflowLogger
from utils.config import PROJECT_ROOT, PROCESSED_RECIPES
import mlflow

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class MultiModelEvaluator:
    """
    Compare multiple embedding models for recipe retrieval
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_models(self):
        """Load all models to compare"""
        print("Loading models for comparison...")
        
        # Sentence-Transformers (MiniLM)
        print("\n1. Loading Sentence-Transformers (MiniLM)...")
        st_retriever = RecipeRetriever()
        st_retriever.load()
        self.models["MiniLM-L6"] = st_retriever
        
        # CLIP
        print("\n2. Loading CLIP...")
        clip_retriever = CLIPRecipeRetriever()
        clip_retriever.load()
        self.models["CLIP-ViT-B32"] = clip_retriever
        
        print(f"\n✓ Loaded {len(self.models)} models")
    
    def create_evaluation_queries(self) -> List[Dict]:
        """Create evaluation queries with expected results"""
        queries = [
            {"query": "chocolate cake", "category": "dessert", "expected_ingredients": ["chocolate", "flour", "sugar"]},
            {"query": "pasta carbonara", "category": "main", "expected_ingredients": ["pasta", "egg", "bacon"]},
            {"query": "caesar salad", "category": "salad", "expected_ingredients": ["lettuce", "parmesan", "crouton"]},
            {"query": "chicken soup", "category": "soup", "expected_ingredients": ["chicken", "broth", "vegetable"]},
            {"query": "banana bread", "category": "dessert", "expected_ingredients": ["banana", "flour", "sugar"]},
            {"query": "beef tacos", "category": "main", "expected_ingredients": ["beef", "tortilla", "cheese"]},
            {"query": "vegetable stir fry", "category": "main", "expected_ingredients": ["vegetable", "soy sauce", "oil"]},
            {"query": "apple pie", "category": "dessert", "expected_ingredients": ["apple", "flour", "sugar", "cinnamon"]},
            {"query": "grilled salmon", "category": "main", "expected_ingredients": ["salmon", "lemon", "herb"]},
            {"query": "mushroom risotto", "category": "main", "expected_ingredients": ["rice", "mushroom", "broth"]},
            {"query": "recipe with chicken and rice", "category": "main", "expected_ingredients": ["chicken", "rice"]},
            {"query": "tomato basil pasta", "category": "main", "expected_ingredients": ["tomato", "basil", "pasta"]},
            {"query": "lemon garlic shrimp", "category": "main", "expected_ingredients": ["lemon", "garlic", "shrimp"]},
            {"query": "peanut butter cookies", "category": "dessert", "expected_ingredients": ["peanut butter", "sugar"]},
            {"query": "spinach and feta", "category": "side", "expected_ingredients": ["spinach", "feta"]},
            {"query": "vegan chocolate dessert", "category": "dessert", "dietary": "vegan", "expected_ingredients": ["chocolate"]},
            {"query": "gluten free pancakes", "category": "breakfast", "dietary": "gluten-free", "expected_ingredients": ["egg", "milk"]},
            {"query": "low carb dinner", "category": "main", "dietary": "low-carb", "expected_ingredients": []},
            {"query": "vegetarian curry", "category": "main", "dietary": "vegetarian", "expected_ingredients": ["curry", "vegetable"]},
            {"query": "dairy free ice cream", "category": "dessert", "dietary": "dairy-free", "expected_ingredients": []},
        ]
        return queries
    
    def calculate_ingredient_match_score(self, results: pd.DataFrame, expected_ingredients: List[str]) -> float:
        """Calculate how many results contain expected ingredients"""
        if not expected_ingredients or len(results) == 0:
            return 0.0
        
        matches = 0
        for _, row in results.iterrows():
            ingredients_str = ' '.join(row['ingredients_parsed']).lower()
            if any(exp.lower() in ingredients_str for exp in expected_ingredients):
                matches += 1
        
        return matches / len(results)
    
    def calculate_name_relevance_score(self, results: pd.DataFrame, query: str) -> float:
        """Calculate how relevant the recipe names are to the query"""
        if len(results) == 0:
            return 0.0
        
        query_words = set(query.lower().split())
        scores = []
        
        for _, row in results.iterrows():
            name_words = set(row['name'].lower().split())
            overlap = len(query_words & name_words) / len(query_words)
            scores.append(overlap)
        
        return np.mean(scores)
    
    def evaluate_model(self, model_name: str, retriever, queries: List[Dict], k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate a single model across all queries and K values"""
        results = {
            "model_name": model_name,
            "accuracy_at_k": {k: [] for k in k_values},
            "ingredient_match_at_k": {k: [] for k in k_values},
            "name_relevance_at_k": {k: [] for k in k_values},
            "similarity_scores": {k: [] for k in k_values},
            "search_times": [],
            "per_query_results": []
        }
        
        for query_data in tqdm(queries, desc=f"Evaluating {model_name}"):
            query = query_data["query"]
            expected_ingredients = query_data.get("expected_ingredients", [])
            dietary = query_data.get("dietary", None)
            
            start_time = time.time()
            search_results = retriever.search(
                query, 
                top_k=max(k_values),
                dietary_filters=[dietary] if dietary else None
            )
            search_time = (time.time() - start_time) * 1000
            
            results["search_times"].append(search_time)
            
            query_result = {
                "query": query,
                "category": query_data.get("category", "unknown"),
                "results_count": len(search_results)
            }
            
            for k in k_values:
                top_k_results = search_results.head(k)
                
                ing_score = self.calculate_ingredient_match_score(top_k_results, expected_ingredients)
                results["ingredient_match_at_k"][k].append(ing_score)
                
                name_score = self.calculate_name_relevance_score(top_k_results, query)
                results["name_relevance_at_k"][k].append(name_score)
                
                accuracy = 1 if ing_score > 0.5 else 0
                results["accuracy_at_k"][k].append(accuracy)
                
                avg_sim = top_k_results['similarity_score'].mean() if len(top_k_results) > 0 else 0
                results["similarity_scores"][k].append(avg_sim)
                
                query_result[f"accuracy_at_{k}"] = accuracy
                query_result[f"ingredient_match_at_{k}"] = ing_score
            
            results["per_query_results"].append(query_result)
        
        results["summary"] = {
            "avg_search_time_ms": np.mean(results["search_times"]),
            "std_search_time_ms": np.std(results["search_times"]),
        }
        
        for k in k_values:
            results["summary"][f"accuracy_at_{k}"] = np.mean(results["accuracy_at_k"][k])
            results["summary"][f"ingredient_match_at_{k}"] = np.mean(results["ingredient_match_at_k"][k])
            results["summary"][f"name_relevance_at_{k}"] = np.mean(results["name_relevance_at_k"][k])
            results["summary"][f"avg_similarity_at_{k}"] = np.mean(results["similarity_scores"][k])
        
        return results
    
    def run_comparison(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Run full comparison across all models"""
        queries = self.create_evaluation_queries()
        
        print(f"\n{'='*70}")
        print(f"MULTI-MODEL COMPARISON")
        print(f"Models: {list(self.models.keys())}")
        print(f"Queries: {len(queries)}")
        print(f"K values: {k_values}")
        print(f"{'='*70}\n")
        
        for model_name, retriever in self.models.items():
            print(f"\nEvaluating {model_name}...")
            self.results[model_name] = self.evaluate_model(model_name, retriever, queries, k_values)
        
        return self.results
    
    def print_summary_table(self):
        """Print a summary comparison table"""
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}\n")
        
        metrics = ["Accuracy@1", "Accuracy@5", "Accuracy@10", 
                   "Ing.Match@5", "Name Rel.@5", "Avg Sim@5", "Latency(ms)"]
        
        print(f"{'Metric':<20}", end="")
        for model_name in self.results.keys():
            print(f"{model_name:<20}", end="")
        print("\n" + "-"*80)
        
        for metric in metrics:
            print(f"{metric:<20}", end="")
            for model_name in self.results.keys():
                summary = self.results[model_name]["summary"]
                
                if "Accuracy@1" in metric:
                    val = summary["accuracy_at_1"]
                elif "Accuracy@5" in metric:
                    val = summary["accuracy_at_5"]
                elif "Accuracy@10" in metric:
                    val = summary["accuracy_at_10"]
                elif "Ing.Match" in metric:
                    val = summary["ingredient_match_at_5"]
                elif "Name Rel" in metric:
                    val = summary["name_relevance_at_5"]
                elif "Avg Sim" in metric:
                    val = summary["avg_similarity_at_5"]
                elif "Latency" in metric:
                    val = summary["avg_search_time_ms"]
                    print(f"{val:<20.1f}", end="")
                    continue
                
                print(f"{val:<20.3f}", end="")
            print()
        
        print(f"\n{'='*80}")
    
    def generate_comparison_charts(self, output_dir: str = None):
        """Generate comparison visualizations"""
        if output_dir is None:
            output_dir = PROJECT_ROOT / "experiments" / "comparison_charts"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_names = list(self.results.keys())
        k_values = [1, 3, 5, 10]
        
        # 1. Accuracy@K Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(k_values))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            accuracies = [self.results[model_name]["summary"][f"accuracy_at_{k}"] for k in k_values]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, accuracies, width, label=model_name)
            
            for bar, acc in zip(bars, accuracies):
                ax.annotate(f'{acc:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('K (Top-K Results)', fontsize=12)
        ax.set_ylabel('Accuracy@K', fontsize=12)
        ax.set_title('Accuracy@K Comparison: MiniLM vs CLIP', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_at_k_comparison.png", dpi=150)
        plt.close()
        print(f"✓ Saved: accuracy_at_k_comparison.png")
        
        # 2. Ingredient Match Score
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, model_name in enumerate(model_names):
            scores = [self.results[model_name]["summary"][f"ingredient_match_at_{k}"] for k in k_values]
            offset = width * (i - len(model_names)/2 + 0.5)
            bars = ax.bar(x + offset, scores, width, label=model_name)
            
            for bar, score in zip(bars, scores):
                ax.annotate(f'{score:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('K (Top-K Results)', fontsize=12)
        ax.set_ylabel('Ingredient Match Score', fontsize=12)
        ax.set_title('Ingredient Match@K: MiniLM vs CLIP', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'@{k}' for k in k_values])
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(output_dir / "ingredient_match_comparison.png", dpi=150)
        plt.close()
        print(f"✓ Saved: ingredient_match_comparison.png")
        
        # 3. Similarity Score Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for i, model_name in enumerate(model_names):
            all_similarities = []
            for k in k_values:
                all_similarities.extend(self.results[model_name]["similarity_scores"][k])
            
            axes[i].hist(all_similarities, bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_xlabel('Similarity Score', fontsize=12)
            axes[i].set_ylabel('Frequency', fontsize=12)
            axes[i].set_title(f'{model_name}\nSimilarity Score Distribution', fontsize=12, fontweight='bold')
            axes[i].axvline(np.mean(all_similarities), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_similarities):.3f}')
            axes[i].legend()
        plt.tight_layout()
        plt.savefig(output_dir / "similarity_distribution.png", dpi=150)
        plt.close()
        print(f"✓ Saved: similarity_distribution.png")
        
        # 4. Search Time Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        times_data = [self.results[m]["search_times"] for m in model_names]
        bp = ax.boxplot(times_data, labels=model_names, patch_artist=True)
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Search Time (ms)', fontsize=12)
        ax.set_title('Search Latency Comparison', fontsize=14, fontweight='bold')
        for i, model_name in enumerate(model_names):
            mean_time = np.mean(self.results[model_name]["search_times"])
            ax.annotate(f'μ={mean_time:.1f}ms', xy=(i+1, mean_time), 
                       xytext=(10, 0), textcoords="offset points", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / "search_time_comparison.png", dpi=150)
        plt.close()
        print(f"✓ Saved: search_time_comparison.png")
        
        # 5. Radar Chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
        categories = ['Accuracy@5', 'Ingredient\nMatch@5', 'Name\nRelevance@5', 
                     'Similarity\nScore', 'Speed\n(normalized)']
        
        for model_name in model_names:
            summary = self.results[model_name]["summary"]
            max_time = max(r["summary"]["avg_search_time_ms"] for r in self.results.values())
            speed_norm = 1 - (summary["avg_search_time_ms"] / max_time)
            
            values = [
                summary["accuracy_at_5"],
                summary["ingredient_match_at_5"],
                summary["name_relevance_at_5"],
                summary["avg_similarity_at_5"],
                speed_norm
            ]
            values += values[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories), endpoint=False))
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_title('Multi-Dimensional Model Comparison', fontsize=14, fontweight='bold', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(output_dir / "radar_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: radar_comparison.png")
        
        # 6. Summary Table Image
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        
        headers = ["Model", "Acc@1", "Acc@5", "Acc@10", "Ing.Match@5", "Similarity", "Latency(ms)"]
        table_data = []
        for model_name in model_names:
            s = self.results[model_name]["summary"]
            table_data.append([
                model_name,
                f"{s['accuracy_at_1']:.3f}",
                f"{s['accuracy_at_5']:.3f}",
                f"{s['accuracy_at_10']:.3f}",
                f"{s['ingredient_match_at_5']:.3f}",
                f"{s['avg_similarity_at_5']:.3f}",
                f"{s['avg_search_time_ms']:.1f}"
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        plt.title('Model Comparison Summary', fontsize=14, fontweight='bold', y=0.85)
        plt.tight_layout()
        plt.savefig(output_dir / "summary_table.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: summary_table.png")
        
        print(f"\n✓ All charts saved to: {output_dir}")
        return output_dir


def log_comparison_to_mlflow(evaluator: MultiModelEvaluator, chart_dir: Path):
    """Log comparison results to MLflow"""
    logger = MLflowLogger("recipe-search-pipeline")
    
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.log_param("num_models", len(evaluator.models))
        mlflow.log_param("models_compared", list(evaluator.models.keys()))
        mlflow.log_param("num_queries", len(evaluator.create_evaluation_queries()))
        
        for model_name, results in evaluator.results.items():
            summary = results["summary"]
            prefix = model_name.replace("-", "_")
            
            mlflow.log_metric(f"{prefix}_accuracy_at_1", summary["accuracy_at_1"])
            mlflow.log_metric(f"{prefix}_accuracy_at_5", summary["accuracy_at_5"])
            mlflow.log_metric(f"{prefix}_accuracy_at_10", summary["accuracy_at_10"])
            mlflow.log_metric(f"{prefix}_ingredient_match_at_5", summary["ingredient_match_at_5"])
            mlflow.log_metric(f"{prefix}_avg_similarity_at_5", summary["avg_similarity_at_5"])
            mlflow.log_metric(f"{prefix}_avg_search_time_ms", summary["avg_search_time_ms"])
        
        for chart_file in chart_dir.glob("*.png"):
            mlflow.log_artifact(str(chart_file), "comparison_charts")
        
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("eval_type", "model_comparison")
        mlflow.set_tag("status", "completed")
        
        print("✓ Logged comparison results to MLflow")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding models for recipe retrieval")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for charts")
    
    args = parser.parse_args()
    
    evaluator = MultiModelEvaluator()
    evaluator.load_models()
    evaluator.run_comparison()
    evaluator.print_summary_table()
    chart_dir = evaluator.generate_comparison_charts(args.output_dir)
    
    if not args.no_mlflow:
        log_comparison_to_mlflow(evaluator, chart_dir)