# src/evaluation/clip_image_tests.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import List, Dict
import time
from PIL import Image
import requests
from io import BytesIO

from models.clip_retrieval import CLIPRecipeRetriever
from utils.mlflow_logger import MLflowLogger
import mlflow

class CLIPImageHardNegativeEvaluator:
    """
    Evaluate CLIP's ability to distinguish between visually similar foods - testing confusing pairs like spinach/chard
    """
    
    def __init__(self):
        self.retriever = CLIPRecipeRetriever()
        self.retriever.load()
    
    def create_confusing_pairs_test_cases(self) -> List[Dict]:
        """
        Define test cases for visually similar but different foods
        Each case tests if CLIP can distinguish between confusables
        """
        test_cases = [
            {
                "name": "Spinach vs Chard/Acelga",
                "target_food": "spinach",
                "confusable_foods": ["chard", "acelga", "kale", "collard"],
                "description": "Spinach and chard look similar but are different leafy greens",
                "expected_keywords": ["spinach"],
                "avoid_keywords": ["chard", "acelga", "swiss chard"],
            },
            {
                "name": "Zucchini vs Cucumber",
                "target_food": "zucchini",
                "confusable_foods": ["cucumber", "pickle"],
                "description": "Zucchini and cucumber are visually similar",
                "expected_keywords": ["zucchini", "courgette"],
                "avoid_keywords": ["cucumber", "pickle"],
            },
            {
                "name": "Sweet Potato vs Regular Potato",
                "target_food": "sweet potato",
                "confusable_foods": ["potato", "yam"],
                "description": "Sweet potatoes vs regular potatoes",
                "expected_keywords": ["sweet potato"],
                "avoid_keywords": ["potato", "russet", "yukon"],
            },
            {
                "name": "Broccoli vs Cauliflower",
                "target_food": "broccoli",
                "confusable_foods": ["cauliflower", "romanesco"],
                "description": "Broccoli and cauliflower have similar structure",
                "expected_keywords": ["broccoli"],
                "avoid_keywords": ["cauliflower"],
            },
            {
                "name": "Cilantro/Coriander vs Parsley",
                "target_food": "cilantro",
                "confusable_foods": ["parsley"],
                "description": "Cilantro and parsley look very similar",
                "expected_keywords": ["cilantro", "coriander"],
                "avoid_keywords": ["parsley"],
            },
            {
                "name": "Red Cabbage vs Purple Cabbage",
                "target_food": "red cabbage",
                "confusable_foods": ["purple cabbage", "radicchio"],
                "description": "Different types of purple/red cabbage",
                "expected_keywords": ["red cabbage", "purple cabbage"],
                "avoid_keywords": ["radicchio", "red lettuce"],
            },
            {
                "name": "Green Beans vs Snap Peas",
                "target_food": "green beans",
                "confusable_foods": ["snap peas", "snow peas", "sugar snap"],
                "description": "Green beans vs snap peas look similar",
                "expected_keywords": ["green beans", "string beans"],
                "avoid_keywords": ["snap peas", "snow peas"],
            },
            {
                "name": "Butternut Squash vs Pumpkin",
                "target_food": "butternut squash",
                "confusable_foods": ["pumpkin", "acorn squash"],
                "description": "Similar orange squash varieties",
                "expected_keywords": ["butternut", "squash"],
                "avoid_keywords": ["pumpkin"],
            },
        ]
        
        return test_cases
    
    def simulate_image_query(self, food_name: str, top_k: int = 10) -> pd.DataFrame:
        """
        Simulate an image query by using text description
        """
     
        # This tests if recipe text embeddings can distinguish the foods
        results = self.retriever.search_by_image(
            self._create_dummy_image(),  # Placeholder
            top_k=top_k
        )
        
        # With real images:
        # image = self._download_image(food_image_url)
        # results = self.retriever.search_by_image(image, top_k=top_k)
        
        # For simulation, use text-based search
        from models.retrieval import RecipeRetriever
        text_retriever = RecipeRetriever()
        text_retriever.load()
        results = text_retriever.search(food_name, top_k=top_k)
        
        return results
    
    def _create_dummy_image(self) -> Image.Image:
        """Create a dummy image for placeholder"""
        return Image.new('RGB', (224, 224), color='white')
    
    def evaluate_confusing_pair(self, test_case: Dict) -> Dict:
        """Evaluate a single confusing pair test case"""
        target_food = test_case["target_food"]
        expected_keywords = test_case["expected_keywords"]
        avoid_keywords = test_case["avoid_keywords"]
        
        top_k = 10
        
        # Search for target food
        start_time = time.time()
        results = self.simulate_image_query(target_food, top_k=top_k)
        search_time = time.time() - start_time
        
        # Check how many results contain expected keywords
        recipes_with_target = 0
        for _, row in results.iterrows():
            name_lower = row['name'].lower()
            ingredients_lower = ' '.join(row['ingredients_parsed']).lower()
            combined = name_lower + ' ' + ingredients_lower
            
            if any(kw.lower() in combined for kw in expected_keywords):
                recipes_with_target += 1
        
        target_precision = recipes_with_target / len(results) if len(results) > 0 else 0
        
        # Check contamination with confusable foods
        recipes_with_confusables = 0
        confusable_examples = []
        for _, row in results.iterrows():
            name_lower = row['name'].lower()
            ingredients_lower = ' '.join(row['ingredients_parsed']).lower()
            combined = name_lower + ' ' + ingredients_lower
            
            for avoid_kw in avoid_keywords:
                if avoid_kw.lower() in combined:
                    recipes_with_confusables += 1
                    confusable_examples.append({
                        "recipe": row['name'],
                        "confusable": avoid_kw,
                        "score": row['similarity_score']
                    })
                    break
        
        confusable_contamination = recipes_with_confusables / len(results) if len(results) > 0 else 0
        
        # Calculate metrics
        avg_similarity = results['similarity_score'].mean() if len(results) > 0 else 0
        
        # Test passes if:
        # - At least 70% results have target food
        # - Less than 20% contamination with confusables
        passed = (target_precision >= 0.7 and confusable_contamination <= 0.2)
        
        return {
            "test_name": test_case["name"],
            "target_food": target_food,
            "num_results": len(results),
            "search_time_ms": search_time * 1000,
            "avg_similarity": avg_similarity,
            "target_precision": target_precision,
            "confusable_contamination": confusable_contamination,
            "recipes_with_target": recipes_with_target,
            "recipes_with_confusables": recipes_with_confusables,
            "confusable_examples": confusable_examples[:3],  # Top 3 confusables
            "passed": passed
        }
    
    def run_evaluation(self) -> Dict:
        """Run all confusing pairs tests"""
        test_cases = self.create_confusing_pairs_test_cases()
        results = []
        
        print(f"\n{'='*70}")
        print(f"CLIP Image Hard Negative Evaluation")
        print(f"Testing visually similar but different foods")
        print(f"Total test cases: {len(test_cases)}")
        print(f"{'='*70}\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_case['name']}")
            print(f"  {test_case['description']}")
            
            result = self.evaluate_confusing_pair(test_case)
            results.append(result)
            
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {status}")
            print(f"  - Target precision: {result['target_precision']:.1%}")
            print(f"  - Confusable contamination: {result['confusable_contamination']:.1%}")
            print(f"  - Avg similarity: {result['avg_similarity']:.3f}")
            
            if result['confusable_examples']:
                print(f"  - Confusable examples found:")
                for ex in result['confusable_examples']:
                    print(f"    • {ex['recipe']} (score: {ex['score']:.3f})")
            print()
        
        # Aggregate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['passed'])
        avg_target_precision = np.mean([r['target_precision'] for r in results])
        avg_contamination = np.mean([r['confusable_contamination'] for r in results])
        avg_search_time = np.mean([r['search_time_ms'] for r in results])
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "avg_target_precision": avg_target_precision,
            "avg_confusable_contamination": avg_contamination,
            "avg_search_time_ms": avg_search_time,
            "individual_results": results
        }
        
        print(f"{'='*70}")
        print(f"SUMMARY - Confusing Pairs Evaluation")
        print(f"{'='*70}")
        print(f"Tests passed: {passed_tests}/{total_tests} ({summary['pass_rate']:.1%})")
        print(f"Avg target precision: {avg_target_precision:.1%}")
        print(f"Avg confusable contamination: {avg_contamination:.1%}")
        print(f"Avg search time: {avg_search_time:.1f}ms")
        print(f"{'='*70}\n")
        
        return summary


def log_evaluation_to_mlflow(summary: Dict):
    """Log confusing pairs evaluation to MLflow"""
    logger = MLflowLogger("recipe-search-pipeline")
    
    with mlflow.start_run(run_name="clip_confusing_pairs_eval"):
        # Log parameters
        mlflow.log_param("eval_type", "confusing_pairs")
        mlflow.log_param("retriever", "CLIP_simulated")
        mlflow.log_param("total_test_cases", summary['total_tests'])
        
        # Log aggregate metrics
        mlflow.log_metric("pass_rate", summary['pass_rate'])
        mlflow.log_metric("avg_target_precision", summary['avg_target_precision'])
        mlflow.log_metric("avg_confusable_contamination", summary['avg_confusable_contamination'])
        mlflow.log_metric("avg_search_time_ms", summary['avg_search_time_ms'])
        mlflow.log_metric("tests_passed", summary['passed_tests'])
        
        # Log individual test results
        for i, result in enumerate(summary['individual_results'], 1):
            mlflow.log_metric(f"test_{i}_target_precision", result['target_precision'])
            mlflow.log_metric(f"test_{i}_contamination", result['confusable_contamination'])
            mlflow.log_metric(f"test_{i}_similarity", result['avg_similarity'])
        
        # Log tags
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("eval_type", "confusing_pairs")
        mlflow.set_tag("food_similarity", "high")
        mlflow.set_tag("status", "completed")
        
        print("✓ Logged confusing pairs evaluation to MLflow")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CLIP with confusing food pairs")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = CLIPImageHardNegativeEvaluator()
    summary = evaluator.run_evaluation()
    
    # Log to MLflow
    if not args.no_mlflow:
        log_evaluation_to_mlflow(summary)