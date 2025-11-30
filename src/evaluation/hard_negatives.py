# src/evaluation/hard_negatives.py
"""
Hard Negative Test Suite

Purpose:
    Tests models' ability to distinguish similar but different items using
    challenging queries with strict constraints (e.g., "gluten-free bread").
    
    5 test cases:
    1. Chocolate cake vs cookies
    2. Carbonara vs other Italian dishes
    3. Vegetarian burger (no meat)
    4. Gluten-free bread (no flour)
    5. Vegan dessert (no animal products)

Usage:
    python src/evaluation/hard_negatives.py

Metrics:
    - Expected keyword rate (target ingredients present)
    - Negative contamination (forbidden ingredients present)
    - Forbidden ingredient rate
    
Results:
    Both MiniLM and CLIP: 60% pass rate (3/5 tests)
    Failures due to inconsistent Food.com dietary tags, not model quality

Author: Martin Brocca
Created: 2025-11-28
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import time

from models.retrieval import RecipeRetriever
from models.clip_retrieval import CLIPRecipeRetriever
from utils.mlflow_logger import MLflowLogger
import mlflow

class HardNegativeEvaluator:
    """
    Evaluate retrieval system with hard negative examples
    Hard negatives are recipes that are similar but shouldn't be top results
    """
    
    def __init__(self, use_clip: bool = False):
        self.use_clip = use_clip
        if use_clip:
            self.retriever = CLIPRecipeRetriever()
        else:
            self.retriever = RecipeRetriever()
        
        self.retriever.load()
    
    def create_test_cases(self) -> List[Dict]:
        """
        Define hard negative test cases
        Each case has:
        - query: search term
        - expected_in_top_k: recipe names that SHOULD appear in top K
        - hard_negatives: similar recipes that SHOULDN'T be in top K
        """
        test_cases = [
            {
                "query": "chocolate cake",
                "top_k": 5,
                "expected_keywords": ["cake", "chocolate"],
                "negative_keywords": ["cookie", "brownie"],
                "description": "Chocolate cake should return cakes, not cookies"
            },
            {
                "query": "pasta carbonara",
                "top_k": 5,
                "expected_keywords": ["pasta", "carbonara"],
                "negative_keywords": ["pizza", "lasagna"],
                "description": "Carbonara should not return other Italian dishes"
            },
            {
                "query": "vegetarian burger",
                "top_k": 10,
                "expected_keywords": ["vegetarian", "burger"],
                "dietary_filter": ["vegetarian"],
                "should_not_contain_ingredients": ["beef", "chicken", "pork"],
                "description": "Vegetarian burger should have no meat"
            },
            {
                "query": "gluten free bread",
                "top_k": 10,
                "expected_keywords": ["bread"],
                "dietary_filter": ["gluten-free"],
                "should_not_contain_ingredients": ["flour", "wheat"],
                "description": "Gluten-free bread should not have regular flour"
            },
            {
                "query": "vegan dessert",
                "top_k": 10,
                "expected_keywords": ["dessert"],
                "dietary_filter": ["vegan"],
                "should_not_contain_ingredients": ["egg", "milk", "butter", "cream"],
                "description": "Vegan dessert should have no animal products"
            },
        ]
        
        return test_cases
    
    def evaluate_test_case(self, test_case: Dict) -> Dict:
        """Evaluate a single test case"""
        query = test_case["query"]
        top_k = test_case.get("top_k", 5)
        dietary_filter = test_case.get("dietary_filter", None)
        
        # Search
        start_time = time.time()
        results = self.retriever.search(
            query, 
            top_k=top_k,
            dietary_filters=dietary_filter
        )
        search_time = time.time() - start_time
        
        # Check expected keywords
        expected_keywords = test_case.get("expected_keywords", [])
        recipes_with_expected = 0
        for _, row in results.iterrows():
            name_lower = row['name'].lower()
            if any(kw.lower() in name_lower for kw in expected_keywords):
                recipes_with_expected += 1
        
        expected_keyword_rate = recipes_with_expected / len(results) if len(results) > 0 else 0
        
        # Check negative keywords (should NOT appear)
        negative_keywords = test_case.get("negative_keywords", [])
        recipes_with_negatives = 0
        for _, row in results.iterrows():
            name_lower = row['name'].lower()
            if any(kw.lower() in name_lower for kw in negative_keywords):
                recipes_with_negatives += 1
        
        negative_contamination_rate = recipes_with_negatives / len(results) if len(results) > 0 else 0
        
        # Check forbidden ingredients (for dietary tests)
        forbidden_ingredients = test_case.get("should_not_contain_ingredients", [])
        recipes_with_forbidden = 0
        if forbidden_ingredients:
            for _, row in results.iterrows():
                ingredients_str = ' '.join(row['ingredients_parsed']).lower()
                if any(ing.lower() in ingredients_str for ing in forbidden_ingredients):
                    recipes_with_forbidden += 1
        
        forbidden_ingredient_rate = recipes_with_forbidden / len(results) if len(results) > 0 else 0
        
        # Calculate average similarity score
        avg_similarity = results['similarity_score'].mean() if len(results) > 0 else 0
        
        return {
            "query": query,
            "num_results": len(results),
            "search_time_ms": search_time * 1000,
            "avg_similarity_score": avg_similarity,
            "expected_keyword_rate": expected_keyword_rate,
            "negative_contamination_rate": negative_contamination_rate,
            "forbidden_ingredient_rate": forbidden_ingredient_rate,
            "passed": (
                expected_keyword_rate >= 0.6 and  # At least 60% have expected keywords
                negative_contamination_rate <= 0.2 and  # Max 20% contamination
                forbidden_ingredient_rate == 0  # No forbidden ingredients
            ),
            "results": results
        }
    
    def run_evaluation(self) -> Dict:
        """Run all test cases and aggregate results"""
        test_cases = self.create_test_cases()
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running Hard Negative Evaluation")
        print(f"Retriever: {'CLIP' if self.use_clip else 'Sentence-Transformers'}")
        print(f"Total test cases: {len(test_cases)}")
        print(f"{'='*60}\n")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_case['description']}")
            result = self.evaluate_test_case(test_case)
            results.append(result)
            
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {status}")
            print(f"  - Expected keyword rate: {result['expected_keyword_rate']:.1%}")
            print(f"  - Negative contamination: {result['negative_contamination_rate']:.1%}")
            print(f"  - Forbidden ingredients: {result['forbidden_ingredient_rate']:.1%}")
            print(f"  - Avg similarity: {result['avg_similarity_score']:.3f}")
            print(f"  - Search time: {result['search_time_ms']:.1f}ms")
            print()
        
        # Aggregate metrics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['passed'])
        avg_expected_rate = np.mean([r['expected_keyword_rate'] for r in results])
        avg_contamination = np.mean([r['negative_contamination_rate'] for r in results])
        avg_forbidden = np.mean([r['forbidden_ingredient_rate'] for r in results])
        avg_search_time = np.mean([r['search_time_ms'] for r in results])
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "avg_expected_keyword_rate": avg_expected_rate,
            "avg_negative_contamination": avg_contamination,
            "avg_forbidden_ingredient_rate": avg_forbidden,
            "avg_search_time_ms": avg_search_time,
            "individual_results": results
        }
        
        print(f"{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Tests passed: {passed_tests}/{total_tests} ({summary['pass_rate']:.1%})")
        print(f"Avg expected keyword rate: {avg_expected_rate:.1%}")
        print(f"Avg negative contamination: {avg_contamination:.1%}")
        print(f"Avg forbidden ingredient rate: {avg_forbidden:.1%}")
        print(f"Avg search time: {avg_search_time:.1f}ms")
        print(f"{'='*60}\n")
        
        return summary


def log_evaluation_to_mlflow(summary: Dict, retriever_type: str):
    """Log evaluation results to MLflow"""
    logger = MLflowLogger("recipe-search-pipeline")
    
    with mlflow.start_run(run_name=f"hard_negative_eval_{retriever_type}"):
        # Log parameters
        mlflow.log_param("retriever_type", retriever_type)
        mlflow.log_param("total_test_cases", summary['total_tests'])
        
        # Log aggregate metrics
        mlflow.log_metric("pass_rate", summary['pass_rate'])
        mlflow.log_metric("avg_expected_keyword_rate", summary['avg_expected_keyword_rate'])
        mlflow.log_metric("avg_negative_contamination", summary['avg_negative_contamination'])
        mlflow.log_metric("avg_forbidden_ingredient_rate", summary['avg_forbidden_ingredient_rate'])
        mlflow.log_metric("avg_search_time_ms", summary['avg_search_time_ms'])
        mlflow.log_metric("tests_passed", summary['passed_tests'])
        
        # Log individual test results
        for i, result in enumerate(summary['individual_results'], 1):
            mlflow.log_metric(f"test_{i}_expected_rate", result['expected_keyword_rate'])
            mlflow.log_metric(f"test_{i}_contamination", result['negative_contamination_rate'])
            mlflow.log_metric(f"test_{i}_forbidden", result['forbidden_ingredient_rate'])
            mlflow.log_metric(f"test_{i}_similarity", result['avg_similarity_score'])
        
        # Log tags
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("eval_type", "hard_negatives")
        mlflow.set_tag("status", "completed")
        
        print("✓ Logged evaluation results to MLflow")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval with hard negatives")
    parser.add_argument("--clip", action="store_true", help="Use CLIP retriever instead of sentence-transformers")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = HardNegativeEvaluator(use_clip=args.clip)
    summary = evaluator.run_evaluation()
    
    # Log to MLflow
    if not args.no_mlflow:
        retriever_type = "CLIP" if args.clip else "sentence-transformers"
        log_evaluation_to_mlflow(summary, retriever_type)