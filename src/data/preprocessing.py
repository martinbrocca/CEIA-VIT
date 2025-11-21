# src/data/preprocessing.py
import pandas as pd
import ast
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import (
    FOODCOM_RAW_RECIPES, 
    PROCESSED_RECIPES,
    setup_directories
)
from utils.mlflow_logger import MLflowLogger

def parse_list_field(field):
    """Safely parse string representation of lists"""
    try:
        return ast.literal_eval(field)
    except:
        return []

def create_recipe_text(row):
    """
    Create a rich text representation of a recipe for embedding.
    Combines: name, ingredients, and key tags
    """
    name = row['name']
    ingredients = ', '.join(row['ingredients_parsed'])
    
    # Extract useful tags (exclude generic ones)
    tags = row['tags_parsed']
    useful_tags = [t for t in tags if any(keyword in t for keyword in 
                   ['cuisine', 'vegetarian', 'vegan', 'gluten', 'dairy', 
                    'main-dish', 'side-dish', 'dessert', 'breakfast', 'healthy'])]
    tags_text = ', '.join(useful_tags[:5])
    
    # Combine into searchable text
    recipe_text = f"{name}. Ingredients: {ingredients}. Tags: {tags_text}"
    return recipe_text

def preprocess_recipes(log_to_mlflow: bool = True):
    """
    Load and preprocess Food.com recipes
    """
    # Setup directories
    setup_directories()
    
    print(f"Loading recipes from: {FOODCOM_RAW_RECIPES}")
    if not FOODCOM_RAW_RECIPES.exists():
        raise FileNotFoundError(f"Raw recipes not found at {FOODCOM_RAW_RECIPES}")
    
    df = pd.read_csv(FOODCOM_RAW_RECIPES)
    total_raw = len(df)
    print(f"Loaded {total_raw} recipes")
    
    # Parse list fields
    print("Parsing ingredients and tags...")
    df['ingredients_parsed'] = df['ingredients'].apply(parse_list_field)
    df['tags_parsed'] = df['tags'].apply(parse_list_field)
    
    # Create recipe text for embedding
    print("Creating recipe text representations...")
    df['recipe_text'] = df.apply(create_recipe_text, axis=1)
    
    # Basic filtering - remove recipes with missing critical data
    df = df[df['ingredients_parsed'].str.len() > 0]
    df = df[df['name'].notna()]
    
    total_processed = len(df)
    filtered_out = total_raw - total_processed
    print(f"After filtering: {total_processed} recipes ({filtered_out} filtered out)")
    
    # Select columns to keep
    columns_to_keep = [
        'id', 'name', 'minutes', 'tags_parsed', 'ingredients_parsed', 
        'n_ingredients', 'steps', 'description', 'recipe_text'
    ]
    
    df_clean = df[columns_to_keep].copy()
    
    # Calculate statistics
    avg_ingredients = df_clean['n_ingredients'].mean()
    avg_text_length = df_clean['recipe_text'].str.len().mean()
    
    print(f"Average ingredients per recipe: {avg_ingredients:.1f}")
    print(f"Average recipe text length: {avg_text_length:.0f} characters")
    
    # Save as parquet
    print(f"Saving to {PROCESSED_RECIPES}...")
    df_clean.to_parquet(PROCESSED_RECIPES, index=False)
    
    print("✓ Done!")
    print(f"Processed recipes saved to: {PROCESSED_RECIPES}")
    
    # Log to MLflow
    if log_to_mlflow:
        print("\nLogging to MLflow...")
        logger = MLflowLogger("recipe-search-pipeline")
        logger.log_preprocessing({
            "raw_path": str(FOODCOM_RAW_RECIPES),
            "total_raw": total_raw,
            "total_processed": total_processed,
            "filtered_out": filtered_out,
            "avg_ingredients": avg_ingredients,
            "avg_text_length": avg_text_length,
            "output_path": str(PROCESSED_RECIPES)
        })
        print("✓ Logged to MLflow")
    
    return df_clean

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    args = parser.parse_args()
    
    preprocess_recipes(log_to_mlflow=not args.no_mlflow)