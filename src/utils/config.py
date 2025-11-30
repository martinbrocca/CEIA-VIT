# src/utils/config.py
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if not (PROJECT_ROOT / "data" / "raw" / "food-com").exists():
    print(f"Warning: Expected data directory not found at {PROJECT_ROOT}")

    
# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Food.com specific paths
FOODCOM_RAW_DIR = RAW_DATA_DIR / "food-com"
FOODCOM_RAW_RECIPES = FOODCOM_RAW_DIR / "RAW_recipes.csv"
PROCESSED_RECIPES_ORIG = PROCESSED_DATA_DIR / "recipes.parquet"
PROCESSED_RECIPES = PROJECT_ROOT / "data" / "processed" / "recipes_corrected.parquet"  # Fixed tags with Carina's script


# Embeddings paths
RECIPE_EMBEDDINGS = EMBEDDINGS_DIR / "recipe_embeddings.npy"
FAISS_INDEX_DIR = EMBEDDINGS_DIR / "faiss_index"

# Model configs
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
# Add to embeddings paths
CLIP_RECIPE_EMBEDDINGS = EMBEDDINGS_DIR / "clip_recipe_embeddings.npy"


# Ensure directories exist
def setup_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [PROCESSED_DATA_DIR, EMBEDDINGS_DIR, FAISS_INDEX_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw data: {RAW_DATA_DIR}")
    print(f"Processed data: {PROCESSED_DATA_DIR}")
    print(f"Food.com recipes: {FOODCOM_RAW_RECIPES}")
    print(f"Exists: {FOODCOM_RAW_RECIPES.exists()}")