# src/models/retrieval.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import (
    PROCESSED_RECIPES,
    RECIPE_EMBEDDINGS,
    FAISS_INDEX_DIR,
    DEFAULT_EMBEDDING_MODEL,
    setup_directories
)
from utils.device import get_device

class RecipeRetriever:
    """
    Recipe retrieval system using FAISS for similarity search
    """
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.recipes_df = None
        self.recipe_ids = None
        self.device = get_device()
        
    def load(self):
        """Load model, embeddings, and build FAISS index"""
        print("Loading retrieval system...")
        
        # Load sentence transformer model
        print(f"Loading model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.model = self.model.to(self.device)
        
        # Load recipes
        print(f"Loading recipes from {PROCESSED_RECIPES}")
        self.recipes_df = pd.read_parquet(PROCESSED_RECIPES)
        
        # Load embeddings
        print(f"Loading embeddings from {RECIPE_EMBEDDINGS}")
        embeddings = np.load(RECIPE_EMBEDDINGS)
        
        # Load recipe IDs
        recipe_ids_path = RECIPE_EMBEDDINGS.parent / "recipe_ids.npy"
        self.recipe_ids = np.load(recipe_ids_path)
        
        print(f"Loaded {len(embeddings)} recipe embeddings (dim={embeddings.shape[1]})")
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Use IndexFlatIP for cosine similarity (after normalization)
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index = faiss.IndexFlatIP(dimension)  # Inner product after normalization = cosine similarity
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ FAISS index built with {self.index.ntotal} recipes")
        
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        dietary_filters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Search for recipes by text query
        
        Args:
            query: Text query (e.g., "pasta with tomatoes")
            top_k: Number of results to return
            dietary_filters: List of dietary requirements (e.g., ['vegetarian', 'gluten-free'])
        
        Returns:
            DataFrame with top matching recipes
        """
        # Encode query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            device=str(self.device)
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search - get more results if filtering
        search_k = top_k * 10 if dietary_filters else top_k
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Get recipe IDs
        result_recipe_ids = self.recipe_ids[indices[0]]
        
        # Get recipes
        results = self.recipes_df[self.recipes_df['id'].isin(result_recipe_ids)].copy()
        
        # Add similarity scores
        id_to_score = dict(zip(result_recipe_ids, distances[0]))
        results['similarity_score'] = results['id'].map(id_to_score)
        results = results.sort_values('similarity_score', ascending=False)
        
        # Apply dietary filters if specified
        if dietary_filters:
            for diet_filter in dietary_filters:
                diet_filter_lower = diet_filter.lower()
                results = results[
                    results['tags_parsed'].apply(
                        lambda tags: any(diet_filter_lower in tag.lower() for tag in tags)
                    )
                ]
        
        # Return top_k after filtering
        return results.head(top_k)
    
    def search_by_ingredients(
        self,
        ingredients: List[str],
        top_k: int = 10,
        dietary_filters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Search for recipes by ingredients list
        
        Args:
            ingredients: List of ingredients (e.g., ["tomato", "pasta", "basil"])
            top_k: Number of results
            dietary_filters: Dietary requirements
        """
        # Create query from ingredients
        query = "Recipe with ingredients: " + ", ".join(ingredients)
        return self.search(query, top_k, dietary_filters)


def test_retrieval():
    """Test the retrieval system"""
    retriever = RecipeRetriever()
    retriever.load()
    
    # Test 1: Simple search
    print("\n" + "="*50)
    print("Test 1: Search for 'chocolate cake'")
    results = retriever.search("chocolate cake", top_k=5)
    print(f"\nTop {len(results)} results:")
    for idx, row in results.iterrows():
        print(f"  {row['similarity_score']:.3f} - {row['name']}")
    
    # Test 2: Ingredient-based search
    print("\n" + "="*50)
    print("Test 2: Search by ingredients: ['chicken', 'rice', 'vegetables']")
    results = retriever.search_by_ingredients(['chicken', 'rice', 'vegetables'], top_k=5)
    print(f"\nTop {len(results)} results:")
    for idx, row in results.iterrows():
        print(f"  {row['similarity_score']:.3f} - {row['name']}")
    
    # Test 3: With dietary filter
    print("\n" + "="*50)
    print("Test 3: Search for 'pasta' with vegetarian filter")
    results = retriever.search("pasta", top_k=5, dietary_filters=['vegetarian'])
    print(f"\nTop {len(results)} results:")
    for idx, row in results.iterrows():
        print(f"  {row['similarity_score']:.3f} - {row['name']}")
        veg_tags = [t for t in row['tags_parsed'] if 'vegetarian' in t or 'vegan' in t]
        print(f"     Dietary tags: {veg_tags}")

if __name__ == "__main__":
    test_retrieval()