# src/models/clip_retrieval.py
"""
CLIP Recipe Retrieval System (Image & Text Search)

Purpose:
    Multimodal recipe search using CLIP embeddings. Enables searching recipes
    by uploading food images OR text queries. Both modalities use the same
    embedding space for consistent results.
    
    Primary use case: Image-to-recipe search
    For pure text search, retrieval.py is faster

Usage:
    # Interactive testing
    python src/models/clip_retrieval.py
    
    # Programmatic usage
    from models.clip_retrieval import CLIPRecipeRetriever
    retriever = CLIPRecipeRetriever()
    retriever.load()
    
    # Image search
    from PIL import Image
    img = Image.open("food.jpg")
    results = retriever.search_by_image(img, top_k=10)
    
    # Text search
    results = retriever.search("chocolate cake", top_k=10)

Features:
    - Image-to-recipe search using CLIP vision encoder
    - Text-to-recipe search using CLIP text encoder
    - Dietary filtering support
    - Shared embedding space for consistent multimodal results

Search Methods:
    1. search_by_image(image, top_k, dietary_filters)
        - Input: PIL Image of food
        - Encodes image with CLIP vision encoder
        - Returns semantically similar recipes
    
    2. search(query, top_k, dietary_filters)
        - Input: Text query (max 77 tokens)
        - Encodes text with CLIP text encoder
        - Same functionality as retrieval.py but using CLIP

Model Architecture:
    CLIP-ViT-B/32:
        - Vision encoder: ViT-B/32 (512-dim)
        - Text encoder: Transformer (512-dim)
        - Shared embedding space
        - Image size: 224x224 (auto-resized)

Performance:
    - Image query time: ~15-25ms (GPU)
    - Text query time: ~5-10ms (GPU)
    - Index: FAISS IndexFlatIP
    - 231,637 recipes searchable

Image Search Quality:
    - Average similarity: 0.318 for relevant matches
    - Works best with clear food photos
    - Handles various angles and presentations
    - Lower similarity than text (expected for cross-modal)

Dietary Filtering:
    - Same as retrieval.py
    - Retrieves top_k * 10, then filters
    - Preserves semantic ranking

Example Usage:
    # Load retriever
    retriever = CLIPRecipeRetriever()
    retriever.load()
    
    # Search by image
    img = Image.open("chocolate_cake.jpg")
    results = retriever.search_by_image(img, top_k=5)
    print(results[['name', 'similarity_score']])
    
    # Output:
    # name                        similarity_score
    # Chocolate Fudge Cake        0.342
    # Dark Chocolate Cake         0.328
    # Chocolate Layer Cake        0.315
    # Chocolate Cupcakes          0.298
    # Chocolate Brownies          0.285
    
    # Search by text (using CLIP)
    results = retriever.search("chocolate cake", top_k=5)
    # Higher similarity scores for text queries (0.7-0.9 range)

When to Use:
    - Use search_by_image() for visual recipe discovery
    - Use search() for text queries when you need CLIP features
    - Use retrieval.py for faster pure text search

Output Format:
    DataFrame with columns:
        - id, name, minutes, tags_parsed, ingredients_parsed
        - n_ingredients, steps, description, recipe_text
        - similarity_score (0.0 to 1.0)

Technical Details:
    - Embeddings: L2 normalized for cosine similarity
    - Image preprocessing: Resize → Normalize → Tensor
    - Text preprocessing: Tokenize (max 77) → Pad → Tensor
    - FAISS index: IndexFlatIP (exact search)

Limitations:
    - Text limited to 77 tokens (vs 512 for sentence-transformers)
    - Image search quality depends on photo clarity
    - Lower absolute similarity scores than text-to-text search

Author: Martin Brocca
Created: 2025-11-29
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import List, Optional
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config import (
    PROCESSED_RECIPES,
    CLIP_RECIPE_EMBEDDINGS,
    CLIP_MODEL,
    setup_directories
)
from utils.device import get_device

class CLIPRecipeRetriever:
    """
    Recipe retrieval using CLIP for image-to-recipe search
    """
    
    def __init__(self, model_name: str = CLIP_MODEL):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.index = None
        self.recipes_df = None
        self.recipe_ids = None
        self.device = get_device()
        
    def load(self):
        """Load CLIP model, embeddings, and build FAISS index"""
        print("Loading CLIP retrieval system...")
        
        # Load CLIP model
        print(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load recipes
        print(f"Loading recipes from {PROCESSED_RECIPES}")
        self.recipes_df = pd.read_parquet(PROCESSED_RECIPES)
        
        # Load CLIP embeddings
        print(f"Loading CLIP embeddings from {CLIP_RECIPE_EMBEDDINGS}")
        embeddings = np.load(CLIP_RECIPE_EMBEDDINGS)
        
        # Load recipe IDs
        recipe_ids_path = CLIP_RECIPE_EMBEDDINGS.parent / "recipe_ids.npy"
        self.recipe_ids = np.load(recipe_ids_path)
        
        print(f"Loaded {len(embeddings)} CLIP recipe embeddings (dim={embeddings.shape[1]})")
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        
        # Embeddings are already normalized in creation
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ CLIP FAISS index built with {self.index.ntotal} recipes")
        
    def search_by_image(
        self, 
        image: Image.Image,
        top_k: int = 10,
        dietary_filters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Search for recipes by uploading a food image
        
        Args:
            image: PIL Image of food
            top_k: Number of results to return
            dietary_filters: List of dietary requirements
        
        Returns:
            DataFrame with top matching recipes
        """
        with torch.no_grad():
            # Process image
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Get image embeddings
            image_embeddings = self.model.get_image_features(**inputs)
            
            # Normalize for cosine similarity
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = image_embeddings.cpu().numpy()
        
        # Search - get more results if filtering
        search_k = top_k * 10 if dietary_filters else top_k
        distances, indices = self.index.search(image_embeddings.astype('float32'), search_k)
        
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


    def search(
        self, 
        query: str,
        top_k: int = 10,
        dietary_filters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Search for recipes by text query using CLIP text encoder
        
        Args:
            query: Text query
            top_k: Number of results to return
            dietary_filters: List of dietary requirements
        
        Returns:
            DataFrame with top matching recipes
        """
        with torch.no_grad():
            # Process text query
            inputs = self.processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            # Get text embeddings
            text_embeddings = self.model.get_text_features(**inputs)
            
            # Normalize for cosine similarity
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.cpu().numpy()
        
        # Search - get more results if filtering
        search_k = top_k * 10 if dietary_filters else top_k
        distances, indices = self.index.search(text_embeddings.astype('float32'), search_k)
        
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


def test_clip_retrieval():
    """Test CLIP image retrieval - requires a test image"""
    retriever = CLIPRecipeRetriever()
    retriever.load()
    
    print("\n" + "="*50)
    print("CLIP Retrieval System Ready!")
    print("To test with an actual image, upload one in the Streamlit app")
    print("="*50)

if __name__ == "__main__":
    test_clip_retrieval()