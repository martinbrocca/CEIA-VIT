"""
Multimodal Recipe Search - Model Comparison Interface

Purpose:
    Interactive Streamlit app for comparing any two vision-language models.
    Supports both text and image queries with side-by-side results display.

Features:
    - Model selector: Choose any 2 models to compare
    - Text search with both models simultaneously
    - Image upload with both models simultaneously
    - Side-by-side result comparison
    - Similarity score comparison
    - Visual highlighting of better results

Usage:
    streamlit run app/streamlit_app_compare.py
    
    Or with venv:
    .venv/bin/python -m streamlit run app/streamlit_app_compare.py

Available Models:
    - CLIP-ViT-B/32: openai/clip-vit-base-patch32
    - CLIP-ViT-L/14: openai/clip-vit-large-patch14
    - SigLIP Baseline: google/siglip-base-patch16-224
    - SigLIP Fine-tuned: models/siglip-food-finetuned

Author: Martin (CEIA Master's Thesis)
Created: 2025-11-29
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import pickle

import streamlit as st
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
import faiss
import numpy as np
from typing import Dict, Tuple

from utils.config import PROCESSED_RECIPES, PROJECT_ROOT
from utils.device import get_device


# Available models
AVAILABLE_MODELS = {
    "CLIP-ViT-B/32": {
        "path": "openai/clip-vit-base-patch32",
        "type": "clip",
        "description": "CLIP Base - Best overall performance (90% accuracy)",
        "color": "#1976D2"
    },
    "SigLIP v1": {  # Renamed from "SigLIP Baseline"
        "path": "google/siglip-base-patch16-224",
        "type": "siglip",
        "description": "SigLIP v1 before fine-tuning (57% accuracy)",
        "color": "#F57C00"
    },
    "SigLIP v2 (SO400M)": {  # NEW!
        "path": "google/siglip-so400m-patch14-384",
        "type": "siglip",
        "description": "SigLIP v2 - Trained on 400M samples, 384px resolution",
        "color": "#00897B"
    },
    "SigLIP Fine-tuned": {
        "path": "models/siglip-food-finetuned",
        "type": "siglip",
        "description": "Fine-tuned on Food-101 (74% accuracy, +16.7%)",
        "color": "#7B1FA2"
    }
}


# Page config
st.set_page_config(
    page_title="Recipe Search - Model Comparison",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-label {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .winner-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .recipe-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .similarity-score {
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name: str, model_info: Dict):
    """Load a single model"""
    device = get_device()
    model_path = model_info["path"]
    model_type = model_info["type"]
    
    # Check if local model exists
    if model_path.startswith("models/"):
        full_path = PROJECT_ROOT / model_path
        if not full_path.exists():
            st.error(f"❌ Model not found: {model_path}")
            st.info("Please run: `python src/training/finetune_siglip.py --epochs 5 --batch-size 32`")
            return None, None
        model_path = str(full_path)
    
    with st.spinner(f"🔄 Loading {model_name}..."):
        if model_type == "clip":
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path)
        else:  # siglip
            model = AutoModel.from_pretrained(model_path)
            processor = AutoProcessor.from_pretrained(model_path)
        
        model = model.to(device)
        model.eval()
    
    return model, processor


@st.cache_data(hash_funcs={pd.DataFrame: id})
def load_recipes():
    """Load recipe database"""
    df = pd.read_parquet(PROCESSED_RECIPES)
    return df


@st.cache_resource
def create_embeddings_and_index(_model, _processor, _model_type, _device, _recipes_df, model_identifier: str):
    """Create embeddings and FAISS index for a model"""
    # model_identifier is the cache key to prevent reuse across different models
    recipe_texts = _recipes_df['recipe_text'].tolist()
    
    all_embeddings = []
    batch_size = 64
    
    progress_bar = st.progress(0, text=f"Creating embeddings for {model_identifier}...")
    total_batches = len(recipe_texts) // batch_size + 1
    
    with torch.no_grad():
        for i in range(0, len(recipe_texts), batch_size):
            batch = recipe_texts[i:i + batch_size]
            
            if _model_type == "clip":
                inputs = _processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(_device)
                embeddings = _model.get_text_features(**inputs)
            else:  # siglip
                inputs = _processor(
                    text=batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                ).to(_device)
                embeddings = _model.get_text_features(**inputs)
            
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
            
            progress_bar.progress((i // batch_size + 1) / total_batches)
    
    progress_bar.empty()
    
    embeddings = np.vstack(all_embeddings)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    return index


def search_text(query: str, model, processor, model_type, index, device, recipes_df, top_k: int = 5):
    """Search by text query"""
    with torch.no_grad():
        if model_type == "clip":
            inputs = processor(
                text=[query],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            query_embedding = model.get_text_features(**inputs)
        else:  # siglip
            inputs = processor(
                text=[query],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            ).to(device)
            query_embedding = model.get_text_features(**inputs)
        
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()
    
    distances, result_indices = index.search(query_embedding.astype('float32'), top_k)  # ← Changed variable name
    
    results = recipes_df.iloc[result_indices[0]].copy()
    results['similarity_score'] = distances[0]
    
    return results


def search_image(image: Image.Image, model, processor, model_type, index, device, recipes_df, top_k: int = 5):
    """Search by image"""
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt").to(device)
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy()
    
    distances, result_indices = index.search(query_embedding.astype('float32'), top_k)  # ← Changed variable name
    
    results = recipes_df.iloc[result_indices[0]].copy()
    results['similarity_score'] = distances[0]
    
    return results


def display_recipe_card(recipe, rank: int, is_winner: bool = False):
    """Display a recipe card"""
    winner_badge = '<span class="winner-badge">BETTER</span>' if is_winner else ''
    
    st.markdown(f"""
    <div class="recipe-card">
        <h4>#{rank} {recipe['name']} {winner_badge}</h4>
        <p class="similarity-score">Similarity: {recipe['similarity_score']:.4f}</p>
        <p><strong>Ingredients:</strong> {', '.join(recipe['ingredients_parsed'][:8])}{'...' if len(recipe['ingredients_parsed']) > 8 else ''}</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_resource
def create_or_load_index(_model, _processor, _model_type, _device, _recipes_df, model_identifier: str):
    """Create embeddings and FAISS index, or load from cache"""
    
    # Define cache paths
    cache_dir = PROJECT_ROOT / "cache" / "indices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique filename based on model and dataset size
    cache_file = cache_dir / f"{model_identifier.replace('/', '_')}_{len(_recipes_df)}.faiss"
    embeddings_file = cache_dir / f"{model_identifier.replace('/', '_')}_{len(_recipes_df)}.npy"
    
    # Try to load from cache
    if cache_file.exists() and embeddings_file.exists():
        st.info(f"Loading cached index for {model_identifier}...")
        try:
            index = faiss.read_index(str(cache_file))
            # embeddings = np.load(embeddings_file)  # Optional: if you need embeddings later
            st.success(f"✅ Loaded cached index ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
            return index
        except Exception as e:
            st.warning(f"Cache load failed: {e}. Rebuilding...")
    
    # Cache miss - create embeddings
    recipe_texts = _recipes_df['recipe_text'].tolist()
    
    all_embeddings = []
    batch_size = 64
    
    progress_bar = st.progress(0, text=f"Creating embeddings for {model_identifier}...")
    total_batches = len(recipe_texts) // batch_size + 1
    
    with torch.no_grad():
        for i in range(0, len(recipe_texts), batch_size):
            batch = recipe_texts[i:i + batch_size]
            
            if _model_type == "clip":
                inputs = _processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(_device)
                embeddings = _model.get_text_features(**inputs)
            else:  # siglip
                inputs = _processor(
                    text=batch,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=64
                ).to(_device)
                embeddings = _model.get_text_features(**inputs)
            
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())
            
            progress_bar.progress((i // batch_size + 1) / total_batches)
    
    progress_bar.empty()
    
    embeddings = np.vstack(all_embeddings)
    
    # Build FAISS index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    # Save to cache
    try:
        faiss.write_index(index, str(cache_file))
        np.save(embeddings_file, embeddings)
        st.success(f"✅ Cached index to {cache_file} ({cache_file.stat().st_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        st.warning(f"Failed to cache index: {e}")
    
    return index

def main():
    # Header
    st.markdown('<p class="main-header">🍳 Recipe Search: Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare Any Two Vision-Language Models Side-by-Side</p>', unsafe_allow_html=True)
    
    # Model Selection (before loading anything)
    st.markdown("---")
    st.subheader("🎯 Select Models to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model1_name = st.selectbox(
            "Model A:",
            options=list(AVAILABLE_MODELS.keys()),
            index=0,  # CLIP-ViT-B/32
            key="model1"
        )
        st.caption(AVAILABLE_MODELS[model1_name]["description"])
    
    with col2:
        model2_name = st.selectbox(
            "Model B:",
            options=list(AVAILABLE_MODELS.keys()),
            index=3,  # SigLIP Fine-tuned
            key="model2"
        )
        st.caption(AVAILABLE_MODELS[model2_name]["description"])
    
    if model1_name == model2_name:
        st.warning("⚠️ Please select two different models to compare!")
        st.stop()
    
    # Load selected models
    device = get_device()
    
    model1_info = AVAILABLE_MODELS[model1_name]
    model2_info = AVAILABLE_MODELS[model2_name]
    
    model1, processor1 = load_model(model1_name, model1_info)
    model2, processor2 = load_model(model2_name, model2_info)
    
    if model1 is None or model2 is None:
        st.stop()
    
    # Load recipes
    recipes_df = load_recipes()
    
    # Create indices with model identifier as cache key
    with st.spinner(f"Loading index for {model1_name}..."):
        index1 = create_or_load_index(
            model1, processor1, model1_info["type"], device, recipes_df, model1_name
        )

    with st.spinner(f"Loading index for {model2_name}..."):
        index2 = create_or_load_index(
            model2, processor2, model2_info["type"], device, recipes_df, model2_name
        )
    
    st.success(f"✅ Loaded {len(recipes_df):,} recipes | Device: {device}")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        st.markdown("### 📊 Selected Models")
        st.markdown(f"**A:** {model1_name}")
        st.markdown(f"**B:** {model2_name}")
        
        st.markdown("---")
        st.markdown("### 📈 Performance Reference")
        st.markdown("""
        **Accuracy@1 on Food-101:**
        - CLIP-ViT-B/32: 90%
        - SigLIP Fine-tuned: 74%
        - SigLIP Baseline: 57%
        
        **Best for:**
        - Text: CLIP-ViT-B/32
        - Images: CLIP or Fine-tuned SigLIP
        """)
        
        # ADD THIS NEW SECTION HERE ↓
        st.markdown("---")
        st.markdown("### 🗂️ Cache Management")
        
        cache_dir = PROJECT_ROOT / "cache" / "indices"
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.faiss"))
            total_size = sum(f.stat().st_size for f in cache_files) / 1024 / 1024 / 1024
            
            st.markdown(f"**Cached indices:** {len(cache_files)}")
            st.markdown(f"**Total size:** {total_size:.2f} GB")
            
            if st.button("🗑️ Clear Cache"):
                import shutil
                shutil.rmtree(cache_dir)
                st.success("✅ Cache cleared! Refreshing...")
                st.rerun()
        else:
            st.markdown("*No cache yet*")
    
    # Search Mode Selection
    st.markdown("---")
    search_mode = st.radio(
        "Search Mode:",
        ["🔍 Text Search", "📷 Image Search"],
        horizontal=True
    )
    
    # Search Interface
    if search_mode == "🔍 Text Search":
        query = st.text_input(
            "Enter your search query:",
            placeholder="e.g., chocolate cake, pasta carbonara, grilled salmon...",
            key="text_query"
        )
        
        if query:
            with st.spinner("Searching..."):
                results1 = search_text(query, model1, processor1, model1_info["type"], 
                                      index1, device, recipes_df, top_k)
                results2 = search_text(query, model2, processor2, model2_info["type"], 
                                      index2, device, recipes_df, top_k)
            
            # Display results side-by-side
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div class="model-label" style="background-color: {model1_info["color"]}22; color: {model1_info["color"]};">📘 {model1_name}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f"**Avg Similarity:** {results1['similarity_score'].mean():.4f}")
                
                for i, (_, recipe) in enumerate(results1.iterrows(), 1):
                    is_winner = recipe['similarity_score'] > results2.iloc[i-1]['similarity_score']
                    display_recipe_card(recipe, i, is_winner)
            
            with col2:
                st.markdown(f'<div class="model-label" style="background-color: {model2_info["color"]}22; color: {model2_info["color"]};">🔥 {model2_name}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f"**Avg Similarity:** {results2['similarity_score'].mean():.4f}")
                
                for i, (_, recipe) in enumerate(results2.iterrows(), 1):
                    is_winner = recipe['similarity_score'] > results1.iloc[i-1]['similarity_score']
                    display_recipe_card(recipe, i, is_winner)
    
    else:  # Image Search
        uploaded_file = st.file_uploader(
            "Upload a food image:",
            type=["jpg", "jpeg", "png"],
            help="Upload an image of food to find similar recipes"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display uploaded image
            st.image(image, caption="Uploaded Image", width=300)
            
            with st.spinner("Searching..."):
                results1 = search_image(image, model1, processor1, model1_info["type"], 
                                       index1, device, recipes_df, top_k)
                results2 = search_image(image, model2, processor2, model2_info["type"], 
                                       index2, device, recipes_df, top_k)
            
            # Display results side-by-side
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<div class="model-label" style="background-color: {model1_info["color"]}22; color: {model1_info["color"]};">📘 {model1_name}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f"**Avg Similarity:** {results1['similarity_score'].mean():.4f}")
                
                for i, (_, recipe) in enumerate(results1.iterrows(), 1):
                    is_winner = recipe['similarity_score'] > results2.iloc[i-1]['similarity_score']
                    display_recipe_card(recipe, i, is_winner)
            
            with col2:
                st.markdown(f'<div class="model-label" style="background-color: {model2_info["color"]}22; color: {model2_info["color"]};">🔥 {model2_name}</div>', 
                           unsafe_allow_html=True)
                st.markdown(f"**Avg Similarity:** {results2['similarity_score'].mean():.4f}")
                
                for i, (_, recipe) in enumerate(results2.iterrows(), 1):
                    is_winner = recipe['similarity_score'] > results1.iloc[i-1]['similarity_score']
                    display_recipe_card(recipe, i, is_winner)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666;'>
        <p>CEIA Master's Thesis | Universidad de Buenos Aires | 2025</p>
        <p>Comparing: {model1_name} vs {model2_name}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()