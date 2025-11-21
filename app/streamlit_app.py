# app/streamlit_app.py
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from models.retrieval import RecipeRetriever
from models.clip_retrieval import CLIPRecipeRetriever
from substitutions.rules import SubstitutionEngine
from strings_es import STRINGS_ES as S

# Page config
st.set_page_config(
    page_title=S["page_title"],
    page_icon="🍳",
    layout="wide"
)

# Initialize systems (cached)
@st.cache_resource
def load_retriever():
    retriever = RecipeRetriever()
    retriever.load()
    return retriever

@st.cache_resource
def load_clip_retriever():
    clip_retriever = CLIPRecipeRetriever()
    clip_retriever.load()
    return clip_retriever

@st.cache_resource
def load_substitution_engine():
    return SubstitutionEngine()

# Title
st.title(S["title"])
st.markdown(S["subtitle"])

# Load systems
with st.spinner(S["loading_db"]):
    retriever = load_retriever()
    clip_retriever = load_clip_retriever()
    sub_engine = load_substitution_engine()

st.success(S["loaded_success"].format(len(retriever.recipes_df)))

# Sidebar - Search options
st.sidebar.header(S["search_options"])

search_mode = st.sidebar.radio(
    S["search_by"],
    [S["image_upload"], S["text_description"], S["ingredients_list"]]
)

# Dietary filters
st.sidebar.subheader(S["dietary_filters"])
dietary_filters = []

if st.sidebar.checkbox(S["vegetarian"]):
    dietary_filters.append("vegetarian")
if st.sidebar.checkbox(S["vegan"]):
    dietary_filters.append("vegan")
if st.sidebar.checkbox(S["gluten_free"]):
    dietary_filters.append("gluten-free")
if st.sidebar.checkbox(S["dairy_free"]):
    dietary_filters.append("dairy-free")

# Substitution preferences
st.sidebar.subheader(S["show_substitutions"])
show_substitutions = st.sidebar.multiselect(
    S["dietary_needs"],
    ["vegan", "vegetarian", "gluten-free", "dairy-free", "low-sugar"],
    help=S["substitutions_help"]
)

# Number of results
top_k = st.sidebar.slider(S["num_results"], 5, 50, 10)

# Display recipe card with substitutions
def display_recipe(row, show_subs=None):
    """Display a recipe card with optional substitutions"""
    with st.expander(f"⭐ {row['similarity_score']:.2%} - {row['name']}"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(S["ingredients"])
            ingredients = row['ingredients_parsed']
            
            # Show ingredients
            for ing in ingredients[:15]:
                st.write(f"• {ing}")
            if len(ingredients) > 15:
                st.write(S["and_more"].format(len(ingredients) - 15))
            
            # Show substitutions if requested
            if show_subs:
                st.markdown("---")
                st.markdown(S["substitutions_title"])
                
                sub_result = sub_engine.suggest_for_recipe(ingredients, show_subs)
                
                if sub_result['substitutions']:
                    for orig_ing, subs in sub_result['substitutions'].items():
                        st.markdown(f"**{orig_ing}** →")
                        for category, options in subs.items():
                            if category in show_subs or category == 'general':
                                st.write(f"  *{category}*: {', '.join(options[:3])}")
                else:
                    st.info(S["no_substitutions"])
            
            if row['description'] and str(row['description']) != 'nan':
                st.markdown("---")
                st.markdown(S["description"])
                st.write(row['description'])
        
        with col2:
            st.metric(S["cooking_time"], S["minutes"].format(row['minutes']))
            st.metric(S["total_ingredients"], f"{row['n_ingredients']}")
            
            # Show dietary tags
            diet_tags = [t for t in row['tags_parsed'] 
                       if any(d in t for d in ['vegetarian', 'vegan', 'gluten', 'dairy', 'low-'])]
            if diet_tags:
                st.markdown(S["dietary_tags"])
                for tag in diet_tags[:5]:
                    st.markdown(f"- `{tag}`")

# Main search area
if search_mode == S["image_upload"]:
    st.markdown(S["upload_image_title"])
    st.caption(S["upload_image_caption"])
    
    uploaded_file = st.file_uploader(
        S["choose_image"],
        type=['png', 'jpg', 'jpeg', 'webp'],
        help=S["upload_help"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=S["uploaded_image"], use_container_width=True)
        
        with col2:
            if st.button(S["find_similar"], type="primary"):
                with st.spinner(S["analyzing"]):
                    results = clip_retriever.search_by_image(
                        image, 
                        top_k=top_k, 
                        dietary_filters=dietary_filters
                    )
                
                if len(results) > 0:
                    st.subheader(S["found_results"].format(len(results)))
                    
                    # Display results
                    for idx, row in results.iterrows():
                        display_recipe(row, show_subs=show_substitutions if show_substitutions else None)
                else:
                    st.warning(S["no_results_warning"])

elif search_mode == S["text_description"]:
    query = st.text_input(
        S["what_looking_for"],
        placeholder=S["text_search_placeholder"]
    )
    
    if st.button(S["search_button"], type="primary"):
        if query:
            with st.spinner(S["searching"]):
                results = retriever.search(query, top_k=top_k, dietary_filters=dietary_filters)
            
            if len(results) > 0:
                st.subheader(S["found_results"].format(len(results)))
                
                # Display results
                for idx, row in results.iterrows():
                    display_recipe(row, show_subs=show_substitutions if show_substitutions else None)
            else:
                st.warning(S["no_results_warning"])
        else:
            st.warning(S["enter_query_warning"])

else:  # Ingredients List
    st.markdown(S["enter_ingredients"])
    ingredients_text = st.text_area(
        S["dietary_needs"],
        placeholder=S["ingredients_placeholder"],
        height=150
    )
    
    if st.button(S["find_recipes"], type="primary"):
        if ingredients_text:
            ingredients = [ing.strip() for ing in ingredients_text.split('\n') if ing.strip()]
            
            with st.spinner(S["finding_recipes"]):
                results = retriever.search_by_ingredients(
                    ingredients, 
                    top_k=top_k, 
                    dietary_filters=dietary_filters
                )
            
            if len(results) > 0:
                st.subheader(S["found_recipes"].format(len(results)))
                st.caption(S["searching_for"].format(', '.join(ingredients)))
                
                # Display results
                for idx, row in results.iterrows():
                    with st.expander(f"⭐ {row['similarity_score']:.2%} - {row['name']}"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(S["ingredients"])
                            recipe_ingredients = row['ingredients_parsed']
                            
                            # Highlight matching ingredients
                            matching = [ing for ing in recipe_ingredients 
                                      if any(search_ing.lower() in ing.lower() 
                                            for search_ing in ingredients)]
                            
                            st.markdown(S["your_ingredients"])
                            st.write(", ".join(matching))
                            
                            other_ingredients = [ing for ing in recipe_ingredients 
                                               if ing not in matching]
                            if other_ingredients:
                                st.markdown(S["additional_needed"])
                                for ing in other_ingredients[:10]:
                                    st.write(f"• {ing}")
                                if len(other_ingredients) > 10:
                                    st.write(S["and_more"].format(len(other_ingredients) - 10))
                            
                            # Show substitutions if requested
                            if show_substitutions:
                                st.markdown("---")
                                st.markdown(S["substitutions_title"])
                                
                                sub_result = sub_engine.suggest_for_recipe(recipe_ingredients, show_substitutions)
                                
                                if sub_result['substitutions']:
                                    for orig_ing, subs in sub_result['substitutions'].items():
                                        st.markdown(f"**{orig_ing}** →")
                                        for category, options in subs.items():
                                            if category in show_subs or category == 'general':
                                                st.write(f"  *{category}*: {', '.join(options[:3])}")
                        
                        with col2:
                            st.metric(S["cooking_time"], S["minutes"].format(row['minutes']))
                            st.metric(S["total_ingredients"], f"{row['n_ingredients']}")
                            st.metric(S["match"], f"{len(matching)}/{len(recipe_ingredients)}")
            else:
                st.warning(S["no_results_warning"])
        else:
            st.warning(S["no_ingredients_warning"])

# Footer
st.markdown("---")
st.caption(S["footer_multimodal"])
st.caption(S["footer_tip"])