# src/data/create_food101_pairs.py
"""
Food-101-Recipe-Pairs Dataset Creator

Purpose:
    Creates a vision-language dataset by pairing Food-101 images with Food.com recipes.
    Matches images to recipes by category name and optionally verifies quality using BLIP.

Output:
    - data/processed/food101_recipe_pairs.json (21K+ image-text pairs)
    - data/processed/food101_recipe_pairs.csv (tabular format)

Usage:
    # Quick version (no BLIP verification)
    python src/data/create_food101_pairs.py --max-images 100 --recipes-per-image 3
    
    # With BLIP quality verification (recommended)
    python src/data/create_food101_pairs.py --max-images 100 --recipes-per-image 3 --use-blip
    
    # Full dataset
    python src/data/create_food101_pairs.py --max-images 200 --recipes-per-image 5 --use-blip

Dataset Statistics:
    - 21,729 image-text pairs
    - 86 food categories
    - 8,600 unique images
    - 80.2% average quality score (BLIP verified)

Author: Martin Brocca
Created: 2025-11-29
"""


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict

from utils.config import PROJECT_ROOT, PROCESSED_RECIPES
from utils.device import get_device


class Food101PairBuilder:
    """
    Build image-text pairs from Food-101 images + Food.com recipes
    
    Strategy:
    1. Match Food-101 categories to Food.com recipe names
    2. Pair images with recipes
    3. (Optional) Use BLIP to verify quality
    4. Export as JSON dataset
    """
    
    FOOD101_PATH = PROJECT_ROOT / "data" / "raw" / "food-101" / "images"
    OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "food101_recipe_pairs.json"
    
    # Map Food-101 categories to Food.com search terms
    CATEGORY_MAPPINGS = {
        # Desserts
        "apple_pie": ["apple pie", "apple tart"],
        "baklava": ["baklava"],
        "cannoli": ["cannoli"],
        "carrot_cake": ["carrot cake"],
        "cheesecake": ["cheesecake", "cheese cake"],
        "chocolate_cake": ["chocolate cake"],
        "chocolate_mousse": ["chocolate mousse", "mousse chocolate"],
        "churros": ["churros", "churro"],
        "creme_brulee": ["creme brulee", "crème brûlée", "brulee"],
        "cup_cakes": ["cupcake", "cupcakes", "cup cake"],
        "donuts": ["donut", "doughnut", "donuts", "doughnuts"],
        "macarons": ["macaron", "macarons"],
        "panna_cotta": ["panna cotta"],
        "red_velvet_cake": ["red velvet cake", "red velvet"],
        "strawberry_shortcake": ["strawberry shortcake"],
        "tiramisu": ["tiramisu"],
        "waffles": ["waffle", "waffles"],
        
        # Main dishes
        "baby_back_ribs": ["baby back ribs", "bbq ribs", "pork ribs", "back ribs"],
        "beef_carpaccio": ["beef carpaccio", "carpaccio"],
        "beef_tartare": ["beef tartare", "steak tartare", "tartare"],
        "bibimbap": ["bibimbap"],
        "chicken_curry": ["chicken curry", "curry chicken"],
        "chicken_quesadilla": ["chicken quesadilla", "quesadilla"],
        "chicken_wings": ["chicken wings", "buffalo wings", "wings"],
        "dumplings": ["dumpling", "dumplings", "potsticker"],
        "filet_mignon": ["filet mignon", "beef tenderloin"],
        "fish_and_chips": ["fish and chips", "fish chips"],
        "fried_rice": ["fried rice"],
        "grilled_salmon": ["grilled salmon", "salmon grilled"],
        "gyoza": ["gyoza"],
        "hamburger": ["hamburger", "burger"],
        "hot_dog": ["hot dog", "hotdog"],
        "lasagna": ["lasagna", "lasagne"],
        "pad_thai": ["pad thai"],
        "paella": ["paella"],
        "peking_duck": ["peking duck", "duck"],
        "pho": ["pho"],
        "pizza": ["pizza"],
        "pork_chop": ["pork chop", "pork chops"],
        "prime_rib": ["prime rib"],
        "ramen": ["ramen"],
        "ravioli": ["ravioli"],
        "risotto": ["risotto"],
        "spaghetti_bolognese": ["spaghetti bolognese", "bolognese"],
        "spaghetti_carbonara": ["spaghetti carbonara", "carbonara"],
        "steak": ["steak"],
        "sushi": ["sushi"],
        "tacos": ["taco", "tacos"],
        
        # Salads & Sides
        "beet_salad": ["beet salad", "beetroot salad"],
        "caesar_salad": ["caesar salad"],
        "caprese_salad": ["caprese salad", "caprese"],
        "greek_salad": ["greek salad"],
        "seaweed_salad": ["seaweed salad"],
        
        # Breakfast
        "breakfast_burrito": ["breakfast burrito"],
        "eggs_benedict": ["eggs benedict", "benedict"],
        "french_toast": ["french toast"],
        "huevos_rancheros": ["huevos rancheros"],
        "omelette": ["omelette", "omelet"],
        "pancakes": ["pancake", "pancakes"],
        
        # Appetizers
        "bruschetta": ["bruschetta"],
        "crab_cakes": ["crab cake", "crab cakes"],
        "deviled_eggs": ["deviled eggs"],
        "edamame": ["edamame"],
        "escargots": ["escargot", "escargots"],
        "falafel": ["falafel"],
        "french_fries": ["french fries", "fries"],
        "fried_calamari": ["fried calamari", "calamari"],
        "garlic_bread": ["garlic bread"],
        "guacamole": ["guacamole"],
        "hummus": ["hummus"],
        "nachos": ["nachos"],
        "onion_rings": ["onion rings"],
        "spring_rolls": ["spring roll", "spring rolls"],
        
        # Soups
        "clam_chowder": ["clam chowder", "chowder"],
        "french_onion_soup": ["french onion soup", "onion soup"],
        "hot_and_sour_soup": ["hot and sour soup"],
        "lobster_bisque": ["lobster bisque", "bisque"],
        "miso_soup": ["miso soup"],
        
        # Other
        "cheese_plate": ["cheese plate", "cheese platter"],
        "club_sandwich": ["club sandwich"],
        "frozen_yogurt": ["frozen yogurt", "froyo"],
        "gnocchi": ["gnocchi"],
        "grilled_cheese_sandwich": ["grilled cheese"],
        "ice_cream": ["ice cream"],
        "macaroni_and_cheese": ["macaroni and cheese", "mac and cheese"],
        "samosa": ["samosa"],
        "takoyaki": ["takoyaki"],
    }
    
    def __init__(self, max_images_per_category: int = 100, 
                 recipes_per_image: int = 3,
                 use_blip_verification: bool = False):
        """
        Args:
            max_images_per_category: Max images to use per Food-101 category
            recipes_per_image: How many recipes to pair with each image
            use_blip_verification: Use BLIP to verify image-text match quality
        """
        self.max_images_per_category = max_images_per_category
        self.recipes_per_image = recipes_per_image
        self.use_blip_verification = use_blip_verification
        self.device = get_device()
        
        self.recipes_df = None
        self.blip_model = None
        self.blip_processor = None
        self.pairs = []
        
    def load_recipes(self):
        """Load Food.com recipes"""
        print(f"Loading recipes from {PROCESSED_RECIPES}...")
        self.recipes_df = pd.read_parquet(PROCESSED_RECIPES)
        print(f"Loaded {len(self.recipes_df)} recipes")
    
    def load_blip(self):
        """Load BLIP for caption-based verification (optional)"""
        if not self.use_blip_verification:
            return
        
        print("Loading BLIP for verification...")
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.blip_model.eval()
        print("✓ BLIP loaded")
    
    def generate_caption(self, image: Image.Image) -> str:
        """Generate caption for image using BLIP"""
        if not self.use_blip_verification:
            return ""
        
        import torch
        
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.blip_model.generate(**inputs, max_length=50)
        
        caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def find_matching_recipes(self, category: str, search_terms: List[str], 
                            n: int = 10) -> pd.DataFrame:
        """Find recipes matching the category"""
        # Build regex pattern from search terms
        pattern = "|".join([f"\\b{term}\\b" for term in search_terms])
        
        # Search in recipe names
        matches = self.recipes_df[
            self.recipes_df['name'].str.contains(pattern, case=False, regex=True, na=False)
        ]
        
        # If not enough matches, also search in recipe_text
        if len(matches) < n:
            text_matches = self.recipes_df[
                self.recipes_df['recipe_text'].str.contains(pattern, case=False, regex=True, na=False)
            ]
            # FIX: Use 'id' column to drop duplicates instead of all columns
            matches = pd.concat([matches, text_matches]).drop_duplicates(subset=['id'])
        
        return matches.head(n * 3)  # Get extra for sampling
    
    def verify_pair_quality(self, image: Image.Image, recipe_text: str, 
                           category: str) -> Dict:
        """Verify image-text pair quality"""
        quality_score = 1.0
        verification = {
            "blip_caption": "",
            "caption_match": False,
            "quality_score": quality_score
        }
        
        if self.use_blip_verification:
            # Generate caption
            caption = self.generate_caption(image)
            verification["blip_caption"] = caption
            
            # Check if caption mentions the dish
            caption_lower = caption.lower()
            category_words = category.replace("_", " ").split()
            
            matches = sum(1 for word in category_words if word in caption_lower)
            verification["caption_match"] = matches > 0
            
            # Adjust quality score
            if verification["caption_match"]:
                quality_score = 1.0
            elif any(food_word in caption_lower for food_word in 
                    ["food", "plate", "dish", "meal", "dessert"]):
                quality_score = 0.7
            else:
                quality_score = 0.4
            
            verification["quality_score"] = quality_score
        
        return verification
    
    def build_pairs(self):
        """Build all image-recipe pairs"""
        print(f"\n{'='*60}")
        print("Building Food-101-Recipe-Pairs Dataset")
        print(f"{'='*60}\n")
        
        self.load_recipes()
        self.load_blip()
        
        stats = {
            "total_categories": 0,
            "successful_categories": 0,
            "total_images": 0,
            "total_pairs": 0,
            "avg_quality": [],
        }
        
        for category, search_terms in tqdm(self.CATEGORY_MAPPINGS.items(), 
                                          desc="Processing categories"):
            category_path = self.FOOD101_PATH / category
            
            if not category_path.exists():
                print(f"⚠️  Category not found: {category}")
                continue
            
            stats["total_categories"] += 1
            
            # Find matching recipes
            matching_recipes = self.find_matching_recipes(category, search_terms)
            
            if len(matching_recipes) == 0:
                print(f"⚠️  No recipes found for: {category}")
                continue
            
            # Get images from this category
            image_files = list(category_path.glob("*.jpg"))
            random.shuffle(image_files)
            image_files = image_files[:self.max_images_per_category]
            
            category_pairs = 0
            
            for img_path in image_files:
                try:
                    # Load image
                    image = Image.open(img_path).convert("RGB")
                    stats["total_images"] += 1
                    
                    # Sample recipes for this image
                    sampled_recipes = matching_recipes.sample(
                        min(self.recipes_per_image, len(matching_recipes))
                    )
                    
                    for _, recipe in sampled_recipes.iterrows():
                        # Verify quality (optional)
                        verification = self.verify_pair_quality(
                            image, recipe['recipe_text'], category
                        )
                        
                        # Only keep high-quality pairs
                        if verification["quality_score"] >= 0.5:
                            pair = {
                                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                                "category": category,
                                "recipe_id": int(recipe['id']),
                                "recipe_name": recipe['name'],
                                "recipe_text": recipe['recipe_text'],
                                "ingredients": recipe['ingredients_parsed'],
                                "tags": recipe['tags_parsed'],
                                "search_terms": search_terms,
                                "quality_score": verification["quality_score"],
                            }
                            
                            if self.use_blip_verification:
                                pair["blip_caption"] = verification["blip_caption"]
                                pair["caption_match"] = verification["caption_match"]
                            
                            self.pairs.append(pair)
                            category_pairs += 1
                            stats["total_pairs"] += 1
                            stats["avg_quality"].append(verification["quality_score"])
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if category_pairs > 0:
                stats["successful_categories"] += 1
        
        # Print statistics
        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Categories processed: {stats['total_categories']}")
        print(f"Categories with pairs: {stats['successful_categories']}")
        print(f"Total images used: {stats['total_images']}")
        print(f"Total pairs created: {stats['total_pairs']}")
        if stats['avg_quality']:
            print(f"Average quality score: {np.mean(stats['avg_quality']):.3f}")
        print(f"{'='*60}\n")
        
        return stats
        
    def save_dataset(self):
        """Save dataset to JSON"""
        # Convert pairs to JSON-serializable format
        json_pairs = []
        for pair in self.pairs:
            json_pair = pair.copy()
            
            # Convert lists/arrays to plain Python lists
            if 'ingredients' in json_pair:
                json_pair['ingredients'] = list(json_pair['ingredients']) if isinstance(json_pair['ingredients'], (list, np.ndarray)) else []
            if 'tags' in json_pair:
                json_pair['tags'] = list(json_pair['tags']) if isinstance(json_pair['tags'], (list, np.ndarray)) else []
            
            # Convert numpy types to Python types
            for key, value in json_pair.items():
                if isinstance(value, np.integer):
                    json_pair[key] = int(value)
                elif isinstance(value, np.floating):
                    json_pair[key] = float(value)
                elif isinstance(value, np.ndarray):
                    json_pair[key] = value.tolist()
            
            json_pairs.append(json_pair)
        
        output_data = {
            "metadata": {
                "name": "Food-101-Recipe-Pairs",
                "description": "Image-text pairs created from Food-101 images and Food.com recipes",
                "version": "1.0",
                "num_pairs": len(self.pairs),
                "num_categories": len(set(p["category"] for p in self.pairs)),
                "created_by": "CEIA-VIT Recipe Search Project",
                "source_images": "Food-101 Dataset",
                "source_recipes": "Food.com (Kaggle)",
            },
            "pairs": json_pairs
        }
        
        # Save
        self.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.OUTPUT_PATH, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Dataset saved to: {self.OUTPUT_PATH}")
        print(f"  Size: {self.OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Also save a CSV for easy inspection
        csv_path = self.OUTPUT_PATH.with_suffix('.csv')
        pairs_df = pd.DataFrame(json_pairs)
        pairs_df.to_csv(csv_path, index=False)
        print(f"✓ CSV saved to: {csv_path}")
        
        return self.OUTPUT_PATH
    
    def show_sample_pairs(self, n: int = 5):
        """Display sample pairs for verification"""
        print(f"\n{'='*60}")
        print(f"SAMPLE PAIRS (showing {n})")
        print(f"{'='*60}\n")
        
        samples = random.sample(self.pairs, min(n, len(self.pairs)))
        
        for i, pair in enumerate(samples, 1):
            print(f"{i}. Category: {pair['category']}")
            print(f"   Recipe: {pair['recipe_name']}")
            print(f"   Image: {pair['image_path']}")
            if 'quality_score' in pair:
                print(f"   Quality: {pair['quality_score']:.2f}")
            if 'blip_caption' in pair:
                print(f"   BLIP: {pair['blip_caption']}")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Food-101-Recipe-Pairs dataset")
    parser.add_argument("--max-images", type=int, default=100, 
                       help="Max images per category")
    parser.add_argument("--recipes-per-image", type=int, default=3,
                       help="Recipes to pair with each image")
    parser.add_argument("--use-blip", action="store_true",
                       help="Use BLIP for quality verification (slower)")
    parser.add_argument("--show-samples", type=int, default=5,
                       help="Number of sample pairs to display")
    
    args = parser.parse_args()
    
    # Build dataset
    builder = Food101PairBuilder(
        max_images_per_category=args.max_images,
        recipes_per_image=args.recipes_per_image,
        use_blip_verification=args.use_blip
    )
    
    stats = builder.build_pairs()
    builder.save_dataset()
    builder.show_sample_pairs(n=args.show_samples)
    
    print("\n✓ Dataset creation complete!")
    print(f"\nNext steps:")
    print(f"1. Inspect the dataset: data/processed/food101_recipe_pairs.csv")
    print(f"2. Use for fine-tuning: python src/training/finetune_siglip.py")



if __name__ == "__main__":
    main()