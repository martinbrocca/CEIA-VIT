# src/substitutions/rules.py
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

SUBSTITUTION_DB_PATH = Path(__file__).parent / "substitution_db.json"

class SubstitutionEngine:
    """Find ingredient substitutions based on dietary needs"""
    
    def __init__(self):
        self.substitutions = self._load_substitutions()
    
    def _load_substitutions(self) -> Dict:
        """Load substitution database"""
        with open(SUBSTITUTION_DB_PATH, 'r') as f:
            return json.load(f)
    
    def find_substitutions(
        self, 
        ingredients: List[str], 
        dietary_preferences: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Find substitutions for ingredients based on dietary preferences
        
        Args:
            ingredients: List of ingredients in recipe
            dietary_preferences: List of dietary needs (e.g., ['vegan', 'gluten-free'])
        
        Returns:
            Dict mapping ingredients to their substitutions by category
        """
        if not dietary_preferences:
            dietary_preferences = ['general']
        
        results = {}
        
        for ingredient in ingredients:
            ingredient_lower = ingredient.lower().strip()
            
            # Check if we have substitutions for this ingredient
            for key in self.substitutions.keys():
                if key in ingredient_lower:
                    subs_for_ingredient = {}
                    
                    # Get substitutions for each dietary preference
                    for pref in dietary_preferences:
                        pref_lower = pref.lower()
                        if pref_lower in self.substitutions[key]:
                            subs_for_ingredient[pref] = self.substitutions[key][pref_lower]
                    
                    # Also add general substitutions
                    if 'general' in self.substitutions[key]:
                        subs_for_ingredient['general'] = self.substitutions[key]['general']
                    
                    if subs_for_ingredient:
                        results[ingredient] = subs_for_ingredient
                    break
        
        return results
    
    def suggest_for_recipe(
        self, 
        recipe_ingredients: List[str], 
        dietary_filters: List[str]
    ) -> Dict:
        """
        Suggest substitutions for an entire recipe
        
        Returns dict with:
        - substitutable_count: Number of ingredients that can be substituted
        - substitutions: Dict of ingredient -> substitutions
        """
        subs = self.find_substitutions(recipe_ingredients, dietary_filters)
        
        return {
            "substitutable_count": len(subs),
            "total_ingredients": len(recipe_ingredients),
            "substitutions": subs
        }


def test_substitutions():
    """Test substitution engine"""
    engine = SubstitutionEngine()
    
    # Test recipe
    ingredients = [
        "butter",
        "milk",
        "eggs",
        "flour",
        "sugar",
        "vanilla extract"
    ]
    
    print("Test Recipe Ingredients:")
    for ing in ingredients:
        print(f"  - {ing}")
    
    print("\n" + "="*50)
    print("Vegan Substitutions:")
    result = engine.suggest_for_recipe(ingredients, ['vegan'])
    print(f"Can substitute {result['substitutable_count']}/{result['total_ingredients']} ingredients")
    
    for ing, subs in result['substitutions'].items():
        print(f"\n{ing}:")
        for category, options in subs.items():
            print(f"  {category}: {', '.join(options)}")
    
    print("\n" + "="*50)
    print("Gluten-Free Substitutions:")
    result = engine.suggest_for_recipe(ingredients, ['gluten-free'])
    print(f"Can substitute {result['substitutable_count']}/{result['total_ingredients']} ingredients")
    
    for ing, subs in result['substitutions'].items():
        print(f"\n{ing}:")
        for category, options in subs.items():
            print(f"  {category}: {', '.join(options)}")

if __name__ == "__main__":
    test_substitutions()