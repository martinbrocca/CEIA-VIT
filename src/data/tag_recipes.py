# Lee archivo de recetas, enriquece el etiquetado y deja los resultados en un nuevo archivo

import pandas as pd
import ast
from pathlib import Path

# carne
MEAT_KEYWORDS = ['beef', 'pork', 'chicken', 'turkey', 'lamb', 'veal', 'bacon', 
                 'sausage', 'ham', 'meat', 'steak', 'rib', 'salami', 'pepperoni',
                 'prosciutto', 'chorizo', 'bison', 'venison', 'duck', 'goose']

# pescado y frutos de mar
SEAFOOD_KEYWORDS = ['fish', 'salmon', 'tuna', 'shrimp', 'crab', 'lobster', 
                    'scallop', 'clam', 'mussel', 'oyster', 'anchov', 'sardine',
                    'cod', 'haddock', 'halibut', 'tilapia', 'trout', 'bass',
                    'seafood', 'prawn', 'caviar']
# lácteos
DAIRY_KEYWORDS = ['milk', 'cream', 'cheese', 'butter', 'yogurt', 'sour cream',
                  'ice cream', 'whey', 'casein', 'ghee', 'buttermilk',
                  'condensed milk', 'evaporated milk', 'half-and-half',
                  'mascarpone', 'ricotta', 'parmesan', 'cheddar', 'mozzarella']

# huevo y derivados
EGG_KEYWORDS = ['egg', 'mayonnaise', 'mayo', 'egg white']

# derivados de trigo y cereales
GLUTEN_KEYWORDS = ['flour', 'wheat', 'barley', 'rye', 'bread', 'pasta', 
                   'couscous', 'seitan', 'cracker', 'breadcrumb', 'bun',
                   'tortilla', 'pita', 'noodle', 'spaghetti', 'macaroni',
                   'orzo', 'farro', 'graham', 'pretzel', 'wafer',
                   'graham cracker', 'seitan', 'vital wheat gluten']

def classify_dietary_from_ingredients(ingredients_list):
    """
    Clasifica restricciones de dieta basada en ingredientes.
    - Recibe una lista de ingredientes.
    - En base a esos ingredientes crea tags usando las listas personalizadas: SEAFOOD_KEYWORDS, etc.
    - Maneja casos especiales de bouillon/caldo/fondo:
        * Excluye las variantes vegetales/veggie/mushroom.
        * Si es ambiguo y el contexto indica vegan, asume que es vegetal.
    - Verifica si hay presencia de carne, excluyendo explícitamente sustitutos vegetarianos/veganos.
    - Detecta productos lácteos, huevos (incluyendo claras de huevo), gluten y mariscos.
    - Devuelve un array de etiquetas inferidas.
    """
    
    ingredients_str = ' '.join(ingredients_list).lower()
    
    # Verifica indicadores vegan/vegetarian explícitos
    has_vegan_indicator = any(
        'vegan' in item.lower() or 'vegetarian' in item.lower() or 'veggie' in item.lower()
        for item in ingredients_list
    )
    
    # Tratamiento especial para el caldo: asumir que es vegetal si no tiene un calificador de origen animal.
    def has_animal_broth():
        for item in ingredients_list:
            item_lower = item.lower()
            if any(keyword in item_lower for keyword in ['bouillon', 'broth', 'stock']):
                # Chequear si tiene un calificador animal
                if any(animal in item_lower for animal in 
                       ['chicken', 'beef', 'pork', 'turkey', 'fish', 'seafood']):
                    return True
                # If it's explicitly vegetable/mushroom, not animal
                if any(veg in item_lower for veg in 
                       ['vegetable', 'veggie', 'mushroom', 'vegetarian']):
                    return False
                # caldo ("broth") ambiguo. Siendo conservadores, asumimos que podría ser animal
                # En verdad, seamos optimistas si el contexto de la receta sugiere que es vegano
                if has_vegan_indicator:
                    return False
                # Otherwise assume might be animal
                return 'broth' in item_lower or 'stock' in item_lower
        return False
    
    # Verificar si contiene carne (excluyendo las alternativas vegetarianas).
    def has_meat_products():
        for item in ingredients_list:
            item_lower = item.lower()
            # Saltear si es explícitamente vegetarian/vegan
            if 'vegetarian' in item_lower or 'veggie' in item_lower or 'vegan' in item_lower:
                continue
            if any(keyword in item_lower for keyword in MEAT_KEYWORDS):
                return True
        return False
    
    has_meat = has_meat_products() or has_animal_broth()
    has_seafood = any(keyword in ingredients_str for keyword in SEAFOOD_KEYWORDS)
    
    # Lácteo, pero excluye alternativas veganas
    def has_dairy_products():
        for item in ingredients_list:
            item_lower = item.lower()
            if 'vegan' in item_lower or 'soy' in item_lower or 'almond' in item_lower or 'coconut milk' in item_lower:
                continue
            if any(keyword in item_lower for keyword in DAIRY_KEYWORDS):
                return True
        return False
    
    has_dairy = has_dairy_products()
    
    # Huevos (incluye clara de huevo)
    has_eggs = any(keyword in ingredients_str for keyword in EGG_KEYWORDS)
    
    # Gluten
    has_gluten = any(keyword in ingredients_str for keyword in GLUTEN_KEYWORDS)
    
    tags = []
    
    if not (has_meat or has_seafood or has_dairy or has_eggs):
        tags.append('vegan')
        tags.append('vegetarian')
    elif not (has_meat or has_seafood):
        tags.append('vegetarian')
    
    if not has_gluten:
        tags.append('gluten_free')
    if not has_dairy:
        tags.append('dairy_free')
    if not has_eggs:
        tags.append('egg_free')
    
    return tags


def parse_list_field(field):
    try:
        return ast.literal_eval(field)
    except:
        return []

def to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            x = x.strip("[]")
            return [t.strip() for t in x.split(",") if t.strip()]
    return []


if __name__ == "__main__":
    
    # Path a los datos
    data_path = Path("../../data/raw/food-com")
    
    
    recetas_raw = f'{data_path}/RAW_recipes.csv'
    print(f"Leyendo archivo de recetas {recetas_raw} ...")

    recipes_raw = pd.read_csv(recetas_raw)    
    print(f"Tamaño del dataset: {recipes_raw.shape[0]}.")

    print(f"Parseando los ingredientes ...")
    recipes_raw['ingredients_parsed'] = recipes_raw['ingredients'].apply(parse_list_field)

    print(f"Parseando las etiquetas...")
    recipes_raw['tags_parsed'] = recipes_raw['tags'].apply(parse_list_field)

    tagged_recipes = recipes_raw.copy()

    print(f"Etiquetando las recetas ...")
    tagged_recipes['inferred_dietary'] = tagged_recipes['ingredients_parsed'].apply(classify_dietary_from_ingredients)
    
    print(f"Unificando las etiquetas ...")
    tagged_recipes['tags'] = tagged_recipes['tags'].apply(to_list)
    tagged_recipes['tags_with_quotes'] = tagged_recipes['inferred_dietary'].apply(to_list)
    tagged_recipes['tags_combined'] = tagged_recipes.apply(
        lambda row: row['tags_with_quotes'] + row['tags'], axis=1)
    tagged_recipes['tags_combined'] = tagged_recipes['tags_combined'].apply(repr)
    
    # exporta
    columns_to_keep = ['name', 'id', 'minutes', 'contributor_id', 'submitted',
                       'tags_combined', 'nutrition', 'n_steps', 'steps',
                       'description', 'ingredients', 'n_ingredients']
    tagged_recipes = tagged_recipes[columns_to_keep]
    tagged_recipes = tagged_recipes.rename(columns={'tags_combined': 'tags'})
    
    recipes_tag_raw = f'{data_path}/RAW_recipes_tag.csv'

    tagged_recipes.to_csv(recipes_tag_raw, index=False)
    print(f"\nArchivo de recetas {recipes_tag_raw} generado.")
