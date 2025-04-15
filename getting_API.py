import requests
import csv
import time
import os

API_KEY = "07bdae4c141b42e5b706796bd28ba02f"
BASE_URL = "https://api.spoonacular.com"
OUTPUT_FILE = "recipes.csv"
NEW_RECIPES_TO_ADD = 150

def get_random_recipes(number=10):
    url = f"{BASE_URL}/recipes/random"
    params = {
        "number": number,
        "includeNutrition": False,
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["recipes"]

def extract_recipe_info(recipe):
    title = recipe.get("title", "N/A")
    image = recipe.get("image", "")
    ready_in_minutes = recipe.get("readyInMinutes", 0)
    instructions = recipe.get("instructions", "Không có hướng dẫn.")
    ingredients = [ing.get("original", "") for ing in recipe.get("extendedIngredients", [])]

    return {
        "id": recipe["id"],
        "title": title,
        "image": image,
        "readyInMinutes": ready_in_minutes,
        "instructions": instructions,
        "ingredients": "; ".join(ingredients)
    }

def read_existing_ids(filepath):
    existing_ids = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(int(row["id"]))
    return existing_ids

def fetch_unique_recipes_to_add(existing_ids, target_count):
    new_recipes = {}
    print(f"🔄 Đang tải thêm {target_count} món ăn KHÔNG trùng với file cũ...")

    while len(new_recipes) < target_count:
        try:
            recipes = get_random_recipes(number=10)
        except Exception as e:
            print(f"Lỗi khi gọi API: {e}")
            time.sleep(5)
            continue

        for recipe in recipes:
            rid = recipe["id"]
            if rid not in existing_ids and rid not in new_recipes:
                new_recipes[rid] = extract_recipe_info(recipe)

        print(f"Đã thu thập: {len(new_recipes)}/{target_count} món mới")

    return list(new_recipes.values())

def append_to_csv(data, output_file):
    file_exists = os.path.exists(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "title", "image", "readyInMinutes", "instructions", "ingredients"])
        for item in data:
            writer.writerow([
                item["id"],
                item["title"],
                item["image"],
                item["readyInMinutes"],
                item["instructions"],
                item["ingredients"]
            ])

if __name__ == "__main__":
    existing_ids = read_existing_ids(OUTPUT_FILE)
    new_recipes = fetch_unique_recipes_to_add(existing_ids, target_count=NEW_RECIPES_TO_ADD)
    append_to_csv(new_recipes, OUTPUT_FILE)
    print(f"\nĐã thêm {len(new_recipes)} món ăn vào file '{OUTPUT_FILE}'")
