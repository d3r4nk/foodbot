import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import requests
from io import BytesIO
import re

# Load data
df = pd.read_csv(r"D:\Data Mining\recipes.csv")

# Main window
root = tk.Tk()
root.title("What to Eat Today?")
root.geometry("600x750")

# Widgets
title_label = ttk.Label(root, text="", font=("Arial", 18, "bold"), wraplength=500, justify="center")
title_label.pack(pady=20)

image_label = tk.Label(root)
image_label.pack(pady=10)

info_text = tk.Text(root, wrap="word", height=20, width=70)
info_text.pack(pady=10)

def strip_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def show_random_recipe():
    recipe = df.sample(1).iloc[0]

    title_label.config(text=recipe['title'])

    try:
        img_url = recipe['image']
        response = requests.get(img_url) # Get the url of the image
        img_data = Image.open(BytesIO(response.content))
        img_data = img_data.resize((300, 200))
        photo = ImageTk.PhotoImage(img_data)
        image_label.config(image=photo) 
        image_label.image = photo
    except:
        image_label.config(image=None)
        image_label.image = None

    ready_time = recipe.get('readyInMinutes', 0)
    calories = recipe.get('calories', 'Unknown')
    try:
        calories = int(float(calories))
    except:
        calories = "Unknown"

    instructions_raw = recipe.get('instructions', 'No instructions available.')
    instructions = strip_html_tags(instructions_raw)

    ingredients_str = recipe.get('ingredients', '')
    ingredients = [i.strip() for i in ingredients_str.split(";") if i.strip()]

    content = f"🕒 Preparation Time: {ready_time} minutes\n"
    content += f"🔥 Calories: {calories} kcal\n"
    content += "\n📋 Ingredients:\n" + "\n".join(f"- {i}" for i in ingredients[:10])
    content += f"\n\n📖 Instructions:\n{instructions[:1000]}..."

    info_text.delete(1.0, tk.END)
    info_text.insert(tk.END, content)

# Button to get another recipe
random_button = ttk.Button(root, text="🎲 Suggest Another Recipe", command=show_random_recipe)
random_button.pack(pady=20)

# Load first recipe
show_random_recipe()

# Start GUI
root.mainloop()
