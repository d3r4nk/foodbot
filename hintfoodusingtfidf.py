import tkinter as tk
from tkinter import messagebox, Scrollbar, Text
from PIL import Image, ImageTk
import requests
from io import BytesIO
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re  # For HTML tag removal
# Function definitions (same as above)
def preprocess_data(df):
    """ Preprocesses recipe data to extract titles and ingredients. """
    df['soup'] = df['title'] + ' ' + df['ingredients'] + ' ' + df['instructions']
    return df
def build_tfidf_recommender(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['soup'])
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix

def recommend(df, sim_matrix, title):
    """ Returns similar recipes based on title, ingredients, and instructions. """
    indices = pd.Series(df.index, index=df['title'])
    idx = indices.get(title)
    
    if idx is None:
        return []

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]  # Get top 5 similar recipes
    result_indices = [i[0] for i in scores]
    
    return df.iloc[result_indices]

# Function to clean HTML tags from text
def clean_html(text):
    """Remove HTML tags and extra spaces from a string."""
    clean_text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Replace multiple spaces with one space
    return clean_text.strip()  # Remove leading/trailing spaces

# Load data and preprocess
df = pd.read_csv('recipes.csv')  # Load your recipe data from CSV
df = preprocess_data(df)
df['soup'] = df['soup'].fillna('').astype(str)

# Build the recommender system
sim_matrix = build_tfidf_recommender(df)

# Tkinter GUI setup
class RecipeRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Recipe Recommender with tfidf")
        self.root.geometry("600x600")
        
        self.label = tk.Label(root, text="Enter recipe title:", font=("Arial", 14))
        self.label.pack(pady=10)
        
        self.title_entry = tk.Entry(root, font=("Arial", 12))
        self.title_entry.pack(pady=10)
        
        self.search_button = tk.Button(root, text="Search", font=("Arial", 12), command=self.search_recipes)
        self.search_button.pack(pady=10)
        
        self.results_listbox = tk.Listbox(root, width=50, height=10, font=("Arial", 12))
        self.results_listbox.pack(pady=10)
        self.results_listbox.bind("<Double-1>", self.show_recipe_details)
        
        self.details_frame = tk.Frame(root)
        self.details_frame.pack(pady=10)
        
        self.recipe_image_label = tk.Label(self.details_frame)
        self.recipe_image_label.grid(row=0, column=0, rowspan=3, padx=10)
        
        self.recipe_title_label = tk.Label(self.details_frame, font=("Arial", 18, "bold"))
        self.recipe_title_label.grid(row=0, column=1, sticky="w")
        
        self.time_label = tk.Label(self.details_frame, font=("Arial", 12))
        self.time_label.grid(row=1, column=1, sticky="w")
        
        self.ingredients_label = tk.Label(self.details_frame, text="Ingredients:", font=("Arial", 14, "bold"))
        self.ingredients_label.grid(row=3, column=0, sticky="w", pady=(10, 5))
        
        # Ingredients Text area
        self.ingredients_text = Text(self.details_frame, height=5, width=50, wrap="word", font=("Arial", 12), state="disabled")
        self.ingredients_text.grid(row=4, column=0, columnspan=2)
        
        # Scrollbar for Ingredients
        self.ingredients_scrollbar = Scrollbar(self.details_frame, command=self.ingredients_text.yview)
        self.ingredients_scrollbar.grid(row=4, column=2, sticky="ns")
        self.ingredients_text.config(yscrollcommand=self.ingredients_scrollbar.set)
        
        self.instructions_label = tk.Label(self.details_frame, text="Instructions:", font=("Arial", 14, "bold"))
        self.instructions_label.grid(row=5, column=0, sticky="w", pady=(10, 5))
        
        # Instructions Text area
        self.instructions_text = Text(self.details_frame, height=5, width=50, wrap="word", font=("Arial", 12), state="disabled")
        self.instructions_text.grid(row=6, column=0, columnspan=2)

        # Scrollbar for Instructions
        self.instructions_scrollbar = Scrollbar(self.details_frame, command=self.instructions_text.yview)
        self.instructions_scrollbar.grid(row=6, column=2, sticky="ns")
        self.instructions_text.config(yscrollcommand=self.instructions_scrollbar.set)

    def search_recipes(self):
        """ Search and display similar recipes based on the title. """
        title = self.title_entry.get()
        if not title:
            messagebox.showwarning("Input Error", "Please enter a recipe title.")
            return

        recommended_recipes = recommend(df, sim_matrix, title)
        
        if recommended_recipes.empty:
            messagebox.showinfo("No Results", f"No similar recipes found for '{title}'.")
            return
        
        # Clear previous results
        self.results_listbox.delete(0, tk.END)
        
        # Add recommended recipe titles to listbox
        for index, row in recommended_recipes.iterrows():
            self.results_listbox.insert(tk.END, row['title'])

    def show_recipe_details(self, event):
        """ Show details of the selected recipe when clicked. """
        selected_index = self.results_listbox.curselection()
        if not selected_index:
            return
        
        selected_recipe = self.results_listbox.get(selected_index)
        recipe_details = df[df['title'] == selected_recipe].iloc[0]
        
        # Update labels and text widgets with recipe details
        self.recipe_title_label.config(text=recipe_details['title'])
        self.time_label.config(text=f"Preparation Time: {recipe_details['readyInMinutes']} minutes")
        
        # Load and display recipe image
        image_url = recipe_details['image']
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.recipe_image_label.config(image=img_tk)
            self.recipe_image_label.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            self.recipe_image_label.config(image="")
            messagebox.showwarning("Image Error", "Could not load recipe image.")
        
        # Clean and display ingredients and instructions in Text widgets
        cleaned_ingredients = clean_html(recipe_details['ingredients'])
        cleaned_instructions = clean_html(recipe_details['instructions'])

        self.ingredients_text.config(state="normal")
        self.ingredients_text.delete(1.0, tk.END)
        self.ingredients_text.insert(tk.END, cleaned_ingredients)
        self.ingredients_text.config(state="disabled")
        
        self.instructions_text.config(state="normal")
        self.instructions_text.delete(1.0, tk.END)
        self.instructions_text.insert(tk.END, cleaned_instructions)
        self.instructions_text.config(state="disabled")

# Set up Tkinter root window and application
root = tk.Tk()
app = RecipeRecommenderApp(root)

# Run the GUI
root.mainloop()
