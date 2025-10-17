🧠 Overview

This project is a Vietnamese recipe recommendation system that combines Natural Language Processing (NLP) and Machine Learning to suggest dishes based on user input.
It includes five different recommendation models and an interactive chatbot capable of understanding Vietnamese queries, recognizing ingredients, and recommending suitable recipes.

Users can explore and compare the performance of all five models through a Flask web interface and chat with an intelligent assistant for personalized suggestions.
The project also integrates with the Spoonacular API to automatically expand and update the recipe dataset.
🌟 Features

🥗 Five Recommendation Models:
Implemented and compared five algorithms — TF-IDF, Doc2Vec, BM25, PyVi, and VnCoreNLP — to evaluate their performance in Vietnamese recipe recommendation.

🤖 Chatbot Integration:
Developed an intelligent chatbot (endpoint /chatbot) that understands natural Vietnamese language, detects ingredients, cooking time, and user intent to recommend recipes dynamically.

⚙️ Model Comparison Interface:
A dedicated web page (endpoint /) that allows users to visually compare the outputs and similarities across all five models.

⚡ Optimized Model Training:
Introduced a caching mechanism to store similarity matrices, dramatically reducing model rebuilding and loading time.

🧄 Expanded Ingredient Database:
Enhanced the recipe dataset with 200+ common Vietnamese ingredients for more detailed and accurate search results.

🍳 Automated Data Collection:
Integrated with the Spoonacular API to continuously fetch and append unique new recipes into recipes.csv.
🛠️ Technologies Used

Backend & Frameworks:

Python 3

Flask

Machine Learning & NLP:

Gensim (Doc2Vec)

Scikit-learn (TF-IDF, cosine similarity)

BM25 (Okapi BM25 implementation)

PyVi (Vietnamese tokenizer)

VnCoreNLP (Vietnamese linguistic processing)

Data Handling & Storage:

Pandas

NumPy

Pickle (for model caching)

CSV (recipe storage)

External API:

Spoonacular Recipe API

Others:

Regex, NLTK, Unicodedata for text preprocessing

HTML/CSS/Bootstrap for front-end templates

📸 Screenshot
