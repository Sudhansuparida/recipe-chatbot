Recipe Chatbot (Ingredient-based Recipe Finder)

Overview:
This project is an AI-powered Recipe Chatbot built using FastAPI and Sentence Transformers. It helps users find the most relevant Indian recipes based on ingredients they enter. For example, if a user enters "Egg, Onions", the chatbot will suggest recipes that best match those ingredients.

Features:

Accepts ingredient-based input from users

Finds and ranks recipes using semantic similarity

Uses pre-trained SentenceTransformer model (all-MiniLM-L6-v2)

Fast and scalable API built with FastAPI

Can be integrated with any frontend (HTML/React/Vue, etc.)

Requirements:
Make sure you have Python 3.9+ installed.
Then install dependencies using:
pip install fastapi uvicorn pandas sentence-transformers

Folder Structure:
recipe-chatbot/
│
├── server.py → Main FastAPI app
├── test_request.py → Test script to check the API
├── indian_recipes.csv → Dataset containing recipe info
└── README.txt → Project documentation

Dataset:
The indian_recipes.csv file should include the following columns:
Recipe Name, Ingredients, Instructions
Example:
Masala Dosa, Rice, Urad Dal, Potato, Onion, Mustard Seeds, Prepare dosa batter from rice and dal...

How to Run:

Start the FastAPI server:
uvicorn server:app --reload --host 127.0.0.1 --port 8000

You’ll see:
✅ Generating recipe embeddings (this may take a minute)...
✅ Embeddings ready!
🚀 Running on http://127.0.0.1:8000

Open your browser and visit:
http://127.0.0.1:8000

Testing the API:
You can test using the provided script:
python test_request.py

Expected Output Example:
{
"query": "Egg, Onion",
"results": [
{
"Recipe Name": "Egg Curry",
"Ingredients": "Eggs, Onions, Tomato, Spices",
"Instructions": "Boil eggs, prepare onion-tomato gravy, add spices and mix well.",
"Similarity Score": 0.87
}
]
}

How It Works:

The model encodes each recipe’s ingredients into numerical vectors (embeddings).

When a user enters ingredients, their input is also converted into an embedding.

The app calculates the cosine similarity between the input and all recipes.

It returns the top 3 most similar recipes.

Author:
Sudhansu Parida
Data Science Learner | AIML Enthusiast | Python Developer