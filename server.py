from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="🍲 Recipe Chatbot API", description="Suggest recipes based on ingredients")


# Define input model
class Query(BaseModel):
    query: str


# Load recipe dataset
df = pd.read_csv("indian_recipes.csv")

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for ingredients
print("✅ Generating recipe embeddings (this may take a minute)...")
df['embedding'] = df['Ingredients'].fillna('').apply(lambda x: model.encode(x, convert_to_tensor=True))
print("✅ Embeddings ready!")


@app.get("/")
def home():
    return {"message": "Recipe Chatbot API is running successfully!"}


@app.post("/search")
def search_recipes(query: Query):
    """Search recipes by ingredient similarity"""
    user_query = query.query
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarity between query and recipe ingredient embeddings
    df['similarity'] = df['embedding'].apply(lambda x: util.cos_sim(query_embedding, x).item())

    # Get top 3 most similar recipes
    top_results = df.sort_values(by='similarity', ascending=False).head(3)

    recipes = []
    for _, row in top_results.iterrows():
        recipes.append({
            "Recipe Name": row.get("Recipe Name", "Unknown"),
            "Ingredients": row.get("Ingredients", ""),
            "Instructions": row.get("Instructions", ""),
            "Similarity Score": round(row["similarity"], 3)
        })

    return {
        "query": user_query,
        "results": recipes if recipes else "No similar recipes found."
    }


if __name__ == "__main__":
    print("🚀 Starting Recipe Chatbot Server at http://127.0.0.1:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)






