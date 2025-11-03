import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 🌟 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Recipe Chatbot", page_icon="🍲", layout="centered")

# -------------------------------
# 🌟 HEADER
# -------------------------------
st.title("🍲 Indian Recipe Chatbot")
st.write("Enter ingredients you have, and I’ll suggest delicious Indian recipes for you!")


# -------------------------------
# 🌟 LOAD MODEL & DATA
# -------------------------------
@st.cache_resource
def load_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


@st.cache_data
def load_data():
    df = pd.read_csv("indian_recipes.csv")
    df['Ingredients'] = df['Ingredients'].fillna('')
    return df


model = load_model()
df = load_data()


# -------------------------------
# 🌟 GENERATE EMBEDDINGS (once)
# -------------------------------
@st.cache_resource
def generate_embeddings(df):
    st.info("Generating recipe embeddings... please wait ⏳")
    df['embedding'] = df['Ingredients'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    st.success("✅ Recipe embeddings ready!")
    return df


df = generate_embeddings(df)


# -------------------------------
# 🌟 SEARCH FUNCTION
# -------------------------------
def search_recipes(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    df['similarity'] = df['embedding'].apply(lambda x: util.cos_sim(query_embedding, x).item())
    top_results = df.sort_values(by='similarity', ascending=False).head(3)
    return top_results


# -------------------------------
# 🌟 USER INPUT SECTION
# -------------------------------
user_input = st.text_input("🔍 Enter ingredients (e.g., 'Egg, Onion, Tomato')")

if st.button("Find Recipes"):
    if user_input.strip():
        with st.spinner("Finding the best recipes for you..."):
            results = search_recipes(user_input)

        st.subheader("🍛 Top Recipe Suggestions:")
        for i, row in results.iterrows():
            st.markdown(f"### 🥘 {row['Recipe Name']}")
            st.write(f"**Ingredients:** {row['Ingredients']}")
            st.write(f"**Instructions:** {row['Instructions']}")
            st.write(f"**Similarity Score:** {round(row['similarity'], 3)}")
            st.divider()
    else:
        st.warning("Please enter some ingredients to search.")

# -------------------------------
# 🌟 FOOTER
# -------------------------------
st.markdown("---")
st.caption("👨‍🍳 Developed by Sudhansu Parida | Powered by Sentence Transformers & Streamlit")
