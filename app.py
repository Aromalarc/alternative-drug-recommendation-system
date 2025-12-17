

import streamlit as st
import pandas as pd
import re
import joblib

st.set_page_config(page_title="Drug Alternative Recommendation", layout="centered")

st.title("Drug Alternative Recommendation System")
st.write("Suggests alternative medicines based on chemical composition, dosage, and price.")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("A_Z_medicines_dataset_of_India.csv")

# Ensure price is numeric
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# -----------------------------
# CLEAN COMPOSITION
# -----------------------------
def clean_composition(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[()\[\],]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["composition"] = (
    df["short_composition1"].fillna("") + " " +
    df["short_composition2"].fillna("")
)

df["clean_composition"] = df["composition"].apply(clean_composition)
df = df[df["clean_composition"] != ""].reset_index(drop=True)

# -----------------------------
# CLEAN NAME
# -----------------------------
df["clean_name"] = df["name"].str.lower().str.strip()

# -----------------------------
# EXTRACT DOSAGE
# -----------------------------
def extract_dosage(text):
    return " ".join(
        re.findall(r"\d+\s*mg|\d+\s*ml|\d+\s*mcg", text.lower())
    )

df["dosage"] = df["name"].apply(extract_dosage)

# -----------------------------
# CREATE DRUG GROUP
# -----------------------------
df["drug_group"] = df["clean_composition"].astype("category").cat.codes

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
model = joblib.load("drug_recommendation_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# RECOMMENDATION FUNCTION
# -----------------------------
def recommend_alternatives(medicine_name, top_n=10):
    key = medicine_name.lower().strip()

    matches = df[df["clean_name"].str.contains(key, regex=False)]
    if matches.empty:
        return None

    row = matches.iloc[0]
    comp = row["clean_composition"]
    dosage = row["dosage"]

    vec = tfidf.transform([comp])
    pred_group = model.predict(vec)[0]
    real_group = label_encoder.inverse_transform([pred_group])[0]

    alternatives = df[
        (df["drug_group"] == real_group) &
        (df["dosage"] == dosage) &
        (df["clean_name"] != row["clean_name"])
    ][["name", "clean_composition", "price"]]

    if alternatives.empty:
        return None

    # ✅ SORT BY PRICE (CHEAPEST FIRST)
    alternatives = alternatives.sort_values(by="price", ascending=True)

    return alternatives.head(top_n).reset_index(drop=True)

# -----------------------------
# STREAMLIT UI
# -----------------------------

# custom css
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max_width: 1200px;
        margin: 0 auto;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .result-title {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
    }
    .result-price {
        font-size: 18px;
        font-weight: bold;
        color: #e74c3c;
    }
    .result-comp {
        color: #7f8c8d;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/883/883407.png", width=100)
    st.title("💊 Drug Recommender")
    st.markdown("Find cheaper alternatives with similar composition.")
    
    st.header("Search Options")
    medicine_input = st.text_input(
        "Enter medicine name",
        placeholder="e.g. Ascoril LS Syrup"
    ).strip()
    
    top_n = st.slider(
        "Max alternatives to show",
        min_value=1,
        max_value=20,
        value=5
    )
    
    search_btn = st.button("🔍 Find Alternatives", type="primary")
    
    st.divider()
    st.markdown("### About")
    st.info("This tool uses ML to find drugs with similar chemical structures and dosages, sorted by price.")

# Main Area
st.title("🏥 drug Alternative Recommendation System")
st.markdown("#### Save money on your prescriptions without compromising quality.")
st.divider()

if search_btn:
    if not medicine_input:
        st.warning("⚠️ Please enter a medicine name to search.")
    else:
        with st.spinner(f"Searching for alternatives to '{medicine_input}'..."):
            results = recommend_alternatives(medicine_input, top_n)

        if results is None or results.empty:
            st.error(f"❌ No alternatives found for '{medicine_input}'. Please check the spelling or try another medicine.")
        else:
            st.success(f"✅ Found {len(results)} alternatives for **{medicine_input}**:")
            
            for i, row in results.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div class="result-card">
                            <div class="result-title">💊 {row['name']}</div>
                            <div class="result-comp">🧪 {row['clean_composition']}</div>
                            <br>
                            <div class="result-price">💰 Price: ₹{row['price']}</div>
                        </div>
                    """, unsafe_allow_html=True)