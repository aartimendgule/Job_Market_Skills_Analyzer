# ================================
# JOB MARKET SKILLS ANALYZER
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# NLTK Imports
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# -------------------------------
# DATABASE FUNCTIONS
# -------------------------------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

def create_table():
    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            name TEXT,
            email TEXT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()

def add_user(name, email, username, password):
    c.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
              (name, email, username, password))
    conn.commit()

def login_user(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, password))
    return c.fetchone()

create_table()

# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -------------------------------
# LOAD DATASET
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("jobs.csv")
    df["Cleaned_Description"] = df["Description"].apply(clean_text)
    return df

# -------------------------------
# STREAMLIT SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------------------
# APP TITLE
# -------------------------------
st.set_page_config(page_title="Job Market Skills Analyzer", layout="wide")
st.title("📊 Job Market Skills Analyzer")

# -------------------------------
# SIDEBAR MENU
# -------------------------------
menu = ["Home", "Register", "Login"]

if st.session_state.logged_in:
    menu = [
        "Dashboard",
        "Upload Dataset",
        "Skill Analysis",
        "Trending Skills",
        "Word Cloud",
        "Job Search",
        "Logout"
    ]

choice = st.sidebar.selectbox("Navigation", menu)

# -------------------------------
# HOME PAGE
# -------------------------------
if choice == "Home":
    st.subheader("Welcome to the Job Market Skills Analyzer System")
    st.write("""
    This system analyzes job descriptions using **Information Retrieval**
    and **Text Mining** techniques to identify in-demand skills.
    """)

# -------------------------------
# REGISTRATION PAGE
# -------------------------------
elif choice == "Register":
    st.subheader("📝 User Registration")
    name = st.text_input("Full Name")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        try:
            add_user(name, email, username, password)
            st.success("Registration Successful!")
        except:
            st.error("Username already exists!")

# -------------------------------
# LOGIN PAGE
# -------------------------------
elif choice == "Login":
    st.subheader("🔐 User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid Username or Password")

# -------------------------------
# DASHBOARD
# -------------------------------
elif choice == "Dashboard":
    st.subheader("📊 Dashboard")
    st.write(f"Welcome, **{st.session_state.username}**!")
    st.info("Use the sidebar to explore system features.")

# -------------------------------
# UPLOAD DATASET
# -------------------------------
elif choice == "Upload Dataset":
    st.subheader("📂 Upload Job Dataset")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["Cleaned_Description"] = df["Description"].apply(clean_text)
        st.session_state.df = df
        st.success("Dataset Uploaded Successfully!")
        st.dataframe(df.head())
    else:
        df = load_data()
        st.session_state.df = df
        st.info("Default dataset loaded.")
        st.dataframe(df.head())

# -------------------------------
# SKILL ANALYSIS
# -------------------------------
elif choice == "Skill Analysis":
    st.subheader("🛠 Extracted Skills")

    df = st.session_state.get("df", load_data())

    skills_list = [
        'python', 'java', 'c++', 'machine learning', 'deep learning',
        'data science', 'sql', 'excel', 'tableau', 'power bi',
        'aws', 'azure', 'nlp', 'tensorflow', 'pandas', 'numpy',
        'statistics', 'html', 'css', 'javascript', 'docker', 'kubernetes'
    ]

    def extract_skills(text):
        return [skill for skill in skills_list if skill in text]

    df["Extracted Skills"] = df["Cleaned_Description"].apply(extract_skills)
    st.dataframe(df[["Job Title", "Extracted Skills"]])

# -------------------------------
# TRENDING SKILLS
# -------------------------------
elif choice == "Trending Skills":
    st.subheader("📈 Top In-Demand Skills")

    df = st.session_state.get("df", load_data())

    skills = df["Cleaned_Description"].str.split().explode()
    skill_counts = skills.value_counts().head(10)

    st.bar_chart(skill_counts)

# -------------------------------
# WORD CLOUD
# -------------------------------
elif choice == "Word Cloud":
    st.subheader("☁ Word Cloud of Skills")

    df = st.session_state.get("df", load_data())
    text = " ".join(df["Cleaned_Description"])

    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text)

    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")
    st.pyplot(fig)

# -------------------------------
# JOB SEARCH USING IR
# -------------------------------
elif choice == "Job Search":
    st.subheader("🔍 Job Search Using Information Retrieval")

    df = st.session_state.get("df", load_data())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Cleaned_Description"])

    query = st.text_input("Enter Skill or Job Role")

    if query:
        query_clean = clean_text(query)
        query_vec = vectorizer.transform([query_clean])
        similarity = cosine_similarity(query_vec, tfidf_matrix)
        indices = similarity.argsort()[0][-5:][::-1]

        results = df.iloc[indices][
            ["Job Title", "Company", "Location"]
        ]

        st.write("### Top Matching Jobs")
        st.dataframe(results)

# -------------------------------
# LOGOUT
# -------------------------------
elif choice == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("Logged out successfully!")
    st.rerun()