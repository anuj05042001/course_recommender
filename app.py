import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import os

# --- 1. NLTK Setup (Optimized for Deployment) ---
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for res in resources:
        nltk.download(res, quiet=True)

download_nltk_resources()

# Initialize global tools after download
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- 2. Sample Data ---
COURSE_DATA = {
    'course_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'title': [
        'Introduction to Python Programming', 'Data Science Fundamentals',
        'Machine Learning Essentials', 'Web Development with Django',
        'Deep Learning with TensorFlow', 'Natural Language Processing',
        'Data Visualization with Python', 'Cloud Computing for Beginners',
        'SQL for Data Analysis', 'Big Data Analytics'
    ],
    'description': [
        'Learn Python basics, data structures, and programming fundamentals',
        'Foundational concepts in data science and statistical analysis',
        'Key machine learning algorithms and practical implementation',
        'Build web applications using Django framework and Python',
        'Neural networks and deep learning techniques with TensorFlow',
        'Process and analyze human language data using NLP techniques',
        'Create effective visualizations using Matplotlib and Seaborn',
        'Introduction to cloud services and deployment strategies',
        'Database management and SQL queries for data professionals',
        'Handling large datasets with distributed computing frameworks'
    ],
    'category': [
        'Programming', 'Data Science', 'AI', 'Web Development', 'AI',
        'AI', 'Data Science', 'Cloud', 'Database', 'Big Data'
    ],
    'difficulty': [
        'Beginner', 'Beginner', 'Intermediate', 'Intermediate', 
        'Advanced', 'Intermediate', 'Intermediate', 'Beginner', 
        'Beginner', 'Advanced'
    ]
}

# --- 3. Preprocessing Logic ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- 4. Recommender Engine (Cached) ---
class CourseRecommender:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self._prepare_data()
        
    def _prepare_data(self):
        self.df['combined_features'] = self.df['title'] + ' ' + self.df['description'] + ' ' + self.df['category']
        self.df['processed_features'] = self.df['combined_features'].apply(preprocess_text)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['processed_features'])
        
    def recommend_courses(self, input_text, top_n=3):
        processed_input = preprocess_text(input_text)
        input_vector = self.tfidf_vectorizer.transform([processed_input])
        cosine_sim = cosine_similarity(input_vector, self.tfidf_matrix)
        similar_indices = cosine_sim.argsort()[0][-top_n:][::-1]
        return self.df.iloc[similar_indices]

@st.cache_resource
def load_recommender():
    return CourseRecommender(COURSE_DATA)

# --- 5. Streamlit UI ---
def main():
    st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")
    
    # Load model once
    recommender = load_recommender()

    # CSS for styling
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .course-card {
            background-color: white; padding: 20px; border-radius: 10px;
            border-left: 5px solid #2e86c1; margin-bottom: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Smart Course Recommender")
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("What do you want to learn?")
        user_input = st.text_area("Enter keywords (e.g., 'I want to build neural networks')", 
                                  placeholder="Type here...", height=150)
        predict_button = st.button("Find Courses", use_container_width=True)

    with col2:
        st.subheader("Recommended for You")
        if predict_button and user_input:
            results = recommender.recommend_courses(user_input)
            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="course-card">
                    <h3>{row['title']}</h3>
                    <p><b>Category:</b> {row['category']} | <b>Level:</b> {row['difficulty']}</p>
                    <p>{row['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Results will appear here after you click 'Find Courses'.")

if __name__ == "__main__":
    main()
