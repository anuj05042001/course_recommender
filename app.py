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

# --- 1. NLTK Setup (Optimized for Cloud) ---
@st.cache_resource
def download_resources():
    # Adding 'punkt_tab' as it is required by newer NLTK versions
    for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(res, quiet=True)

download_resources()

# Initialize tools once
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# --- 2. Dataset ---
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

# --- 3. Preprocessing ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in STOP_WORDS]
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- 4. Recommender Engine ---
class CourseRecommender:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self._prepare_data()
        
    def _prepare_data(self):
        # Create a single string of features for the ML model to read
        self.df['combined_features'] = (
            self.df['title'] + ' ' + 
            self.df['description'] + ' ' + 
            self.df['category']
        )
        self.df['processed_features'] = self.df['combined_features'].apply(preprocess_text)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_features'])
        
    def get_recommendations(self, query, top_n=3):
        processed_query = preprocess_text(query)
        query_vec = self.vectorizer.transform([processed_query])
        similarity = cosine_similarity(query_vec, self.tfidf_matrix)
        indices = similarity.argsort()[0][-top_n:][::-1]
        return self.df.iloc[indices]

@st.cache_resource
def init_model():
    return CourseRecommender(COURSE_DATA)

# --- 5. Main App Function ---
def main():
    # CRITICAL: This MUST be the first Streamlit command
    st.set_page_config(page_title="Course Finder", page_icon="🎓", layout="wide")
    
    recommender = init_model()

    # Custom CSS
    st.markdown("""
        <style>
        .stTextArea textarea { font-size: 1.1rem; }
        .course-card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            border-left: 6px solid #2e86c1;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 AI Course Recommendation System")
    st.write("Enter your interests below, and our model will find the best courses for you.")

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Your Interests")
        user_query = st.text_area(
            "What do you want to learn?",
            placeholder="e.g. I want to learn data analysis and visualization using python",
            height=150
        )
        search_btn = st.button("Recommend Courses", use_container_width=True)

    with col2:
        st.subheader("Top Matches")
        if search_btn and user_query:
            results = recommender.get_recommendations(user_query)
            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="course-card">
                    <h3 style="margin:0;">{row['title']}</h3>
                    <p style="color:#2e86c1;"><b>{row['category']}</b> | Level: {row['difficulty']}</p>
                    <p>{row['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        elif not search_btn:
            st.info("Waiting for your input... Describe your goals in the box on the left.")

    st.divider()
    with st.expander("See all available courses"):
