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

# --- 1. NLTK Setup ---
@st.cache_resource
def download_resources():
    # Only download if not already present
    for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(res, quiet=True)

download_resources()

# Initialize tools
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
    # MUST BE FIRST
    st.set_page_config(page_title="Course Finder", page_icon="🎓", layout="wide")
    
    recommender = init_model()

    st.title("🎓 AI Course Recommendation System")
    
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("Your Interests")
        user_query = st.text_area("What do you want to learn?", height=150)
        search_btn = st.button("Recommend Courses", use_container_width=True)

    with col2:
        st.subheader("Top Matches")
        if search_btn and user_query:
            results = recommender.get_recommendations(user_query)
            for _, row in results.iterrows():
                with st.container():
                    st.write(f"### {row['title']}")
                    st.caption(f"{row['category']} | Level: {row['difficulty']}")
                    st.write(row['description'])
                    st.divider()
        else:
            st.info("Results will appear here.")

if __name__ == "__main__":
    main()
