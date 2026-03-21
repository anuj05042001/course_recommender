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

# --- 1. Cached Resource Loading ---
@st.cache_resource
def load_all_resources():
    """Initializes NLTK and Recommender in one safe go."""
    # 1. Download NLTK bits
    for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(res, quiet=True)
    
    # 2. Setup Preprocessing Tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # 3. Sample Data
    course_data = {
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
    
    return stop_words, lemmatizer, course_data

# --- 2. Preprocessing Function ---
def preprocess_text(text, stop_words, lemmatizer):
    text = str(text).lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- 3. Recommender Class ---
class CourseRecommender:
    def __init__(self, data, stop_words, lemmatizer):
        self.df = pd.DataFrame(data)
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer
        self._prepare_data()
        
    def _prepare_data(self):
        self.df['combined_features'] = self.df['title'] + ' ' + self.df['description'] + ' ' + self.df['category']
        self.df['processed_features'] = self.df['combined_features'].apply(
            lambda x: preprocess_text(x, self.stop_words, self.lemmatizer)
        )
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['processed_features'])
        
    def get_recommendations(self, query, top_n=3):
        proc_query = preprocess_text(query, self.stop_words, self.lemmatizer)
        query_vec = self.vectorizer.transform([proc_query])
        sim = cosine_similarity(query_vec, self.tfidf_matrix)
        indices = sim.argsort()[0][-top_n:][::-1]
        return self.df.iloc[indices]

# --- 4. Main App ---
def main():
    # STEP 1: MUST BE THE FIRST LINE
    st.set_page_config(page_title="AI Course Finder", page_icon="🎓", layout="wide")
    
    # STEP 2: Load resources inside main
    stop_words, lemmatizer, course_data = load_all_resources()
    
    # STEP 3: Initialize model (cached)
    @st.cache_resource
    def get_model():
        return CourseRecommender(course_data, stop_words, lemmatizer)
    
    recommender = get_model()

    # UI Elements
    st.title("🎓 Smart Course Recommender")
    st.write("Find your next skill using AI-powered search.")
    
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("What are you interested in?")
        user_query = st.text_area("Example: 'I want to learn deep learning and AI'", height=150)
        if st.button("Recommend Courses", use_container_width=True):
            if user_query:
                st.session_state.results = recommender.get_recommendations(user_query)
            else:
                st.warning("Please enter some text first!")

    with col2:
        st.subheader("Your Recommendations")
        if 'results' in st.session_state:
            for _, row in st.session_state.results.iterrows():
                with st.container():
                    st.markdown(f"### {row['title']}")
                    st.caption(f"{row['category']} • Level: {row['difficulty']}")
                    st.write(row['description'])
                    st.divider()
        else:
            st.info("Enter your interests on the left to see results here.")

if __name__ == "__main__":
    main()
