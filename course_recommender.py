# course_recommender.py
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

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample course dataset (replace with your actual data)
COURSE_DATA = {
    'course_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'title': [
        'Introduction to Python Programming',
        'Data Science Fundamentals',
        'Machine Learning Essentials',
        'Web Development with Django',
        'Deep Learning with TensorFlow',
        'Natural Language Processing',
        'Data Visualization with Python',
        'Cloud Computing for Beginners',
        'SQL for Data Analysis',
        'Big Data Analytics'
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
    'difficulty': ['Beginner', 'Beginner', 'Intermediate', 'Intermediate', 
                  'Advanced', 'Intermediate', 'Intermediate', 'Beginner', 
                  'Beginner', 'Advanced']
}

# Preprocessing functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f'[{string.punctuation}]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Create recommendation engine
class CourseRecommender:
    def __init__(self, data):
        self.df = pd.DataFrame(data)
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._prepare_data()
        
    def _prepare_data(self):
        # Create combined text features
        self.df['combined_features'] = self.df['title'] + ' ' + self.df['description'] + ' ' + self.df['category']
        
        # Preprocess text
        self.df['processed_features'] = self.df['combined_features'].apply(preprocess_text)
        
        # Create TF-IDF matrix
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['processed_features'])
        
    def recommend_courses(self, input_text, top_n=5):
        # Preprocess input
        processed_input = preprocess_text(input_text)
        
        # Transform input using TF-IDF
        input_vector = self.tfidf_vectorizer.transform([processed_input])
        
        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(input_vector, self.tfidf_matrix)
        
        # Get top N recommendations
        similar_indices = cosine_similarities.argsort()[0][-top_n:][::-1]
        
        # Return recommended courses
        recommendations = self.df.iloc[similar_indices]
        return recommendations[['course_id', 'title', 'description', 'category', 'difficulty']]

# Streamlit app
def main():
    st.set_page_config(
        page_title="Course Recommendation System",
        page_icon="🎓",
        layout="wide"
    )
    
    # Initialize recommender
    recommender = CourseRecommender(COURSE_DATA)
    
    # Custom CSS
    st.markdown("""
    <style>
        .header {
            color: #2e86c1;
            text-align: center;
            font-size: 36px;
            padding: 20px;
        }
        .subheader {
            color: #3498db;
            font-size: 24px;
            margin-top: 20px;
        }
        .recommendation-box {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #2e86c1;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="header">🎓 Course Recommendation System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("This system recommends relevant courses based on your interests using machine learning.")
        st.subheader("How to Use:")
        st.write("1. Enter your interests or course preferences")
        st.write("2. Click the 'Recommend Courses' button")
        st.write("3. Explore the recommended courses")
        st.divider()
        st.subheader("Sample Keywords:")
        st.code("python programming")
        st.code("data analysis")
        st.code("machine learning")
        st.code("web development")
        st.code("cloud computing")
    
    # Main content
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Your Interests")
        user_input = st.text_area(
            "Describe your interests or desired skills:", 
            "I want to learn Python for data analysis...",
            height=200
        )
        
        if st.button("Recommend Courses", use_container_width=True):
            st.session_state.recommendations = recommender.recommend_courses(user_input)
    
    with col2:
        st.subheader("Recommended Courses")
        
        if 'recommendations' in st.session_state:
            recommendations = st.session_state.recommendations
            for _, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4>{row['title']}</h4>
                        <p><strong>Category:</strong> {row['category']} | <strong>Level:</strong> {row['difficulty']}</p>
                        <p>{row['description']}</p>
                        <small>Course ID: {row['course_id']}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Enter your interests and click 'Recommend Courses' to get suggestions")
    
    # Course explorer at the bottom
    st.divider()
    st.subheader("All Available Courses")
    st.dataframe(recommender.df[['title', 'category', 'difficulty', 'description']], 
                hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()