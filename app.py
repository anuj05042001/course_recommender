import streamlit as st
from course_recommender import recommend_courses  # adjust function name if different

st.title("📚 Course Recommender")

user_input = st.text_input("Enter a course or topic:")

if st.button("Recommend"):
    results = recommend_courses(user_input)
    st.write(results)
