import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ---- THIS MUST BE FIRST ----
st.set_page_config(page_title="Job Recommender", page_icon="💼", layout="wide")

# ---- Load saved model ----
@st.cache_resource
def load_model():
    with open('src/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('src/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    df = pd.read_csv('data/cleaned_jobs.csv')
    return tfidf, tfidf_matrix, df

tfidf, tfidf_matrix, df = load_model()

# ---- Recommendation Function ----
def recommend_jobs(user_skills, top_n=10):
    user_skills = user_skills.lower().strip()
    user_vector = tfidf.transform([user_skills])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarity_scores[0].argsort()[::-1][:top_n]
    results = df.iloc[top_indices][['Job Title',
                                    'Key Skills',
                                    'Location',
                                    'Industry',
                                    'Job Experience Required']].copy()
    results['Match Score'] = similarity_scores[0][top_indices]
    results['Match Score'] = results['Match Score'].apply(lambda x: f"{x*100:.1f}%")
    return results.reset_index(drop=True)

# ---- UI ----
st.title("💼 Job Recommendation System")
st.markdown("#### Find the most relevant jobs based on your skills")
st.markdown("---")

# Sidebar
st.sidebar.header("🔧 Search Settings")
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Example Skills")
st.sidebar.markdown("- python machine learning sql")
st.sidebar.markdown("- javascript react nodejs html")
st.sidebar.markdown("- sales communication b2b negotiation")
st.sidebar.markdown("- java spring boot microservices")

# Main input
user_input = st.text_area(
    "Enter your skills (comma or space separated):",
    placeholder="e.g. python machine learning data science pandas numpy",
    height=100
)

if st.button("🔍 Find Jobs", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter at least one skill!")
    else:
        with st.spinner("Finding best matches..."):
            user_input_clean = user_input.replace(',', ' ')
            results = recommend_jobs(user_input_clean, top_n)

        st.success(f"Found top {top_n} jobs matching your skills!")
        st.markdown("---")

        for i, row in results.iterrows():
            with st.expander(f"#{i+1} — {row['Job Title']} | 📍 {row['Location']} | 🎯 Match: {row['Match Score']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🏢 Industry:** {row['Industry']}")
                    st.markdown(f"**⏳ Experience:** {row['Job Experience Required']}")
                with col2:
                    st.markdown(f"**🛠️ Skills Required:**")
                    st.markdown(f"{row['Key Skills']}")

st.markdown("---")
st.markdown("Built with ❤️ using Python, Scikit-learn & Streamlit")