# Job Recommendation System

I built this project to get hands-on with Machine Learning concepts I've been 
studying — specifically how to work with real messy data and build something 
that actually works end to end.

The idea is simple: you enter your skills, and the system finds the most relevant 
job listings for you from 28,000+ real Naukri.com job postings, ranked by how 
well they match.

## How it works

I used TF-IDF to convert job skills into numbers and Cosine Similarity to measure 
how close a user's skills are to each job listing. No black box — every part of 
this is something I understand and can explain.

1. Cleaned and processed 30,000 raw job listings
2. Built a 28,729 x 6,521 TF-IDF matrix (one row per job, one column per unique skill)
3. At query time — convert user input to the same vector space and find nearest jobs
4. Return top N matches ranked by similarity score

## What I found in the data

- 28,729 jobs across 123 industries after cleaning
- 6,521 unique skills across all listings
- Most jobs are in Bengaluru
- Most in-demand skill is Sales, not Python — which genuinely surprised me

## Tech Stack

| Tool | Why I used it |
|------|--------------|
| Pandas and NumPy | Data cleaning and EDA |
| Scikit-learn | TF-IDF Vectorizer |
| Streamlit | To make it actually usable by anyone |
| Matplotlib and Seaborn | Visualizations during EDA |

## Run it yourself

```bash
git clone https://github.com/apoorva-sahu09/job-recommendation-system.git
cd job-recommendation-system
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
job-recommendation-system/
├── data/
│   ├── naukri_jobs.csv          # Raw dataset
│   └── cleaned_jobs.csv         # Cleaned version
├── notebooks/
│   ├── 01_EDA.ipynb             # Data exploration
│   └── 02_Recommender.ipynb     # Model building
├── src/
│   ├── tfidf_vectorizer.pkl     # Saved vectorizer
│   └── tfidf_matrix.pkl         # Saved matrix
├── app.py                       # Streamlit app
└── README.md
```

## Example

Input: `python machine learning data science pandas`

| Job Title | Location | Match Score |
|-----------|----------|-------------|
| Data Science - Freshers | Hyderabad | 64.9% |
| Trainee Data Scientist | Hyderabad, Pune | 59.1% |
| Python Developer | Bengaluru | 52.5% |

## What I learned

- Real data is always messier than expected — 1,271 missing skill entries
- TF-IDF automatically gives more weight to rare, specific skills over common ones
- Cosine similarity works surprisingly well for text matching problems

---
Built by Apoorva Sahu | Part of my ML learning journey
