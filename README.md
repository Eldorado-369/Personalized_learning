# Personalized Learning Recommendation System 📚

> ML-powered engine that tailors educational content to individual student profiles

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

---

## Overview

This recommendation engine analyses individual student learning profiles — including performance history, learning pace, and topic strengths — and applies machine learning techniques to surface the most relevant content for each student. The goal is to move beyond one-size-fits-all curricula and improve learning outcomes through personalised pathways.

---

## Features

- **Student Profile Analysis** — Processes individual learning data including performance, engagement, and history
- **ML-Based Recommendations** — Applies supervised/unsupervised learning techniques to match content to learner profiles
- **Personalised Learning Paths** — Generates ranked content suggestions tailored to each student
- **Performance Metrics** — Evaluates recommendation quality using standard ML evaluation metrics

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| ML / Data | scikit-learn, Pandas, NumPy |
| Notebook | Jupyter Notebook |

---

## Local Setup

```bash
# Clone the repo
git clone https://github.com/Eldorado-369/Personalized_learning.git
cd Personalized_learning

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Open the main notebook and follow the cells in sequence.

---

## Project Structure

```
Personalized_learning/
├── data/                  # Sample student datasets
├── notebooks/             # Jupyter notebooks for exploration & model building
├── models/                # Saved trained models
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Data Ingestion** — Load student profiles with features like quiz scores, time-on-topic, and completion rates
2. **Preprocessing** — Clean, normalise, and encode categorical features
3. **Model Training** — Train recommendation model (collaborative filtering / content-based)
4. **Recommendation Generation** — Predict and rank content items per student
5. **Evaluation** — Measure accuracy, precision, and recall of recommendations

---

## About the Author

**Eldho K Shajee** — M.Sc. Data Science & Analytics, Jain University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/eldho-k-shajee-723102279)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/Eldorado-369)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
