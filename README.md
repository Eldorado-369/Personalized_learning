# 🎓 Personalized Learning Recommendation System

## 📌 Overview

With the explosion of online learning platforms, students face difficulty choosing the most relevant courses among thousands of options. Generic recommendation systems often fail to consider individual learning needs, resulting in poor engagement and suboptimal learning outcomes.

This project presents an **AI-powered Personalized Learning Recommendation System** that delivers tailored course suggestions based on a learner’s profile, behavior, and performance.

---

## 🚀 Features

* 🎯 **Personalized Course Recommendations**

  * Hybrid recommendation engine (KNN + Collaborative Filtering)
* 👥 **Similar Student Detection**

  * Find learners with similar profiles using cosine similarity
* 📉 **Learning Gap Analysis**

  * Identify weak areas using assessment scores & completion rates
* 📊 **System Analytics Dashboard**

  * Visual insights into performance, ratings, and engagement
* 🎛️ **Interactive UI**

  * Built with Streamlit for real-time exploration

---

## 🧠 Models Used

### 1. K-Nearest Neighbors (KNN)

* Finds similar students based on feature vectors
* Uses **cosine similarity** as the distance metric

### 2. Collaborative Filtering (SVD)

* Matrix factorization using `TruncatedSVD`
* Predicts missing ratings from latent features

### 3. Hybrid Model

* Combines both approaches:

  `Final Score = α × KNN Score + (1 - α) × CF Score`

---

## 🛠️ Tech Stack

* **Python**
* **Pandas & NumPy** – Data processing
* **Scikit-learn** – ML models (KNN, SVD, scaling)
* **Matplotlib & Seaborn** – Visualization
* **Streamlit** – Interactive web app

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit application
├── Images               # Screenshots
├── README.md            # Project documentation
```

---

## 📊 Dataset

The dataset is **synthetically generated** and includes:

### 👤 Student Features

* Age, Gender, GPA
* Study hours, attendance
* Learning style
* Favorite subject
* Internet access
* Academic performance metrics

### 📚 Course Features

* Course name & subject
* Difficulty level
* Average rating
* Enrollment count

### 🔗 Interaction Data

* Ratings
* Completion rate
* Time spent
* Assessment scores

---

## ⚙️ How It Works

### Step 1: Data Processing

* Encoding categorical features
* Feature scaling using `MinMaxScaler`

### Step 2: Model Building

* KNN trained on student features
* Ratings matrix built for CF
* SVD applied for latent factorization

### Step 3: Recommendation

* Identify unseen courses
* Compute:

  * KNN-based score
  * CF-based score
* Combine scores (Hybrid mode)

---

## 📈 Evaluation

### Metrics Used:

* ✅ **Precision@K**
* ✅ **Hit Rate@K**

### Sample Results:

| K  | Precision@K | Hit Rate |
| -- | ----------- | -------- |
| 3  | 0.3167      | 0.95     |
| 5  | 0.1900      | 0.95     |
| 7  | 0.1386      | 0.97     |
| 10 | 0.0980      | 0.98     |

---

## 📊 Key Insights

* 📈 Higher study hours → improved GPA
* 🏫 Attendance strongly impacts performance
* 🎯 Subject-aligned recommendations improve completion rates
* 🤝 Hybrid model outperforms individual models

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd <repo-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🎮 Usage

* Select a student from the sidebar
* Choose recommendation mode:

  * 🔀 Hybrid
  * 👥 KNN
  * 🧮 Collaborative Filtering
* Adjust parameters like:

  * Top-K recommendations
  * Weight (α)
* Explore:

  * Recommendations
  * Similar students
  * Learning gaps
  * System analytics

---

## 🔒 Constraints Followed

✔ Used only allowed libraries
✔ Implemented KNN and Collaborative Filtering
✔ Used similarity metrics (cosine similarity)
✔ Evaluated using Precision@K
❌ No deep learning used

---

## 🌟 Future Improvements

* Use real-world datasets (Coursera, Udemy, etc.)
* Add content-based filtering
* Deploy as a web service
* Add user authentication & profiles
* Real-time feedback loop

---

## 👨‍💻 Author

**Eldho K Shajee**

---

## 📜 License

This project is for academic and educational purposes.
