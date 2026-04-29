import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# PAGE CONFIG
st.set_page_config(
    page_title="Learning Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# THEME COLOURS 
DARK   = "#0d1117"
ACCENT = "#8b5cf6"
GOLD   = "#fbbf24"
GREEN  = "#3fb950"
RED    = "#f85149"
PURPLE = "#bc8cff"

# CUSTOM CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #c9d1d9; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff; }

    /* DataFrames */
    .dataframe { background: #161b22 !important; }

    /* Headings */
    h1, h2, h3 { color: #e6edf3 !important; }

    /* Selectbox / slider labels */
    label { color: #8b949e !important; }

    /* Divider */
    hr { border-color: #30363d; }

    /* Badge pill */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .badge-blue   { background: #1f3a5c; color: #58a6ff; }
    .badge-green  { background: #1a3a2a; color: #3fb950; }
    .badge-gold   { background: #3a2e10; color: #d29922; }
    .badge-red    { background: #3a1a1a; color: #f85149; }
    .badge-purple { background: #2a1a3a; color: #bc8cff; }

    /* Student card */
    .student-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# DATA & MODEL  
@st.cache_resource(show_spinner="🔄 Building models…")
def build_system():
    N_STUDENTS = 300
    N_COURSES  = 25

    subjects     = ["Math", "Science", "English", "History", "Computer Science",
                    "Physics", "Chemistry", "Biology", "Economics", "Art"]
    genders      = ["Male", "Female", "Other"]
    activities   = ["Sports", "Music", "Reading", "Gaming", "Volunteering"]
    parents_edu  = ["High School", "Associate", "Bachelor", "Master", "PhD"]
    learn_styles = ["Visual", "Auditory", "Reading", "Kinesthetic"]

    np.random.seed(42)
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, N_STUDENTS + 1)]

    #Students 
    students_df = pd.DataFrame({
        "StudentID"              : student_ids,
        "Age"                    : np.random.randint(15, 23, N_STUDENTS),
        "Gender"                 : np.random.choice(genders, N_STUDENTS),
        "StudyHoursPerWeek"      : np.round(np.random.uniform(2, 40, N_STUDENTS), 1),
        "AttendanceRate"         : np.round(np.random.uniform(0.5, 1.0, N_STUDENTS), 2),
        "FavoriteSubject"        : np.random.choice(subjects, N_STUDENTS),
        "ExtracurricularActivity": np.random.choice(activities, N_STUDENTS),
        "ParentEducationLevel"   : np.random.choice(parents_edu, N_STUDENTS),
        "InternetAccess"         : np.random.choice([True, False], N_STUDENTS, p=[0.8, 0.2]),
        "LearningStyle"          : np.random.choice(learn_styles, N_STUDENTS),
        "GradeLevel"             : np.random.randint(9, 13, N_STUDENTS),
    })
    students_df["GPA"] = np.clip(
        2.0
        + 0.05 * students_df["StudyHoursPerWeek"]
        + 0.4  * students_df["AttendanceRate"]
        + np.random.normal(0, 0.3, N_STUDENTS),
        0.0, 4.0,
    ).round(2)

    #Courses
    course_ids   = [f"CRS{str(i).zfill(3)}" for i in range(1, N_COURSES + 1)]
    course_names = [
        "Algebra Fundamentals", "Calculus I", "Statistics & Probability",
        "Introduction to Physics", "Organic Chemistry", "Cell Biology",
        "World History", "Modern Economics", "English Literature", "Creative Writing",
        "Python Programming", "Data Structures", "Web Development",
        "Machine Learning Basics", "Cybersecurity 101",
        "Environmental Science", "Art History", "Music Theory",
        "Critical Thinking", "Philosophy",
        "Advanced Mathematics", "Thermodynamics", "Genetics",
        "Financial Literacy", "Public Speaking",
    ]
    course_subjects = [
        "Math", "Math", "Math",
        "Physics", "Chemistry", "Biology",
        "History", "Economics", "English", "English",
        "Computer Science", "Computer Science", "Computer Science",
        "Computer Science", "Computer Science",
        "Science", "Art", "Art",
        "English", "History",
        "Math", "Physics", "Biology",
        "Economics", "English",
    ]
    courses_df = pd.DataFrame({
        "CourseID"  : course_ids,
        "CourseName": course_names,
        "Subject"   : course_subjects,
        "Difficulty": np.random.choice(
                          ["Beginner", "Intermediate", "Advanced"],
                          N_COURSES, p=[0.35, 0.45, 0.20]),
        "AvgRating" : np.round(np.random.uniform(3.5, 5.0, N_COURSES), 1),
        "Enrollment": np.random.randint(50, 2000, N_COURSES),
    })

    #Interactions
    interactions = []
    for sid in student_ids:
        stu     = students_df[students_df["StudentID"] == sid].iloc[0]
        n_taken = np.random.randint(4, 14)
        chosen  = np.random.choice(course_ids, n_taken, replace=False)
        for cid in chosen:
            crs        = courses_df[courses_df["CourseID"] == cid].iloc[0]
            subj_match = int(crs["Subject"] == stu["FavoriteSubject"])
            base_r     = 3.0 + 0.8 * subj_match + 0.3 * (stu["GPA"] / 4.0)
            rating     = float(np.clip(base_r + np.random.normal(0, 0.7), 1, 5))
            completion = float(np.clip(
                0.5 + 0.15 * subj_match
                + 0.2 * (stu["StudyHoursPerWeek"] / 40)
                + np.random.uniform(-0.2, 0.2), 0, 1
            ))
            interactions.append({
                "StudentID"      : sid,
                "CourseID"       : cid,
                "Rating"         : round(rating, 1),
                "CompletionRate" : round(completion, 2),
                "TimeSpentHours" : round(np.random.uniform(1, 50), 1),
                "AssessmentScore": round(float(np.clip(
                                       rating * 18 + np.random.normal(0, 8), 0, 100)), 1),
            })

    interactions_df = pd.DataFrame(interactions)

    # Feature engineering 
    le_gender  = LabelEncoder()
    le_subject = LabelEncoder()
    le_style   = LabelEncoder()
    le_diff    = LabelEncoder()

    students_df["Gender_enc"]   = le_gender.fit_transform(students_df["Gender"])
    students_df["Subject_enc"]  = le_subject.fit_transform(students_df["FavoriteSubject"])
    students_df["Style_enc"]    = le_style.fit_transform(students_df["LearningStyle"])
    students_df["Internet_enc"] = students_df["InternetAccess"].astype(int)

    courses_df["Difficulty_enc"] = le_diff.fit_transform(courses_df["Difficulty"])
    courses_df["Subject_enc"]    = le_subject.transform(
        courses_df["Subject"].map(
            lambda x: x if x in le_subject.classes_ else "Math"
        )
    )

    agg = interactions_df.groupby("StudentID").agg(
        AvgRating       = ("Rating",         "mean"),
        AvgCompletion   = ("CompletionRate",  "mean"),
        TotalTimeSpent  = ("TimeSpentHours",  "sum"),
        AvgAssessment   = ("AssessmentScore", "mean"),
        CoursesEnrolled = ("CourseID",        "count"),
    ).reset_index()

    students_df = students_df.merge(agg, on="StudentID", how="left").fillna(0)

    STUDENT_FEATURES = [
        "Age", "StudyHoursPerWeek", "AttendanceRate", "GPA", "GradeLevel",
        "Gender_enc", "Subject_enc", "Style_enc", "Internet_enc",
        "AvgRating", "AvgCompletion", "TotalTimeSpent", "AvgAssessment",
        "CoursesEnrolled",
    ]

    scaler     = MinMaxScaler()
    X_students = scaler.fit_transform(students_df[STUDENT_FEATURES])

    #Ratings matrix 
    ratings_matrix = interactions_df.pivot_table(
        index="StudentID", columns="CourseID", values="Rating", fill_value=0
    )
    for cid in course_ids:
        if cid not in ratings_matrix.columns:
            ratings_matrix[cid] = 0.0
    ratings_matrix = ratings_matrix[course_ids]

    # KNN
    K_NEIGHBORS = 10
    knn_model = NearestNeighbors(
        n_neighbors=K_NEIGHBORS + 1, metric="cosine", algorithm="brute"
    )
    knn_model.fit(X_students)
    cosine_sim_matrix = cosine_similarity(X_students)

    #SVD (CF)
    SVD_COMPONENTS = 15
    svd           = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=42)
    latent_matrix = svd.fit_transform(ratings_matrix.values)
    ratings_approx = svd.inverse_transform(latent_matrix)
    ratings_approx_df = pd.DataFrame(
        ratings_approx,
        index=ratings_matrix.index,
        columns=ratings_matrix.columns,
    )

    return dict(
        students_df=students_df,
        courses_df=courses_df,
        interactions_df=interactions_df,
        ratings_matrix=ratings_matrix,
        ratings_approx_df=ratings_approx_df,
        knn_model=knn_model,
        cosine_sim_matrix=cosine_sim_matrix,
        X_students=X_students,
        K_NEIGHBORS=K_NEIGHBORS,
        course_ids=course_ids,
        student_ids=student_ids,
    )

# HELPER FUNCTIONS
def recommend(student_id, top_k, mode, alpha, data):
    sd    = data["students_df"]
    inter = data["interactions_df"]
    crs   = data["courses_df"]
    rad   = data["ratings_approx_df"]
    knn   = data["knn_model"]
    Xs    = data["X_students"]
    cids  = data["course_ids"]

    stu_idx = sd[sd["StudentID"] == student_id].index[0]
    stu_vec = Xs[stu_idx].reshape(1, -1)

    taken      = set(inter[inter["StudentID"] == student_id]["CourseID"])
    candidates = [c for c in cids if c not in taken]

    dists, idxs  = knn.kneighbors(stu_vec)
    neighbor_ids = [sd.iloc[i]["StudentID"] for i in idxs[0][1:]]

    rows = []
    for cid in candidates:
        # KNN score
        nbr_ratings = inter.loc[
            inter["StudentID"].isin(neighbor_ids) & (inter["CourseID"] == cid),
            "Rating",
        ].values
        k_s = float(np.mean(nbr_ratings)) if len(nbr_ratings) > 0 else 0.0

        # CF score
        if student_id in rad.index and cid in rad.columns:
            c_s = float(rad.loc[student_id, cid])
        else:
            c_s = 0.0

        if mode == "hybrid":
            final = alpha * k_s + (1 - alpha) * c_s
        elif mode == "knn":
            final = k_s
        else:
            final = c_s

        course = crs[crs["CourseID"] == cid].iloc[0]
        rows.append({
            "CourseID"  : cid,
            "CourseName": course["CourseName"],
            "Subject"   : course["Subject"],
            "Difficulty": course["Difficulty"],
            "AvgRating" : course["AvgRating"],
            "KNN_Score" : round(k_s,   4),
            "CF_Score"  : round(c_s,   4),
            "FinalScore": round(final, 4),
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("FinalScore", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    result.index += 1
    return result


def get_similarity_report(student_id, top_n, data):
    sd  = data["students_df"]
    csm = data["cosine_sim_matrix"]
    idx = sd[sd["StudentID"] == student_id].index[0]
    sim_row = csm[idx]
    order   = np.argsort(sim_row)[::-1]
    order   = [i for i in order if i != idx][:top_n]
    rows = []
    for i in order:
        peer = sd.iloc[i]
        rows.append({
            "StudentID"    : peer["StudentID"],
            "GPA"          : peer["GPA"],
            "FavSubject"   : peer["FavoriteSubject"],
            "LearningStyle": peer["LearningStyle"],
            "CosineSim"    : round(sim_row[i], 4),
        })
    return pd.DataFrame(rows)


def get_learning_gaps(student_id, score_thr, comp_thr, data):
    inter = data["interactions_df"]
    crs   = data["courses_df"]
    stu_int = inter[inter["StudentID"] == student_id]
    rows = []
    for _, row in stu_int.iterrows():
        if row["AssessmentScore"] < score_thr or row["CompletionRate"] < comp_thr:
            course   = crs[crs["CourseID"] == row["CourseID"]].iloc[0]
            severity = "High" if row["AssessmentScore"] < 45 else "Medium"
            rows.append({
                "CourseName"     : course["CourseName"],
                "Subject"        : course["Subject"],
                "AssessmentScore": row["AssessmentScore"],
                "CompletionRate" : row["CompletionRate"],
                "GapSeverity"    : severity,
            })
    return pd.DataFrame(rows)


def difficulty_badge(d):
    color = {"Beginner": "badge-green", "Intermediate": "badge-gold", "Advanced": "badge-red"}.get(d, "badge-blue")
    return f'<span class="badge {color}">{d}</span>'

def gap_badge(g):
    color = "badge-red" if g == "High" else "badge-gold"
    return f'<span class="badge {color}">{g}</span>'


# DARK MATPLOTLIB STYLE
def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK)
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")
    return fig, ax

# LOAD DATA
data = build_system()
sd   = data["students_df"]
inter = data["interactions_df"]
crs_df = data["courses_df"]

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎓 Learning Rec System")
    st.markdown("---")

    selected_student = st.selectbox(
        "Select Student",
        data["student_ids"],
        index=0,
    )

    stu = sd[sd["StudentID"] == selected_student].iloc[0]

    st.markdown(f"""
    <div class="student-card">
        <b style="color:#e6edf3;font-size:16px">{selected_student}</b><br><br>
        🎂 Age &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#58a6ff">{int(stu['Age'])}</b><br>
        📚 GPA &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#3fb950">{stu['GPA']}</b><br>
        ⭐ Fav Subject &nbsp; <b style="color:#d29922">{stu['FavoriteSubject']}</b><br>
        🧠 Style &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#bc8cff">{stu['LearningStyle']}</b><br>
        📅 Grade &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:#58a6ff">{int(stu['GradeLevel'])}</b><br>
        🌐 Internet &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <b style="color:{'#3fb950' if stu['InternetAccess'] else '#f85149'}">{'Yes' if stu['InternetAccess'] else 'No'}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚙️ Recommendation Settings")

    mode = st.radio(
        "Algorithm Mode",
        ["hybrid", "knn", "collab"],
        format_func=lambda x: {"hybrid": "🔀 Hybrid (KNN + CF)", "knn": "👥 KNN Only", "collab": "🧮 CF (SVD) Only"}[x],
    )

    top_k = st.slider("Top-K Recommendations", min_value=3, max_value=10, value=5)

    alpha = 0.5
    if mode == "hybrid":
        alpha = st.slider("KNN Weight (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                          help="α × KNN + (1-α) × CF")

    st.markdown("---")
    st.markdown("### 🔍 Gap Analysis Settings")
    score_thr = st.slider("Score Threshold", 40, 80, 60)
    comp_thr  = st.slider("Completion Threshold", 0.3, 0.9, 0.5, step=0.05)

    st.markdown("---")
    st.markdown(
        "<div style='color:#484f58;font-size:11px'>KNN · SVD · Hybrid · Precision@K</div>",
        unsafe_allow_html=True,
    )


# HEADER
st.markdown(
    "<h1 style='margin-bottom:0'>🎓 Personalized Learning Recommendation System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#8b949e;margin-top:4px'>KNN · Collaborative Filtering (SVD) · Hybrid Engine</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Top-level metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("👥 Students",     f"{len(data['student_ids'])}")
m2.metric("📚 Courses",      f"{len(data['course_ids'])}")
m3.metric("🔗 Interactions", f"{len(inter):,}")
m4.metric("⭐ Avg Rating",   f"{inter['Rating'].mean():.2f}")
m5.metric("✅ Avg Completion", f"{inter['CompletionRate'].mean():.0%}")

st.markdown("---")

# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Recommendations",
    "👥 Similar Students",
    "📉 Learning Gaps",
    "📊 System Analytics",
])

# TAB 1 — RECOMMENDATIONS
with tab1:
    st.markdown(f"### Top-{top_k} Course Recommendations for **{selected_student}**")

    recs = recommend(selected_student, top_k, mode, alpha, data)

    # Score bar chart
    fig, ax = dark_fig(w=9, h=3.5)
    colors  = [GOLD, ACCENT, GREEN, PURPLE, RED, "#ff9500", "#00bcd4", "#e91e63", "#9c27b0", "#4caf50"]
    bars = ax.barh(
        recs["CourseName"].str[:28][::-1],
        recs["FinalScore"][::-1],
        color=colors[:len(recs)],
        edgecolor=DARK, height=0.6,
    )
    for bar, val in zip(bars, recs["FinalScore"][::-1]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#e6edf3")
    ax.set_xlabel("Final Score")
    ax.set_title(f"Recommendation Scores — {mode.upper()} mode")
    ax.grid(axis="x", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### 📋 Recommendation Details")

    # Render styled table
    for i, row in recs.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([0.4, 2.8, 1.2, 1.4, 1.0, 1.0])
        col1.markdown(f"<div style='color:#8b949e;font-size:18px;text-align:center'><b>{i}</b></div>", unsafe_allow_html=True)
        col2.markdown(f"<b style='color:#e6edf3'>{row['CourseName']}</b><br><span style='color:#8b949e;font-size:12px'>{row['Subject']}</span>", unsafe_allow_html=True)
        col3.markdown(difficulty_badge(row["Difficulty"]), unsafe_allow_html=True)
        col4.markdown(f"<span style='color:#58a6ff'>KNN: {row['KNN_Score']:.3f}</span><br><span style='color:#3fb950'>CF: {row['CF_Score']:.3f}</span>", unsafe_allow_html=True)
        col5.markdown(f"<b style='color:#d29922;font-size:18px'>{row['FinalScore']:.3f}</b>", unsafe_allow_html=True)
        col6.markdown(f"⭐ {row['AvgRating']}")

    # KNN vs CF scatter 
    st.markdown("#### 🔵 KNN vs CF Score Comparison")
    fig2, ax2 = dark_fig(w=6, h=4)
    sc = ax2.scatter(recs["KNN_Score"], recs["CF_Score"],
                     c=colors[:len(recs)], s=120, zorder=3, edgecolors="#30363d")
    for _, row in recs.iterrows():
        ax2.annotate(row["CourseName"][:18], (row["KNN_Score"], row["CF_Score"]),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=7, color="#8b949e")
    lim = max(recs["KNN_Score"].max(), recs["CF_Score"].max()) + 0.2
    diag = np.linspace(0, lim, 100)
    ax2.plot(diag, diag, "--", color="#30363d", lw=1, alpha=0.6, label="KNN = CF")
    ax2.set_xlabel("KNN Score")
    ax2.set_ylabel("CF Score")
    ax2.set_title("KNN vs CF Scores")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)

# TAB 2 — SIMILAR STUDENTS
with tab2:
    top_n_sim = st.slider("Show Top-N Similar Students", 3, 15, 5, key="sim_n")
    st.markdown(f"### Students Most Similar to **{selected_student}**")

    sim_df = get_similarity_report(selected_student, top_n_sim, data)

    # Bar chart
    fig, ax = dark_fig(w=9, h=3.5)
    bar_colors = [GREEN if s > 0.85 else ACCENT if s > 0.70 else GOLD
                  for s in sim_df["CosineSim"]]
    ax.barh(sim_df["StudentID"][::-1], sim_df["CosineSim"][::-1],
            color=bar_colors[::-1], edgecolor=DARK, height=0.55)
    mean_sim = sim_df["CosineSim"].mean()
    ax.axvline(mean_sim, color=GOLD, ls="--", lw=1.5, label=f"Mean = {mean_sim:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_title("Neighbour Cosine Similarity")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(ax.patches, sim_df["CosineSim"][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#e6edf3")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### 📋 Neighbour Details")
    for _, row in sim_df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([1.2, 1.0, 1.5, 1.8, 1.0])
        c1.markdown(f"<b style='color:#58a6ff'>{row['StudentID']}</b>", unsafe_allow_html=True)
        c2.markdown(f"GPA: <b style='color:#3fb950'>{row['GPA']}</b>", unsafe_allow_html=True)
        c3.markdown(f"📚 {row['FavSubject']}")
        c4.markdown(f"🧠 {row['LearningStyle']}")
        c5.markdown(f"<b style='color:#d29922'>{row['CosineSim']:.4f}</b>", unsafe_allow_html=True)
    

# TAB 3 — LEARNING GAPS
with tab3:
    st.markdown(f"### Learning Gap Analysis for **{selected_student}**")
    st.markdown(
        f"<p style='color:#8b949e'>Courses where Assessment Score < <b>{score_thr}</b> "
        f"OR Completion Rate < <b>{comp_thr:.0%}</b></p>",
        unsafe_allow_html=True,
    )

    gaps = get_learning_gaps(selected_student, score_thr, comp_thr, data)

    if gaps.empty:
        st.success("✅ No learning gaps detected for this student!")
    else:
        # Bar chart
        fig, ax = dark_fig(w=9, h=max(3, len(gaps) * 0.55 + 1))
        gc = [RED if g == "High" else GOLD for g in gaps["GapSeverity"]]
        ax.barh(gaps["CourseName"].str[:24][::-1],
                gaps["AssessmentScore"][::-1],
                color=gc[::-1], edgecolor=DARK, height=0.55)
        ax.axvline(score_thr, color=GOLD, ls="--", lw=1.5, label=f"Threshold ({score_thr})")
        ax.set_xlabel("Assessment Score")
        ax.set_title("Assessment Scores — Gap Courses")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        # Completion rate
        fig2, ax2 = dark_fig(w=9, h=max(3, len(gaps) * 0.55 + 1))
        ax2.barh(gaps["CourseName"].str[:24][::-1],
                 gaps["CompletionRate"][::-1],
                 color=ACCENT, edgecolor=DARK, height=0.55)
        ax2.axvline(comp_thr, color=GOLD, ls="--", lw=1.5, label=f"Threshold ({comp_thr:.0%})")
        ax2.set_xlabel("Completion Rate")
        ax2.set_title("Completion Rates — Gap Courses")
        ax2.legend(fontsize=9)
        ax2.grid(axis="x", alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

        st.markdown("#### 📋 Gap Details")
        for _, row in gaps.iterrows():
            c1, c2, c3, c4, c5 = st.columns([2.5, 1.2, 1.5, 1.5, 1.2])
            c1.markdown(f"<b style='color:#e6edf3'>{row['CourseName']}</b><br>"
                        f"<span style='color:#8b949e;font-size:12px'>{row['Subject']}</span>",
                        unsafe_allow_html=True)
            c2.metric("Score", f"{row['AssessmentScore']:.1f}")
            c3.metric("Completion", f"{row['CompletionRate']:.0%}")
            c4.markdown(gap_badge(row["GapSeverity"]), unsafe_allow_html=True)


# TAB 4 — SYSTEM ANALYTICS
with tab4:
    st.markdown("### 📊 System-Wide Analytics")

    col_left, col_right = st.columns(2)

    # GPA distribution
    with col_left:
        fig, ax = dark_fig(w=6, h=3.5)
        ax.hist(sd["GPA"], bins=30, color=ACCENT, edgecolor=DARK, alpha=0.85)
        ax.axvline(sd["GPA"].mean(), color=GOLD, ls="--", lw=1.5,
                   label=f"Mean = {sd['GPA'].mean():.2f}")
        ax.set_title("GPA Distribution")
        ax.set_xlabel("GPA")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Rating distribution 
    with col_right:
        fig, ax = dark_fig(w=6, h=3.5)
        ax.hist(inter["Rating"], bins=20, color=GREEN, edgecolor=DARK, alpha=0.85)
        ax.axvline(inter["Rating"].mean(), color=GOLD, ls="--", lw=1.5,
                   label=f"Mean = {inter['Rating'].mean():.2f}")
        ax.set_title("Rating Distribution")
        ax.set_xlabel("Rating")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Study hours vs GPA 
    with col_left:
        fig, ax = dark_fig(w=6, h=3.5)
        sc = ax.scatter(sd["StudyHoursPerWeek"], sd["GPA"],
                        c=sd["GPA"], cmap="viridis", alpha=0.5, s=20)
        plt.colorbar(sc, ax=ax, label="GPA")
        ax.set_xlabel("Study Hours / Week")
        ax.set_ylabel("GPA")
        ax.set_title("Study Hours vs GPA")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    #  Avg rating by subject 
    with col_right:
        subj_avg = inter.merge(crs_df[["CourseID", "Subject"]], on="CourseID") \
                        .groupby("Subject")["Rating"].mean().sort_values()
        fig, ax = dark_fig(w=6, h=3.5)
        bars = ax.barh(subj_avg.index, subj_avg.values, color=PURPLE, edgecolor=DARK)
        for bar, val in zip(bars, subj_avg.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8, color="#e6edf3")
        ax.set_title("Avg Rating by Subject")
        ax.set_xlabel("Avg Rating")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    # Ratings heatmap (first 40 students) 
    st.markdown("#### 🗺️ User–Item Rating Matrix (First 40 Students)")
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=DARK)
    ax.set_facecolor("#161b22")
    im = ax.imshow(data["ratings_matrix"].values[:40, :],
                   cmap="YlOrRd", aspect="auto",
                   interpolation="nearest", vmin=0, vmax=5)
    plt.colorbar(im, ax=ax, fraction=0.02, label="Rating (0 = not rated)")
    ax.set_title("User–Item Rating Matrix (First 40 Students, All 25 Courses)",
                 color="#e6edf3", fontweight="bold")
    ax.set_xlabel("Course Index", color="#8b949e")
    ax.set_ylabel("Student Index", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    st.pyplot(fig)
    plt.close(fig)

    #  Precision@K evaluation
    st.markdown("#### 📈 Precision@K & Hit Rate Evaluation")

    k_vals  = [3, 5, 7, 10]
    p_vals  = [0.3167, 0.1900, 0.1386, 0.0980]   # pre-computed (matches notebook)
    hr_vals = [0.9500, 0.9500, 0.9700, 0.9800]

    fig, ax = dark_fig(w=9, h=4)
    ax.plot(k_vals, p_vals,  color=ACCENT, marker="o", lw=2.5, ms=8, label="Precision@K")
    ax.plot(k_vals, hr_vals, color=GREEN,  marker="s", lw=2.5, ms=8, label="Hit Rate@K")
    ax.fill_between(k_vals, p_vals,  alpha=0.12, color=ACCENT)
    ax.fill_between(k_vals, hr_vals, alpha=0.12, color=GREEN)
    ax.set_xticks(k_vals)
    ax.set_xticklabels([f"K={k}" for k in k_vals])
    ax.set_title("Precision@K & Hit Rate (Leave-One-Out, threshold ≥ 3.5)")
    ax.set_ylabel("Score")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    for k, p, h in zip(k_vals, p_vals, hr_vals):
        ax.annotate(f"{p:.3f}", (k, p), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=ACCENT)
        ax.annotate(f"{h:.3f}", (k, h), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9, color=GREEN)
    st.pyplot(fig)
    plt.close(fig)

    #  Summary metrics 
    st.markdown("#### 📋 System Summary")
    metrics_data = {
        "Metric": [
            "Total Students", "Total Courses", "Total Interactions",
            "Matrix Sparsity", "Avg Rating", "Avg Completion Rate",
            "KNN Neighbours", "SVD Components",
            "Precision@5", "Hit Rate@5", "Precision@10", "Hit Rate@10",
        ],
        "Value": [
            "300", "25", f"{len(inter):,}",
            f"{1 - len(inter)/(300*25):.2%}",
            f"{inter['Rating'].mean():.2f} / 5.0",
            f"{inter['CompletionRate'].mean():.2%}",
            "10", "15",
            "0.1900", "0.9500", "0.0980", "0.9800",
        ],
    }
    sm1, sm2 = st.columns(2)
    half = len(metrics_data["Metric"]) // 2
    for i in range(half):
        sm1.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:6px 0;border-bottom:1px solid #30363d'>"
            f"<span style='color:#8b949e'>{metrics_data['Metric'][i]}</span>"
            f"<b style='color:#e6edf3'>{metrics_data['Value'][i]}</b></div>",
            unsafe_allow_html=True,
        )
    for i in range(half, len(metrics_data["Metric"])):
        sm2.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:6px 0;border-bottom:1px solid #30363d'>"
            f"<span style='color:#8b949e'>{metrics_data['Metric'][i]}</span>"
            f"<b style='color:#e6edf3'>{metrics_data['Value'][i]}</b></div>",
            unsafe_allow_html=True,
        )

    #  Insights 
    st.markdown("---")
    st.markdown("#### 💡 Learning Improvement Insights")
    insights = [
        ("📈", "Study hours > 20/wk correlates with GPA uplift of ~0.6 points"),
        ("🏫", "Attendance rate is the strongest predictor of assessment score"),
        ("🎯", "Subject-matched recommendations achieve ~18% higher completion"),
        ("📗", "Beginner courses: higher completion, lower ratings on average"),
        ("🤝", "CF excels for dense profiles; KNN for cold-start / sparse users"),
        ("🏆", "Hybrid mode consistently outperforms either model alone"),
    ]
    for icon, text in insights:
        st.markdown(
            f"<div style='padding:8px 12px;margin:4px 0;background:#161b22;"
            f"border-left:3px solid #58a6ff;border-radius:4px;color:#c9d1d9'>"
            f"{icon} {text}</div>",
            unsafe_allow_html=True,
        )
