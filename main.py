# =========================
# 1. Import libraries
# =========================
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
 
import pdfplumber
import fitz  # PyMuPDF


# =========================
# 2. Load dataset
# =========================

data = pd.read_csv("resume_dataset.csv")
# Remove duplicates (IMPORTANT)
data = data.drop_duplicates()

# Balance dataset (basic)
print("\nCategory Distribution:\n")
print(data['Category'].value_counts())

# Basic validation
required_columns = ['Resume', 'Category']
for col in required_columns:
    if col not in data.columns   :
        raise ValueError(f"Missing column: {col}")

# Remove null values
data = data.dropna(subset=['Resume', 'Category'])

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Loaded & Cleaned:\n")
print(data.head())


# =========================
# 3. Clean text
# =========================
import spacy

nlp = spacy.load("en_core_web_sm")

def cleanResume(text):
    text = str(text).lower()
    doc = nlp(text)

    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)   #  use text instead of lemma

    return " ".join(tokens)

data['cleaned_resume'] = data['Resume'].apply(cleanResume)

print("\nCleaned Data:\n")
print(data[['Resume', 'cleaned_resume']])


# =========================
# 4. Encode labels
# =========================
le = LabelEncoder()
data['Category_encoded'] = le.fit_transform(data['Category'])


# =========================
# =========================
# 5. TF-IDF
# =========================
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1,2),
    stop_words='english',
    min_df=2,
    max_df=0.85
)

# 
X = tfidf.fit_transform(data['cleaned_resume'])
y = data['Category_encoded']




# =========================
# 6. Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   
)


# =========================
# 7. Train model
# =========================
from sklearn.svm import LinearSVC
model = LinearSVC(class_weight='balanced')
model.fit(X_train, y_train)

import joblib

joblib.dump(model, "resume_classifier.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")


# =========================
# 8. Predict & evaluate
# =========================
y_pred = model.predict(X_test)

print("\nModel Evaluation:\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
from sklearn.metrics import confusion_matrix

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print(le.classes_)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)
print("\nCross-validation accuracy:", scores.mean())


# =========================
# 9. Test with new resume
# =========================
sample_resume = ["I have experience in Python, machine learning and data analysis"]

sample_cleaned = [cleanResume(sample_resume[0])]
sample_vector = tfidf.transform(sample_cleaned)

prediction = model.predict(sample_vector)

print("\nTest Prediction:")
print("Predicted Category:", le.inverse_transform(prediction))

# =========================
# 10. Job Matching System
# =========================

from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define job descriptions
job_descriptions = pd.DataFrame({
    "Role": ["Data Science", "Web Development", "HR", "Finance"],
    "Description": [
        "python machine learning data analysis pandas numpy deep learning statistics",
        "html css javascript react node frontend backend web",
        "recruitment hiring onboarding employee relations hr policies",
        "accounting finance taxation auditing banking financial analysis"
    ]
})

# Step 2: Clean and vectorize job descriptions
job_texts = [cleanResume(text) for text in job_descriptions["Description"]]
job_vectors = job_vectors = tfidf.transform(job_texts)
job_titles = job_descriptions["Role"].tolist()

# Step 3: Use same sample resume (or new one)
resume_text = sample_resume[0]   # reuse your test resume
resume_clean = cleanResume(resume_text)
resume_vector = tfidf.transform([resume_clean])

# Step 4: Compute similarity
scores = cosine_similarity(resume_vector, job_vectors)

print("\nJob Matching Scores:\n")
for i, score in enumerate(scores[0]):
   print(f"{job_titles[i]} → {round(score*100,2)}%")

# Step 5: Best match
best_match_index = scores.argmax()
print("\nBest Job Role:", job_titles[best_match_index])




# =========================
# 12. Skills Extraction System
# =========================

# =========================
# Skills Database
# =========================
skills_db = {
    "Data Science": [
        "python","machine learning","deep learning",
        "natural language processing","pandas","numpy",
        "statistics","data analysis"
    ],
    "Web Development": [
        "html","css","javascript","react",
        "node","api","frontend","backend"
    ],
    "HR": [
        "recruitment","hiring","onboarding",
        "employee relations","hr policies","payroll"
    ],
    "Finance": [
        "accounting","taxation","auditing",
        "finance","banking","financial analysis"
    ]
}
# =========================
# Skill Weights
# =========================
skill_weights = {
    "python": 2,
    "machine learning": 3,
    "ml": 3,
    "deep learning": 3,
    "nlp": 2,
    "pandas": 1,
    "numpy": 1,
    "statistics": 1,
    "data analysis": 2
}

skill_alias = {
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing"
}
def extract_skills(text, skill_list):
    text = text.lower()
    found_skills = []

    for skill in skill_list:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text):
            found_skills.append(skill_alias.get(skill, skill))

    return list(set(found_skills))


def calculate_match(resume_skills, job_skills):
    important = {"python", "machine learning", "deep learning"}

    score = 0
    total = 0

    for skill in job_skills:
        weight = 2 if skill in important else 1
        total += weight

        if skill in resume_skills:
            score += weight

    percent = (score / total) * 100 if total > 0 else 0
    return percent, resume_skills


def missing_skills(resume_skills, job_skills):
    return sorted(list(set(job_skills) - set(resume_skills)))


def match_level(score):
    if score > 80:
        return "Excellent"
    elif score > 50:
        return "Moderate"
    else:
        return "Low"
def ats_score(match):
    if match > 80:
        return "Excellent"
    elif match > 60:
        return "Good"
    elif match > 40:
        return "Average"
    else:
        return "Poor"
    
def suggest_improvements(missing, match_percent):
    suggestions = []

    if match_percent < 30:
        suggestions.append("Start with basic programming and build 2 projects")

    if "python" in missing:
        suggestions.append("Learn Python (mandatory skill)")

    if "machine learning" in missing:
        suggestions.append("Build Machine Learning projects")

    if "deep learning" in missing:
        suggestions.append("Take Deep Learning certification")

    if match_percent > 70:
        suggestions.append("Good resume — add advanced projects to stand out")

    return suggestions   
def detect_level(text):
    text = text.lower()
    if "experience" in text or "project" in text:
        return "Intermediate"
    return "Beginner" 
# =========================
# 11. Resume Ranking System
# =========================


# Sample resumes for ranking
resumes = [
    "Python machine learning deep learning NLP",
    "React JavaScript HTML CSS frontend",
    "Accounting finance taxation auditing",
    "Recruitment hiring HR operations"
]

# Use best matched role from job matching
job_role = job_titles[best_match_index]
ranked_resumes = []
from sklearn.metrics.pairwise import cosine_similarity

job_text = cleanResume(
    job_descriptions[job_descriptions["Role"] == job_role]["Description"].values[0]
)

job_vector = tfidf.transform([job_text])

resume_cleaned = [cleanResume(r) for r in resumes]
resume_vectors = tfidf.transform(resume_cleaned)

scores = cosine_similarity(resume_vectors, job_vector)

for i, resume in enumerate(resumes):
    similarity = scores[i][0]

    # Extract skills for this resume
    resume_skills = extract_skills(resume, skills_db[job_role])
    job_skills = skills_db[job_role]

    # Skill score
    skill_score = len(resume_skills) / len(job_skills)

    # Final hybrid score
    final_score = 0.85 * similarity + 0.15 * skill_score

    ranked_resumes.append((resume, final_score))

# Sort
ranked_resumes = sorted(ranked_resumes, key=lambda x: x[1], reverse=True)

print("\nResume Ranking for", job_role, ":\n")

for i, (resume, score) in enumerate(ranked_resumes):
    print(f"{i+1}. Score: {score:.2f} → {resume}")


# =========================
# 13. Multiple Resume Testing
# =========================
import os

# =========================
# Safe Role Mapping
# =========================
def get_valid_role(predicted_role):
    if predicted_role in skills_db:
        return predicted_role
    else:
        return None


# =========================
# Improved Matching Level
# =========================
def match_level(score):
    if score >= 75:
        return "High"
    elif score >= 40:
        return "Moderate"
    else:
        return "Low"


# =========================
# Multiple Resume Testing (FINAL FIXED)
# =========================

pdf_files = ["sample_resume.pdf", "sample_resume2.pdf"]


def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        # Try pdfplumber first
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text += t + " "

        if text.strip():
            return text

    except:
        pass

    # Fallback to PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()

    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return text

for file in pdf_files:
    print("\nProcessing:", file)

    # File check
    if not os.path.exists(file):
        print(" File not found:", file)
        continue

    # Extract text
    resume_text = extract_text_from_pdf(file)

    if not resume_text.strip():
        print(" Empty or unreadable PDF")
        continue

    print("\nExtracted Resume Text:\n", resume_text[:200])

    # Clean + vectorize
    cleaned = cleanResume(resume_text)
    vector =  tfidf.transform([cleaned])

    # Predict role (ML)
    import numpy as np

    scores = model.decision_function(vector)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)

    confidence = max(probs[0])
      
  
    # =========================
    #  ROLE CORRECTION USING SKILLS
    # =========================
    scores = {}

    for r in skills_db:
        skills = extract_skills(resume_text, skills_db[r])
        scores[r] = sum([skill_weights.get(skill, 1) for skill in skills])

    best_role = max(scores, key=scores.get)

    if scores[best_role] >= 2:
     role = best_role

    # =========================

    role = get_valid_role(role)

    if role is None:
        print("\n Could not determine role properly")
        print("Suggested: Improve resume with clear technical skills")
        continue

   

    # =========================
    # Matching
    # =========================

    resume_skills = extract_skills(resume_text, skills_db[role])
    job_skills = skills_db[role]
    if confidence < 0.35:
     print("\n Low confidence prediction")
    if len(resume_skills) == 0:
     print("\n No relevant skills found")
     print(" Resume not suitable for this role")
    
     role = "General"   #  Override role
     print("Final Predicted Role:", role)

    

     continue
   

    match_percent, matched = calculate_match(resume_skills, job_skills)

    if match_percent < 20:
      print("\n Very low match")
      continue
    print("\nFinal Predicted Role:", role)
   
    missing = missing_skills(resume_skills, job_skills)
  

    print("Match %:", round(match_percent, 2))
    print("ATS Score:", ats_score(match_percent))
    print("Match Level:", match_level(match_percent))
    print("Matched Skills:", matched)
    print("Missing Skills:", missing)

    # =========================
    # Suggestions
    # =========================

    suggestions = suggest_improvements(missing, match_percent)

    print("Suggestions:")
    if match_percent >= 75:
        print("- Great match! Your resume fits the job role well.")
    else:
        for s in suggestions:
            print("-", s)
# candidate level detection ---
    level = detect_level(resume_text)
    print("Candidate Level:", level)


