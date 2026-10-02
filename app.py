from flask import Flask, render_template, request
import pdfplumber
import joblib
import re

app = Flask(__name__)

# load model

model = joblib.load("resume_classifier.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")


# SKILLS DATABASE
# =========================

skills_db = {

    "Data Science": [
        "python",
        "machine learning",
        "pandas",
        "numpy",
        "deep learning"
    ],

    "Web Development": [
        "html",
        "css",
        "javascript",
        "react"
    ],

    "HR": [
        "recruitment",
        "payroll",
        "hiring"
    ],

    "Finance": [
        "accounting",
        "finance",
        "taxation",
        "auditing",
        "excel"
    ],

    "General": [
        "communication",
        "ms office",
        "teamwork"
    ]
}


# CLEAN RESUME FUNCTION
# =========================

def cleanResume(text):

    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    return text.lower()

# PDF EXTRACTION FUNCTION
# =========================

def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

    return text


# =========================
# SKILL EXTRACTION
# =========================

def extract_skills(resume_text, skills):

    found = []

    for skill in skills:

        if skill.lower() in resume_text.lower():
            found.append(skill)

    return found

# MATCH CALCULATION
#---------------

def calculate_match(resume_skills, job_skills):

    matched = list(set(resume_skills) & set(job_skills))

    score = (len(matched) / len(job_skills)) * 100

    return score, matched

# route starts here
#---------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/hr-login')
def hr_login():
    return render_template('hr_login.html')


@app.route('/candidate-login')
def candidate_login():
    return render_template('candidate_login.html')


@app.route('/hr-dashboard')
def hr_dashboard():
    return render_template('hr_dashboard.html')

@app.route('/multi-results', methods=['POST'])
def multi_results():

    uploaded_files = request.files.getlist('resumes')

    results = []

    candidate_names = []
    candidate_scores = []

    ds = 0
    web = 0
    hr = 0
    finance = 0
    general = 0

    for uploaded_file in uploaded_files:

        file_path = "uploads/" + uploaded_file.filename

        uploaded_file.save(file_path)

        resume_text = extract_text_from_pdf(file_path)

        cleaned_resume = cleanResume(resume_text)

        vector = tfidf.transform([cleaned_resume])

        prediction = model.predict(vector)[0]

        role = le.inverse_transform([prediction])[0]

        if role not in skills_db:
            continue

        resume_skills = extract_skills(
            cleaned_resume,
            skills_db[role]
        )

        match_percent, matched = calculate_match(
            resume_skills,
            skills_db[role]
        )

        score = round(match_percent, 2)

        results.append({

            "filename": uploaded_file.filename,
            "role": role,
            "score": score

        })

        candidate_names.append(uploaded_file.filename)

        candidate_scores.append(score)

        # ROLE COUNTS

        if role == "Data Science":
            ds += 1

        elif role == "Web Development":
            web += 1

        elif role == "HR":
            hr += 1

        elif role == "Finance":
            finance += 1

        elif role == "General":
            general += 1

    return render_template(

        'multi_results.html',

        results=results,

        candidate_names=candidate_names,

        candidate_scores=candidate_scores,

        ds=ds,
        web=web,
        hr=hr,
        finance=finance,
        general=general
    )


@app.route('/candidate-dashboard')
def candidate_dashboard():
    return render_template('candidate_dashboard.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/results', methods=['POST'])
def results():

    uploaded_file = request.files['resume']

    # save uploaded file
    file_path = "uploads/" + uploaded_file.filename

    uploaded_file.save(file_path)

    # extract text from pdf
    resume_text = extract_text_from_pdf(file_path)

    # clean text
    cleaned_resume = cleanResume(resume_text)

    # convert text into vector
    vector = tfidf.transform([cleaned_resume])

    # predict role
    prediction = model.predict(vector)[0]

    # convert encoded label back to role name
    role = le.inverse_transform([prediction])[0]

    # handle unknown categories safely
    if role not in skills_db:

        return f"""
        Predicted Role: {role}
        <br><br>
        No skills database found for this category.
        """

    # extract skills
    resume_skills = extract_skills(
        cleaned_resume,
        skills_db[role]
    )

    # calculate ATS match
    match_percent, matched = calculate_match(
        resume_skills,
        skills_db[role]
    )
    missing = list(
    set(skills_db[role]) - set(matched)
    )
    if match_percent >= 80:

     level = "Advanced"

    elif match_percent >= 50:

     level = "Intermediate"

    else:

     level = "Beginner" 

    # send results to html page
    return render_template(
    'results.html',
    role=role,
    score=round(match_percent, 2),
    matched=matched,
    missing=missing,
    level=level
    )

# run flask


from threading import Timer
import webbrowser

def open_browser():
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(2, open_browser).start()
    app.run(debug=True, use_reloader=False)






