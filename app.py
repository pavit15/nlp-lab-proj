from flask import Flask, request, jsonify, render_template
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

app = Flask(__name__)

# load embedding model for semantic similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# list of skills we want to detect
SKILL_KEYWORDS = [
    "python", "sql", "aws", "gcp", "docker", "kubernetes",
    "spark", "hadoop", "airflow", "kafka"
]

# basic text cleaning
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# find skills present in text
def extract_skills(text):
    return list(set([s for s in SKILL_KEYWORDS if s in text]))

# compute tf-idf similarity
def tfidf_similarity(jd, resume):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([jd, resume])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# compute embedding-based similarity
def embedding_similarity(jd, resume):
    embeddings = embedder.encode([jd, resume])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# calculate how many jd skills are matched
def skill_coverage(jd_skills, res_skills):
    if not jd_skills:
        return 0
    return len(set(jd_skills) & set(res_skills)) / len(jd_skills)

# combine all scoring methods
def compute_score(jd, resume):
    jd_clean = preprocess_text(jd)
    res_clean = preprocess_text(resume)

    jd_skills = extract_skills(jd_clean)
    res_skills = extract_skills(res_clean)

    tfidf = tfidf_similarity(jd_clean, res_clean)
    embed = embedding_similarity(jd_clean, res_clean)
    skill = skill_coverage(jd_skills, res_skills)

    final_score = (0.4 * tfidf) + (0.3 * embed) + (0.3 * skill)

    return {
        "score": round(final_score * 100, 2),
        "jd_skills": jd_skills,
        "comparison": {
            "matched": [{"skill": s, "context": ""} for s in set(jd_skills) & set(res_skills)],
            "missing": list(set(jd_skills) - set(res_skills)),
            "extra": list(set(res_skills) - set(jd_skills)),
            "coverage": round(skill * 100, 2)
        }
    }

# extract text from uploaded file (pdf or txt)
def extract_text_from_file(file):
    if file.filename.endswith('.pdf'):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode('utf-8', errors='ignore')

# homepage route
@app.route('/')
def home():
    return render_template('index.html')

# main screening endpoint
@app.route('/screen', methods=['POST'])
def screen():
    try:
        jd = request.form.get('job_description', '')

        resume_texts = request.form.getlist('resume_texts[]')
        resume_names = request.form.getlist('resume_names[]')

        # read uploaded files and convert to text
        uploaded_files = request.files.getlist('resumes[]')

        for file in uploaded_files:
            if file and file.filename:
                text = extract_text_from_file(file)
                if text.strip():
                    resume_texts.append(text)
                    resume_names.append(file.filename)

        results = []

        for i, res in enumerate(resume_texts):
            if not res.strip():
                continue

            score_data = compute_score(jd, res)

            results.append({
                "name": resume_names[i] if i < len(resume_names) else f"Resume {i+1}",
                "rank": 0,
                "preview": res[:300],
                **score_data
            })

        # sort resumes by score
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        # assign rank based on sorted order
        for i, r in enumerate(results):
            r['rank'] = i + 1

        return jsonify({
            "success": True,
            "total": len(results),
            "jd_skills": extract_skills(preprocess_text(jd)),
            "results": results
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# run the app
if __name__ == '__main__':
    app.run(debug=True)