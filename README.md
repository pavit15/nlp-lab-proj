# Resume Screening System

An intelligent web-based application that automates resume screening using Natural Language Processing (NLP) and Machine Learning techniques.

The system evaluates resumes against a job description and ranks candidates based on relevance using a hybrid scoring approach.

<img width="1216" height="908" alt="WhatsApp Image 2026-04-29 at 1 41 24 AM" src="https://github.com/user-attachments/assets/8fe452cc-401e-4624-b1d4-e83b5ec12520" />

---

## Features

-  Upload resumes (PDF / TXT)
-  Paste resume text directly
-  Input job description
-  Hybrid scoring system:
    - TF-IDF similarity (keyword-based)
    - Semantic similarity (Sentence Transformers)
    - Skill coverage analysis
-  Skill breakdown:
    - Matched skills
    - Missing skills
   - Extra skills
-  Resume ranking system
-  Fast and efficient processing
-  Flask-based web interface

---

##  How It Works

1. User inputs job description and resumes  
2. System extracts and preprocesses text  
3. Skills are identified using keyword matching  
4. Text is converted into vectors using TF-IDF  
5. Semantic embeddings are generated using SentenceTransformer  
6. Cosine similarity is computed  
7. Final score is calculated using hybrid scoring  

###  Scoring Formula
Final Score = 0.4 × TF-IDF + 0.3 × Embedding + 0.3 × Skill Coverage

---

## Tech Stack

- **Backend:** Flask (Python)
- **ML/NLP:**
  - Scikit-learn (TF-IDF, Cosine Similarity)
  - Sentence-Transformers (Semantic Embeddings)
- **PDF Parsing:** PyPDF2
- **Frontend:** HTML, CSS, JavaScript

##  Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/pavit15/nlp-lab-proj.git
cd nlp-lab-proj
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. Run the application
```
python app.py
```

4. Open in browser
```
http://127.0.0.1:5000/
```

## Sample Output

<img width="1181" height="706" alt="WhatsApp Image 2026-04-29 at 1 41 34 AM" src="https://github.com/user-attachments/assets/6e4f8ade-3b90-4629-b90d-286e2af26960" />

Match Score (e.g., 85.88%)

Skill Coverage (e.g., 100%)

Matched Skills (Python, SQL, AWS, etc.)

Ranked resumes

Resume preview
