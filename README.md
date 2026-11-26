
# SmartCV AI

An intelligent resume analysis and job matching platform
powered by Natural Language Processing and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![NLP](https://img.shields.io/badge/NLP-spaCy-green)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

## Project Overview

SmartCV AI is a full-stack artificial intelligence project
designed to help candidates analyze their resumes, identify
relevant skills, and assess their compatibility with job
descriptions.

The platform aims to provide transparent, explainable,
and actionable resume improvement recommendations.

## Features

### Implemented

- PDF resume text extraction with PyMuPDF
- Text cleaning and Unicode normalization
- English and French language detection
- Tokenization and lemmatization
- Part-of-speech tagging
- Named entity recognition
- Bilingual ESCO skill repository
- Technical skill recognition
- Ambiguous synonym filtering
- Contextual skill classification
- Negation detection
- Precision, Recall and F1-score evaluation
- Automated unit testing

### Upcoming

- Resume section detection
- Professional experience extraction
- Education and certification extraction
- Job description analysis
- TF-IDF and BM25 matching
- Word embeddings
- Sentence Transformers
- Hybrid semantic matching
- Explainable resume recommendations
- REST API with FastAPI
- Spring Boot backend
- React and TypeScript frontend
- PostgreSQL database
- Docker and CI/CD

## Architecture

The project is designed around four main components:

| Component | Technology | Status |
|---|---|---|
| AI Engine | Python, spaCy, scikit-learn | In development |
| AI API | FastAPI | Planned |
| Backend | Java, Spring Boot | Planned |
| Frontend | React, TypeScript | Planned |
| Database | PostgreSQL | Planned |
| Deployment | Docker, GitHub Actions | Planned |

## NLP Pipeline

    PDF Resume
        |
        v
    Text Extraction
        |
        v
    Text Cleaning
        |
        v
    Linguistic Analysis
        |
        v
    ESCO Skill Matching
        |
        v
    Alias Filtering
        |
        v
    Contextual Classification
        |
        v
    Structured Skill Profile

## Datasets

The project uses external datasets and knowledge bases
for research and development.

| Dataset | Size |
|---|---:|
| Resume Dataset | 2,484 resumes |
| LinkedIn Job Postings | 123,849 postings |
| ESCO Skills | 13,896 concepts per language |
| Synthetic Candidate Matching | 10,000 resumes |
| Skill Evaluation Dataset | 24 synthetic examples |

Original datasets and personal resume documents are not
included in this repository.

## Experimental Results

### Skill Extractor V3.2

| Metric | Result |
|---|---:|
| Processed resumes | 2,482 |
| Processing failures | 0 |
| Unique detected skills | 1,419 |
| Document-skill associations | 19,125 |
| Average skills per resume | 7.71 |
| Contextual rejections | 397 |
| Batch execution time | 31.60 seconds |

### Synthetic Evaluation

| Metric | Skill Detection | Context Classification |
|---|---:|---:|
| Precision | 100% | 76.92% |
| Recall | 100% | 76.92% |
| Micro F1-score | 100% | 76.92% |
| Macro F1-score | 100% | 85% |
| Exact Match Accuracy | 100% | 87.50% |

These results were obtained on a small bilingual synthetic
evaluation dataset containing 24 examples and 39 expected
document-skill associations.

They should not be interpreted as performance guarantees
on unseen real-world resumes.

Independent evaluation is planned.

## Skill Extractor Evolution

| Version | Approach | Document-skill pairs |
|---|---|---:|
| V1 | Lexical matching | 39,382 |
| V2 | Alias filtering | 19,269 |
| V3 | Contextual analysis | 18,957 |
| V3.2 | Repository and tokenizer corrections | 19,125 |

## Project Structure

    SmartCV-AI/
    ├── ai_engine/
    │   ├── extraction/
    │   ├── preprocessing/
    │   ├── eda/
    │   ├── nlp/
    │   ├── matching/
    │   ├── evaluation/
    │   ├── models/
    │   └── api/
    ├── backend/
    ├── frontend/
    ├── data/
    ├── reports/
    ├── tests/
    ├── docs/
    ├── requirements.txt
    ├── .gitignore
    └── README.md

## Getting Started

### Create a virtual environment

    python -m venv .venv

### Activate the environment on Windows

    .venv\Scripts\Activate.ps1

### Install dependencies

    pip install -r requirements.txt

### Install spaCy language models

    python -m spacy download en_core_web_sm
    python -m spacy download fr_core_news_sm

### Run unit tests

    python -m pytest tests/ -v

### Build the skill repository

Requires the English and French ESCO CSV files
in the expected local dataset directories.

    python -m ai_engine.nlp.skill_repository

### Run skill evaluation

    python -m ai_engine.evaluation.evaluate_skills

## Privacy and Responsible AI

- Personal resume data must remain private.
- Original datasets are excluded from version control.
- Matching results assess document compatibility,
  not the probability of being hired.
- Recommendations must not encourage candidates
  to invent qualifications or professional experience.
- The system is designed to support candidates,
  not replace human recruitment decisions.

## Roadmap

- [x] Project initialization
- [x] Dataset collection
- [ ] Exploratory data analysis validation
- [x] PDF extraction
- [x] Text preprocessing
- [x] Linguistic analysis
- [x] Skill extraction
- [x] Synthetic skill evaluation
- [ ] Structured candidate profile
- [ ] Job description analysis
- [ ] Machine learning matching
- [ ] Semantic matching
- [ ] Recommendation engine
- [ ] Independent evaluation
- [ ] REST API
- [ ] Spring Boot backend
- [ ] React frontend
- [ ] Docker deployment
- [ ] CI/CD pipeline

## Project Status

Currently developing the NLP engine.

Current milestone: Skill Extractor V3.2.

Next milestone: Resume Section Detection.

## Author

Personal software engineering and artificial intelligence
portfolio project.
