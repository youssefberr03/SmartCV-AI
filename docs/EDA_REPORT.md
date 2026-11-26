
# Exploratory Data Analysis Report

## 1. Objective

This report presents the exploratory data analysis conducted
during the development of SmartCV AI.

The objectives are to:

- Understand the available datasets.
- Identify data quality limitations.
- Analyze document length and vocabulary distributions.
- Identify professional category imbalances.
- Validate multilingual ESCO identifiers.
- Understand the synthetic matching evaluation dataset.
- Guide future NLP and machine learning decisions.

## 2. Datasets Overview

| Dataset | Available Records | Analyzed Records |
|---|---:|---:|
| Resume Dataset | 2,484 | 2,483 |
| LinkedIn Job Postings | 123,849 | 9,908 |
| ESCO English | 13,896 | 13,896 |
| ESCO French | 13,896 | 13,896 |
| Synthetic Resumes | 10,000 | 10,000 |
| Synthetic Jobs | 2,500 | 2,500 |
| Synthetic Matches | 2,500 | 2,500 |

## 3. Methodology

The exploratory analysis was implemented using:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

The analysis includes:

- Descriptive statistics
- Document length analysis
- Vocabulary analysis
- Lexical diversity
- Word frequency analysis
- Bigram and trigram analysis
- Zipf distribution visualization
- Professional category distribution
- Missing value analysis
- ESCO identifier consistency analysis
- Structured dataset inspection

## 4. Resume Dataset Analysis

### 4.1 Corpus Statistics

| Metric | Value |
|---|---:|
| Analyzed documents | 2,483 |
| Professional categories | 24 |
| Total tokens | 2,029,482 |
| Vocabulary size | 40,565 |
| Corpus lexical diversity | 2.00% |
| Mean document lexical diversity | 47.34% |

### 4.2 Document Length

| Statistic | Tokens |
|---|---:|
| Minimum | 108 |
| First quartile | 655 |
| Median | 761 |
| Mean | 817.35 |
| Third quartile | 940 |
| Maximum | 5,175 |
| Standard deviation | 376.50 |

The corpus contains substantial variability in document
length.

Long documents may require section-based processing or
chunking when using transformer models.

Long resumes should not be automatically considered
invalid documents.

### 4.3 Category Distribution

The dataset contains 24 professional categories.

Most categories contain between 96 and 120 resumes.

However, several categories are underrepresented:

| Category | Documents |
|---|---:|
| Information Technology | 120 |
| Business Development | 120 |
| Agriculture | 63 |
| Automobile | 36 |
| BPO | 22 |

This imbalance should be considered when evaluating
performance across professional domains.

### 4.4 Limitations

- Unequal professional category representation.
- Limited evidence of multilingual coverage.
- Variable resume formats and document lengths.
- One record difference between the available inventory
  and the analyzed text corpus.
- Independent skill annotations are not yet available.

## 5. LinkedIn Job Postings Analysis

### 5.1 Sampling Strategy

The original dataset contains 123,849 job postings.

An approximately 8% sample was analyzed.

| Parameter | Value |
|---|---:|
| Original records | 123,849 |
| Sampled records | 9,908 |
| Sampling fraction | 0.08 |
| Random state | 42 |

Sampling reduces computational requirements while
supporting reproducible exploratory analysis.

The sample is not guaranteed to represent every
professional category equally.

### 5.2 Corpus Statistics

| Metric | Value |
|---|---:|
| Total tokens | 5,438,341 |
| Vocabulary size | 86,448 |
| Corpus lexical diversity | 1.59% |
| Mean document lexical diversity | 57.10% |

### 5.3 Document Length

| Statistic | Tokens |
|---|---:|
| Minimum | 1 |
| First quartile | 314 |
| Median | 502 |
| Mean | 548.88 |
| Third quartile | 729 |
| Maximum | 2,849 |
| Standard deviation | 318.51 |

Some job descriptions contain insufficient textual
information.

Input quality validation will therefore be introduced
during the job analysis milestone.

## 6. ESCO Knowledge Base Analysis

| Metric | English | French |
|---|---:|---:|
| Concepts | 13,896 | 13,896 |
| Unique identifiers | 13,896 | 13,896 |
| Duplicate identifiers | 0 | 0 |
| Missing identifiers | 0 | 0 |

All 13,896 identifiers are shared between the two
language versions.

This supports a multilingual architecture based on
language-independent concept identifiers.

Identifier consistency does not guarantee that all
alternative labels are unambiguous.

Additional filtering and contextual analysis remain
necessary.

## 7. Synthetic Candidate Matching Dataset

### 7.1 Candidate Profiles

The dataset contains 10,000 structured candidate profiles.

Each profile includes:

- Candidate identifier
- Professional role
- Seniority
- Years of experience
- Industry
- Education
- Skills
- Professional summary
- Experience descriptions

| Metric | Value |
|---|---:|
| Candidate profiles | 10,000 |
| Minimum skills per profile | 5 |
| Maximum skills per profile | 8 |
| Mean skills per profile | 6.4888 |
| Total skill entries | 64,888 |
| Experience bullets per profile | 3 |
| Total experience bullets | 30,000 |

### 7.2 Job Profiles

The dataset contains 2,500 structured job profiles.

| Metric | Value |
|---|---:|
| Job profiles | 2,500 |
| Minimum mandatory skills | 3 |
| Maximum mandatory skills | 5 |
| Mean mandatory skills | 3.9972 |
| Optional skills per offer | 1 |
| Responsibilities per offer | 3 |
| Requirements per offer | 3 |

The distinction between mandatory and optional skills
supports experimentation with weighted matching.

### 7.3 Matching References

The matching dataset contains:

- 2,500 job-level matching records.
- 30 relevant candidate identifiers per record.
- 75,000 relevance associations.

These references will be used to benchmark candidate
retrieval and ranking algorithms.

Potential evaluation metrics include:

- Precision@K
- Recall@K
- Mean Reciprocal Rank
- Mean Average Precision
- Normalized Discounted Cumulative Gain

### 7.4 Data Quality

No null values were detected in the three synthetic
matching datasets.

However, additional validation is required for:

- Identifier uniqueness
- Referential integrity
- Duplicate relevance associations
- Empty strings
- Invalid experience values
- Distribution of professional roles
- Relevance label quality
- Potential generation bias

### 7.5 Synthetic Dataset Limitations

The dataset is artificially generated and highly
structured.

Candidate profiles and job descriptions have limited
structural variability.

The matching references must not be interpreted as
real recruitment decisions.

Performance on synthetic data does not demonstrate
equivalent performance on real-world resumes.

Independent evaluation will be required.

## 8. Impact on Technical Architecture

| Observation | Technical Decision |
|---|---|
| Long resume documents | Section detection and chunking |
| Short job descriptions | Input quality validation |
| Category imbalance | Domain-specific evaluation |
| Repeated vocabulary | TF-IDF weighting |
| Variable document lengths | Normalized similarity and BM25 |
| Shared ESCO identifiers | Multilingual skill representation |
| Mandatory and optional skills | Weighted matching experiments |
| Synthetic relevance references | Ranking algorithm benchmarks |
| Large datasets | Batch processing |

## 9. Reproducibility

Run the EDA pipeline from the project root:

    python -m ai_engine.eda.run_eda

Generated reports are stored in:

    reports/eda/
    ├── resumes/
    ├── jobs/
    ├── esco/
    └── matching/

Original datasets and detailed generated reports are
excluded from version control for privacy and storage
reasons.

## 10. Conclusion

The exploratory data analysis provides a foundation
for the development of SmartCV AI.

The results support the existing bilingual skill
extraction architecture and guide future document
structuring, matching and evaluation strategies.

No immediate changes to the existing PDF extraction,
text preprocessing, linguistic analysis or skill
extraction pipelines are required based on the
observed statistics.

Further data validation and independent real-world
evaluation remain necessary.

The next development milestone is automatic resume
section detection.
