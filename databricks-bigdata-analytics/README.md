# Big Data Analytics on Databricks — Distributed SQL & Scalable Recommender Systems

**Two production-shaped Big Data workloads on the Databricks Lakehouse: distributed SQL analytics over 573,000 clinical trial records, and an MLflow-tracked ALS recommender system trained on 200,000 Steam user–game interactions.**

[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat-square&logo=databricks&logoColor=white)]()
[![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=flat-square&logo=apachespark&logoColor=white)]()
[![PySpark](https://img.shields.io/badge/PySpark-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white)]()
[![Spark SQL](https://img.shields.io/badge/Spark_SQL-E25A1C?style=flat-square)]()
[![MLlib](https://img.shields.io/badge/MLlib_ALS-E25A1C?style=flat-square)]()

---

## Why This Project

Most of my portfolio demonstrates the modelling and deployment layers of data science. This project demonstrates the **scale layer**: what changes when the data no longer fits comfortably on one machine, and how the Databricks Lakehouse platform (Unity Catalog, distributed Spark compute, managed MLflow) handles it. Every query and training run here executed on a Databricks cluster against data governed in Unity Catalog.

Two workloads, two engines:

| Workload | Engine | Scale | Question answered |
| --- | --- | --- | --- |
| Clinical trials registry analysis | Spark SQL + PySpark | 573K records, 30 columns | What does three decades of US clinical research look like? |
| Steam game recommender | MLlib ALS + MLflow | 200K interactions, 12.4K users × 5.2K games (99.8% sparse) | Which games should each user see next? |

---

## Workload 1 — Clinical Trials at Scale (Spark SQL)

**Notebook:** [`notebooks/01-clinical-trials-spark-sql.ipynb`](notebooks/01-clinical-trials-spark-sql.ipynb)

<details>
<summary><strong>📖 Expand Full Details</strong></summary>

### The Problem

ClinicalTrials.gov publishes every registered US clinical trial — over 570,000 records spanning study types, conditions, sponsors, statuses, and dates. A health-research analyst needs registry-level answers: what kinds of trials dominate, which conditions attract the most research, how long trials actually take, and how research activity in a specific disease area (Alzheimer's) has evolved over three decades.

### What I Built

A Spark SQL analysis on Databricks, reading from **Unity Catalog** volumes, mixing `%sql` and `%python` cells the way real Databricks work is done — SQL for set-based analysis, PySpark for data loading, quality checks, and visualisation of aggregated results.

### Engineering Decisions Worth Noting

| Decision | Rationale |
| --- | --- |
| `inferSchema` for exploration, with explicit note on production trade-off | Schema inference costs an extra pass over the data; a production pipeline would define schemas explicitly for performance and reliability |
| Null audit before any analysis | Completion Date: 16,702 nulls (2.9%) — expected, since many trials are ongoing; every downstream question filters accordingly |
| `LATERAL VIEW EXPLODE` for multi-valued condition strings | Conditions arrive pipe-delimited ("HIV Infections\|Sexually Transmitted Diseases"); exploding is the SQL-native way to count each condition independently |
| Data-quality guard on Study Type | Raw frequencies surfaced numeric junk values ("100", "60") masquerading as study types — filtered before reporting |
| Aggregate in Spark, visualise in pandas | The per-year Alzheimer's aggregation collapses 573K rows to a few dozen — only then is `toPandas()` safe and sensible |

### Key Findings

- **Interventional + observational studies account for over 99.5%** of the registry
- **"Healthy" is the most-studied condition** (10,873 trials) — a signature of Phase I safety trials recruiting healthy volunteers — followed by breast cancer (8,511) and prostate cancer (4,306)
- **Mean trial duration is 35.12 months** (~2 years 11 months), computed from 555,263 valid studies — a 96.9% inclusion rate after date-quality filtering
- **Completed Alzheimer's trials grew from single digits in the 1990s to 60–95 per year through 2009–2018**, tracking the rise of amyloid-targeting research and ageing-population funding priorities

</details>

---

## Workload 2 — Steam Recommender System (MLlib ALS + MLflow)

**Notebook:** [`notebooks/02-steam-als-recommender-mlflow.ipynb`](notebooks/02-steam-als-recommender-mlflow.ipynb)

<details>
<summary><strong>📖 Expand Full Details</strong></summary>

### The Problem

A games platform holds 200,000 user–game interaction records — purchases and play-hours — for 12,393 users across 5,155 games. It wants personalised recommendations, but the raw signal is messy: **45% of purchased games were never played**, play-hours are extremely right-skewed (median 4.5 hours, max 11,754), and the user–item matrix is **99.8% sparse**, with 46% of users having played only a single game.

### What I Built

An end-to-end recommender pipeline on Databricks using **Spark MLlib's Alternating Least Squares (ALS)**:

1. **EDA in Spark SQL** — quantified the purchase-without-play problem (58,750 of 129,511 purchases), skewness, and sparsity
2. **Signal design** — chose play-hours over purchases as the preference signal (a purchase without play is awareness, not preference), log-transformed to compress the heavy tail
3. **ID engineering** — generated integer game IDs with `StringIndexer` rather than relying on an external mapping file
4. **Baseline → tuned** — implicit-feedback ALS baseline, then an **18-run hyperparameter grid** (rank × regParam × implicitPrefs), every run logged to **MLflow** with parameters, metrics, and artifacts
5. **Qualitative validation** — inspected recommendations for real users against their play histories, not just aggregate metrics

### Results

| Model | RMSE | Notes |
| --- | --- | --- |
| Baseline ALS (implicit, rank=10, reg=0.1) | 2.41 | Confidence-weighted preference model |
| Best implicit configuration | 2.3973 | Tuning gains marginal in implicit mode |
| **Best explicit configuration** | **1.4568** | **Log-transformed hours behave like explicit ratings** |

### The Interesting Finding

Play-hours are textbook *implicit* feedback — yet **explicit-mode ALS substantially outperformed implicit mode** (RMSE 1.4568 vs 2.3973). The log transformation converts raw hours into a continuous preference scale that explicit ALS can optimise directly, while implicit mode only treats values as confidence weights on a binary signal. Qualitatively, the explicit model also produced richer, less over-specialised recommendations (the implicit model tunnel-visioned on a single genre for FPS-heavy users). The lesson: **the right ALS mode depends on how the signal is transformed, not just on how the data was collected** — an assumption worth testing, not defaulting.

Tuning also showed **lower rank wins under extreme sparsity** — rank 10 beat rank 50 consistently, because 99.8% sparsity cannot meaningfully populate 50 latent dimensions.

</details>

---

## Tech Stack

`Databricks` · `Unity Catalog` · `Apache Spark` · `Spark SQL` · `PySpark` · `MLlib (ALS)` · `MLflow` · `StringIndexer` · `Matplotlib` · `pandas (post-aggregation)`

## Data

| Workload | Source | Records | Shape |
| --- | --- | --- | --- |
| Clinical trials | [ClinicalTrials.gov](https://clinicaltrials.gov/) registry extract | 572,935 trials | 30 columns incl. multi-valued condition strings |
| Recommender | [steam-200k](https://www.kaggle.com/datasets/tamber/steam-video-games) | 200,000 interactions | 12,393 users × 5,155 games, 99.8% sparse |

## Repository Structure

```
databricks-bigdata-analytics/
├── notebooks/
│   ├── 01-clinical-trials-spark-sql.ipynb      # Distributed SQL analytics (Unity Catalog)
│   └── 02-steam-als-recommender-mlflow.ipynb   # ALS recommender + MLflow experiment tracking
└── README.md
```

> **Note on reproducibility:** Both notebooks were developed and executed on a Databricks workspace with data mounted in Unity Catalog volumes. Cell outputs are preserved so the full analysis is readable directly on GitHub without a cluster. To re-run, upload the datasets to your own workspace and update the two volume paths.

---

📧 [Email](mailto:michizler@gmail.com) · 💼 [LinkedIn](https://linkedin.com/in/bright-uzosike) · 🐙 [Back to full portfolio](https://github.com/michizler/data-science-projects)
