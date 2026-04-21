# Data Science & Analytics Portfolio

**Turning complex data into decisions that matter — across finance, construction, telecommunications, transportation, and macroeconomics.**

[![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)]()
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)]()
[![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)]()

---

## About This Portfolio

This repository brings together five end-to-end data science projects spanning different industries, techniques, and business problems. Each project follows a consistent approach: start with a real business question, explore and prepare the data rigorously, build and validate models against statistical standards, and translate the results into actionable recommendations with quantified impact.

The work covers the full analytics spectrum — from interactive dashboards and hypothesis testing through regression modelling, time series forecasting, and machine learning classification deployed via REST APIs and containerised microservices.

### What You'll Find

| Domain             | Techniques                                       | Deliverables                                 |
| ------------------ | ------------------------------------------------ | -------------------------------------------- |
| Macroeconomics     | Dashboard design, DAX, data modelling            | Interactive Power BI report                  |
| Construction       | MLR, logistic regression, hypothesis testing     | Predictive formula, interactive calculator   |
| Telecommunications | Decision trees, RFM analysis, FastAPI deployment | Deployed churn prediction API                |
| Finance            | Holt-Winters, ARIMA, ARIMA+GARCH                 | 365-day stock price forecast with volatility |
| Transportation     | Gradient boosting, SHAP, MLflow, Docker Compose  | Dockerised dynamic pricing service           |

---

## Projects

### 1. 🌍 Global Economic Outlook — Interactive Dashboard (2001–2020)

**Exploring 20 years of economic performance across 180+ countries using the IMF World Economic Outlook dataset.**

[![Power BI](https://img.shields.io/badge/View_Live_Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://app.powerbi.com/groups/me/reports/68c0fd03-43b3-4ce8-a175-1539d2cb1cb0/c35dfa7c01014060d9a9?experience=power-bi)

<details>
<summary><strong>📖 Expand Full Project Details</strong></summary>

#### The Problem

A thinktank scenario required an analytical tool to help policymakers and the general public understand patterns of economic performance at the country and country-group level. The raw IMF dataset — 44 indicators across 180+ countries over 20 years — was too dense for manual analysis.

#### What I Built

An interactive Power BI dashboard built on a **star schema** data model with one fact table and two dimension tables. The dashboard presents six headline indicators through multiple coordinated visualisations, all responsive to country, region, income group, and year-range filters.

#### Data Model

```
┌──────────────────┐       ┌──────────────────┐
│   Countries       │       │   Indicators      │
│ (Dimension)       │       │ (Dimension)       │
├──────────────────┤       ├──────────────────┤
│ CountryKey (PK)   │       │ Indicator Code    │
│ ISO               │◄──┐   │ Description       │
│ Country           │   │   └────────┬─────────┘
│ Region            │   │            │
│ Income Group      │   │            │ 1:*
└──────────────────┘   │   ┌────────▼─────────┐
              1:* (ISO) └───┤  Economic Data     │
                            │  (Fact)            │
                            └──────────────────┘
```

#### Key Visualisations

- **KPI Cards with Sparklines** — six headline indicators with embedded trendlines and colour-coded cues
- **Top 5 Bar Chart** — dynamically ranks best-performing countries by GDP growth
- **Interactive Line Chart** — tracks any indicator over time via tile slicer switching
- **Radar Chart** — multi-dimensional economic profile powered by a `SWITCH` DAX measure
- **Flexible Slicers** — Country, Region, Income Group, Year Range

#### DAX Measures

Custom measures for each indicator (card callout + sparkline pairs), a `DATATABLE`-based radar chart, and a field parameter for dynamic line chart switching.

#### Tech Stack

`Power BI` · `Power Query` · `DAX` · `Star Schema` · `Excel`

#### Data

| Source                     | Records        | Variables     | Period    |
| -------------------------- | -------------- | ------------- | --------- |
| IMF World Economic Outlook | 180+ countries | 44 indicators | 2001–2020 |

</details>

📂 **[View Project →](https://github.com/michizler/imf-globaleconomy-analytics/)**

---

### 2. 🏗️ Predicting Concrete Compressive Strength

**Reducing material costs by £21,600 per project through regression analysis on 1,030 historical mix trials.**

<details>
<summary><strong>📖 Expand Full Project Details</strong></summary>

#### The Problem

StrataForge Construction Materials Ltd., a UK-based firm specialising in commercial foundations and high-load structural slabs, had accumulated 1,030 concrete mix trial records but still relied on experience-based decision making. Conservative cement-heavy mixes were inflating costs and carbon emissions, while occasional 28-day strength failures triggered expensive rework cycles.

The business needed answers to three questions: _What drives compressive strength? Does fly ash reduce performance? Can we predict strength before pouring?_

#### What I Built

A complete statistical analysis pipeline in R, progressing through **12 candidate regression models** via forward stepwise selection, a **logistic regression** classifier for fly ash detection, and **three hypothesis tests** that proved fly ash can safely replace cement.

#### The Final Model

```
concrete_strength = 23.914
                  + 0.0974 × cement
                  − 2.545 × ln(superplasticizer)
                  − 0.2374 × water
                  + 9.759 × ln(age)
                  + 0.0683 × slag
```

| Metric                       | Value                   |
| ---------------------------- | ----------------------- |
| R²                           | **81.35%**              |
| All coefficients significant | p < 0.001               |
| VIF (multicollinearity)      | All between 1.01 – 1.49 |
| Assumptions passed           | 5 / 5                   |

#### Key Findings

| Finding                              | Evidence                      | Business Impact                                              |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------ |
| Cement and age are strongest drivers | Highest positive coefficients | Optimise cement dosage based on required curing time         |
| Water reduces strength               | β = −0.2374                   | Minimise water content; use superplasticizer for workability |
| **Fly ash does NOT reduce strength** | Kruskal-Wallis p = 0.2324     | Confidently substitute cement with cheaper, greener fly ash  |
| Concrete category irrelevant         | Kruskal-Wallis p = 0.3364     | Coarse vs fine texture has no effect on strength             |

#### Quantified Business Impact

| Metric                            | Before (Conservative) | After (Optimised)   |
| --------------------------------- | --------------------- | ------------------- |
| Cement per m³                     | 400 kg                | 310 kg              |
| Slag per m³                       | 0 kg                  | 100 kg              |
| Cement cost per m³                | £48.00                | £37.20              |
| **Total cost (2,000 m³ project)** | **£96,000**           | **£74,400**         |
| **Saving**                        |                       | **£21,600 (22.5%)** |
| CO₂ reduction                     |                       | ~180 tonnes         |

#### Deliverables

- Multiple Linear Regression model (R² = 81.35%)
- Logistic regression fly ash classifier (AIC = 145.12)
- Three hypothesis tests (Kruskal-Wallis, Chi-Square)
- Interactive React/Vite presentation with **live strength calculator**

#### Tech Stack

`R` · `ggplot2` · `dplyr` · `readxl` · `corrplot` · `car` · `caret` · `RVAideMemoire` · `React` · `Vite`

#### Data

| Source                    | Records          | Variables    | Type                     |
| ------------------------- | ---------------- | ------------ | ------------------------ |
| StrataForge materials lab | 1,030 mix trials | 9 continuous | Clean, no missing values |

</details>

📂 **[View Project →](https://github.com/michizler/strataforge-prediction/)**

---

### 3. 📡 Predicting Customer Attrition — Reder Telecommunications

**Building and deploying a 97%-accuracy churn prediction model via FastAPI for a telecom provider with 15% annual attrition.**

<details>
<summary><strong>📖 Expand Full Project Details</strong></summary>

#### The Problem

Reder Telecommunications, a Norwegian telecom provider headquartered in Oslo, was losing customers at a rate of 15% annually — with acquisition costs 5–7x higher than retention. The company had over 2,000 customer records with 20+ attributes (many stored as nested JSON), but no predictive capability to identify at-risk customers before they left.

Four key obstacles compounded the problem: limited per-segment analytics, pricing inconsistencies across identical usage patterns, no proactive outage management, and purely reactive retention — interventions only after intent to leave was signalled.

#### What I Built

An end-to-end machine learning pipeline from raw nested-JSON data to a deployed **FastAPI prediction endpoint**:

1. **Data Preprocessing** — parsed 9 nested JSON columns using `ast.literal_eval`, normalised into separate DataFrames, merged via CustomerID (12,483 rows × 55 columns)
2. **Feature Engineering** — RFM analysis (Recency, Frequency, Monetary) with quintile binning, one-hot/label/target encoding
3. **Feature Selection** — Mutual Information scoring identified top 20 predictors; payment behaviour features dominated
4. **Model Training** — Logistic Regression baseline (97% accuracy), then Decision Tree with `RandomizedSearchCV` hyperparameter tuning (4-fold CV)
5. **Deployment** — FastAPI REST API with Pydantic validation, serialised model via `pickle`

#### Top Features by Mutual Information

| Rank | Feature               | MI Score |
| ---- | --------------------- | -------- |
| 1    | `late_payment_rate`   | 0.97     |
| 2    | `payment_risk_score`  | 0.95     |
| 3    | `total_late_payments` | 0.93     |
| 4    | `total_interactions`  | 0.58     |
| 5    | `TimeSpent(minutes)`  | 0.57     |
| 6    | `NPS`                 | 0.52     |

Payment behaviour is the single strongest predictor of churn — more than engagement, satisfaction, or demographics.

#### Model Performance

| Metric    | Logistic Regression | Decision Tree (Tuned) |
| --------- | ------------------- | --------------------- |
| Accuracy  | 97.0%               | 96.9%                 |
| Precision | ~97%                | ~97%                  |
| Recall    | ~97%                | ~97%                  |
| F1-Score  | ~97%                | ~97%                  |

Best hyperparameters: `criterion='log_loss'`, `splitter='random'`, `max_depth=300`, `min_samples_split=6`, `min_samples_leaf=4`, `max_features='log2'`

#### API Deployment

```python
# POST /predict
{
  "records": [{ "late_payment_rate": 0.4, "NPS": 3, ... }]
}

# Response
{
  "prediction": 1,        # 1 = will churn
  "probability": 0.847    # 84.7% confidence
}
```

#### Cloud Cost Analysis

| Provider          | Estimated Monthly | Best For               |
| ----------------- | ----------------- | ---------------------- |
| GCP Cloud Run     | $30–75            | POC (pay-per-request)  |
| Azure App Service | $45–90            | Enterprise integration |
| AWS ECS Fargate   | $55–95            | Max flexibility        |

#### Deliverables

- Trained Decision Tree classifier (model.pkl)
- FastAPI prediction endpoint (app.py)
- RFM customer segmentation
- Cloud deployment cost comparison (AWS vs Azure vs GCP)
- Interactive React/Vite presentation

#### Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `FastAPI` · `Pydantic` · `Uvicorn` · `Pickle` · `Jupyter` · `React` · `Vite`

#### Data

| Source            | Records          | Raw Variables           | Engineered Features    |
| ----------------- | ---------------- | ----------------------- | ---------------------- |
| Reder Telecom CRM | 2,000+ customers | 20+ (incl. nested JSON) | 55 after normalisation |

</details>

📂 **[View Project →](https://github.com/michizler/reder-analytics/)**

---

### 4. 📈 Time Series Modelling for Share Price Prediction — US Bancorp

**Forecasting USB stock prices using an iterative approach through Holt-Winters, ARIMA, and ARIMA-GARCH — demonstrating why financial data requires specialised treatment.**

<details>
<summary><strong>📖 Expand Full Project Details</strong></summary>

#### The Problem

Financial markets exhibit **volatility clustering** — periods where large price swings follow large swings, and calm follows calm. This violates the constant-variance assumption of standard forecasting methods. An investment firm needed a model that could predict not just the expected price direction, but the _range of uncertainty_ around those forecasts.

#### What I Built

Three progressively sophisticated models fitted to 47 years (11,835 daily observations) of US Bancorp stock data, with rigorous diagnostic testing at each stage:

| Model                           | Ljung-Box p-value | Verdict                     |
| ------------------------------- | ----------------- | --------------------------- |
| Holt-Winters (α=0.942, γ=FALSE) | < 2.2e-16         | ✗ Residuals autocorrelated  |
| ARIMA(1,1,0) via auto.arima     | < 2.2e-16         | ✗ Same volatility problem   |
| **ARIMA(1,0,1) + sGARCH(1,1)**  | **0.2369**        | **✓ White noise residuals** |

The iterative failure-diagnosis-improvement cycle is intentional — it builds a rigorous justification for why GARCH is necessary on financial data.

#### The Breakthrough: GARCH

By modelling on **log returns** instead of raw prices and adding a sGARCH(1,1) conditional variance equation via the `rugarch` package, the model captured both the mean dynamics and time-varying volatility.

| Parameter   | Value     | Interpretation                         |
| ----------- | --------- | -------------------------------------- |
| μ           | 0.000603  | Mean daily return (~16% annualised)    |
| ar1         | −0.7185   | Autoregressive coefficient             |
| ma1         | 0.7068    | Moving average coefficient             |
| α₁          | 0.0938    | ARCH effect (impact of previous shock) |
| β₁          | 0.8924    | GARCH persistence (volatility memory)  |
| **α₁ + β₁** | **0.986** | **Highly persistent volatility**       |

All parameters significant at p < 0.01. Log-Likelihood: 34,139.66.

#### 365-Day Forecast

| Horizon       | Predicted Price | Change     |
| ------------- | --------------- | ---------- |
| Last observed | $31.93          | —          |
| Day 90        | ~$33.73         | +5.6%      |
| Day 180       | ~$35.39         | +10.8%     |
| **Day 365**   | **~$39.79**     | **+24.6%** |

Unlike Holt-Winters and ARIMA, the GARCH model also provides a **volatility forecast** (sigma) at each step — essential for confidence intervals and risk management.

#### Deliverables

- Three fitted models with full diagnostic output
- 365-day price forecast with return-to-price conversion
- Volatility (sigma) forecast showing decay from 6.7% to 1.8%
- Model performance comparison screenshots
- Interactive React/Vite presentation (17 slides)

#### Tech Stack

`R` · `quantmod` · `xts` · `forecast` · `rugarch` · `highcharter` · `tseries` · `ggplot2` · `React` · `Vite`

#### Data

| Source              | Records                   | Variables             | Period              |
| ------------------- | ------------------------- | --------------------- | ------------------- |
| Yahoo Finance (USB) | 11,835 daily observations | 7 (OHLCV + Adj Close) | May 1973 – Apr 2020 |

</details>

📂 **[View Project →](https://github.com/michizler/usb-stock-timeseries-model/)**

---

### 5. 🚆 Dynamic Ticket Pricing & Revenue Management — VoyageRail Ltd.

**An explainable machine learning framework for demand forecasting and price optimisation in the rail travel sector — containerised and served through a FastAPI + Streamlit stack.**

<details>
<summary><strong>📖 Expand Full Project Details</strong></summary>

#### The Problem

VoyageRail Ltd., a regional rail operator, had seen post-pandemic demand recover — but **revenue per seat had not kept pace**. Ticket volumes were stable, yet yield varied significantly across routes and travel windows, particularly during peak and near-peak periods. Static fare rules written a decade ago no longer responded to how modern travellers book.

Five core issues compounded the challenge: rule-based pricing unable to track real-time demand, revenue leakage from underpriced high-demand seats and overpriced low-demand ones, no segmentation-aware personalisation, operational complexity across multiple routes and classes, and — most critically — previous analytical models offered limited transparency, making it difficult for commercial teams to trust or act on their recommendations.

#### What I Built

A full end-to-end dynamic pricing solution delivered as a **containerised microservices architecture**:

1. **Data Pipeline** — cleaned and engineered 17 booking features including route characteristics, booking lead time, demand index, load factor, customer segment, and loyalty status
2. **Model Training** — compared Linear Regression, Random Forest, Gradient Boosting, and XGBoost; selected the best performer via 5-fold CV on RMSE/MAE/R²
3. **Explainability First** — SHAP values computed for every prediction, producing global feature importance and per-decision explanations that commercial teams can defend
4. **Experiment Tracking** — MLflow logs every run's parameters, metrics, and artifacts to `mlflow.db`, enabling reproducibility and model governance
5. **Two-Service Deployment** — FastAPI inference backend + Streamlit demo UI, orchestrated via Docker Compose

#### Top SHAP Features

| Rank | Feature              | Relative Importance |
| ---- | -------------------- | ------------------- |
| 1    | `demand_index`       | Highest             |
| 2    | `load_factor`        | Very High           |
| 3    | `days_before_travel` | High                |
| 4    | `seat_class`         | Moderate–High       |
| 5    | `route_category`     | Moderate            |

Demand signals dominate the pricing decision; customer and booking context refine it.

#### Solution Architecture

```
┌────────────────────────────────────────────────────────┐
│                Docker Compose Network                  │
│                                                        │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │   streamlit      │─────▶│    backend       │        │
│  │   (port 8501)    │ HTTP │   (FastAPI)      │        │
│  │                  │◀─────│   (port 8000)    │        │
│  │   User Interface │ JSON │   Inference API  │        │
│  └──────────────────┘      └────────┬─────────┘        │
│                                     │                  │
│                                     ▼                  │
│                           ┌──────────────────┐         │
│                           │   model.pkl +    │         │
│                           │   MLflow runs    │         │
│                           └──────────────────┘         │
└────────────────────────────────────────────────────────┘
```

#### Example Prediction

For a Leeds → York, Standard class, 21 days before travel booking with demand index 0.92:

| Metric                     | Value    |
| -------------------------- | -------- |
| **Predicted Ticket Price** | £37.30   |
| **Demand Index**           | 0.92     |
| **Days Before Travel**     | 21       |
| **Estimated Fare Band**    | Moderate |

Streamlit surfaces three result views: **Insight View** (with fare band and comparative chart), **API Response** (raw JSON), and **cURL** (exportable for testing outside the UI).

#### Cloud Cost Analysis (Containerised Stack)

| Provider             | Estimated Monthly | Best For                             |
| -------------------- | ----------------- | ------------------------------------ |
| **GCP Cloud Run**    | $41–78            | POC (pay-per-request, scale-to-zero) |
| Azure Container Apps | $72–102           | Enterprise integration               |
| AWS ECS Fargate      | $72–104           | Max ML ecosystem (SageMaker)         |

#### Total Project Investment

| Cost Type                                         | Range                 |
| ------------------------------------------------- | --------------------- |
| One-time setup (dev, testing, training)           | £11,300 – £20,000     |
| Recurring monthly (infra, monitoring, retraining) | £100 – £220           |
| **Year 1 total**                                  | **£12,500 – £22,640** |

ROI projection: payback in 3–6 months at typical rail operator scale via incremental yield gains.

#### Deliverables

- Trained regression model (model.pkl) with SHAP explainability
- FastAPI inference endpoint with Pydantic validation
- Streamlit interactive demo for stakeholders
- Docker Compose orchestration (2 services)
- MLflow experiment tracking (SQLite backend)
- Interactive React/Vite presentation (23 slides)
- Full project budget breakdown

#### Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SHAP` · `MLflow` · `FastAPI` · `Pydantic` · `Streamlit` · `Plotly` · `Docker` · `Docker Compose` · `React` · `Vite`

#### Data

| Source                    | Records             | Variables                                              | Target                      |
| ------------------------- | ------------------- | ------------------------------------------------------ | --------------------------- |
| VoyageRail booking system | Historical bookings | 17 features (route, timing, pricing, demand, customer) | `ticket_price` (regression) |

</details>

📂 **[View Project →](https://github.com/michizler/voyage-rails-analytics/)**

---

## Skills & Tools

### Languages & Frameworks

| Category                  | Technologies                                                         |
| ------------------------- | -------------------------------------------------------------------- |
| **Statistical Computing** | R (ggplot2, dplyr, forecast, rugarch, car, caret, corrplot)          |
| **Machine Learning**      | Python (Scikit-learn, XGBoost, SHAP, MLflow, Pandas, NumPy, Jupyter) |
| **API Development**       | FastAPI, Pydantic, Uvicorn                                           |
| **Frontend & Demos**      | React, Vite, Streamlit, Plotly, JavaScript/JSX                       |
| **Containerisation**      | Docker, Docker Compose                                               |
| **Dashboarding**          | Power BI (DAX, Power Query, Star Schema)                             |

### Techniques

| Area                    | Methods                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| **Regression**          | Multiple Linear Regression, Logistic Regression, Gradient Boosting, XGBoost, Forward Stepwise Selection |
| **Classification**      | Decision Trees, Logistic Regression, RandomizedSearchCV, Mutual Information                             |
| **Time Series**         | Holt-Winters, ARIMA, sGARCH, Decomposition, ADF Testing, Ljung-Box                                      |
| **Hypothesis Testing**  | Kruskal-Wallis, Chi-Square, Shapiro-Wilk, VIF                                                           |
| **Feature Engineering** | RFM Analysis, Log Transformations, Interaction Features, Target/One-Hot/Label Encoding                  |
| **Explainability**      | SHAP (global & per-prediction), feature importance, dependence plots                                    |
| **MLOps**               | MLflow experiment tracking, model registry, Dockerised deployment                                       |
| **Data Preparation**    | JSON Normalisation, Outlier Analysis, Correlation Matrices                                              |
| **Deployment**          | REST APIs, Model Serialisation (Pickle), Docker Compose, Cloud Cost Analysis (AWS/Azure/GCP)            |

---

## Repository Structure

```
data-science-projects/
│
├── imf-globaleconomy-analytics/        # Power BI dashboard (IMF 2001–2020)
│   ├── dashboard/                       # .pbix file + live URL
│   ├── dax-measures/                    # DAX formulas for all visuals
│   ├── source-data/                     # IMF Excel datasets
│   └── report-documentation/            # Full project report (PDF)
│
├── strataforge-prediction/             # Concrete strength prediction (R)
│   ├── models/                          # Linear + logistic regression scripts
│   ├── hypothesis-tests/                # Kruskal-Wallis, Chi-Square
│   ├── preprocessing/                   # Data loading & transformation
│   ├── source-data/                     # 1,030 mix trial records
│   ├── strataforge-presentation/        # Interactive React slide deck
│   └── report-documentation/            # Full project report (PDF)
│
├── reder-analytics/                    # Telecom churn prediction (Python)
│   ├── preprocessing/                   # JSON parsing, normalisation, RFM
│   ├── model/                           # Training, tuning, serialisation
│   ├── app.py                           # FastAPI prediction endpoint
│   ├── reder-presentation/              # Interactive React slide deck
│   └── report-documentation/            # Full project report (PDF)
│
├── usb-stock-timeseries-model/         # Share price forecasting (R)
│   ├── eda/                             # Exploration & decomposition
│   ├── models/                          # Holt-Winters, ARIMA, GARCH scripts
│   ├── public/model-performance/        # Diagnostic plot screenshots
│   ├── source-data/                     # USB.csv (11,835 records)
│   ├── usbforecast-presentation/        # Interactive React slide deck
│   └── report-documentation/            # Full project report (PDF)
│
├── voyage-rails-analytics/             # Rail dynamic pricing (Python + Docker)
│   ├── preprocessing/                   # EDA, feature engineering, SHAP analysis
│   │   ├── preprocess.ipynb             # Training pipeline notebook
│   │   └── mlruns/                      # MLflow experiment tracking
│   ├── source-data/                     # Historical booking records
│   ├── app.py                           # FastAPI inference service
│   ├── streamlit_app.py                 # Streamlit demo UI
│   ├── Dockerfile.backend               # FastAPI container image
│   ├── Dockerfile.streamlit             # Streamlit container image
│   ├── docker-compose.yml               # Orchestrates both services
│   ├── voyage-presentation/             # Interactive React slide deck
│   └── report-documentation/            # Full project report (PDF)
│
└── README.md                           # ← You are here
```

---

## What Ties These Projects Together

Each project follows the same disciplined approach, regardless of domain:

**1. Start with the business question.** Every project begins with a real problem facing a real organisation — not a dataset looking for a use case. The StrataForge project asks "can we cut cement costs without sacrificing strength?" The Reder project asks "which customers will leave, and can we stop them?" The VoyageRail project asks "can we price each seat based on its actual demand rather than a rule table?" The question shapes every subsequent decision.

**2. Let the data dictate the method.** The USB project is the clearest example: Holt-Winters and ARIMA both failed because the data exhibited volatility clustering. Rather than forcing a method, the iterative diagnostic process revealed _why_ it failed and _what_ was needed — leading to GARCH as a principled solution, not an arbitrary choice.

**3. Validate rigorously.** No model is presented without diagnostic evidence. The concrete regression passed all five classical assumptions (linearity, independence, normality, homoscedasticity, no multicollinearity). The GARCH model passed the Ljung-Box test where two predecessors failed. The churn classifier was evaluated on precision, recall, F1, and accuracy. The VoyageRail pricing model is tracked through MLflow and explained via SHAP for every prediction. Trust is earned through evidence.

**4. Quantify the impact.** Results are translated into the language the business cares about. Not just "R² = 81.35%" but "£21,600 saved per project." Not just "97% accuracy" but "5–7x cheaper than acquiring a new customer." Not just "p = 0.2369" but "this is the only model whose residuals are indistinguishable from white noise." Not just "predicted £37.30" but a defensible narrative explaining _why_ that price.

**5. Make it accessible.** Every project includes an interactive React/Vite presentation that communicates the full analysis to non-technical stakeholders — complete with live calculators, model comparisons, worked cost examples, and (for VoyageRail) a full project budget breakdown.

---

## Contact

Open to opportunities in data science, analytics engineering, and quantitative modelling.

📧 [Email](mailto:michizler@gmail.com) · 💼 [LinkedIn](https://linkedin.com/in/bright-uzosike) · 🐙 [GitHub](https://github.com/michizler)
