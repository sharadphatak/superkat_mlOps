# Additional Project: Advanced Machine Learning and MLOps — SuperKart

## 1. Project administration

| Item | Supplied requirement |
|---|---|
| Available from | 02 October 2026, 8:00 AM |
| Due date | 11 October 2026, 11:59 PM |
| Total marks | 60 |
| Submission type | File upload |
| Final submission | One executed Python notebook exported to HTML (`.html`) |
| Preferred development environment | Google Colaboratory (Colab) |

Times are reproduced from the supplied brief; the institutional timezone was not stated.

## 2. Business context

A sales forecast predicts future sales revenue based on historical data, industry trends, and the status of the current sales pipeline. Businesses use sales forecasts to estimate weekly, monthly, quarterly, and annual sales totals. Accurate forecasts add value across an organization and help different verticals determine their future course of action. Forecasting supports regional sales operations and provides supply chain teams with insights into the procurement of goods and materials.

An accurate forecasting process improves decisions about the future, reduces sales pipeline and forecast risks, reduces the time spent planning territory coverage, and establishes benchmarks for assessing future trends.

## 3. Objective

As the hired MLOps Engineer, build an automated MLOps pipeline with CI/CD to deliver accurate and reliable sales forecasts. Leverage historical sales data, industry trends, and current pipeline status to predict weekly, monthly, quarterly, and annual revenues. Automate ingestion, preprocessing, training, evaluation, and deployment to ensure scalability, consistency, and minimal manual intervention.

Continuously update and seamlessly deploy predictions through CI/CD so business verticals can plan regional sales operations, optimize supply chain procurement, reduce sales pipeline risks, and establish benchmarks for future trend analysis. The intended outcomes are improved decisions, streamlined planning, operational efficiency, and business growth.

## 4. Data input and dictionary

The input is the assignment's CSV dataset containing product and store attributes. **The CSV has not been supplied with this specification.** Obtain the actual assignment dataset before implementation; do not invent data or results.

| Column | Supplied meaning | Expected semantic type |
|---|---|---|
| `Product_Id` | Unique product identifier: two initial letters followed by a number | String identifier |
| `Product_Weight` | Weight of each product | Numeric |
| `Product_Sugar_Content` | Sugar content, such as low sugar, regular, and no sugar | Categorical |
| `Product_Allocated_Area` | Ratio of product display area to total product display area in a store | Numeric ratio |
| `Product_Type` | Meat, snack foods, hard drinks, dairy, canned, soft drinks, health and hygiene, baking goods, bread, breakfast, frozen foods, fruits and vegetables, household, seafood, starchy foods, or others | Categorical |
| `Product_MRP` | Maximum retail price of each product | Numeric |
| `Store_Id` | Unique store identifier | String identifier |
| `Store_Establishment_Year` | Year the store was established | Integer year |
| `Store_Size` | Store size based on square feet: high, medium, or low | Categorical |
| `Store_Location_City_Type` | Tier 1, Tier 2, or Tier 3; Tier 1 has comparatively higher living standards | Categorical |
| `Store_Type` | Departmental Store, Supermarket Type 1, Supermarket Type 2, or Food Mart | Categorical |
| `Product_Store_Sales_Total` | Total revenue from a particular product at a particular store | Numeric target |

Semantic types are implementation guidance, not additional source requirements. Units, currency, observation period, sample size, filename, missing-value patterns, and category spellings must be established from the actual dataset or documented as unknown.

### 4.1 Scope limitation to resolve

The listed schema has no sales date, period, pipeline, or industry-trend fields. It directly supports supervised product/store revenue regression. It does **not** establish genuine weekly/monthly/quarterly/annual time-series forecasting. Document this gap and seek clarification or additional time-indexed data. Do not relabel row-level regression as temporal forecasting or invent aggregation periods.

## 5. Submission guidelines — preserved requirements

1. Submit a single Python notebook in HTML (`.html`) format.
2. In Jupyter Notebook, export the notebook as HTML from the File menu.
3. In Google Colab (preferred), first download the notebook as `.ipynb`, then convert/export it as HTML. The pasted brief referred to a conversion link but did not include its URL. A reproducible `nbconvert` conversion may be used.
4. Copied/plagiarized assignments will not be evaluated and receive zero marks.
5. Submit on time; late submissions may be rejected or penalized.

## 6. Important points to note — preserved requirements

- Read the problem statement and rubric thoroughly before starting.
- Apply concepts and techniques learned in previous weeks and summarize insights at the end.
- Run all code sequentially from start to finish, add observations and insights where applicable, and submit in the specified format.
- Cover every rubric section and make code outputs clearly visible.
- Include the public Hugging Face Spaces frontend link and GitHub repository link in the notebook.
- Correct but unexecuted code incurs a 50% deduction in the corresponding rubric section.
- Contact the Program Manager if anything is unclear.

**Power Ahead!**

## 7. Scoring rubric — 60 marks

| Criterion | Marks | Required implementation | Required evidence |
|---|---:|---|---|
| Data Registration | 3 | Create a master folder and `data` subfolder; add the CSV; register the dataset using a validation script that checks expected columns and prints a summary | Repository layout, validation code, executed summary |
| Data preparation | 7 | Load directly from repository `data`; clean data; remove unnecessary columns; split into train/test; save locally; expose splits to the next workflow job as an artifact | Cleaning rationale, split outputs, artifact upload/download |
| Model Building with Experimentation Tracking | 13 | Load train/test from the workflow artifact; define model and parameters; tune; log all tuned parameters; evaluate; save best model for repository commit. Allowed algorithms: Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost | Experiment records, tuning results, metrics, saved best model |
| Model Deployment | 11 | Build a Streamlit app loading the model committed to the repository; collect inputs in a dataframe; define deployment dependencies; deploy on Streamlit Community Cloud | App code, dependencies, public working deployment |
| Automated GitHub Actions Workflow | 15 | Create `.github/workflows/pipeline.yml`; define all ML steps; push files to GitHub; automate end-to-end workflow; automatically push generated code updates to `main` | YAML, successful run, artifacts, automatic update commit |
| Output Evaluation | 4 | Supply GitHub repository link; screenshots of folder structure and executed workflow; Streamlit Community Cloud app link and screenshot | Real URLs and screenshots embedded in notebook/HTML |
| Notebook Overall Quality | 7 | Good structure and flow; commented code; executed outputs; no errors; organized repository | Sequentially executed, readable final HTML |
| **Total** | **60** | | |

### 7.1 Deployment discrepancy and working interpretation

The instructions require a public **Hugging Face Spaces** frontend, while the scored deployment and output criteria require **Streamlit Community Cloud**. Preserve both requirements. The recommended working interpretation is to deploy the same Streamlit frontend to both services, using a Docker Space if needed for Hugging Face compatibility, and include both public URLs. Confirm with the Program Manager if only one deployment is intended; do not silently substitute one for the other.

## 8. Recommended implementation contract — engineering guidance

This section supplements the supplied brief; it does not change the rubric.

### 8.1 Repository structure

| Path | Purpose |
|---|---|
| `data/<actual_dataset_filename>.csv` | Original assignment dataset |
| `src/validate_data.py` | Schema validation and summary |
| `src/prepare_data.py` | Cleaning and deterministic split creation |
| `src/train.py` | Tuning, experiment logging, evaluation, model selection |
| `src/predict.py` | Shared inference interface |
| `app.py` | Streamlit frontend |
| `models/model.joblib` | Committed fitted preprocessing-and-model pipeline |
| `models/metadata.json` | Schema, model version, data hash, metrics, provenance |
| `reports/experiments/` | Persisted tuning parameters and CV results |
| `reports/figures/` | EDA and evaluation graphics |
| `evidence/` | Actual workflow and app screenshots |
| `notebooks/SuperKart_Submission.ipynb` | Executed academic notebook |
| `submission/SuperKart_Submission.html` | Final upload artifact |
| `.github/workflows/pipeline.yml` | Automated multi-job pipeline |
| `requirements.txt` | Deployment dependencies |
| `requirements-dev.txt` | Training/notebook dependencies |
| `Dockerfile` | Hugging Face Docker Space runtime |
| `README.md` | Reproduction, links, setup and documented limitations |
| `.gitignore` | Secrets, temporary data splits and unnecessary outputs exclusions |

### 8.2 Modeling and reproducibility

- Inspect shape, types, missingness, duplicates, categorical labels, distributions and target relationships before defining cleaning rules.
- Treat IDs deliberately: product IDs repeat across stores; store identity may be predictive. Do not discard all IDs mechanically or use raw product ID as an unexplained memorization feature.
- Remove rows with missing target with a documented count; do not impute target values.
- Fit imputation and encoding only within training/CV folds using a scikit-learn `Pipeline` and `ColumnTransformer`.
- Select the split strategy to match deployment use. Use a deterministic random holdout for the rubric's standard existing-store use case; evaluate grouped splitting if predicting unseen stores/products. Use temporal splits only when authentic timestamps exist.
- Include a simple baseline, compare at least two allowed algorithms with bounded tuning, select by cross-validation, and use the holdout only for final evaluation.
- Report RMSE, MAE and R², with units where known; avoid unsupported accuracy claims and misleading MAPE near zero.
- Persist the entire fitted pipeline, all tried parameters and CV results, selected settings, seed, data fingerprint, package versions and final metrics. JSON/CSV logging is acceptable unless a tracking service is mandated; MLflow can add a richer tracking interface.
- Validate inference schema and unknown categories; keep training and deployment preprocessing identical.

### 8.3 CI/CD and deployment

Use separate validation, preparation, training/evaluation, and publish jobs. Transfer train/test files through GitHub Actions workflow artifacts. Publish the model, metadata and experiment records only after checks pass. Use narrowly scoped workflow write permissions, prevent bot commit loops, and handle branch policies honestly. If `main` is protected, document the exact policy constraint and use the permitted update mechanism rather than bypassing it.

Deploy the repository-backed Streamlit app on Community Cloud. For Hugging Face, reuse the app and pinned compatible runtime. Confirm each deployment loads the intended model version and handles a real sample. Keep exported HTML screenshots embedded so they remain visible without local files.

## 9. Acceptance checklist

- [ ] Actual supplied dataset is used; validation prints summary.
- [ ] Leakage-safe preparation and reproducible train/test artifact exchange are demonstrated.
- [ ] Allowed model algorithms are tuned; every trial's parameters and results are persisted.
- [ ] Best fitted pipeline is committed and loaded by the app.
- [ ] GitHub Actions completes with evidence of automatic repository updates.
- [ ] Public Community Cloud app and public Hugging Face Space are verified, or discrepancy is explicitly resolved.
- [ ] Real repository/deployment links and required screenshots are embedded.
- [ ] All notebook cells execute sequentially; outputs and interpretations are visible.
- [ ] HTML opens correctly and covers all 60 marks.
- [ ] No secrets, fabricated evidence or unsupported time-series claims appear in deliverables.
