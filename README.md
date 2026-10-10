# SuperKart: reproducible academic MLOps project

This build implements the 60-mark assignment in `SuperKart_Project_Specification.md`. This tested package includes the uploaded assignment CSV, a fully tuned fitted model and real evaluation reports. Account authentication and live deployments remain pending. See `ACTUAL_DATA_TEST_REPORT.md` for measured results and verification limits.

## Start in Colab

1. Put **`SupeKart.csv`** in **`MyDrive/Colab Notebooks/`** (exact spelling).
2. Upload/open `SuperKart_Colab.ipynb` in Google Colab and run cells in order. The notebook contains the project source, so it does not need a repository or ZIP download to bootstrap.
3. Mount Drive when Google asks. The notebook reads `/content/drive/MyDrive/Colab Notebooks/SupeKart.csv`, validates the schema, trains models and stores outputs under `MyDrive/Colab Notebooks/SuperKart_Project`.
4. Model development runs in an isolated Python 3.12 environment. Code and dependency locks are the same in Colab, Actions and deployed inference. Notebook display uses the Colab kernel; modeling never imports Colab's preinstalled scientific packages.
5. Read the measured outputs and add your own interpretation where needed. Complete the deployment/evidence cells before exporting the final HTML.

## Local or runner reproduction

Use Python **3.12**:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip check
.venv/bin/python tools/check_environment.py
.venv/bin/python -m src.validate_data
.venv/bin/python -m src.prepare_data
.venv/bin/python -m src.train
.venv/bin/python -m src.analysis
.venv/bin/python -m tools.check_model
.venv/bin/python -m pytest -q
.venv/bin/python -m streamlit run app.py
```

On Windows use `.venv/Scripts/python.exe`. `--quick` is for integration tests only. Full training compares Random Forest and Gradient Boosting over bounded parameter grids using five-fold CV. Baseline and final metrics are printed and saved. The model is selected by CV RMSE, never by holdout performance. Raw product ID is removed; store ID is retained for existing-store predictions. No timestamps exist in the documented schema, so temporal forecasting is not claimed.

## GitHub and CI/CD

The Colab publisher targets **https://github.com/sharadphatak/superkat_mlOps**. It creates the repository if absent, initializes it if empty, or updates files listed in the SuperKart publish manifest. Unrelated remote files are preserved; unmanaged path conflicts fail for review. Ref updates never force-push. The repository's existing public/private visibility is retained. Set `PUBLISH_GITHUB=True` in the notebook and add `GH_TOKEN` to **Colab Secrets** for the `sharadphatak` account. For this existing repository, a fine-grained token needs Contents and Workflows read/write permissions. Creating a missing repository may require a token type that supports repository creation (classic `repo`/`workflow` scopes or creation through the GitHub UI).

Publishing is optional until account setup is ready. The notebook defaults to no upload. Use public visibility only after confirming assignment data publication is allowed. Passwords and MFA stay in provider sign-in pages.

Workflow: registration → preparation → artifact download and training → checks → model/reports commit to `main` → optional HF Space sync. `GITHUB_TOKEN` gets write permission only in the publish job. Repository settings and branch rules must permit the specified update. Bot commits do not trigger retraining; source-specific triggers and `[skip ci]` provide an additional guard. If the source changes while training, publication fails rather than pairing a stale model with new code. Rerun on the newest commit.

The workflow uploads `data/processed/` as `train-test` and downloads it in the next job. Hashes verify the downloaded files. Models are compressed and complete preprocessing is serialized with the regressor. Inference checks model hash and the six primary package versions. Full runtime and development dependency locks capture tested transitive resolutions.

## Streamlit Community Cloud

Sign in at https://share.streamlit.io/ and create an app using the repository, `main`, and `app.py`. **Select Python 3.12 in advanced settings before deployment.** Ensure `models/model.joblib` and `models/metadata.json` have been committed by the successful workflow. The app loads the repository model, gathers inputs into a dataframe and reports estimated revenue in unspecified source units. The committed runtime `requirements.txt` locks deployment dependencies.

Record the actual public app URL and a screenshot showing a successful prediction. Verify the app again after a workflow publishes a new model.

## Hugging Face Spaces

The assignment also asks for a public HF frontend. Use a **Docker Space**, serving the same app on port **7860**. Set GitHub repository variable `HF_SPACE_ID` to `owner/space` and repository secret `HF_TOKEN` to a scoped write token. Rerun the workflow to synchronize the app, runtime dependency file, trusted model, src folder and Dockerfile. Dataset, notebook and secrets are excluded from this upload. Alternatively use the optional Colab publish cell.

Source upload is not proof of a live deployment: inspect the Space build logs, confirm public access, perform a sample prediction and record the real URL/screenshots. Streamlit Community Cloud remains required by the scored rubric.

## Academic submission and evidence

The notebook has all seven rubric sections, displayed code for source files, actual-data exploration, cleaning/split summaries, experiment tracking, model reload, CI/deployment instructions and evidence checks. Fill in real deployment URLs and add:

- `evidence/repository_structure.png`
- `evidence/workflow_success.png`
- `evidence/streamlit_app.png`
- Optional `evidence/huggingface_app.png`

The evidence cell embeds screenshots into notebook output and lists outstanding requirements. Download the fully executed notebook from Colab as `SuperKart_Submission.ipynb`, put it in `MyDrive/Colab Notebooks/`, then run the export cell. Export rejects unexecuted cells or cell errors. The readiness guard also rejects missing required evidence or quick-test metrics. Final HTML is saved in the project `submission/` folder on Drive.

## Tests and limitations

Tests use temporary, explicitly synthetic fixtures solely to verify CSV → split → tuning → serialization → inference → Streamlit form submission. Synthetic models are not shipped as assignment models. Tests also reject invalid schema, invalid ratios, corrupt artifacts and tampered models, and exercise missing features and unknown categories. See `BUILD_VALIDATION.md` for completed checks and unverified external integrations.

No finite dependency test can guarantee error-free execution on all future platform changes, new data patterns or account policies. This build fails early with specific errors when an assumption is violated. Identical retained features with conflicting targets trigger a diagnostic requiring a business rule rather than arbitrary deduplication. Predictions for unseen stores and actual future periods need additional validation/data.

## Official integration references

- Streamlit dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Streamlit deploy/Python selection: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- HF Docker Spaces: https://huggingface.co/docs/hub/spaces-sdks-docker
- GitHub artifact transfer: https://docs.github.com/actions/configuring-and-managing-workflows/persisting-workflow-data-using-artifacts
- GitHub automatic token permissions: https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication
- sklearn persistence/version compatibility: https://scikit-learn.org/stable/model_persistence.html
