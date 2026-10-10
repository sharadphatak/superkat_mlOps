# SuperKart build validation

**Updated:** the actual uploaded dataset has now passed full training and app integration checks. See `ACTUAL_DATA_TEST_REPORT.md` for the latest actual-data results. The tables below record the earlier source-build tests; the earlier actual-dataset limitation is superseded by that report.

## Completed locally

Verification used Python 3.12.14 and temporary **synthetic integration fixtures**, never purported assignment data.

| Check | Result |
|---|---|
| Clean runtime dependency installation | Passed |
| Runtime `pip check` | No broken requirements |
| Development `pip check` | No broken requirements |
| Exact runtime lock alignment in development and inference environments | Passed |
| Six integration tests | Passed |
| All Python source and notebook code cells compile | Passed |
| Notebook format/schema validation | Passed |
| Self-contained notebook bootstrap/source writing | Passed in local harness |
| Full five-fold tuning grids on synthetic fixture | Passed |
| CSV registration, cleaning, saved splits and SHA256 integrity | Passed |
| Persisted parameters, CV records and final metrics | Passed |
| Development model loaded in separate runtime-only environment | Passed |
| Unknown categories and missing features | Passed |
| All-missing feature columns retained in preprocessing | Passed |
| Tampered model/artifact and malformed-schema rejection | Passed |
| Streamlit form renders, sample loads, prediction submitted | Passed via AppTest |
| Actual Streamlit HTTP health endpoint | 200 OK in runtime-only environment |
| Workflow YAML parse and job dependencies | Passed local static checks |
| HTML conversion of executed test notebook | Passed |
| Unexecuted notebook rejected by export utility | Passed |

The local notebook harness substituted only account mounting/Drive paths and reused the installed isolated environment. All substantive notebook code ran sequentially, including full tuning, displayed-result generation and readiness checks. It correctly reported missing remote evidence and did not export an academic submission. Mocking Drive does not verify actual Google authorization.

## Dependency issue resolved during testing

An initially unpinned transitive Arrow resolution crashed on native import in this host. Arrow was pinned to **20.0.0**, and installation, metadata, import, application and inference checks were rerun. Both full dependency lock files include this version. The shared stack is NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, joblib 1.4.2 and Streamlit 1.45.1. Tested compatible versions are deliberately retained rather than choosing untested newest releases.

## Not verified in this session

- Actual `SupeKart.csv`: it is on the user's Drive, inaccessible from this session.
- Google Drive mount, Colab account login and any interpreter fallback on a future non-3.12 Colab runtime.
- GitHub API upload with user token, workflow execution on GitHub-hosted runners, repository write policies and automatic `main` commit.
- Streamlit Community Cloud build and publicly accessible prediction.
- Docker image build: Docker is unavailable in this host.
- Hugging Face account authorization, Docker build, public Space and prediction.
- Actual academic final metrics and executed HTML with required deployment screenshots.

These checks require the actual input/account environment. Source and Docker configuration have been reviewed against official integration documentation, but that does not replace a real deployment test. No “guaranteed error-free” or rubric-score guarantee is claimed.

## Remaining acceptance actions

1. Run the Colab notebook against the exact Drive CSV.
2. Review the real EDA/model results and add student-specific conclusions.
3. Publish a new project repository and verify the full Actions run/model commit.
4. Deploy Community Cloud with Python 3.12 and the HF Docker Space with port 7860.
5. Verify public access and predictions; embed real links and screenshots.
6. Download the executed notebook, then use its final export utility to create the HTML submission.
