# SuperKart — Quick start

## Included files

- `SuperKart_Colab.ipynb`: self-contained Google Colab notebook with embedded repository source and dependency locks.
- `SuperKart_MLOps_Project.zip`: complete repository, workflow, tests, Docker deployment files and setup/validation documentation.

## Run in Google Colab

1. Put the actual **`SupeKart.csv`** in **Google Drive → MyDrive → Colab Notebooks**. The expected path is `/content/drive/MyDrive/Colab Notebooks/SupeKart.csv`.
2. Open https://colab.research.google.com/ and upload `SuperKart_Colab.ipynb` through File → Upload notebook.
3. Use a CPU runtime and run cells in order. Sign in/mount Drive when prompted. Models and outputs are written to `MyDrive/Colab Notebooks/SuperKart_Project`.
4. The notebook installs its tested dependencies into an isolated Python 3.12 environment; no global Colab scientific-package replacement is needed.
5. Check the printed cleaning counts, experiment trials, holdout metrics and plots. Add your own measured conclusions.

## GitHub, Streamlit and Hugging Face

The notebook contains optional publishing cells, disabled until you fill non-secret account configuration. Enter access tokens in **Colab Secrets**, never in notebook text.

- GitHub: use the configured `sharadphatak/superkat_mlOps` repository. Add `GH_TOKEN` to Colab Secrets with access for the notebook and appropriate contents/workflow permissions. Enable the optional publishing cell, or publish manually. Set public visibility only if the dataset may be public.
- GitHub Actions: run **SuperKart MLOps** on `main`; ensure repository policy permits the workflow model commit. Check the successful run and model/reports commit.
- Streamlit Community Cloud: sign in, select the repository, `main`, `app.py`, and **Python 3.12**. Verify a prediction and record the public URL.
- Hugging Face: create/publish a public **Docker Space**. Set repository variable `HF_SPACE_ID=owner/space` and secret `HF_TOKEN`, then rerun Actions, or use the optional Colab HF publishing cell. Inspect the actual build and verify a prediction.

See the generated `README.md` for exact setup, permissions, commands and official references. Do not provide passwords in Markdown or chat.

## Final academic submission

Fill the real links and verification flag in Section 6. Save required screenshots under the generated project's `evidence` folder:

- `repository_structure.png`
- `workflow_success.png`
- `streamlit_app.png`

Rerun the evidence/review cells. Download the fully executed notebook as `SuperKart_Submission.ipynb`, upload that file to `MyDrive/Colab Notebooks`, then run the final export cell. It generates `SuperKart_Project/submission/SuperKart_Submission.html` and offers a download.

The delivered notebook is **source ready to run**, not an already executed submission using your CSV. Local tests passed on synthetic fixtures. Actual-data training, authenticated workflow builds and live cloud deployments remain to be verified in your accounts.

## Repository-specific update

Target: https://github.com/sharadphatak/superkat_mlOps. Open the revised notebook and set `PUBLISH_GITHUB=True` in its publishing cell. Add `GH_TOKEN` to Colab Secrets for your sharadphatak account, with Contents and Workflows write access to this repository. An empty repository is initialized; managed project files are safely updated on repeat runs. Existing visibility is retained. Unrelated files are preserved. No force-push is performed.
