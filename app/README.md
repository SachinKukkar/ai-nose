# AI nose: Gradio version (Docker)

The same app as `streamlit_app.py`, built with Gradio and packaged with Docker.

1. Run `notebooks/07_app_prep.ipynb` to create `models/`, `catalogue.npz` and `tasks.json` in this folder (needs the trained ensemble).
2. Build and run the image:

    docker build -t ai-nose .
    docker run --rm -p 7860:7860 ai-nose

3. Open http://localhost:7860
