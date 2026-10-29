FROM python:3.10-slim

# System libraries: RDKit's molecule drawings need X render/ext, expat, fontconfig and freetype; DGL needs OpenMP (libgomp1)
RUN apt-get update && apt-get install -y --no-install-recommends libxrender1 libxext6 libexpat1 libfontconfig1 libfreetype6 libgomp1 && rm -rf /var/lib/apt/lists/*

# Run as a normal user with id 1000 (what Hugging Face Spaces expects)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH DGLBACKEND=pytorch PYTHONUNBUFFERED=1 GRADIO_SERVER_NAME=0.0.0.0
WORKDIR /home/user/app

# A newer pip resumes downloads that get cut off, instead of failing
RUN python -m pip install --no-cache-dir --user --upgrade pip

# Install the Python libraries first, so rebuilding after a code change is fast
COPY --chown=user requirements.txt .
RUN python -m pip install --no-cache-dir --user --retries 10 --timeout 120 --resume-retries 10 -r requirements.txt

# Import check: fail the build now if any library can't load, instead of when the app starts
RUN python -c "from rdkit.Chem import Draw; import dgl, deepchem, gradio; from openpom.models.mpnn_pom import MPNNPOMModel; print('all imports OK')"

# Then copy the app, the models and the odor-map data
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
