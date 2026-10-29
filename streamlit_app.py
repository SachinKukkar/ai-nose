# AI nose: predict how a molecule smells (Streamlit version)
import os, json, tempfile, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("DGLBACKEND", "pytorch")
import numpy as np, pandas as pd, streamlit as st, torch
import deepchem as dc
from huggingface_hub import snapshot_download
from rdkit import Chem
from rdkit.Chem import Draw
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

HF_REPO = "sachin-kukkar/ai-nose-openpom"   # trained weights + odor-map catalogue on the Hugging Face Hub

st.set_page_config(page_title="AI nose", page_icon="👃", layout="wide")

def make_model(n_tasks):
    return MPNNPOMModel(
        n_tasks=n_tasks, batch_size=128, learning_rate=0.001, class_imbalance_ratio=None,
        loss_aggr_type="sum", node_out_feats=100, edge_hidden_feats=75, edge_out_feats=100,
        num_step_message_passing=5, mpnn_residual=True, message_aggregator_type="sum",
        mode="classification", number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM, n_classes=1,
        readout_type="set2set", num_step_set2set=3, num_layer_set2set=2,
        ffn_hidden_list=[392, 392], ffn_embeddings=256, ffn_activation="relu",
        ffn_dropout_p=0.12, ffn_dropout_at_input_no_act=False, weight_decay=1e-5,
        self_loop=False, optimizer_name="adam", log_frequency=32,
        model_dir=tempfile.mkdtemp(), device_name="cpu")

@st.cache_resource(show_spinner="Loading the 10 models (only on the first visit)...")
def load():
    path = snapshot_download(HF_REPO)
    tasks = json.load(open(os.path.join(path, "tasks.json")))
    cat = dict(np.load(os.path.join(path, "catalogue.npz")))
    models = []
    for i in range(10):
        m = make_model(len(tasks))
        m.model.load_state_dict(torch.load(os.path.join(path, "weights", f"model_{i}.pt"), map_location="cpu"))
        models.append(m)
    emb = cat["emb"] / np.linalg.norm(cat["emb"], axis=1, keepdims=True)
    index = {c: i for i, c in enumerate(cat["canon"])}
    return tasks, cat, emb, index, models, int(cat["best_model"])

WHERE = {"train": "the **training set**: the model has seen it, so this isn't a fair test",
         "valid": "the **validation set**: not trained on, but used to pick the best models",
         "test":  "the **test set**: never used in training, so this is a fair test"}
EXAMPLES = {"Vanillin": "COc1cc(C=O)ccc1O",
            "Isoamyl acetate (banana)": "CC(C)CCOC(C)=O",
            "Limonene (citrus)": "CC1=CCC(CC1)C(=C)C",
            "Ethyl butyrate (fruity)": "CCCC(=O)OCC",
            "Butanal (test set)": "CCCC=O",
            "Aspirin (not a perfume molecule)": "CC(=O)Oc1ccccc1C(=O)O"}

st.title("👃 AI nose: predict how a molecule smells")
st.write("A 10-model graph neural network ensemble (OpenPOM, test ROC-AUC 0.891 on 138 smell words) "
         "predicts how a molecule smells, and a learned odor map finds the catalogue molecules that smell most alike.")

tasks, cat, emb, index, models, best = load()

choice = st.selectbox("Pick an example, or type your own SMILES below", list(EXAMPLES))
smiles = st.text_input("Molecule (SMILES)", value=EXAMPLES[choice]).strip()
mol = Chem.MolFromSmiles(smiles) if smiles else None
if mol is None:
    st.error("Couldn't read that SMILES. Try one of the examples.")
    st.stop()

try:
    X = dc.data.NumpyDataset(GraphFeaturizer().featurize([smiles]))
    p = np.mean([np.asarray(m.predict(X)).reshape(-1) for m in models], axis=0)
    e = np.asarray(models[best].predict_embedding(X)).reshape(-1)
except Exception:
    st.error("The model couldn't process this molecule (it may be too small or unusual).")
    st.stop()

left, right = st.columns(2)
with left:
    st.image(Draw.MolToImage(mol, size=(350, 350)), caption=smiles)
with right:
    top = np.argsort(p)[::-1][:10]
    st.subheader("Top 10 predicted smells")
    st.dataframe(pd.DataFrame({"Smell": [tasks[j] for j in top], "Score": p[top]}), hide_index=True,
                 column_config={"Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})

key = Chem.MolToSmiles(mol)
i = index.get(key)
if i is None:
    st.info("**New molecule**: it isn't in the catalogue, so this is a genuine prediction.")
else:
    st.info(f"This molecule is in {WHERE.get(str(cat['split'][i]), 'the catalogue')}.\n\n"
            f"Catalogue smells: **{cat['smells'][i]}**")

sims = emb @ (e / np.linalg.norm(e))
nearest = [k for k in np.argsort(sims)[::-1] if cat["canon"][k] != key][:5]
st.subheader("Closest molecules on the odor map")
st.dataframe(pd.DataFrame({"Molecule": [str(cat["smiles"][k]) for k in nearest],
                           "Its real smells": [str(cat["smells"][k]) for k in nearest],
                           "Similarity": [round(float(sims[k]), 3) for k in nearest]}), hide_index=True)
st.caption("Scores are a ranking, not true probabilities (training gives rare smells extra weight). "
           "Trained on perfume and flavour materials, so predictions for very different molecules are less reliable.")
