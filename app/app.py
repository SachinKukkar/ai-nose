"""AI nose: predict how a molecule smells (10-model OpenPOM ensemble)."""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import gradio as gr
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

HERE  = os.path.dirname(os.path.abspath(__file__))
TASKS = json.load(open(os.path.join(HERE, "tasks.json")))
CAT   = np.load(os.path.join(HERE, "catalogue.npz"))
SMILES, CANON, SMELLS, SPLIT = CAT["smiles"], CAT["canon"], CAT["smells"], CAT["split"]
EMB   = CAT["emb"] / np.linalg.norm(CAT["emb"], axis=1, keepdims=True)
BEST  = int(CAT["best_model"])
INDEX = {c: i for i, c in enumerate(CANON)}

def make_model(model_dir):
    return MPNNPOMModel(
        n_tasks=len(TASKS), batch_size=128, learning_rate=0.001, class_imbalance_ratio=None,
        loss_aggr_type="sum", node_out_feats=100, edge_hidden_feats=75, edge_out_feats=100,
        num_step_message_passing=5, mpnn_residual=True, message_aggregator_type="sum",
        mode="classification", number_atom_features=GraphConvConstants.ATOM_FDIM,
        number_bond_features=GraphConvConstants.BOND_FDIM, n_classes=1,
        readout_type="set2set", num_step_set2set=3, num_layer_set2set=2,
        ffn_hidden_list=[392, 392], ffn_embeddings=256, ffn_activation="relu",
        ffn_dropout_p=0.12, ffn_dropout_at_input_no_act=False, weight_decay=1e-5,
        self_loop=False, optimizer_name="adam", log_frequency=32,
        model_dir=model_dir, device_name="cpu")

# Load all 10 models once, when the app starts
MODELS = []
for i in range(10):
    d = os.path.join(HERE, "models", f"model_{i}")
    m = make_model(d); m.restore(model_dir=d); MODELS.append(m)
FEAT = GraphFeaturizer()

WHERE = {"train": "the **training set**: the model has seen it, so this isn't a fair test",
         "valid": "the **validation set**: not trained on, but used to pick the best models",
         "test":  "the **test set**: never used in training, so this is a fair test"}

def smell(smiles):
    smiles = (smiles or "").strip()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, {}, "Couldn't read that SMILES. Try one of the examples.", []
    picture = Draw.MolToImage(mol, size=(300, 300))
    try:
        X = dc.data.NumpyDataset(FEAT.featurize([smiles]))
        p = np.mean([np.asarray(m.predict(X)).reshape(-1) for m in MODELS], axis=0)
        e = np.asarray(MODELS[BEST].predict_embedding(X)).reshape(-1)
    except Exception:
        return picture, {}, "The model couldn't process this molecule (it may be too small or unusual).", []

    scores = {TASKS[j]: float(p[j]) for j in np.argsort(p)[::-1][:10]}

    # Is this molecule already in the catalogue? (be honest about what the model has seen)
    key = Chem.MolToSmiles(mol)
    i = INDEX.get(key)
    if i is None:
        status = "**New molecule**: it isn't in the catalogue, so this is a genuine prediction."
    else:
        status = f"This molecule is in {WHERE.get(str(SPLIT[i]), 'the catalogue')}.  \nCatalogue smells: **{SMELLS[i]}**"

    # Closest molecules on the odor map (cosine similarity of the learned embedding)
    sims = EMB @ (e / np.linalg.norm(e))
    nearest = [k for k in np.argsort(sims)[::-1] if CANON[k] != key][:5]
    rows = [[str(SMILES[k]), str(SMELLS[k]), round(float(sims[k]), 3)] for k in nearest]
    return picture, scores, status, rows

EXAMPLES = [["COc1cc(C=O)ccc1O"],     # vanillin
            ["CC(C)CCOC(C)=O"],       # isoamyl acetate (banana)
            ["CC1=CCC(CC1)C(=C)C"],   # limonene (citrus)
            ["CCCC(=O)OCC"],          # ethyl butyrate (fruity)
            ["CCCC=O"]]               # butanal (test set)

with gr.Blocks(title="AI nose") as demo:
    gr.Markdown("# AI nose: predict how a molecule smells\n"
                "Type a molecule as a SMILES string. A 10-model graph neural network ensemble "
                "(OpenPOM, test ROC-AUC 0.891 on 138 smell words) predicts how it smells, and the odor map "
                "finds the catalogue molecules that smell most alike.")
    with gr.Row():
        inp = gr.Textbox(label="Molecule (SMILES)", placeholder="e.g. COc1cc(C=O)ccc1O (vanillin)", scale=4)
        btn = gr.Button("Smell it", variant="primary", scale=1)
    gr.Examples(EXAMPLES, inputs=inp)
    with gr.Row():
        img = gr.Image(label="Molecule", type="pil")
        lab = gr.Label(label="Top 10 predicted smells", num_top_classes=10)
    status = gr.Markdown()
    nbrs = gr.Dataframe(headers=["Similar-smelling molecule", "Its real smells", "Similarity"],
                        label="Closest molecules on the odor map", wrap=True)
    gr.Markdown("Scores are a ranking, not true probabilities (training gives rare smells extra weight). "
                "The model was trained on perfume and flavour materials, so predictions for very different "
                "molecules are less reliable.")
    btn.click(smell, inp, [img, lab, status, nbrs])
    inp.submit(smell, inp, [img, lab, status, nbrs])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)