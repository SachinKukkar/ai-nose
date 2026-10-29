# AI nose: results

Data: 4,983 molecules, 138 smell words (GoodScents + Leffingwell, OpenPOM curated set).
Split: second-order iterative stratification. Train 3,978 / valid 510 / test 495.

## Main result (test set, mean ROC-AUC over 138 smell words)
| Model | Test AUROC | 95% CI |
|---|---|---|
| GNN ensemble (10 × MPNN, OpenPOM) | 0.891 | 0.880–0.901 |
| Single GNN (average of 10 runs) | 0.876 | range 0.872–0.879 |
| Random forest, count Morgan fingerprint (radius 2) | 0.862 | 0.849–0.874 |
| GNN minus RF | +0.029 | 0.021–0.036 |

For reference: OpenPOM benchmark 0.887 (5-fold CV, 10-model ensembles).
Sanchez-Lengeling et al. 2019: GNN 0.894, RF on count fingerprints 0.845.

## Per smell word
GNN beats RF on 109 of 138 smell words.
Same exception as the 2019 paper: anisic (ours GNN 0.81 vs RF 0.88; paper 0.79 vs 0.88).

## Odor map
kNN (k=20) on the learned embedding: 0.818 test AUROC (paper: 0.818).
kNN on fingerprint bits (Jaccard): 0.794 (paper: 0.782).
Example: butanal's 5 nearest neighbours share 15 of its smell words on the GNN map, vs 7 with fingerprints.

## Training notes
- Single model: validation AUROC peaked at epoch ~40 (0.882). Training longer only overfits.
- Ensemble: early stopping (patience 30), about 2 hours total on an RTX 3050 Laptop GPU (4 GB).
- Figures: figures/gnn_vs_rf_per_smell.png, figures/odor_map.png

## Can an LLM smell? (Qwen3-4B-Instruct, zero-shot, SMILES only, 495 test molecules)
Task: pick the 5 most likely smells from the 138-word list (forced JSON output, temperature 0).

| Method | Precision@5 | Recall@5 | First pick right |
|---|---|---|---|
| GNN ensemble | 0.480 | 0.500 | 72% |
| Always the 5 most common smells | 0.275 | 0.255 | 39% |
| LLM (qwen3:4b-instruct) | 0.193 | 0.183 | 38% |
| Random 5 smells | 0.039 | 0.037 | 5% |

- LLM minus most-common baseline: −0.082 (95% CI −0.099 to −0.066), significantly worse than the trivial baseline.
- LLM minus GNN: −0.287 (95% CI −0.307 to −0.266).
- Bias: the LLM said "fruity" for 461/495 molecules (93%; true rate 39%) and "banana" 177 times; "floral" (4th most common real smell) is missing from its top 8.
- Lesson: prompt v1 didn't mention JSON, which caused empty outputs under forced-format decoding. Prompt v2 fixed it (all 495 answered, ~1 s each on an RTX 3050).