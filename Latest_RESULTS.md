# AI nose: results

Teaching a model to predict smell from molecular structure: a reproduction and extension of Google Research's
Principal Odor Map work (Sanchez-Lengeling et al. 2019; Lee et al., Science 2023), using OpenPOM.

## Setup
- Hardware: Windows 11 laptop, NVIDIA RTX 3050 Laptop GPU (4 GB), Ubuntu via WSL2
- Software: Python 3.10 (conda), PyTorch 2.4.0+cu124, DGL 2.4.0+cu124, DeepChem 2.8.1.dev, OpenPOM, NumPy 1.26.4
- LLM experiments: Ollama with qwen3:4b-instruct, run locally on the same GPU

## Data
- 4,983 molecules, 138 smell words (GoodScents + Leffingwell, OpenPOM curated set)
- Split: second-order iterative stratification. Train 3,978 / valid 510 / test 495
- Every smell word appears in both the validation and the test set

## Main result (test set, mean ROC-AUC over 138 smell words)
| Model | Test AUROC | 95% CI |
|---|---|---|
| GNN ensemble (10 × MPNN, OpenPOM) | 0.891 | 0.880–0.901 |
| Single GNN (average of 10 runs) | 0.876 | range 0.872–0.879 |
| Random forest, count Morgan fingerprint (radius 2) | 0.862 | 0.849–0.874 |
| GNN minus RF | +0.029 | 0.021–0.036 |

- CIs: bootstrap, 1,000 resamples of the test set (paired for the difference)
- For reference: OpenPOM benchmark 0.887 (5-fold CV, 10-model ensembles). Sanchez-Lengeling et al. 2019: GNN 0.894, RF on count fingerprints 0.845
- Random forest: 200 trees, min 5 molecules per leaf. Radius 2 chosen on validation (0.854 vs 0.853 for radius 3)

## Per smell word
- GNN beats RF on 109 of 138 smell words (figures/gnn_vs_rf_per_smell.png)
- Biggest GNN wins (gassy, lavender, black currant, sharp, grapefruit) were also GNN wins in the 2019 paper
- Same exception as the paper: anisic (ours GNN 0.81 vs RF 0.88; paper 0.79 vs 0.88)
- Extreme per-word scores come from rare words with only 3–9 test molecules, so they are noisy

## Odor map
- The learned 256-number embedding groups smells by family: floral, meaty and alcoholic regions are separate, with
  specific smells (e.g. lavender, jasmine; beefy, roasted) inside their family. Fingerprints show heavy overlap
  (figures/odor_map.png)
- kNN (k=20) on the learned embedding (cosine): 0.818 test AUROC (paper: 0.818)
- kNN on fingerprint bits (Jaccard): 0.794 (paper: 0.782)
- Example: butanal's 5 nearest neighbours share 15 of its smell words on the GNN map vs 7 with fingerprints.
  Best GNN neighbour (2-methylbutanal) shares 6 of its 7 smells; fingerprints instead pick long-chain aldehydes
  that smell waxy and fatty

## Example predictions (10-model ensemble)
- Butanal (test set, never seen in training): 4 of its 7 real smells in the top 10, "fruity" ranked first
- Vanillin (training set): vanilla, sweet, creamy as the top 3
- Note: scores are best read as a ranking. Class weighting pushes all probabilities up

## Training notes
- Single model: validation AUROC peaked at epoch ~40 (0.882) and stayed flat; training AUROC kept rising to 0.964
  (overfitting). 150 epochs took 20 minutes
- Ensemble: 10 seeds, early stopping (patience 30, max 100 epochs), best epochs 30–90, about 2 hours in total

## Can an LLM smell? (Qwen3-4B-Instruct, zero-shot, SMILES only, 495 test molecules)
Task: pick the 5 most likely smells from the 138-word list (forced JSON output, temperature 0).

| Method | Precision@5 | Recall@5 | First pick right |
|---|---|---|---|
| GNN ensemble | 0.480 | 0.500 | 72% |
| Always the 5 most common smells | 0.275 | 0.255 | 39% |
| LLM (qwen3:4b-instruct) | 0.193 | 0.183 | 38% |
| Random 5 smells | 0.039 | 0.037 | 5% |

- LLM minus most-common baseline: −0.082 (95% CI −0.099 to −0.066), significantly worse than the trivial baseline
- LLM minus GNN: −0.287 (95% CI −0.307 to −0.266)
- Bias: the LLM said "fruity" for 461/495 molecules (93%; true rate 39%) and "banana" 177 times.
  "Floral", the 4th most common real smell, is missing from its top 8
- LLM's most-used words: fruity 461, sweet 339, creamy 229, banana 177, green 139, nutty 137, dairy 127, woody 116
- Most common real smells: fruity 191, green 152, sweet 144, floral 114, herbal 80, woody 66, fatty 60, fresh 53

## Does retrieval help the LLM? (same 495 test molecules)
The LLM is shown the 5 most similar training molecules (Tanimoto on Morgan fingerprints, no GNN involved) with their known smells.

| Method | Precision@5 | Recall@5 | First pick right |
|---|---|---|---|
| GNN ensemble | 0.480 | 0.500 | 72% |
| Copy neighbours' smells (similarity-weighted, no LLM) | 0.417 | 0.433 | 64% |
| LLM + retrieved neighbours | 0.362 | 0.360 | 39% |
| Always the 5 most common smells | 0.275 | 0.255 | 39% |
| LLM zero-shot | 0.193 | 0.183 | 38% |

- Retrieval vs zero-shot: +0.168 (95% CI +0.149 to +0.189). Evidence nearly doubles the LLM's precision
- LLM + retrieval vs copying the neighbours: −0.055 (95% CI −0.072 to −0.038). The LLM makes the evidence worse
- LLM + retrieval vs GNN: −0.118 (95% CI −0.136 to −0.101)
- Bias shrinks but remains: "fruity" in 297/495 answers (true rate: 191); "floral" still not in its top 8
- First-pick check: the LLM led with "fruity" 123 times vs 101 for neighbour-copying (only slightly more).
  The bigger difference: the LLM never led with "floral" (neighbour-copying did 58 times) and often led with
  literal or specific words ("apple" 34, "alcoholic" 29, "aldehydic" 23)
- Literal-reading check: when the LLM led with "aldehydic" (23×), 48% of those molecules had an aldehyde group
  (vs 5% of all test molecules) and the pick was right only 17% of the time. For "alcoholic" (29×), 34% had an
  alcohol group (vs 12%), right 14% of the time, vs 39% for its first picks overall.
  The LLM partly matches chemical-group names to smell words
- Takeaway: for this task, the specialist model should make the prediction; an LLM is better suited to explaining it

## Engineering notes
- Ubuntu's /tmp is RAM-backed, so large pip installs failed with "No space left on device"; fixed with TMPDIR on disk
- The 138-output random forest ran out of RAM; fixed with smaller trees and one forest in memory at a time
- Ollama initially split the model between CPU and GPU; fixed with OLLAMA_NUM_PARALLEL=1, flash attention,
  q8_0 KV cache and a smaller context window (about 1 s per molecule after the fix)
- Forced JSON output without mentioning JSON in the prompt produced empty answers; stating the exact format in the
  prompt (v2) fixed it. Answers are capped at 120 tokens. Every molecule in a run uses the same prompt version

## Limitations
- One fixed split, no cross-validation (OpenPOM's benchmark averages 5 folds)
- Labels come from perfumery catalogues: noisy, biased toward pleasant materials, single molecules only,
  no smell intensity or mixtures
- LLM tests use one small model (4B) and one prompt style, with SMILES only. Larger models may do better, and the
  LLM may have seen these public catalogues during training

## Notebooks
- 01_explore_data: load data, build molecular graphs, split
- 02_train: first training runs (5 and 150 epochs)
- 03_ensemble: 10-model ensemble, prediction function
- 04_baseline: random forest baseline, confidence intervals, per-smell chart
- 05_odor_map: odor-map figure, kNN test, nearest neighbours
- 06_llm_eval: LLM zero-shot and retrieval experiments (answers in llm_eval/)
- 07_app_prep: files for the web app (in progress)