# Reproducibility Guide

This guide describes the main JetGraph workflow from a clean environment to the final report. Run commands from the repository root unless noted otherwise.

## 1. Environment Creation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate jetgraph
```

The environment includes NumPy, pandas, matplotlib, scikit-learn, PyTorch, PyTorch Geometric, and EnergyFlow.

## 2. Dataset Loading

Download and preprocess a default sample of 10,000 EnergyFlow quark/gluon jets:

```bash
python scripts/01_load_qg_dataset.py --num-jets 10000
```

This creates:

```text
data/processed/qg_jets_sample.npz
```

The script also prints dataset shape, label distribution, and summaries of the physics observables.

## 3. Baseline Training

Train observable-only classifiers:

```bash
python scripts/02_baseline_bdt.py
```

This trains logistic regression, random forest, and gradient boosting models using:

- multiplicity
- total pT
- jet mass
- eta width
- phi width

Outputs include ROC curves and feature-importance figures.

## 4. Graph Construction

Build particle-level k-nearest-neighbour graphs with hybrid node features:

```bash
python scripts/03_build_graph_dataset.py --k 8 --feature-mode hybrid
```

This creates:

```text
data/processed/qg_graphs_k8_hybrid.pt
```

Other supported feature modes include `raw`, `physics`, `raw_no_pid`, and `hybrid_no_pid`.

## 5. GNN Training

Train the EdgeConv graph neural network:

```bash
python scripts/04_train_gnn.py \
  --graph-path data/processed/qg_graphs_k8_hybrid.pt \
  --feature-mode hybrid \
  --epochs 20
```

The training loop reports train loss, validation accuracy, and validation ROC AUC for each epoch, then evaluates on the test set.

## 6. Model Comparison

Generate the model-comparison figure:

```bash
python scripts/05_compare_models.py
```

This creates:

```text
figures/model_comparison.png
```

## 7. k-Connectivity Study

Run the k-nearest-neighbour connectivity scan:

```bash
python scripts/06_k_study.py
```

This evaluates `k = 4, 8, 12, 16` with hybrid node features and writes:

```text
data/processed/k_study_results.csv
figures/k_study_auc.png
```

## 8. Generator Robustness

Run the Pythia-vs-Herwig robustness study:

```bash
python scripts/07_generator_robustness.py
```

This trains EdgeConv models on Pythia and Herwig samples and evaluates both in-domain and cross-generator performance.

Outputs include:

```text
data/processed/generator_robustness_results.csv
figures/generator_robustness.png
```

## 9. Jet Visualizations

Create jet-display and kNN-graph visualization figures:

```bash
python scripts/08_visualize_jets.py
```

This produces quark and gluon jet displays in both eta-phi and centered delta_eta-delta_phi coordinates.

## 10. Report Compilation

Compile the LaTeX report:

```bash
cd reports
make
```

This creates:

```text
reports/main.pdf
```

Clean intermediate LaTeX files with:

```bash
make clean
```
