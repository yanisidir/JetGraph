# JetGraph: Particle-Level Graph Learning for Quark/Gluon Jet Tagging

## Abstract

JetGraph is a compact high-energy physics and machine-learning study of
quark/gluon jet tagging using the EnergyFlow `qg_jets` dataset. The project
combines interpretable physics-inspired observables with particle-level graph
neural networks. Baseline classifiers using only global jet observables already
reach strong discrimination, with gradient boosting achieving ROC AUC 0.8616.
A first EdgeConv graph neural network reaches comparable performance, with the
best hybrid node-feature configuration achieving ROC AUC 0.8617. Additional
ablation studies quantify the role of feature representation, graph connectivity,
particle identity information, and Pythia-vs-Herwig generator dependence.

## Physics Motivation

Jets arise from the fragmentation and hadronization of energetic quarks and
gluons. Because gluons carry a larger color factor than quarks, gluon-initiated
jets tend to have higher particle multiplicity and broader radiation patterns.
Quark/gluon tagging is therefore a useful benchmark for studying how machine
learning models exploit both global jet substructure and local particle-level
correlations. JetGraph uses this task to connect physically motivated observables
with graph-based representation learning.

## Dataset

The project uses the EnergyFlow `qg_jets` dataset, with a default sample of
10,000 jets. Each jet is represented as a padded array of particle constituents
with columns `(pT, eta, phi, pid)`. Padded particles are removed by requiring
positive transverse momentum. Pythia is used for the primary development sample,
and Herwig is used later to probe generator dependence.

## Baseline Observables

The first stage computes five interpretable jet-level observables:

- particle multiplicity, excluding padded constituents
- total scalar constituent pT
- approximate jet mass from massless constituent four-vectors
- pT-weighted eta width
- pT-weighted phi width

These observables provide a transparent baseline before introducing graph neural
networks. Three standard classifiers are trained: logistic regression, random
forest, and gradient boosting.

## Graph Construction

Each jet is converted into a PyTorch Geometric graph. Nodes correspond to
particles, and directed edges are built with k-nearest neighbours in wrapped
eta-phi space. The default connectivity is `k=8`.

Several node-feature modes are studied:

- `raw`: `pT, eta, phi, pid`
- `physics`: `log(pT), pT fraction, delta_eta, delta_phi, delta_R, pid`
- `hybrid`: `pT, log(pT), pT fraction, eta, phi, delta_eta, delta_phi, delta_R, pid`
- `raw_no_pid`: `pT, eta, phi`
- `hybrid_no_pid`: hybrid features without `pid`

This design separates the graph topology from the node-feature representation,
allowing controlled ablations.

## EdgeConv Architecture

The first GNN is intentionally simple: two EdgeConv layers with ReLU activations,
followed by global mean pooling and a small MLP classifier. The model is trained
with cross-entropy loss for binary classification. Unless otherwise stated,
training uses 20 epochs, stratified train/validation/test splits, hybrid features,
and `k=8`.

## Results

### Baseline Comparison

Physics-inspired observables already provide strong discrimination:

| Model | ROC AUC |
|---|---:|
| Logistic regression | 0.8579 |
| Random forest | 0.8451 |
| Gradient boosting | 0.8616 |

The gradient-boosted baseline is competitive with the first graph model, showing
that the simple global observables capture much of the quark/gluon separation.

### Feature Ablation

The EdgeConv model is sensitive to the node-feature representation:

| EdgeConv feature mode | ROC AUC |
|---|---:|
| raw | 0.8580 |
| physics | 0.8102 |
| hybrid | 0.8617 |

Raw features already perform well. Relative physics-only features underperform,
suggesting that absolute kinematic information remains useful for this setup.
Hybrid features recover the best performance by combining absolute and relative
particle information.

### k-Connectivity Study

Using hybrid node features, the graph connectivity parameter was scanned over
`k = 4, 8, 12, 16`:

| k | Test accuracy | Test ROC AUC |
|---:|---:|---:|
| 4 | 0.7653 | 0.8525 |
| 8 | 0.7707 | 0.8579 |
| 12 | 0.7720 | 0.8511 |
| 16 | 0.7640 | 0.8455 |

The best AUC in this sweep is obtained at `k=8`. Increasing connectivity beyond
this point does not improve performance, indicating that moderately local
particle neighbourhoods are sufficient for this first EdgeConv model.

### pid/no-pid Study

Particle identity contributes a small but measurable gain:

| EdgeConv feature mode | ROC AUC |
|---|---:|
| raw | 0.8580 |
| raw_no_pid | 0.8533 |
| hybrid | 0.8617 |
| hybrid_no_pid | 0.8585 |

Most of the discrimination is carried by kinematics and geometry, while `pid`
adds complementary information.

### Pythia vs Herwig Robustness

Generator dependence is evaluated by training on one generator and testing on
both Pythia and Herwig:

| Train generator | Test generator | Accuracy | ROC AUC |
|---|---|---:|---:|
| Pythia | Pythia | 0.7707 | 0.8579 |
| Pythia | Herwig | 0.7147 | 0.7835 |
| Herwig | Pythia | 0.7693 | 0.8464 |
| Herwig | Herwig | 0.7053 | 0.7955 |

The Pythia-trained model drops significantly when evaluated on Herwig, revealing
clear generator dependence. Cross-generator validation is therefore essential for
judging whether a tagger has learned robust jet physics rather than
generator-specific patterns.

## Discussion

The study shows that simple jet-level observables remain highly competitive for
quark/gluon discrimination. The EdgeConv model matches this performance only
after careful feature design. The relative physics-only representation is not
sufficient by itself, while hybrid features combining raw kinematics and
jet-centered coordinates perform best. The k-connectivity scan suggests that
local particle structure matters, but overly dense neighbourhoods can dilute the
signal. The Pythia-vs-Herwig study is the strongest cautionary result: in-domain
performance does not guarantee generator robustness.

## Conclusion and Future Work

JetGraph now provides a complete, reproducible mini-study of quark/gluon tagging:
dataset preparation, interpretable baselines, graph construction, EdgeConv
training, feature ablations, connectivity scans, and generator-robustness tests.
The current best result is a hybrid-feature EdgeConv model with ROC AUC 0.8617
on the Pythia test set, comparable to the best gradient-boosted observable
baseline.

Future work should focus on stronger and more robust graph architectures,
including dynamic graphs and ParticleNet-like networks; improved normalization
and preprocessing; systematic hyperparameter scans; uncertainty-aware
Pythia-vs-Herwig evaluation; and training strategies designed to reduce
generator dependence.
