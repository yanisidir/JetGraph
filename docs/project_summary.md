# JetGraph Project Summary

## Objective

JetGraph is a compact HEP/ML project for quark/gluon jet tagging using the EnergyFlow `qg_jets` dataset. The goal is to compare interpretable physics-inspired observables with particle-level graph neural networks, while keeping the full workflow reproducible and scientifically motivated.

## Methods

The project starts from padded jet constituent arrays containing particle pT, eta, phi, and particle identity. It first computes baseline observables such as multiplicity, total pT, approximate jet mass, and angular widths. These observables are used to train standard machine-learning classifiers.

Jets are then represented as particle-level graphs. Padded particles are removed, nodes correspond to jet constituents, and k-nearest-neighbour edges are built in eta-phi space. A simple EdgeConv graph neural network is trained on raw, physics-inspired, hybrid, and no-particle-identity node feature sets.

The project also includes a graph-connectivity scan and a Pythia-vs-Herwig robustness study to test sensitivity to representation choices and generator dependence.

## Main Results

- Gradient boosting on five physics observables reaches ROC AUC 0.8616.
- The best first EdgeConv setup with hybrid node features reaches ROC AUC 0.8617.
- Physics-only relative graph features underperform, while hybrid absolute-and-relative features recover the strongest graph performance.
- Removing particle identity causes only a modest drop, indicating that most discrimination is carried by kinematics and geometry.
- A moderate graph connectivity of `k=8` performs best in the initial k-scan.
- Cross-generator tests reveal clear Pythia-vs-Herwig dependence, highlighting the need for robustness studies in jet-tagging workflows.

## Scientific Conclusions

The study shows that simple physics observables remain highly competitive for quark/gluon discrimination. Graph neural networks can match these baselines, but their performance depends strongly on node-feature representation and graph construction. The generator-robustness study emphasizes that high in-domain performance is not sufficient for physics reliability.

## Relevance for PhD Applications

JetGraph demonstrates the full arc of a research-style HEP/ML project: dataset preparation, physics baselines, graph construction, neural-network training, ablation studies, robustness evaluation, visualization, and scientific reporting. It is designed to communicate both technical implementation ability and physics-aware machine-learning judgment.
