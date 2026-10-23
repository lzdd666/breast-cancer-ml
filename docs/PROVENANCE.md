# Provenance and changes

## Original coursework

The starting materials were the archives `DataMining代码20250417.zip` and
`DataMining.zip`, the course analysis document `数据挖掘过程.docx`, and the team
presentation `医学数据挖掘汇报20250417.pptx`. Their shared main script identifies
Tianqi Liu, School of Basic Medicine, Jilin University, in its author header,
dated April 15, 2025. This attribution is preserved here.

The README's project background follows the course document and presentation.
Dataset structure and feature definitions were checked against `wdbc.names` in
the supplied code archive and the data loaded by the current package. Feature
names in the README match scikit-learn's column names; the three features
highlighted by the original report are identified separately from the new
benchmark's feature-importance results.

The course workflow covered WDBC data inspection, five classifier comparisons,
random-forest tree-count exploration, feature-importance analysis, and clustering.
The source notebooks contain execution outputs; they do not establish which
individual wrote or modified each part of the code. Original teaching scripts
and archives are not included in this repository.

## October 2026 extension

The current package, tests, and documentation were prepared with AI assistance
as a subsequent engineering and evaluation extension. Its experiments and results
are separate from the original 2025 course outputs.

| Original implementation | Current implementation |
| --- | --- |
| Some scripts depend on shared interactive state | Importable modules with one command-line entry point |
| Scaling experiment fits before train/test split | Predictive scalers fitted inside model pipelines and CV |
| Test scores used to choose random-forest tree count | Prespecified search spaces, inner-CV selection, outer-CV scoring |
| Some stochastic classifiers lack seeds | Explicit seeds for splitting, models, clustering and permutations |
| IQR outlier counts described as handling | Explicit detection only; no row deletion or capping |
| No multi-model probability-weighted ensemble | New soft-voting experiment with inner-selected weights |
| Metrics from different runs/settings mixed in prose | Per-fold metrics, predictions, configurations and split manifests |
| DBSCAN noise can be treated as a cluster | Exclude noise from clustering metrics and report coverage |
| Exploratory feature importance from a single forest | Outer-fold impurity and held-out permutation importance |

The original document and notebook have different recorded random-forest CV
accuracies (96.24% and 95.54%). The original comparison did not fix the random
forest's seed. Neither number is reused as an outcome of the new protocol.
The original 100-tree test run records accuracy 0.9371 and AUC 0.9822; the
5-tree run records accuracy 0.9510 after consulting the same test set during
tree-count exploration. These are historical observations, not new benchmark
results.

## Dataset attribution

Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
*Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5DW2B>. Dataset license: CC BY 4.0.

The repository loads the dataset bundled with scikit-learn and explicitly maps
its labels to malignant=1 and benign=0. This reverses scikit-learn's original
numeric label convention. Patient identifiers and local source-file paths are
not part of the experiment outputs.
