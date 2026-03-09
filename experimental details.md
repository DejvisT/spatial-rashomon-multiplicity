Primary experiment

Loss: Brier score
Splits: 60/20/20; 10 outer seeds; report mean ± std across runs
Datasets: Adult, COMPAS, Folk Mobility, Folk Travel
Models: LogReg, kNN, RF, GBM, MLP
Candidate generation: random search, 50 candidates per family per run (seeded)


Rashomon set construction (per run)
Global Rashomon (optional): top-K models by validation Brier across all candidates
Per-family Rashomon (required): top-K models by validation Brier within each family
choose K (e.g., 10–15) given 50 candidates
Save model predictions/trained models so no need to retrain
Metrics (computed on test unless noted)
Performance
Accuracy, Brier (report summary over Rashomon models; optionally also ensemble)
Predictive multiplicity
Mean variance
Ambiguity
Disagreement rate (ε=0.05)
Discrepancy


Spatial multiplicity
Pointwise variance
Moran’s I (k=30)
LISA (k=30)
Neighborhood agreement
LCAE (k=30): neighbors from validation, averaged over test
Also add:
Report K and candidate pool size
If you use ε instead of K: report Rashomon size per run
Null experiment: permutation test on test-set prediction variance


Compute prediction variance for each point in the test set


Compute Moran’s I / LISA  using test-set kNN graph (k=30)


Generate null by permuting predictions across test points:
independent permutation per model (permute the predictions of each model independently and recompute variance)
Repeat permutations R=100 per run; compare observed Moran’s I and LISA cluster rates to null


Experiment 3: Hotspot Robustness Across Runs
Goal: Systematically test whether HH hotspots are reproducible across random splits/seeds, rather than Monte-Carlo noise.
Inputs reused: All artifacts from the Primary experiment (same 10 runs per dataset/model family):
test sets per run, prediction variance vvv per test point, Moran’s I, LISA labels, kNN graph (k=30)
Key idea: Compare hotspots across runs at three levels.
(1) Point-level stability (“How often is point x in HH?”)
For each run r∈{1,…,10}, compute LISA categories on the test set and record HH indicator HH_r(i).
Compute per-point HH frequency
Summarize:
histogram of HH frequencies
fraction of points with freq ≥ 0.8, ≥ 0.5, etc.
Note: To make “point i” comparable across runs, use a stable identifier:
If you have a unique row ID → use it.
If not, use a deterministic hash of raw features + label (or keep original dataset row indices before splitting).


(2) Region-level stability (connected components / hotspot regions)
For each run, build the HH subgraph induced by HH points using the same kNN graph.
Extract connected components (hotspot regions).
Compare regions across runs using cluster/region similarity measures, e.g.:
Jaccard overlap between component node sets
Adjusted Rand Index / NMI after mapping points to region labels (only if you can align point IDs)
“Region persistence”: how often a similar component reappears (match components by maximum Jaccard)
(3) Summary stability statistics
Report (mean ± std over runs):
number of HH points
number of HH components
size of largest HH component
total HH mass (sum of variance over HH points)

Experiment 4: Sensitivity to Analysis Parameters
Goal: Show hotspots are not an artifact of specific parameter choices (ε, k, number of Rashomon models). Identify stabilization / elbow behavior.
This experiment has three sensitivity sweeps.
4A) Sensitivity to Rashomon tolerance (ε or top-K)
Choose one approach (recommended: top-K, since you already use it pragmatically).
If you use top-K: evaluate K ∈ {5, 10, 15, 25} (must be ≤ candidate pool size per family).
If you use ε: evaluate ε ∈ {ε₁, ε₂, ε₃, …} (absolute or relative), and report resulting Rashomon size.
For each setting, recompute:
pointwise variance on test
Moran’s I, LISA HH set
hotspot stability measures (from Experiment 3) within that setting


Stabilization criterion: Look for an “elbow” where increasing K (or ε) changes:
mean variance a lot initially but then plateaus
hotspot map similarity stops changing much (Jaccard/ARI stabilizes)


4B) Sensitivity to k in the kNN graph
Evaluate k ∈ {10, 20, 30, 50}. For each k:
build kNN graph on test features
recompute Moran’s I and LISA (HH sets)
compute hotspot stability summaries


4C) Sensitivity to Monte-Carlo sampling (candidate models / random search budget)
Evaluate random search budget B ∈ {25, 50, 100} candidates per model family (or a subset if compute is tight).
For each B:
train B candidates per family per run (same 10 outer seeds)
construct Rashomon set (top-K or ε)
recompute variance and hotspots
assess whether hotspots stabilize as B grows


Interpretation: If hotspots stabilize with larger B, that supports that they’re not driven by sampling noise.

What to Report (for both experiments)
For each dataset × model family (and optionally pooled):
HH frequency distribution (point-level)
hotspot region persistence / overlap metrics (region-level)
sensitivity curves:
Moran’s I vs K (or ε)
#HH points vs K (or ε)
Jaccard similarity of HH sets across parameter settings
Moran’s I vs k
stabilization with candidate budget B


Thesis claim enabled: Hotspots are reproducible across runs and stable across reasonable parameter ranges; observed patterns are not threshold/parameter artifacts nor Monte-Carlo sampling noise.

Experiment 5: Calibration Robustness Check
Goal
Test whether spatial multiplicity (variance, Moran’s I, LISA hotspots) is driven by probability miscalibration across model families.

Setup
Reuse trained models from the primary experiment (loaded from pickles).


Use the same 60/20/20 splits and the same Rashomon sets (selected via validation Brier).


No retraining of base models.



Calibration Procedure
For each run and each model in the Rashomon set:
Predict probabilities on the validation set.


Fit a calibration mapping (Platt scaling; optional: isotonic).


Apply the mapping to test-set probabilities.


Test labels are never used for calibration.

Metrics Recomputed (on calibrated test probabilities)
Mean variance


Ambiguity, Disagreement Rate, Discrepancy


Pointwise variance


Moran’s I (k = 30)


LISA (k = 30)


Number and size of HH hotspots


Jaccard overlap of HH sets (before vs after calibration)


Results reported as mean ± std over 10 runs.

Interpretation
If hotspots persist after calibration → spatial multiplicity is structural.
 If hotspots weaken → part of the effect is due to probability scaling differences.
Calibration is treated purely as a robustness check; the Rashomon set is not redefined.

Experiment 6: Hyperparameter-Space Drivers of Predictive Variance
Goal
Identify whether specific hyperparameters (or ranges) systematically induce higher predictive multiplicity (e.g., higher mean/pointwise prediction variance), and whether this effect is stable across runs.
Setup
Reuse the primary experiment artifacts: all trained candidate models (random search: 50 per model family per run), their hyperparameters, validation Brier, and test predictions.


Construct Rashomon sets per model family using validation Brier (top-K or ε).


Compute variance metrics on the test set as before.


Variables
For each model mmm in the Rashomon set, define:
HP vector hmh_mhm​ (hyperparameters)


Model-level variance score VmV_mVm​, e.g.:


Vm=1ntest∑i(fm(xi)−fˉ(xi))2V_m = \frac{1}{n_{test}}\sum_i \big(f_m(x_i) - \bar f(x_i)\big)^2Vm​=ntest​1​∑i​(fm​(xi​)−fˉ​(xi​))2 (distance to ensemble mean)


or use contribution to total variance (drop-one model effect)


Also keep:
validation loss LmvalL_m^{val}Lmval​ (for controlling performance)


Analyses
A) Univariate HP effects (per family)
For each hyperparameter:
Bin or smooth (for continuous HPs) and plot:


HP value → VmV_mVm​


Report effect size via:


Spearman correlation (continuous HPs)


ANOVA / Kruskal-Wallis (categorical HPs)


Include controls:


restrict to Rashomon models only (to avoid “bad models create variance”)


optionally regress out LmvalL_m^{val}Lmval​


B) Conditional effects / interactions
Fit a simple surrogate model predicting VmV_mVm​ from hyperparameters:
Random Forest regressor on (hm→Vm)(h_m \rightarrow V_m)(hm​→Vm​)


Report:


feature importance (per HP)


partial dependence / ALE curves for top HPs


C) Hyperparameter-space hotspots
Define “high-variance models”:
top 20% by VmV_mVm​ within each run/family
 Then:


check if they cluster in HP space:


kNN graph in HP space + Moran’s I on VmV_mVm​ (optional)


density comparison: HP distribution in high-variance vs low-variance sets


simple rules: decision tree depth 2–3 to describe “high variance region” (interpretable)


Metrics to Report
Per dataset × model family (mean ± std over 10 runs):
correlation/effect size per HP


top HPs ranked by importance (surrogate model)


stability of top HPs across runs (how often each HP appears in top-3)


example “high variance HP ranges” (e.g., depth > 12, learning_rate < 0.05, etc.)


Interpretation
If specific HP ranges repeatedly associate with high VmV_mVm​, you can claim:
predictive multiplicity is not only spatial (feature space), but also systematically induced by model capacity/regularization choices.

Experiment 7: Variance Decomposition (Hyperparameters vs. Seeds)
Goal
Decompose predictive variance into:
Between-hyperparameter variance (effect of different HP settings)


Within-hyperparameter variance (effect of training randomness / seeds)


This identifies whether multiplicity is primarily driven by model design choices or optimization stochasticity.

Scope
To control computation:
Conducted on 1–2 representative datasets


Conducted on 1–2 model families (e.g., RF and MLP)


Same 60/20/20 splits and 10 outer seeds as in the primary experiment



Training Design
For each run:
Sample H fixed hyperparameter configurations (e.g., H = 15–20).


For each configuration, train S different seeds (e.g., S = 5).


Total models per run per family: H × S.
Rashomon set is defined using validation Brier (top-K or ε), ensuring multiple seeds per HP survive.

Decomposition
On the test set, for each point:
Vartotal=VarHP+Varseed\text{Var}_{total} = \text{Var}_{HP} + \text{Var}_{seed}Vartotal​=VarHP​+Varseed​
Where:
Between-HP variance: variance of mean predictions across HP settings


Within-HP variance: average variance across seeds within each HP


Both are averaged over test points.

Reported Metrics
Per dataset × model family (mean ± std over 10 runs):
Total predictive variance


Proportion attributable to HP vs. seeds


Spatial Moran’s I for:


total variance


HP-induced variance


seed-induced variance



Interpretation
If HP variance dominates → multiplicity is structurally induced by model capacity/regularization.
 If seed variance dominates → multiplicity is mainly due to optimization randomness.
This experiment clarifies the source of predictive multiplicity.

Experiment 8: Clustering and Partitioning by Predictive Variance
Goal
Identify structured subgroups of observations that exhibit similar predictive variance behavior, and determine whether variance is explainable from input features or hyperparameters.

Data Used
From the primary experiment (per run):
Pointwise variance viv_ivi​ on the test set


Input features xix_ixi​


Model hyperparameters for Rashomon models


Validation/test predictions


No retraining required.

8.1 Unsupervised Clustering of Observations
Procedure
Cluster test observations using:


viv_ivi​ alone, or


augmented feature space (xi,vi)(x_i, v_i)(xi​,vi​)


Methods:


k-means


hierarchical clustering


Outputs
Cluster-level mean variance


Proportion of HH points per cluster


Spatial distribution of clusters


Purpose
Identify structured subgroups with systematically high or low multiplicity.

8.2 Supervised Regression of Variance
Procedure
Train a simple decision tree or random forest regressor:
 xi→vix_i \rightarrow v_ixi​→vi​
Evaluate:


R²


feature importance


tree splits (interpretable rules)


Purpose
Determine whether predictive variance is explainable by specific feature regions (e.g., age < 25, rare category, etc.).

8.3 Hyperparameter-Stratified Variance
Procedure
For each hyperparameter setting hhh:


Compute model-level variance contribution VmV_mVm​


Analyze:


correlation between HP values and VmV_mVm​


partial dependence / surrogate model importance


Purpose
Identify hyperparameter dimensions that systematically induce multiplicity.

Experiment 9: Subgroup Characterization of High-Variance Regions
Goal
Understand structural properties of high-variance subgroups.

9.1 Density and Data Geometry
Test whether high-variance points:
Lie in low-density regions (kNN density estimate)


Are near decision boundary (mean predicted probability ≈ 0.5)


Belong disproportionately to minority class


Metrics:
Compare density distributions (HH vs non-HH)


Compare prediction confidence distributions


Class proportion differences



9.2 Feature Involvement
Identify features overrepresented in HH clusters


Compare feature distributions between HH and non-HH


Use logistic regression:
 1{HH}∼xi\mathbf{1}\{HH\} \sim x_i1{HH}∼xi​

9.3 Explanation Instability (Optional Extension)
For selected datasets/models:
Compute SHAP values across Rashomon models


Compute variance of SHAP values per feature per point


Test correlation between:


predictive variance viv_ivi​


explanation variance


Alternative:
Compare counterfactual or rule-based explanation stability across models.



Interpretation
These analyses determine whether predictive multiplicity:
Is concentrated in low-density or boundary regions,


Is linked to specific features or population subgroups,


Propagates into explanation instability.

Notes on experimental details:

Rashomon set will once be computed family wise and once for all families together. Probably family-wise is the best for the experiments and all together can go in appendix.
Choose K to be 25 for global and 10 for family wise
Permutation-based p-values with FDR correction for LISA
Number of permutations for LISA p-values: choose and fix it (e.g., R=999 or R=499). Don’t reuse the “null R=100” unless you’re okay with coarse p-values
Define the frequency of a point being HH conditional on the point being in the test set since the test set is different for each seed.
Neighborhood definition: kNN graphs are built using Euclidean distance in the preprocessed, standardized feature space. Standardization parameters are fit on the training set only and applied to validation/test to avoid leakage. The same feature space is used for Moran’s I, LISA, and LCAE.
One-hot encoding + scaling: one-hot encoded features are included in the standardized space (fit on train only).
Check the library of LISA for what definition of high variance it uses by default
LISA implementation: Local Moran’s I is computed using pysal.esda.Moran_Local.
The variance vector vvv is standardized within each run prior to analysis (z-score).
HH classification follows PySAL quadrant definitions: a point is labeled HH if its standardized variance and the spatial lag of standardized variance are both positive and its FDR-corrected local p-value ≤ 0.05.
Spatial weights: We construct a kNN graph (k = 30) on the test features and use binary weights (neighbor = 1). We then row-standardize the weights (each row sums to 1) before computing Moran’s I and LISA in PySAL.
No per-family constraint: The global Rashomon set (top K = 25) is selected purely by validation Brier across all candidate models. No cap is imposed on the number of models per family.


Composition reporting: For each run, we report the number of selected models per family within the global Rashomon set.
The outer seed controls the train/val/test split, hyperparameter sampling, and model training randomness to ensure independent runs.
Compare hotspots using numeric-only vs full one-hot space or numeric-only vs Gower (if feasible).
Report whether Moran/LISA hotspots are qualitatively stable.
Ensemble prediction: For any method-level prediction f^(x)\hat f(x)f^​(x), we use the mean predicted probability across the selected Rashomon models:
 f^(x)=1M∑m=1Mfm(x).\hat f(x) = \frac{1}{M}\sum_{m=1}^{M} f_m(x).f^​(x)=M1​m=1∑M​fm​(x).
Performance metrics (Accuracy, Brier) are computed both (i) per-model across Rashomon models and (ii) on the ensemble prediction f^\hat ff^​ (reported as mean ± std over runs).


LCAE uses f^(xtest)\hat f(x_{test})f^​(xtest​) as the test prediction in its definition.
Nested sampling for budget sensitivity: To compare candidate budgets fairly, we use a fixed hyperparameter sampling stream per run and model family. Budgets are nested: the B=25 candidate set is a subset (prefix) of the B=50 set, and B=50 is a subset of B=100, ensuring differences are attributable to increased sampling rather than different HP draws.
kNN graph structure: The kNN graph (k = 30) is constructed as a directed graph (each observation has exactly k outgoing neighbors). The graph is not symmetrized. Binary weights are row-standardized prior to Moran’s I and LISA computation
LISA inference: Local Moran’s I p-values are computed using 999 permutations per run, followed by Benjamini–Hochberg FDR correction at 0.05.


Null experiment: The permutation-based null distribution is generated using 100 permutations per run (independent permutation per model), recomputing pointwise variance and spatial statistics for each permutation.
Model weighting: All models in the selected Rashomon set are treated equally. Ensemble predictions and variance-based metrics use uniform weighting across the K selected models (no performance-based weights).
Pointwise variance: We compute the population variance of predicted probabilities across the K selected Rashomon models (ddof = 0).
Candidate-level randomness: For stochastic learners , each candidate uses a deterministic but distinct seed random_state = outer_seed + candidate_id (per model family). This preserves reproducibility while avoiding identical randomness across candidates within a run.
Stratified splits: All train/validation/test splits are stratified by the target label to preserve class proportions across splits.
Classification threshold: Accuracy is computed using a fixed probability threshold of 0.5 for all models and datasets. No threshold optimization is performed.
For logreg use only lbfgs
Use bootstrap True for RF
MLP configuration: MLP models are trained with solver='adam' and early_stopping=True. Early stopping uses an internal validation fraction of 0.1 (default), with n_iter_no_change=10. Maximum iterations are set to 1000. Each candidate uses a deterministic seed as defined in the experiment protocol.
GBM configuration: We allow subsample ∈ {0.6, 0.8, 1.0}, enabling both deterministic (subsample=1.0) and stochastic gradient boosting. Randomness is controlled by candidate-level seeds as specified in the protocol.
