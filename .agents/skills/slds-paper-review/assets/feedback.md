# Feedback on `Master_Thesis.pdf`

## Review setup

- **Mode:** Draft-development review, because no target venue/template or submission-readiness level was specified.
- **Manuscript source of truth:** `/mnt/data/Master_Thesis.pdf`, cross-checked against `overleaf_bundle/chapters/*.tex`, generated tables in `overleaf_bundle/presentation_assets/tab/`, and selected implementation files in the uploaded repository.
- **Repository reviewed:** `/mnt/data/repo_unzipped/rashomon-multiplicity-main`
- **Git provenance:** Not applicable. The uploaded zip does not contain a `.git/` directory, so no commit hash, branch, remote alignment, or uncommitted-change status could be recorded.
- **Reviewed files/artifacts:** `Master_Thesis.pdf`; LaTeX chapters for introduction, related work, methodology, experimental setup, results, discussion, conclusion, appendix; `bibliography.bib`; generated result tables; selected code in `analysis/run_analysis.py`, `analysis/spatial.py`, `analysis/hp_decomposition.py`, `analysis/rule_extraction.py`, `analysis/calibration.py`, `analysis/experiment_runner.py`; result summaries in `results/*/*.csv`.
- **External literature spot-check:** Targeted searches for predictive multiplicity, Rashomon sets, hyperparameters, calibration, and individual-level reporting. Confirmed that the thesis cites the core predictive multiplicity/hyperparameter/legal-AI-Act line, but one very relevant 2026 calibration/multiplicity paper appears missing and should be checked before final submission.

## Manuscript summary

The thesis argues that predictive multiplicity should not only be summarized globally, but localized in feature space. It constructs finite top-$K$ Rashomon approximations from multiple model families, computes observation-wise predictive variance and conflict ratio, builds kNN graphs on preprocessed features, and uses Moran's $I$ and LISA to detect High--High multiplicity hotspots. The empirical story is coherent: COMPAS shows the clearest localized structure, German Credit has larger aggregate disagreement but weaker spatial concentration, Adult has broad/diffuse structure, and synthetic datasets validate the ability to recover known ambiguous regions. The thesis is already much stronger than a minimal master's thesis draft: it has clear research questions, multiple robustness checks, synthetic validation, a useful distinction between soft variance and hard conflict, and a cautious discussion of point-level instability.

The main remaining issue is not a missing experiment, but **claim calibration and methodological defensibility**. The thesis currently presents a lot of empirical evidence, but some definitions, null procedures, and interpretation steps need to be made more exact so that a skeptical reviewer/supervisor cannot dismiss the results as artifacts of graph choices, top-$K$ selection, calibration choices, or descriptive post-hoc analysis.

## Overall assessment

This is a promising and mostly coherent empirical thesis. The strongest parts are the central spatial framing, the soft-vs-hard disagreement comparison, the repeated robustness analyses, the synthetic validation, and the honest discussion that exact HH labels are unstable. The weakest parts are: (1) the operational Rashomon definition and top-$K$ approximation are still too easy to confuse with the classical $
$-Rashomon set; (2) the statistical testing story mixes several permutation procedures and should more clearly distinguish what each p-value tests; (3) the hyperparameter attribution section is descriptive but sometimes reads stronger than the method warrants; (4) figures/tables are numerous and sometimes not self-contained enough; and (5) the discussion/conclusion repeat the same message several times instead of sharpening the final contribution.

I would not add many more experiments now. The best payoff is to tighten the methodology, reduce overclaiming, and make the result chapter easier to audit.

## Highest-priority issues

### high-1: Make the nonstandard Rashomon approximation explicit earlier and more defensible

**Where:** Methodology, `overleaf_bundle/chapters/methodology.tex`, lines 31--54; Experimental setup, Section 4.5; Introduction/Abstract claims.

**Evidence:** The thesis first defines the classical tolerance Rashomon set, `R_epsilon = { f : L(f) <= L* + epsilon }`, then immediately replaces it with a top-$K$ selection rule because the epsilon needed for fixed size varies across datasets and splits (`methodology.tex`, lines 31--54). This is reasonable, but the text still often says "Rashomon set" without reminding the reader that this is a finite, rank-based approximation.

**Why it matters:** This is likely the first technical objection a supervisor/reviewer will raise. A top-25 set by validation Brier is not the same object as an $
$-Rashomon set, and top-$K$ can include models at very different absolute loss gaps across datasets/seeds. If this is not framed clearly, the thesis risks looking like it overclaims theoretical Rashomon-set conclusions from a practical candidate-pool procedure.

**Concrete fix:** Add a short boxed or italic clarification at the end of Section 3.2 and echo it in Section 4.5:

> In the empirical chapters, "Rashomon set" refers to a finite top-$K$ approximation within the sampled candidate pool, not to the full tolerance-defined Rashomon set over an unrestricted hypothesis space. The analysis therefore characterizes multiplicity among retained near-best candidates under the chosen model families and search spaces.

Then add one small diagnostic table or sentence reporting the validation Brier spread of the selected top-$K$ models by dataset, e.g. mean/max `val_brier - best_val_brier`. This would make the "near-optimal" claim auditable without adding a large experiment.

### high-2: Clarify exactly what the null tests prove; avoid using them as general proof of non-randomness

**Where:** Methodology Section 3.5; Experimental setup Section 4.8; Results Sections 5.3 and 5.12.

**Evidence:** The thesis uses at least two related but different null procedures: PySAL's internal Moran permutation over the variance field (`methodology.tex`, lines 139--143; `analysis/run_analysis.py`, lines around 290--307) and the custom prediction-matrix permutation null (`methodology.tex`, Section 3.5; `experimental_setup.tex`, Section 4.8). The result text then says the clustering is "not randomly distributed" and "not a byproduct of the marginal distribution of disagreement values alone". That is mostly right, but the exact null being rejected is narrower: it rejects spatial alignment after independently permuting each model's predictions across observations while preserving each model's marginal prediction distribution.

**Why it matters:** The null does not rule out all artifacts. It does not test whether the kNN graph is the right representation, whether the preprocessing metric is semantically meaningful, whether top-$K$ selection induces structure, or whether correlated model errors under realistic resampling could produce similar patterns. The thesis actually discusses graph dependence later, but the null-test language is stronger than the null design.

**Concrete fix:** In Section 5.3, replace broad phrasing like "spatially structured rather than randomly distributed" with:

> Under the prediction-permutation null, the observed Moran's $I$ is larger than expected when model predictions are detached from feature-space locations while preserving each model's marginal prediction distribution.

Then add one sentence:

> This does not by itself validate the feature-space metric; that question is addressed separately through kNN sensitivity and alternative graph constructions.

Also mention both p-value resolutions clearly: PySAL local/global permutations use 999 permutations, while the prediction-matrix null uses 100 permutations, so the empirical p-value floor is 1/101. The current text states this for the custom null but not always when discussing figures that also include internal Moran p-values.

### high-3: Strengthen the mathematical description of Moran's I/LISA to match PySAL and row-standardization

**Where:** Methodology Section 3.4, especially `methodology.tex`, lines 129--157.

**Evidence:** The global Moran's $I$ equation omits the usual $N/S_0$ factor (`methodology.tex`, lines 133--137). Because $W$ is row-standardized, $S_0 = N$, so the factor is 1. That should be stated explicitly. Similarly, the local statistic is written as $I_i = z_i \cdot l_i$ (`methodology.tex`, lines 148--153), while common local Moran formulations include a scale factor involving the sample variance. PySAL handles these normalizations internally.

**Why it matters:** A statistics examiner may notice the missing normalization and wonder whether the thesis uses a simplified statistic or the exact PySAL statistic. This is a small issue, but it affects technical credibility.

**Concrete fix:** After the global Moran equation add:

> Since the weights are row-standardized, $S_0=\sum_i\sum_j W_{ij}=N$, so the usual $N/S_0$ factor equals one and is omitted.

For LISA, either write the standard formula with the variance normalization or state that the displayed expression gives the sign/quadrant intuition, while implementation uses `esda.Moran_Local` with row-standardized weights. Best option: use the standard local Moran notation and then explain HH classification by signs of centered value and spatial lag.

### high-4: Hyperparameter importance is useful, but its interpretation must be weakened and better separated from causal language

**Where:** Experimental setup Section 4.11; Results Section 5.7; Discussion lines 23--24; Conclusion finding 3.

**Evidence:** The method computes a model-level contribution $V_m$ and decomposes its variation by grouped hyperparameter values (`experimental_setup.tex`, Section 4.11). The results say that hyperparameters "explain" remaining disagreement and that specific parameters are "drivers" (e.g., Section 5.7.2 and 5.7.3). The limitations correctly say the analysis is descriptive, but the main results repeatedly use stronger driver language.

**Why it matters:** Hyperparameters are sampled jointly, correlated, and filtered by top-$K$ performance. Grouping continuous values into tertiles also changes the estimand. A one-way grouped variance decomposition cannot isolate independent effects when hyperparameters are correlated. The meta-model diagnostic helps, but it is also descriptive.

**Concrete fix:** Keep "driver" in section titles if you like, but add a precise caveat at the start of 5.7:

> Throughout this section, "driver" means an observed association between a modeling choice and variation in model-level disagreement within the sampled candidate pool; it should not be read as a causal effect of changing that hyperparameter while holding all others fixed.

Then change repeated statements such as "regularization strength C explains nearly all" to "variation across grouped values of C accounts for most of the observed variation in $V_m$ within the retained logistic-regression models." This is more defensible and still clear.

### high-5: The thesis needs a sharper "nearest-neighbor work" comparison, especially around calibration and local/individual multiplicity

**Where:** Related Work Sections 2.2--2.5; Results 5.6.3; Discussion 6.6.

**Evidence:** The bibliography covers Marx et al., Watson-Daniels et al., Hsu/Calmon, RashomonGB, Cavus et al. on hyperparameters, Sokol et al., and Frohnapfel et al. This is good. However, a targeted search found a directly relevant recent paper: **CITATION TO VERIFY: Mustafa Cavus, "Mitigating the Multiplicity Burden: The Role of Calibration in Reducing Predictive Multiplicity of Classifiers", arXiv:2603.11750**. It specifically studies calibration and predictive multiplicity using Platt and isotonic methods, which overlaps with your Section 5.6.3.

**Why it matters:** Your calibration robustness section currently reads as an internal robustness check. If a recent paper studies calibration as a multiplicity mitigation mechanism, you should either cite it or explain how your use differs: you are not mainly asking whether calibration reduces multiplicity globally, but whether spatial hotspot conclusions survive calibration.

**Concrete fix:** Add one paragraph in Related Work or Section 5.6.3:

> Recent work has also examined calibration as a way to reduce predictive multiplicity. Our calibration analysis has a different role: it is a robustness check for spatial hotspot structure rather than a proposed mitigation strategy.

Verify the exact paper details before citing. If you do not want to add this citation, at least avoid implying that your calibration analysis is isolated from related work.

## Section-by-section critique

## Title and front matter

### title-1: Title is accurate but slightly generic

**Current title:** "Variance-Based Analysis of Predictive Multiplicity in Rashomon Sets"

**Assessment:** Accurate and searchable, but it hides the most distinctive part: spatial localization/hotspots. A stronger title would advertise the actual novelty.

**Possible alternatives:**

- "Localizing Predictive Multiplicity in Rashomon Sets with Spatial Hotspot Analysis"
- "Spatial Hotspots of Predictive Multiplicity in Rashomon Sets"
- "Where Near-Optimal Models Disagree: Spatial Analysis of Predictive Multiplicity"

No need to change if LMU title constraints exist, but a spatial/hotspot word would make the contribution clearer.

### title-2: Date appears future relative to current draft

**Where:** Title page shows "Munich, June 29th, 2026".

**Why it matters:** If this is a placeholder for expected submission date, fine. If not, it can look careless. Before submission, ensure the date matches the actual signed declaration date.

## Abstract

### abstract-1: Strong abstract, but it under-specifies the finite top-$K$ nature of the Rashomon set

**Evidence:** The abstract says "across a Rashomon set of near-optimal models" and "Experiments on three benchmark datasets with five model families". It does not mention that the empirical Rashomon set is a finite top-$K$ approximation selected by validation Brier score.

**Why it matters:** The abstract is where the strongest version of the contribution is formed. A reader may assume a tolerance-defined or exhaustive Rashomon set.

**Concrete fix:** Add one phrase:

> "across finite top-$K$ approximations of Rashomon sets selected by validation Brier score"

or, less technical:

> "across finite candidate-pool approximations of near-optimal models"

### abstract-2: "Confirm" and "non-random" are slightly too strong

**Evidence:** The abstract says experiments "confirm that predictive multiplicity is consistently spatially structured and non-random." This is supported under the tested datasets and nulls, but "confirm" sounds definitive.

**Concrete fix:** Replace with:

> "show that predictive multiplicity is consistently spatially structured under the tested benchmark settings and permutation nulls."

This keeps the result strong but bounded.

## Introduction

### intro-1: The motivation examples are useful but make Chapter 1 long and slightly fictional

**Evidence:** Section 1.1 contains three fictional examples. They are readable, but the thesis later repeats their conceptual points in the problem statement, research questions, and contributions.

**Why it matters:** Your supervisor already noted length/redundancy. Chapter 1 is a good place to cut without losing scientific content.

**Concrete fix:** Compress the three examples into one shorter running example, or keep the headings but reduce each to one paragraph of 4--5 lines. The key ideas are: individual instability, clustering, and drivers. You do not need as much narrative detail about Alice.

### intro-2: Add the nonstandard Rashomon caveat already in Chapter 1

**Evidence:** Research question 1 asks about "equally accurate models" and contributions say "Rashomon set". The operational top-$K$ approximation only becomes clear later.

**Concrete fix:** In Section 1.2 after the classical definition, add:

> Empirically, this thesis approximates such sets by finite top-$K$ candidate pools ranked by validation Brier score; hence conclusions concern this operational approximation rather than the full continuous Rashomon set.

This directly addresses a likely supervisor comment.

### intro-3: Contributions are clear but could separate method vs empirical finding

**Evidence:** Contribution 1 is the framework, 2 is evidence, 3 is attribution, 4 bundles robustness/validation/interpretability.

**Concrete fix:** Slightly sharpen contribution 3 by saying "descriptive variance decomposition" instead of "develop a variance decomposition" unless you want to argue it is technically novel. This avoids overclaiming a standard law-of-total-variance decomposition as a new method.

## Background and related work

### related-1: Good coverage, but the gap statement should be more precise

**Evidence:** Section 2.5 says what is missing is a framework linking per-observation multiplicity signals to feature-space geometry. This is good. However, the section should explicitly say that the closest prior works already provide per-instance measures, but not spatial autocorrelation/local cluster tests.

**Concrete fix:** Add a two-sentence bridge:

> Prior per-instance approaches can identify observations with large feasible prediction ranges or high conflict, but they treat these observations independently. The distinct step in this thesis is to test whether high-disagreement observations are locally dependent on a feature-space graph and to summarize them as statistically significant HH regions.

### related-2: Verify and possibly add calibration-multiplicity citation

**Evidence:** Section 5.6.3 uses Platt and isotonic calibration as a robustness check. A targeted search found **CITATION TO VERIFY: Cavus, "Mitigating the Multiplicity Burden: The Role of Calibration in Reducing Predictive Multiplicity of Classifiers", arXiv:2603.11750**.

**Concrete fix:** Add this only after verifying the source. It is directly relevant to calibration/multiplicity and would strengthen the positioning of your robustness analysis.

### related-3: Some references need bibliographic cleanup

**Evidence:** In the PDF reference list, several entries are incomplete or preprint-like: Hsu and Calmon (2022) lacks venue details; RashomonGB (2024) lacks venue/arXiv metadata; several arXiv URLs are shown as raw URLs. This may be acceptable for a thesis, but it looks uneven.

**Concrete fix:** Before final submission, normalize all bibliography entries: author names, year, title capitalization, venue/proceedings/journal where available, arXiv identifier if no published version exists, and protected capitalization for terms like `{RashomonGB}`, `{COMPAS}`, `{AI}`.

## Methodology

### method-1: Brier-score choice is plausible, but needs a short justification against accuracy/AUC

**Evidence:** Section 3.2 says Brier is aligned because the main response is predicted-probability spread (`methodology.tex`, lines 43--46). That is good but very short.

**Why it matters:** Since predictive multiplicity is often defined over classification decisions, a reader may ask why Brier is the selection criterion and not accuracy, log loss, or AUC.

**Concrete fix:** Add one sentence:

> Brier score is a proper scoring rule for probabilistic predictions, so it selects models that are near-optimal with respect to calibrated probability quality rather than only thresholded classification accuracy.

You can also state that conflict-ratio analyses partially address threshold-level disagreement.

### method-2: The prediction variance estimator convention should be specified

**Evidence:** The thesis defines $v_i = \mathrm{Var}_m(\hat p_{mi})$ but does not say whether this is population variance with denominator $M$ or sample variance with denominator $M-1$. NumPy defaults to `ddof=0`, and the exact scale matters slightly for reproducibility.

**Concrete fix:** Add "computed as population variance over the retained model set" if using NumPy default. If using `pandas.var` or `ddof=1` anywhere, standardize and state it.

### method-3: kNN graph construction on one-hot encoded features needs one limitation earlier

**Evidence:** Section 4.4 says categorical features are one-hot encoded and the same transformed space is used for kNN graph construction. Section 5.6.4 later checks PCA/cosine alternatives. The limitation appears in Discussion, but the method section could acknowledge it upfront.

**Why it matters:** Euclidean distance in high-dimensional one-hot encoded space is not always semantically meaningful. Your alternative graph robustness helps, but the reader should not have to wait until later to know you are aware of this.

**Concrete fix:** In Section 3.4 or 4.7 add:

> Because Euclidean distance on mixed numeric/one-hot features is only one operational notion of similarity, graph-dependent conclusions are later checked using PCA-reduced and cosine-distance graphs.

### method-4: Local multiple testing and FDR need exact implementation detail

**Evidence:** The thesis says Benjamini--Hochberg at $\alpha=0.05$ controls FDR across all $N$ local tests. It does not specify whether BH is applied to all local Moran p-values or only to candidate HH/LL quadrants, and whether p-values are one-sided/two-sided.

**Why it matters:** LISA cluster labeling depends on this detail. A reviewer can reproduce only if the correction procedure is explicit.

**Concrete fix:** Add one sentence in Section 3.4/4.7:

> BH correction is applied to the vector of local Moran pseudo p-values for all test observations; only observations passing the corrected threshold and satisfying the sign conditions are labeled HH/LL/HL/LH.

If the code differs, update the text accordingly.

## Experimental setup

### exp-1: Dataset sizes and feature counts are missing from the main setup

**Evidence:** Section 4.1 describes the three datasets qualitatively but does not give sample sizes after filtering, number of features before/after preprocessing, class balance, or test-set sizes.

**Why it matters:** Many result interpretations depend on test-set size: Adult has 631 HH points partly because its test set has 9769 observations, while German has only 200. Without a dataset table, this context comes too late and indirectly.

**Concrete fix:** Add a compact Table in Section 4.1:

Dataset | n after preprocessing/splitting | test n | positive rate | numeric features | categorical features | transformed dimension.

This would also make HH rates and graph parameters easier to interpret.

### exp-2: Search spaces should be in an appendix table or electronic appendix reference

**Evidence:** Section 4.3 lists which hyperparameters vary but not their actual ranges/distributions.

**Why it matters:** The candidate pool defines the empirical Rashomon approximation and hyperparameter importance results. Without search spaces, the experiment is not fully reproducible from the thesis text.

**Concrete fix:** Add an appendix table with search ranges for all five families. It can be compact and copied from the code. In the main text, add "Full search spaces are reported in Appendix X."

### exp-3: Calibration protocol likely needs a clearer anti-leakage statement

**Evidence:** Section 4.10 says each model's calibration mapping is fitted on validation predicted probabilities and validation labels, then applied to test predictions. Good. But because the same validation set is used for selecting top-$K$ models and fitting calibration, there is a possible double-use concern.

**Why it matters:** This is probably acceptable for a robustness check, but it should be acknowledged. Calibration is not used to select the Rashomon set, but the validation labels are reused.

**Concrete fix:** Add:

> Calibration is used only as a post-selection robustness check; the selected model set remains fixed. Since the validation set is also used for Rashomon selection, calibrated metrics are interpreted descriptively rather than as an independent performance estimate.

### exp-4: Fixed-test vs independent-split protocols are a strength; make the result sections consistently remind the reader

**Evidence:** Section 4.2 clearly distinguishes primary independent splits and fixed-test-set splits. Section 5.4 correctly states it uses fixed-test. Some later summaries talk about "across runs" without re-emphasizing that stability results use a different split protocol.

**Concrete fix:** In the Results summary item 3, write "under the fixed-test-set stability protocol" so the reader does not confuse it with the independent split results.

## Results

### results-1: Result chapter is comprehensive but too table/figure dense

**Evidence:** Results spans many subsections and includes many tables/figures: core metrics, conflict metrics, null summary, HH frequency, Jaccard heatmap, component table, per-family panels, K sensitivity, k sensitivity, calibration panels, graph comparison, family decomposition, HP decomposition, synthetic figures, rule tables, margin table, fairness table.

**Why it matters:** The evidence is strong, but the narrative may feel like a sequence of diagnostics. A reader may miss the three central findings.

**Concrete fix:** At the beginning of Chapter 5, add a short "reading guide" paragraph with three claims:

1. Multiplicity is spatially clustered under the prediction-permutation null.
2. Hotspot membership is regionally but not pointwise stable.
3. Model family and selected hyperparameters are associated with much of the disagreement.

Then move one or two auxiliary items to the appendix if page count matters: the full calibration panel for all datasets, the alternative graph table/figure duplication, and some rule-frequency tables.

### results-2: Avoid saying Adult has "weakest" clustering without normalizing context

**Evidence:** Section 5.1 says Adult has the weakest Moran's I, yet later conflict Moran's I for Adult is stronger than German and Adult has large HH counts. This is not inconsistent, but it can confuse readers.

**Concrete fix:** Use more precise phrasing:

> For variance-based Moran's $I$, Adult has the smallest mean value among the three datasets, although its large test set still yields many HH observations and its conflict-based clustering is stronger than German Credit's.

### results-3: German Credit conclusions should be more cautious throughout

**Evidence:** German Credit has only 200 test observations per run and a mean HH count of 5.4 with std 5.4. Yet several cross-dataset statements include German Credit alongside COMPAS/Adult.

**Why it matters:** Small HH counts make LISA, component, and HH-specific decomposition unstable. You already mention this in places, but the conclusion still groups all datasets together.

**Concrete fix:** In the Results summary and Discussion, say:

> German Credit supports the global Moran/null conclusion, but hotspot-level interpretations are limited by the small number of detected HH observations.

This keeps the finding but bounds the weakest dataset.

### results-4: Figures should have more self-contained captions

**Evidence:** Many captions identify what is plotted but not the takeaway. Example: Figure 1 says what panels show; it does not state the key conclusion that observed values exceed the prediction-permutation null across datasets. Figure 9 explains bars but not why HH vs non-HH matters.

**Concrete fix:** Add one final takeaway sentence to major captions:

- Figure 1: "Observed Moran's $I$ is separated from the prediction-permutation null in all datasets."
- Figure 9: "Family effects are larger in HH regions for COMPAS and Adult."
- Figure 10: "Within-family disagreement is concentrated in a small number of hyperparameters per family."

### results-5: K-sensitivity result needs quantitative support for "stabilizes"

**Evidence:** Section 5.6.1 says Moran's I stabilizes once $K$ reaches 20--25 and remains in a narrow range. The plot supports this visually, but there is no numerical criterion.

**Why it matters:** "Stabilizes" can look subjective.

**Concrete fix:** Add one sentence with a simple range:

> For example, for $K \ge 25$, COMPAS variance Moran's $I$ remains between approximately X and Y across the mean curve.

Fill X/Y from the plotted data or exported CSV. If you do not want to add numbers, write "visually stabilizes" and treat it as descriptive.

### results-6: Calibration robustness could be positioned better relative to new literature

**Evidence:** Section 5.6.3 finds Platt often reduces variance on COMPAS while isotonic can increase variance. The thesis says calibration does not eliminate spatial structure. This is good, but recent work on calibration as multiplicity mitigation makes this result more interesting.

**Concrete fix:** Add a short contrast:

> Unlike work that evaluates calibration as a multiplicity mitigation method, here calibration is used to ask whether hotspot conclusions are artifacts of probability scaling. The answer is mixed at the mask level but stable at the spatial-autocorrelation level.

Verify and cite the relevant work first.

### results-7: Rule extraction results are useful but the reported rules contain redundant thresholds

**Evidence:** Table 13 includes rules like `priors_count > 14.5 AND priors_count > 17.5 AND age > 33`, where the first priors threshold is redundant. Similar redundancy appears in multiple rules.

**Why it matters:** Redundant conjuncts make rules look automatically generated and less polished. They also distract from interpretability.

**Concrete fix:** Simplify rules before reporting by removing dominated numeric thresholds on the same feature. For example, replace:

`priors_count > 14.5 AND priors_count > 17.5 AND age > 33`

with:

`priors_count > 17.5 AND age > 33`.

This is a presentation cleanup; it does not require rerunning experiments if you only simplify logically equivalent displayed rules.

### results-8: Fairness/subgroup exposure is valuable but should be explicitly auxiliary in headings

**Evidence:** Section 5.11 reports a significant race-based HH exposure disparity. The text is cautious, but this is a sensitive claim.

**Why it matters:** A reader may overinterpret it as a fairness result, even though it is observational and exploratory.

**Concrete fix:** Rename the subsection to:

> Exploratory Subgroup Exposure Diagnostic (COMPAS)

Also add in the first sentence:

> This section is not a full fairness evaluation; it only asks whether hotspot exposure differs descriptively across available COMPAS subgroup labels.

## Discussion and conclusion

### discussion-1: Discussion is conceptually good but repeats the same core claim too often

**Evidence:** The phrases "localized", "structured", "not only global", "feature-space regions", and "model choice matters" recur in Sections 6.1, 6.2, 6.3, 6.6, 7.1, and 7.2.

**Why it matters:** The repetition makes the final chapters feel longer than necessary and slightly LLM-polished. Your supervisor already noticed that some parts read LLM-generated and redundant.

**Concrete fix:** Cut 15--25% from Chapters 6 and 7. Suggested deletions/merges:

- Merge 6.6 into the end of 6.1 or 6.2; it is only one short paragraph and repeats the positioning.
- In 7.1, reduce the five findings to 3--4 more compact findings or point back to 5.12 instead of restating the same list.
- Remove one of the two paragraphs in 7.2 that restates average performance is insufficient.

### discussion-2: Practical deployment workflow is plausible but should be explicitly framed as speculative

**Evidence:** Section 6.3 proposes retaining near-optimal sets, computing hotspot maps, and treating hotspot predictions differently.

**Why it matters:** The thesis did not evaluate deployment interventions, human review, abstention policies, runtime overhead in deployment, or user outcomes.

**Concrete fix:** Add "A possible" or "In applications where retaining multiple models is feasible" to the workflow. Also add one limitation: hotspot maps would need recalibration/revalidation under distribution shift before deployment.

### conclusion-1: Conclusion should explicitly answer the thesis title/problem in one sentence

**Evidence:** The conclusion is clear but somewhat general. It says multiplicity is localized and structured, but the final takeaway could be sharper.

**Concrete fix:** Add a final sentence like:

> The main lesson is that model multiplicity should be audited not only by asking how much near-optimal models disagree, but by asking where that disagreement concentrates and which modeling choices make those regions unstable.

## Numerical and internal consistency checks

### consistency-1: Main reported numerical values match generated result tables in the repository

I spot-checked the main tables against generated `.tex` tables and CSV summaries. The following values are consistent:

- Dataset summary: COMPAS mean variance 0.0013, Moran's $I$ 0.210, HH count 128.4; German 0.0050/0.088/5.4; Adult 0.0014/0.075/631.0.
- Conflict summary: COMPAS mean conflict 0.035, conflict Moran's $I$ 0.17; German 0.059/0.05; Adult 0.017/0.12.
- Null significance: 100% for COMPAS/Adult, 90% for German, for both variance and conflict.
- Rule tables and feature-frequency tables in the PDF match `presentation_assets/tab/*.tex`.

No major numeric contradiction was found in the spot-check.

### consistency-2: Dataset naming should be standardized

**Evidence:** Tables alternate between `Compas`, `COMPAS`, `German`, and `German Credit`; text uses `German Credit`.

**Why it matters:** Small style inconsistency, but it affects polish.

**Concrete fix:** Use `COMPAS`, `German Credit`, and `Adult` everywhere in tables, captions, and plots. If figure scripts output lowercase labels (`compas`, `german`, `adult`), update plot label mapping.

### consistency-3: Some table formatting uses `\include` instead of `\input`

**Evidence:** In `results.tex`, Table conflict summary uses `\include{presentation_assets/tab/conflict_summary}` while most tables use `\input{...}`.

**Why it matters:** `\include` forces page breaks and is intended for larger document units. It can create layout artifacts.

**Concrete fix:** Replace with `\input{presentation_assets/tab/conflict_summary.tex}`.

### consistency-4: The AI tools declaration is transparent, but the declaration of authorship may conflict in wording

**Evidence:** The AI tools page says generative AI tools were used for linguistic, coding, and literature-orientation support. The declaration page says "my own unaided work."

**Why it matters:** Many universities still use a fixed declaration wording, so this may be required. But if not required, "unaided" can appear in tension with the AI declaration.

**Concrete fix:** If LMU requires the exact declaration, leave it. If editable, consider wording such as "except for the aids transparently declared above" or ask your supervisor/examination office.

## Code and reproducibility checks

### repro-1: Repository is mostly well organized, but thesis should point to exact reproduction entry points

**Evidence:** The repository contains `run_training_pipeline.py`, `run_training_pipeline_fixed_test.py`, notebooks 01--10, `scripts/export_thesis_assets.py`, and generated results. The thesis electronic appendix only says "Data, code and figures are provided in electronic form."

**Why it matters:** A reader/examiner should know what to run or inspect first.

**Concrete fix:** Expand Appendix B by 5--8 lines:

- main training entry point: `run_training_pipeline.py`
- fixed-test stability entry point: `run_training_pipeline_fixed_test.py`
- analysis notebooks: `notebooks/01_...` to `10_...`
- generated thesis tables/figures: `overleaf_bundle/presentation_assets/`
- requirements: `requirements.txt`

### repro-2: Requirements/version information may be enough in repo, but the thesis should state the key libraries

**Evidence:** The user previously considered removing detailed versions. That is okay, but the thesis should still mention core libraries for reproducibility: scikit-learn, PySAL/libpysal/esda, NumPy/Pandas, SciPy.

**Concrete fix:** Add one sentence in Section 4.9:

> The implementation uses scikit-learn for model training and preprocessing and PySAL/libpysal/esda for spatial weights and Moran/LISA statistics; exact package versions are listed in the electronic appendix requirements file.

### repro-3: Top-K loss gaps would greatly improve auditability

This is repeated because it is the single highest-value small addition. You do not need a new analysis pipeline; the metadata already contains validation Brier scores. Export a small table:

Dataset | global top-25 mean Brier gap | max Brier gap | per-family top-25 max gap range.

This would defend the "near-optimal" label much better than verbal explanation.

## Figures and tables

### figs-1: Consider reducing duplicate table+figure evidence

**Evidence:** Alternative graph constructions are shown as both Table 10 and Figure 8. For a thesis this is acceptable, but if page count matters, keep one in the main text and move the other to appendix.

**Concrete fix:** Keep Table 10 in main text if exact values matter; move Figure 8 to appendix. Or keep Figure 8 and move Table 10 to appendix.

### figs-2: Some multi-panel figures are likely too dense in print

**Evidence:** Figure 7 contains three full-width calibration figures stacked. Figure 10 is a large grid of hyperparameter importance plots.

**Concrete fix:** For Figure 7, either split by dataset or keep only COMPAS in the main text and move German/Adult to appendix. For Figure 10, ensure font sizes remain readable in the final PDF; otherwise include a compact top-2 hyperparameter table in main text and move the full grid to appendix.

### figs-3: Use consistent capitalization in plot titles

**Evidence:** Plots show labels like `compas`, `german`, `adult` in lower case. Tables use mixed capitalization.

**Concrete fix:** Apply display-name mapping in plotting scripts: `{'compas': 'COMPAS', 'german': 'German Credit', 'adult': 'Adult'}`.

## Style and concision

### style-1: Reduce repeated "not merely global, but localized" phrasing

This phrase is correct but appears throughout the thesis. Replace some repeats with more specific claims:

- "HH regions recur at the regional but not pointwise level."
- "Family choice contributes more inside COMPAS/Adult hotspots."
- "Conflict and variance identify different affected observations."

This will make the writing feel more human and less template-like.

### style-2: Avoid overly broad words: "confirm", "prove", "drivers", "non-random"

Suggested replacements:

- "confirm" -> "show", "support", "provide evidence that"
- "drivers" -> "associated sources", "contributors", "variance components" when causal interpretation is not established
- "non-random" -> "larger than expected under the prediction-permutation null"

### style-3: Some paragraphs are very polished but generic

The most generic-sounding areas are Discussion 6.1--6.3 and Conclusion 7.1--7.2. Make them more concrete by tying claims to your actual findings: COMPAS vs German vs Adult, top-$K$ approximation, HH instability, family decomposition, and synthetic recovery metrics.

## Suggested priority action list

1. **Add the top-$K$/finite Rashomon caveat in Abstract, Introduction, Methodology, and Experimental setup.** This is the biggest credibility fix.
2. **Export/report validation Brier gaps for selected top-$K$ models.** This is a small table with high payoff.
3. **Tighten the null-test interpretation.** State exactly what is permuted and what conclusion follows; avoid broad "non-random" language.
4. **Fix the Moran/LISA formula explanation.** Add the row-standardization normalization note and align local Moran notation with PySAL.
5. **Weaken hyperparameter causal language.** Keep the analysis, but call it descriptive grouped association unless you add causal/conditional controls.
6. **Add a dataset summary table with sample sizes/class balance/feature dimensions.** This will clarify Adult/German/COMPAS comparisons.
7. **Simplify displayed rule conjuncts by removing redundant thresholds.** This will make the interpretability section look much cleaner.
8. **Check and possibly cite recent calibration-multiplicity work.** Especially relevant to Section 5.6.3.
9. **Cut repetition in Discussion/Conclusion by about 15--25%.** Merge repeated positioning statements and make final takeaways more specific.
10. **Polish figure/table consistency.** Standardize dataset names, use `\input` for tables, and move one or two dense auxiliary visuals to the appendix if page count is a concern.

## Bottom line

The thesis is scientifically promising and already contains enough empirical material. The next revision should not mainly add more results. It should make the central methodological object more precise: a finite top-$K$ candidate-pool Rashomon approximation; a prediction-permutation spatial null; descriptive, not causal, attribution; and region-level, not pointwise, hotspot stability. If those boundaries are stated clearly, the thesis will read as much more rigorous and less overclaimed. 
