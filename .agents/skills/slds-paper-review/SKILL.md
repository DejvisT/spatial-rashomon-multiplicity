---
name: slds-paper-review
description: >
  Reviews ML/statistics manuscripts and produces one rigorous feedback file with section-by-section critique, math/notation checks, explicit numerical/internal-consistency checks, claim-evidence calibration, SOTA positioning, experiment/evaluation review, theoretical-strengthening suggestions, and revision priorities. Use for paper feedback, manuscript review, submission readiness, scientific critique, numerical inconsistency checks, overclaim detection, or clarity/structure review. Does not edit manuscript or source files.
tools: Read, Edit, Glob, Grep, Bash, WebSearch, AskUserQuestion
---

# Skill: Review ML/Statistics Papers

## Task contract

Act as a strict, constructive co-author and scientific editor. Help the author improve the paper's clarity, rigor, evidence, structure, and readability.

Review the manuscript. Do not rewrite, patch, rename, or otherwise edit manuscript or source files.

## Step 1 - Determine scope

Ask the user, unless the prompt already gives enough context to proceed:

> "Which manuscript should I review, what is the target venue or template if any, and should this be submission-ready, draft-development, or targeted feedback?"

Review modes:

- **Submission-ready:** judge as near-final.
- **Draft-development:** focus on high-leverage draft improvements.
- **Targeted:** focus on requested sections or concerns.

If the user asks you to proceed without answering, assume **draft-development** and record that assumption in `Review setup`.

Resolve the source of truth:

- Prefer the user's explicit manuscript file or pasted text; ask if none is identified.
- For LaTeX, review the assembled paper (main file, included sections, figures, tables, bibliography, appendix, supplement).
- Include venue template and layout checks when available.
- Ignore unrelated notes, old drafts, and build artifacts unless needed for a claim.

## Step 2 - Inspect provenance

If the manuscript is in a git repo, record review provenance for the feedback file:

```bash
git rev-parse HEAD
git status --short
git status --branch --short
```

Include commit, remote/branch alignment when available, reviewed files, and whether manuscript files have uncommitted changes. If not in git, state provenance as not applicable.

## Step 3 - Read the manuscript

Read the assembled manuscript end-to-end before drafting feedback so load-bearing claims, scope, and section order are understood as a whole. Map the paper's real section order, then inspect the bibliography, figures, tables, appendix, supplement, and any source files needed to understand load-bearing claims.

Build a private review map before writing:

- Main claim, central mechanism or theorem, and what evidence actually supports it.
- Which artifacts were checked or unavailable.
- Highest-risk gaps a reviewer would question (novelty, correctness, validity, clarity).

Evidence rules:

- Do not invent experiments, proofs, baselines, datasets, or empirical conclusions.
- Separate what the manuscript shows from what it implies or still needs to verify; mark gaps as `TODO` with what would resolve them.
- Anchor substantive points with section/line references and the smallest useful quote.

## Step 4 - Check nearest-neighbor and SOTA work

Before drafting, check the manuscript, bibliography, repo, and available literature/search tools for nearest-neighbor or SOTA methods that may already solve the main problem. Keep the search bounded: use the paper's title/abstract keywords, claimed contribution, and bibliography to run a few targeted searches rather than turning the review into a literature survey.

Use external leads conservatively:

- Do not invent bibliographic details.
- Label suggested missing citations or uncertain leads as `CITATION TO VERIFY: <keywords / author-topic hint>`.
- Focus on whether the manuscript clearly distinguishes its contribution from the closest known work.

## Step 5 - Apply review criteria

Use these criteria as a checklist while drafting. Preserve the paper's notation; do not silently rename symbols or redefine variables. Put issues in the relevant section critique, not as a detached rubric. Favor fewer, sharper issues over exhaustive low-signal coverage; every substantive criticism should have evidence, why it matters, and a concrete repair path.

Paper-wide checks:

- **Readiness and risk:** apply the selected mode consistently; identify strengths, blockers, reviewer trigger points, and what remains for a strong submission.
- **Contribution and positioning:** check whether the contribution is sharp, visible early, novel without exaggeration, and distinguished from nearest-neighbor or SOTA work with enough context and citations. Flag originality or integrity risks only when they follow directly from the manuscript or repo, such as weak differentiation from very close work or possible duplicate/self-overlap. Check whether the paper tells one clear story (e.g., in a goal-problem-solution rhythm); ensure aims, definitions, abbreviations and claims appear before use.
- **Significance and audience fit:** check whether the paper meaningfully advances the field, target community, or venue audience relative to the closest prior work; whether the practical, empirical, or theoretical payoff is clear; and whether the scope of the claims matches the actual importance of the results.
- **Claims and evidence:** flag overstatements, unsupported claims, missing citations, contradictions, internal inconsistencies, misleading figure or plot presentation, conclusions that outrun the paper's own data/results, and limitations that do not bound the contribution.
- **Numerical and internal consistency:** cross-check values across abstract, text, tables, figures, appendix, and supplement; verify reported deltas or percentage improvements against the cited numbers; check sample sizes, averages, percentages, p-values, confidence intervals, and effect sizes when present; ensure equation/figure/table references resolve; and check acronym first use, terminology consistency, and citation formatting/completeness. Treat repeated small inconsistencies as credibility issues.
- **Technical rigor:** check theorem-claim alignment, proof gaps, assumptions, undefined or overloaded symbols, notation consistency, indexing, dimensions/shapes, quantifiers/scope, overloaded formalism, and possible theorem/proposition additions that would strengthen the paper; treat new theorem ideas as proposals to verify (interpretation, what it buys, short proof sketch). If code that implements the methods is available, check whether equations, algorithms, and experimental descriptions match the implementation.
- **Evaluation and validity:** check research questions, baseline relevance/fairness, ablations, metrics, tuning, data splits/leakage, seeds, uncertainty, statistical tests, robustness, sensitivity, runtime, memory, scaling, and failure cases.
- **Presentation and reproducibility:** check venue/template compliance, title, keywords, teaser, figures, tables, captions, reproducibility artifacts, bibliography quality, limitations, redundancy, repetition, and concision.

Section checks:

- **Title / acronym / keywords / teaser:** check whether the title is attractive, relevant, concise, searchable, and jargon-light while advertising the core insight without overclaiming or cute obscurity. Check that keywords cover the main research areas in priority order; any acronym is short, distinctive, pronounceable, and inferable from the method; and the teaser or first figure teaches the main idea, problem, or result at a glance.
- **Abstract:** check that it stands alone as a searchable summary with purpose/problem, method or design, main results, and conclusion/implication. Flag abstracts that motivate without findings, report methods without results, overgeneralize specific results, or omit why the reader should continue.
- **Introduction:** by the end of the introduction, a reviewer should know the audience, goal, unsolved problem, importance, research question or hypothesis, core insight or "nugget", main contribution, evidence plan, and scope. Flag missing hypothesis/test logic, vague novelty language, "we are interested in" framing, unsupported "more/better" claims, and promises not delivered later.
- **Related Work / Background:** check that related work is organized by themes and history rather than as a laundry list. Each group should explain what prior work gets right, what it misses, and how this paper goes further; the section should end by clarifying the gap that justifies the paper. Flag missing definitive or older work, weak closest-work boundaries, and unsupported "first" or "only" claims.
- **Problem Setup / Method / Theory:** definitions should appear before use, assumptions and hypotheses should be explicit, notation should be consistent, and the method should be understandable, replicable, and verifiable. Check that important concepts are explained through text, equations, and figures when useful; for empirical work, check data/sample collection, randomization or splits, measurements, preprocessing, tools/materials, computations, and statistical techniques. Flag proof gaps, undefined symbols, overloaded variables, dimension/indexing mismatches, hidden assumptions, unjustified implications, and claims that outrun formal results.
- **Experiments / Results / Discussion:** experiments should test the central claims rather than only display numbers, and results should support or reject hypotheses rather than "prove" them. Check for a strong simple baseline, fair nearest-neighbor/SOTA comparisons, one-change-at-a-time ablations, clear metrics, tuning fairness, data leakage risks, uncertainty, robustness, runtime/memory/scaling, failure cases, negative results, alternative explanations, relation to similar studies, and text explaining what each result teaches before moving to the next.
- **Figures / Tables:** figures and tables should work as a paper within the paper. Check that each is referenced in text, has a self-contained caption with the intended takeaway, uses readable text and labeled axes/units, explains colors and markers, highlights important details when needed, and is not cherry-picked without support from broader results.
- **Discussion / Limitations / Conclusion:** takeaways should match evidence, limitations should be honest and bounded, and future work should show a plausible path without claiming too much of the field. The conclusion should answer "so what?" by tying the findings back to the research problem, prior-work gap, broader field, and practical or scientific implications. Flag conclusions that merely repeat the abstract, introduce new unsupported claims, or hide key limitations.
- **References / Acknowledgments:** check reference style against the venue when known, citation order and formatting, completeness of bibliographic entries, definitive published versions rather than stale preprints when appropriate, missing major work, and capitalization protection for names/acronyms. Check acknowledgments only for obvious required funding, data, compute, or contributor recognition issues.
- **Appendix / Supplement:** review appendix and supplement material with the same rigor when main-text claims depend on them. Check that appendices hold important but distracting details, supplements do what the main text promises, failure cases or extra evidence appear when relevant, and all such material is referenced from the main paper.
- **Cross-section consistency, style, and repetition:** check for drift between abstract, introduction, method, results, and conclusion. Flag repeated motivation, duplicated definitions/statements, repeated claims, undefined acronyms, unexplained technical terms, imprecise use of terms like approach/method/methodology/framework/measure/measurement/model, inconsistent tense/capitalization, contractions, exclamation marks, citation-start sentences, vague verbs like "provides/enables/allows" when they hide mechanism, orphan sentences, weak paragraph flow, and sections that repeat without adding information. Suggest concrete deletions, merges, or tightening edits with a justification.

Severity labels (use them only when useful):

- `High` - blocks understanding, credibility, or correctness.
- `Medium` - meaningfully weakens the paper but is locally fixable.
- `Low` - local polish or readability issue.

## Step 6 - Write feedback file

Write exactly one output file: `feedback-YYYYMMDD-HHMM.md` unless the user names a file.

- User-specified location, otherwise beside the main manuscript, otherwise current directory.
- If the target exists, ask before overwriting; in autonomous mode, use a timestamped variant.
- No scratch files or extra outputs unless requested; rewrite suggestions stay in the feedback file.

Use `assets/feedback.md` as the skeleton. Delete unused optional sections, template comments, and placeholders. Fill `Manuscript summary` early.

- Open with a short summary of what the manuscript appears to claim, its intended contribution, and what already works.
- Keep the section-by-section critique as the main body; fold recurring paper-wide patterns into `Overall assessment`, `Cross-cutting checks`, or `Priority action list` instead of duplicating a formal major/minor review template.
- For substantive issues, cite the section and give line references of the source; quote the smallest useful fragment; explain why it matters; give the smallest concrete fix.
- When the feedback is long, use stable labels for long reviews (e.g. `intro-1`, `method-2`).
- State what would most improve submission readiness and tie priorities to that.
- End with a high-leverage priority list ordered by payoff relative to effort; use roughly 5-10 items when the manuscript still needs substantial work, otherwise keep it shorter.

## Step 7 - Quality gate

Before finalizing, re-read the drafted feedback against the manuscript for a focused pass:

- Scope and mode match the request; critique follows manuscript section order.
- Strong criticisms are evidence-backed and traceable; main issues have concrete fixes.
- Numerical/reference consistency findings are evidenced rather than speculative.
- External leads and theorem ideas are labeled provisional; no invented evidence.
- Template comments, placeholders, and unused sections are removed.
- Priority actions are ordered by impact relative to effort and surface the main blocker early.
- Exactly one feedback file was created; no manuscript or source file was modified.

## Step 8 - Report to user

Final chat response:

- Feedback file path.
- Review mode.
- The single most important issue.

## Bundled Files

- `assets/feedback.md` - output skeleton.
