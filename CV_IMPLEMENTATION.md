# Cross-Validation Implementation for Rashomon Sets

## Overview

The Rashomon set construction has been updated to support **cross-validation (CV) or repeated holdout splits** instead of a single train/val split. This makes the Rashomon membership less dependent on a single lucky split and more comparable to multiplicity papers that emphasize stability.

## Key Changes

### 1. New Function: `make_cv_splits()` in `src/data.py`

Creates CV splits for train/val data (test set is held out separately).

**Parameters:**
- `cv_method`: `"kfold"` or `"repeated_holdout"`
- `n_folds`: Number of folds for K-fold CV (default: 5)
- `n_repeats`: Number of repeats for repeated holdout (default: 5)
- `test_size`: Fraction of data to hold out as test (default: 0.2)

**Returns:**
- `test_split`: Dict with test indices
- `cv_splits`: List of dicts, each with train/val indices for CV

### 2. New Function: `fit_and_eval_model_cv()` in `src/rashomon.py`

Computes mean CV loss across all folds and fits a final model on all train+val data.

**Returns:**
- `mean_cv_loss`: Mean log loss across CV folds (used for Rashomon membership)
- `pipeline`: Final pipeline fit on all train+val data (for test predictions)

### 3. Updated: `build_rashomon_set()` in `src/rashomon.py`

Now accepts:
- `split`: Single train/val/test split (original behavior)
- `test_split` + `cv_splits`: CV mode (new)

When CV is used:
- Rashomon membership is based on **mean CV loss**: `L(m) = mean(log_loss across folds)`
- Threshold: `L(m) ≤ L* + ε` where L* is the best mean CV loss

### 4. Updated CLI: `run_rashomon_experiment.py`

New command-line arguments:
- `--use_cv`: Enable CV mode
- `--cv_method`: `"kfold"` or `"repeated_holdout"` (default: `"kfold"`)
- `--n_folds`: Number of folds for K-fold (default: 5)
- `--n_repeats`: Number of repeats for repeated holdout (default: 5)

## Usage Examples

### K-fold Cross-Validation (5 folds)

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --use_cv --cv_method kfold --n_folds 5
```

### Repeated Holdout (5 repeats)

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --use_cv --cv_method repeated_holdout --n_repeats 5
```

### Per-family with CV

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42 --family RF --use_cv --n_folds 5
```

### Original Single Split (backward compatible)

```powershell
python run_rashomon_experiment.py --dataset compas --epsilon 0.01 --seed 42
```

## How It Works

1. **Test set is held out first** (20% by default)
2. **CV splits are created on remaining 80%** (train+val portion)
3. **For each model:**
   - Fit on each CV fold's training set
   - Evaluate on each CV fold's validation set
   - Compute mean CV loss: `mean(log_loss across all folds)`
   - Fit final model on all train+val data (for test predictions)
4. **Rashomon membership:**
   - Models with `mean_cv_loss ≤ best_mean_cv_loss + ε` are included
5. **Final predictions:**
   - All Rashomon models predict on the held-out test set

## Benefits

- **Less split-dependent**: Rashomon membership is averaged across multiple splits
- **More stable**: Reduces variance in Rashomon set composition
- **Comparable to literature**: Matches practices in multiplicity papers
- **Backward compatible**: Original single-split mode still works

## Output Changes

When using CV, the `split.npz` file contains:
- `test`: Test indices
- `seed`: Random seed
- `cv_method`: "kfold" or "repeated_holdout"
- `n_folds` or `n_repeats`: CV parameters

Additionally, `cv_splits.npz` contains all CV fold splits for reference.

## Migration Notes

- Existing results using single splits are still valid
- New experiments should use `--use_cv` for more robust results
- The `config.json` now includes `use_cv` flag
- Notebooks may need updates to handle CV split structure (if they load splits)
