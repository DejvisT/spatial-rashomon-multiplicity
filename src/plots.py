"""
Visualization functions for Rashomon multiplicity analysis.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import os


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def save_figure(fig, filename: str, save_dir: str = None, dpi: int = 150):
    """Save figure if save_dir is provided."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches='tight')


# Rashomon set plots

def plot_rashomon_composition(rashomon_result: Dict, save_dir: str = None) -> None:
    """
    Plot composition of Rashomon set by model type.
    """
    set_style()
    
    # Count model types
    types = {}
    for p in rashomon_result['params']:
        t = p['model_type']
        types[t] = types.get(t, 0) + 1
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    
    bars = ax.bar(types.keys(), types.values(), color=colors, edgecolor='black')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Count')
    ax.set_title(f'Rashomon Set Composition (n={len(rashomon_result["models"])})')
    
    # Add count labels on bars
    for bar, count in zip(bars, types.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, 'rashomon_composition.png', save_dir)
    plt.show()


def plot_score_distribution(rashomon_result: Dict, save_dir: str = None) -> None:
    """
    Plot distribution of model scores in Rashomon set.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    scores = rashomon_result['scores']
    ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(rashomon_result['best_score'], color='green', linestyle='--',
               linewidth=2, label=f"Best: {rashomon_result['best_score']:.4f}")
    ax.axvline(rashomon_result['threshold'], color='red', linestyle='--',
               linewidth=2, label=f"Threshold: {rashomon_result['threshold']:.4f}")
    
    ax.set_xlabel('Model Score (AUC)')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution in Rashomon Set')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'score_distribution.png', save_dir)
    plt.show()


# Observation-wise variance plots

def plot_variance_distribution(metrics: Dict, save_dir: str = None) -> None:
    """
    Plot distribution of observation-wise prediction variance.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variance = metrics['variance']
    ax.hist(variance, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(variance.mean(), color='red', linestyle='--', linewidth=2,
               label=f"Mean = {variance.mean():.4f}")
    ax.axvline(np.median(variance), color='orange', linestyle='--', linewidth=2,
               label=f"Median = {np.median(variance):.4f}")
    
    ax.set_xlabel('Prediction Variance', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Observation-wise Variance', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'variance_distribution.png', save_dir)
    plt.show()


def plot_variance_vs_prediction(metrics: Dict, save_dir: str = None) -> None:
    """
    Plot variance vs mean prediction (shows where uncertainty concentrates).
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(
        metrics['mean_pred'], 
        metrics['variance'],
        c=metrics['flip_probability'],
        cmap='Reds',
        alpha=0.5,
        s=20
    )
    
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5, label='Decision boundary')
    ax.set_xlabel('Mean Prediction')
    ax.set_ylabel('Prediction Variance')
    ax.set_title('Variance vs Mean Prediction')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Flip Probability')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'variance_vs_prediction.png', save_dir)
    plt.show()


def plot_variance_metrics(metrics: Dict, save_dir: str = None) -> None:
    """
    Plot 4-panel variance metrics visualization.
    """
    set_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Panel 1: Rashomon Prediction Intervals
    ax = axes[0, 0]
    n_display = min(100, len(metrics['mean_pred']))
    idx = np.argsort(metrics['mean_pred'])[::max(1, len(metrics['mean_pred'])//n_display)][:n_display]
    x_pos = np.arange(len(idx))
    
    ax.fill_between(x_pos, metrics['rpi_lower_95'][idx], metrics['rpi_upper_95'][idx],
                    alpha=0.4, color='steelblue', label='95% RPI')
    ax.fill_between(x_pos, metrics['rpi_lower_50'][idx], metrics['rpi_upper_50'][idx],
                    alpha=0.6, color='steelblue', label='50% RPI')
    ax.plot(x_pos, metrics['mean_pred'][idx], 'k-', linewidth=1, label='Mean')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Samples (sorted by mean prediction)')
    ax.set_ylabel('Predicted Probability')
    ax.set_title('Rashomon Prediction Intervals')
    ax.legend(loc='upper left')
    
    # Panel 2: Normalized variance by prediction level
    ax = axes[0, 1]
    scatter = ax.scatter(
        metrics['mean_pred'], 
        metrics['normalized_variance'],
        c=metrics['flip_probability'],
        cmap='Reds',
        alpha=0.5,
        s=20
    )
    ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Prediction')
    ax.set_ylabel('Normalized Variance')
    ax.set_title('Normalized Variance by Prediction Level')
    plt.colorbar(scatter, ax=ax, label='Flip Prob')
    
    # Panel 3: RPI width distribution
    ax = axes[1, 0]
    ax.hist(metrics['rpi_width_95'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(metrics['rpi_width_95'].mean(), color='red', linestyle='--',
               label=f"Mean = {metrics['rpi_width_95'].mean():.3f}")
    ax.set_xlabel('95% RPI Width')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Prediction Interval Widths')
    ax.legend()
    
    # Panel 4: Flip probability vs prediction
    ax = axes[1, 1]
    colors = ['#E74C3C' if c else '#3498DB' for c in metrics['crosses_boundary']]
    ax.scatter(metrics['mean_pred'], metrics['flip_probability'],
               c=colors, alpha=0.5, s=20)
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Prediction')
    ax.set_ylabel('Flip Probability')
    ax.set_title(f"Classification Instability (red = crosses 0.5)")
    
    plt.tight_layout()
    save_figure(fig, 'enhanced_metrics.png', save_dir)
    plt.show()


def plot_controversial_cases(metrics: Dict, y_test: np.ndarray = None, 
                             save_dir: str = None) -> None:
    """
    Highlight controversial cases (where 95% RPI crosses decision boundary).
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    controversial = metrics['crosses_boundary']
    certain = ~controversial
    
    ax.scatter(metrics['mean_pred'][certain], metrics['rpi_width_95'][certain],
               alpha=0.3, s=20, c='steelblue', label='Certain')
    ax.scatter(metrics['mean_pred'][controversial], metrics['rpi_width_95'][controversial],
               alpha=0.7, s=40, c='red', marker='x', label='Controversial')
    
    ax.axvline(0.5, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Prediction')
    ax.set_ylabel('95% RPI Width')
    ax.set_title(f'Controversial Cases: {controversial.sum()} ({100*controversial.mean():.1f}%)')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, 'controversial_cases.png', save_dir)
    plt.show()


# PDP plots

def plot_pdp_overlay(pdp_result: Dict, feature_name: str, 
                     n_models: int = 30, save_dir: str = None) -> None:
    """
    Plot PDP overlay for a single feature.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle categorical grids (strings) vs numeric grids
    grid = pdp_result['grid']
    is_categorical = isinstance(grid, (list, tuple, np.ndarray)) and len(grid) > 0 and isinstance(grid[0], str)
    x = np.arange(len(grid)) if is_categorical else grid

    # Plot individual PDPs
    n_plot = min(n_models, pdp_result['pdp_matrix'].shape[0])
    if is_categorical:
        for i in range(n_plot):
            ax.plot(
                x,
                pdp_result['pdp_matrix'][i],
                linestyle='-',
                marker='o',
                markersize=5,
                linewidth=0.8,
                alpha=0.25,
                color='steelblue',
            )
    else:
        for i in range(n_plot):
            ax.plot(
                x,
                pdp_result['pdp_matrix'][i],
                alpha=0.3,
                color='steelblue',
                linewidth=0.8,
            )
    
    # Plot mean PDP
    if is_categorical:
        ax.plot(x, pdp_result['mean_pdp'], '-o', color='red', linewidth=2.0, markersize=7, label='Mean')
        ax.set_xticks(x)
        ax.set_xticklabels(list(grid), rotation=30, ha='right')
    else:
        ax.plot(x, pdp_result['mean_pdp'], 'r-', linewidth=2.5, label='Mean')
    
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Predicted Probability')
    ax.set_title(f'PDP Overlay: {feature_name} (Mean Variance = {pdp_result["mean_variance_pdp"]:.4f})')
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, f'pdp_{feature_name}.png', save_dir)
    plt.show()


def plot_pdp_top_features(pdp_results: Dict, top_n: int = 4, 
                          save_dir: str = None) -> None:
    """
    Plot PDP overlays for top N most unstable features.
    """
    set_style()
    
    # Get ranking
    ranking = sorted(pdp_results.items(), key=lambda x: x[1]['mean_variance_pdp'], reverse=True)
    top_features = ranking[:top_n]
    
    n_cols = 2
    n_rows = (top_n + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
    axes = axes.flatten()
    
    for i, (fname, res) in enumerate(top_features):
        ax = axes[i]

        grid = res['grid']
        is_categorical = isinstance(grid, (list, tuple, np.ndarray)) and len(grid) > 0 and isinstance(grid[0], str)
        x = np.arange(len(grid)) if is_categorical else grid
        
        # Plot individual PDPs
        n_plot = min(30, res['pdp_matrix'].shape[0])
        if is_categorical:
            for j in range(n_plot):
                ax.plot(
                    x,
                    res['pdp_matrix'][j],
                    linestyle='-',
                    marker='o',
                    markersize=5,
                    linewidth=0.8,
                    alpha=0.25,
                    color='steelblue',
                )
            ax.plot(x, res['mean_pdp'], '-o', color='red', linewidth=2.0, markersize=7, label='Mean')
            ax.set_xticks(x)
            ax.set_xticklabels(list(grid), rotation=30, ha='right')
        else:
            for j in range(n_plot):
                ax.plot(x, res['pdp_matrix'][j], alpha=0.3, color='steelblue')
            ax.plot(x, res['mean_pdp'], 'r-', linewidth=2, label='Mean')
        ax.set_xlabel(fname)
        ax.set_ylabel('Predicted Probability')
        ax.set_title(f'{fname} (Mean Variance = {res["mean_variance_pdp"]:.4f})')
        ax.legend(loc='best')
    
    # Hide empty subplots
    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('PDP Overlay: Top Unstable Features', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, 'pdp_top_features.png', save_dir)
    plt.show()


def plot_pdp_ranking(pdp_ranking: pd.Series, save_dir: str = None) -> None:
    """
    Plot horizontal bar chart of RPDP-II values.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(pdp_ranking)*0.4)))
    
    colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, len(pdp_ranking)))
    ax.barh(range(len(pdp_ranking)), pdp_ranking.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(pdp_ranking)))
    ax.set_yticklabels(pdp_ranking.index)
    ax.set_xlabel('RPDP-II (Mean PDP Variance)')
    ax.set_title('Feature Instability Ranking (Global Effects)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_figure(fig, 'pdp_ranking.png', save_dir)
    plt.show()


# FME plots

def plot_fme_ranking(fme_ranking: pd.Series, save_dir: str = None) -> None:
    """
    Plot horizontal bar chart of FME variance values.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(fme_ranking)*0.4)))
    
    colors = plt.cm.coolwarm(np.linspace(0.8, 0.2, len(fme_ranking)))
    ax.barh(range(len(fme_ranking)), fme_ranking.values, color=colors, edgecolor='black')
    ax.set_yticks(range(len(fme_ranking)))
    ax.set_yticklabels(fme_ranking.index)
    ax.set_xlabel('Mean FME Variance')
    ax.set_title('Feature Instability Ranking (Local Effects)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    save_figure(fig, 'fme_ranking.png', save_dir)
    plt.show()


def plot_fme_distribution(fme_result: Dict, feature_name: str,
                          save_dir: str = None) -> None:
    """
    Plot distribution of FME values for a feature.
    """
    set_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Mean FME distribution
    ax = axes[0]
    mean_fme = fme_result['mean_fme']
    if isinstance(mean_fme, np.ndarray) and mean_fme.ndim > 1:
        mean_fme = mean_fme.ravel()
    ax.hist(mean_fme, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Mean FME')
    ax.set_ylabel('Count')
    ax.set_title(f'{feature_name}: Mean FME Distribution')
    
    # Right: FME variance distribution
    ax = axes[1]
    var = fme_result['variance']
    if isinstance(var, np.ndarray) and var.ndim > 1:
        var = var.ravel()
    ax.hist(var, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('FME Variance')
    ax.set_ylabel('Count')
    ax.set_title(f'{feature_name}: FME Variance Distribution')
    
    plt.suptitle(f'FME Analysis: {feature_name}', fontsize=14, y=1.02)
    plt.tight_layout()
    save_figure(fig, f'fme_dist_{feature_name}.png', save_dir)
    plt.show()


# Summary plots


def plot_variance_by_feature(
    metrics_df: pd.DataFrame,
    feature_name: str,
    save_dir: str = None
) -> None:
    """
    Plot variance distribution by a categorical or binned feature.
    """
    set_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # If numeric with many values, bin it
    if metrics_df[feature_name].nunique() > 10:
        metrics_df = metrics_df.copy()
        metrics_df[f'{feature_name}_binned'] = pd.qcut(
            metrics_df[feature_name], q=5, duplicates='drop'
        )
        group_col = f'{feature_name}_binned'
    else:
        group_col = feature_name
    
    metrics_df.boxplot(column='variance', by=group_col, ax=ax)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Prediction Variance')
    ax.set_title(f'Variance Distribution by {feature_name}')
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    save_figure(fig, f'variance_by_{feature_name}.png', save_dir)
    plt.show()


def plot_heatmap_variance_features(
    metrics_df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    save_dir: str = None
) -> None:
    """
    Plot 2D heatmap of variance across two features.
    """
    set_style()
    
    df = metrics_df.copy()
    
    # Bin continuous features
    for feat in [feature_x, feature_y]:
        if df[feat].nunique() > 10:
            df[f'{feat}_bin'] = pd.qcut(df[feat], q=5, duplicates='drop')
        else:
            df[f'{feat}_bin'] = df[feat]
    
    # Create pivot table
    pivot = df.pivot_table(
        values='variance',
        index=f'{feature_y}_bin',
        columns=f'{feature_x}_bin',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'Mean Variance: {feature_x} vs {feature_y}')
    
    plt.tight_layout()
    save_figure(fig, f'heatmap_{feature_x}_{feature_y}.png', save_dir)
    plt.show()

