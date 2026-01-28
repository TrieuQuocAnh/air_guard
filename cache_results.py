"""
Script to save notebook results to pickle cache for Streamlit dashboard
Run this after executing the notebook to generate the cache file
"""

import pickle
import sys
from pathlib import Path
import pandas as pd

def save_notebook_results_to_cache(
    st_results,
    baseline_test_acc,
    baseline_test_f1,
    best_tau,
    baseline_report,
    best_st_metrics,
    y_test_filtered,
    y_test_pred_filtered,
    best_st_y_test_filtered=None,
    best_st_y_test_pred_filtered=None,
    AQI_CLASSES=None
):
    """
    Save notebook results to pickle cache for Streamlit dashboard
    
    Parameters:
    -----------
    st_results : dict
        Results from self-training (histories, metrics, predictions)
    baseline_test_acc : float
        Baseline test accuracy
    baseline_test_f1 : float
        Baseline F1-Macro score
    best_tau : float
        Best threshold value
    baseline_report : dict
        Baseline classification report
    best_st_metrics : dict
        Best self-training metrics
    y_test_filtered : array-like
        Filtered test labels for baseline
    y_test_pred_filtered : array-like
        Filtered test predictions for baseline
    best_st_y_test_filtered : array-like, optional
        Filtered test labels for best self-training model
    best_st_y_test_pred_filtered : array-like, optional
        Filtered test predictions for best self-training model
    AQI_CLASSES : list, optional
        AQI class names
    """
    
    cache_data = {
        'st_results': st_results,
        'baseline_test_acc': baseline_test_acc,
        'baseline_test_f1': baseline_test_f1,
        'best_tau': best_tau,
        'baseline_report': baseline_report,
        'best_st_metrics': best_st_metrics,
        'y_test_filtered': y_test_filtered,
        'y_test_pred_filtered': y_test_pred_filtered,
        'best_st_y_test_filtered': best_st_y_test_filtered if best_st_y_test_filtered is not None else y_test_filtered,
        'best_st_y_test_pred_filtered': best_st_y_test_pred_filtered if best_st_y_test_pred_filtered is not None else y_test_pred_filtered,
        'AQI_CLASSES': AQI_CLASSES
    }
    
    project_root = Path(__file__).parent.resolve()
    cache_dir = project_root / "data/processed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / "st_results_cache.pkl"
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✓ Cache saved to: {cache_file}")
    return cache_file

if __name__ == "__main__":
    print("Use this function in the notebook to save results:")
    print("save_notebook_results_to_cache(st_results, baseline_test_acc, ...)")
