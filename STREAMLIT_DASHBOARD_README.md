# Self-Training Model Evaluation Dashboard

A comprehensive Streamlit dashboard for visualizing and comparing **Baseline Supervised Learning** vs **Self-Training** with different threshold (τ) values for Air Quality Index (AQI) classification.

## 📊 Dashboard Features

### Pages:

1. **📊 Overview**
   - Quick metrics comparison (Accuracy, F1-Macro)
   - Improvement percentages
   - All tau values comparison table
   - Test accuracy bar chart

2. **📈 Validation Evolution**
   - Validation accuracy curves for all tau values
   - Detailed training history table
   - Pseudo-labeled samples per iteration
   - F1-Macro score evolution

3. **🎯 Test Metrics**
   - Classification reports for both models
   - Confusion matrices (Baseline vs Self-Training)
   - Per-class performance analysis

4. **🔍 Best Model Details**
   - Detailed metrics for the best tau model
   - Per-class performance with highlighting
   - Comprehensive confusion matrix
   - Full classification report

5. **🗂️ Training Progression**
   - Interactive tau selection
   - Iteration-by-iteration analysis
   - Validation accuracy tracking
   - F1-Macro evolution
   - Pseudo-labeled samples visualization

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- streamlit>=1.28
- pandas>=2.0
- numpy>=1.24
- scikit-learn>=1.3
- matplotlib>=3.7
- seaborn>=0.12

### Running the Dashboard

#### Step 1: Generate Cache Data

First, run the Jupyter notebook to generate results:

```bash
# Execute the notebook
jupyter notebook notebooks/semi_self_training_detailed.ipynb
```

Or use papermill to run it automatically:

```bash
python run_papermill.py
```

This will:
- Train baseline supervised model
- Run self-training with τ = [0.70, 0.80, 0.90, 0.95]
- Generate visualizations
- Save cache to `data/processed/st_results_cache.pkl`

#### Step 2: Launch Dashboard

```bash
streamlit run streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501` in your default browser.

## 📈 Dashboard Metrics Explained

### Baseline Metrics
- **Accuracy**: Proportion of correct predictions on test set
- **F1-Macro**: Unweighted mean of F1-scores across all classes
- **Classification Report**: Precision, recall, f1-score per class

### Self-Training Metrics
- **Threshold (τ)**: Confidence threshold for pseudo-labeling (0.70, 0.80, 0.90, 0.95)
- **Validation Accuracy**: Accuracy on validation set at each iteration
- **Pseudo-Labeled Samples**: Number of samples added at each iteration
- **Unlabeled Pool Remaining**: How many unlabeled samples remain

### Improvement Metrics
- **Absolute Change**: Difference between self-training and baseline
- **Percentage Change**: Relative improvement ((New - Baseline) / Baseline × 100%)

## 🎨 Visualizations

1. **Test Accuracy Comparison**: Bar chart showing accuracy for each tau
2. **Validation Accuracy Evolution**: Line chart tracking accuracy across iterations
3. **Pseudo-Labeled Samples**: Bar chart of samples added per iteration
4. **F1-Macro Evolution**: Line chart of F1-score progression
5. **Confusion Matrices**: Heatmaps for both baseline and best model
6. **Classification Reports**: Detailed per-class metrics

## 📁 File Structure

```
air_guard/
├── streamlit_dashboard.py      # Main dashboard application
├── cache_results.py             # Helper for saving results
├── notebooks/
│   └── semi_self_training_detailed.ipynb   # Source notebook
├── data/processed/
│   └── st_results_cache.pkl    # Generated cache (after notebook run)
└── requirements.txt             # Dependencies
```

## 🔧 Customization

### Modify Dashboard Theme

Edit the CSS styling section in `streamlit_dashboard.py`:

```python
st.markdown("""
    <style>
    .metric-card { ... }
    .metric-title { ... }
    ...
    </style>
""", unsafe_allow_html=True)
```

### Add Custom Metrics

Add new calculation in the appropriate section:

```python
if page == "📊 Overview":
    # Add your custom metric here
    st.metric("Custom Metric", value, delta=change)
```

### Change Page Layout

Modify sidebar navigation:

```python
page = st.radio(
    "Select a page:",
    ["Your Custom Page 1", "Your Custom Page 2", ...]
)
```

## 📊 Data Cache Details

The cache file (`st_results_cache.pkl`) contains:

```python
{
    'st_results': {
        0.70: {'history': DataFrame, 'test_metrics': dict, ...},
        0.80: {...},
        0.90: {...},
        0.95: {...}
    },
    'baseline_test_acc': float,
    'baseline_test_f1': float,
    'best_tau': float,
    'baseline_report': dict,
    'best_st_metrics': dict,
    'y_test_filtered': array,
    'y_test_pred_filtered': array,
    'AQI_CLASSES': list
}
```

## 🐛 Troubleshooting

### "Cache file not found"
- Ensure you've run the notebook first
- Check that `data/processed/st_results_cache.pkl` exists
- Re-run the notebook: `jupyter notebook notebooks/semi_self_training_detailed.ipynb`

### Dashboard loads but shows no data
- Make sure the notebook executed successfully (no errors)
- Check that all cells, especially the cache saving cell, were executed
- Verify cache file was created: `ls data/processed/st_results_cache.pkl`

### Streamlit not installed
```bash
pip install streamlit>=1.28
```

### Port already in use
```bash
streamlit run streamlit_dashboard.py --server.port 8502
```

## 📝 Notes

- The dashboard uses Streamlit's caching to improve performance
- All visualizations are interactive and can be expanded/zoomed
- The cache is loaded once and reused within the session
- To update with new results, re-run the notebook and refresh the browser

## 📞 Support

For issues or improvements, refer to:
- Streamlit Documentation: https://docs.streamlit.io
- Project README: [../README.md](../README.md)

---

**Last Updated**: 2026-01-28
**Version**: 1.0
