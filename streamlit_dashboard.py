"""
Streamlit Dashboard for Self-Training Model Evaluation
Visualizes comparison between baseline supervised learning and self-training with different τ values
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix
import sys

# Setup paths
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure page
st.set_page_config(
    page_title="Self-Training Model Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #666666;
        font-weight: 500;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    .improvement-positive {
        color: #31a354;
        font-weight: bold;
    }
    .improvement-negative {
        color: #d62728;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load or create results data
@st.cache_resource
def load_notebook_data():
    """Load data from the notebook execution"""
    # Try to load from saved pickle or recreate from notebook
    results_cache = PROJECT_ROOT / "data/processed/st_results_cache.pkl"
    
    if results_cache.exists():
        import pickle
        with open(results_cache, 'rb') as f:
            return pickle.load(f)
    
    # Fallback: load from CSV files if they exist
    return None

@st.cache_data
def load_results_from_files():
    """Load results from saved CSV and JSON files"""
    results = {
        "tau_values": [0.70, 0.80, 0.90, 0.95],
        "histories": {},
        "test_metrics": {},
        "baseline_metrics": {}
    }
    
    # Try to load history files
    data_dir = PROJECT_ROOT / "data/processed"
    
    return results

def create_tau_comparison_chart(st_results, baseline_acc, tau_test_accuracies):
    """Create bar chart comparing test accuracy across tau values"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    taus = sorted(tau_test_accuracies.keys())
    test_accs = [tau_test_accuracies[t] for t in taus]
    colors = ['#31a354' if acc > baseline_acc else '#ff7f0e' if acc >= baseline_acc - 0.01 else '#d62728' 
              for acc in test_accs]
    
    bars = ax.bar([f"τ={t:.2f}" for t in taus], test_accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2.5, label=f"Baseline = {baseline_acc:.4f}")
    
    ax.set_ylabel("Test Accuracy", fontsize=12, fontweight='bold')
    ax.set_xlabel("Threshold (τ)", fontsize=12, fontweight='bold')
    ax.set_title("Test Accuracy Comparison: Self-Training vs Baseline", fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_validation_evolution_chart(st_results, baseline_acc):
    """Create line chart of validation accuracy evolution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for tau in sorted(st_results.keys()):
        history = st_results[tau]["history"]
        ax.plot(history["iter"], history["val_accuracy"], "o-", label=f"τ = {tau}", linewidth=2, markersize=8)
    
    ax.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2.5, label=f"Baseline = {baseline_acc:.4f}")
    ax.set_xlabel("Iteration", fontsize=12, fontweight='bold')
    ax.set_ylabel("Validation Accuracy", fontsize=12, fontweight='bold')
    ax.set_title("Validation Accuracy Evolution (Different τ values)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_pseudo_labeled_chart(history_best):
    """Create bar chart of pseudo-labeled samples per iteration"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(history_best["iter"], history_best["new_pseudo"], alpha=0.7, color='steelblue', 
                   edgecolor='black', linewidth=1.5)
    ax.set_xlabel("Iteration", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Samples Added", fontsize=12, fontweight='bold')
    ax.set_title("Pseudo-Labeled Samples per Iteration", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, history_best["new_pseudo"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 100, 
                f'{int(count):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_chart(y_true, y_pred, title, class_names=None):
    """Create confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_f1_evolution_chart(history_best, baseline_f1, best_tau):
    """Create line chart of F1-Macro evolution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(history_best["iter"], history_best["val_f1_macro"], "s-", 
            label=f"Best τ={best_tau}", linewidth=2.5, markersize=10, color='green')
    ax.axhline(y=baseline_f1, color='r', linestyle='--', linewidth=2.5, label=f"Baseline F1={baseline_f1:.4f}")
    
    ax.set_xlabel("Iteration", fontsize=12, fontweight='bold')
    ax.set_ylabel("F1-Macro Score", fontsize=12, fontweight='bold')
    ax.set_title(f"F1-Macro Evolution (Best τ = {best_tau})", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def format_classification_report(report_dict):
    """Format classification report dictionary into DataFrame"""
    df_report = pd.DataFrame(report_dict).transpose()
    return df_report[['precision', 'recall', 'f1-score', 'support']]

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("📊 Self-Training Model Evaluation Dashboard")
st.markdown("**Comparison between Baseline Supervised Learning and Self-Training with Different τ Values**")

# Try to load data from notebook
try:
    import pickle
    cache_file = PROJECT_ROOT / "data/processed/st_results_cache.pkl"
    
    # Check if cache exists, otherwise guide user
    if not cache_file.exists():
        st.warning("⚠️ Please run the notebook first to generate the results cache.")
        st.info("Run the Jupyter notebook `semi_self_training_detailed.ipynb` to generate the required data.")
        st.stop()
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    st_results = cached_data['st_results']
    baseline_test_acc = cached_data['baseline_test_acc']
    baseline_test_f1 = cached_data['baseline_test_f1']
    best_tau = cached_data['best_tau']
    baseline_report = cached_data['baseline_report']
    best_st_metrics = cached_data['best_st_metrics']
    y_test_filtered = cached_data['y_test_filtered']
    y_test_pred_filtered = cached_data['y_test_pred_filtered']
    best_st_y_test_filtered = cached_data.get('best_st_y_test_filtered', y_test_filtered)
    best_st_y_test_pred_filtered = cached_data.get('best_st_y_test_pred_filtered', y_test_pred_filtered)
    AQI_CLASSES = cached_data.get('AQI_CLASSES', None)
    
except Exception as e:
    st.error(f"❌ Error loading cache: {e}")
    st.info("Please ensure you've run the notebook `semi_self_training_detailed.ipynb` first.")
    st.stop()

# Calculate improvement metrics (used across multiple pages)
pct_acc_improvement = 100 * (best_st_metrics['accuracy'] - baseline_test_acc) / baseline_test_acc
pct_f1_improvement = 100 * (best_st_metrics['f1_macro'] - baseline_test_f1) / baseline_test_f1
tau_test_accuracies = {tau: st_results[tau]['test_metrics']['accuracy'] for tau in st_results.keys()}

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio(
        "Select a page:",
        ["📊 Overview", "📈 Validation Evolution", "🎯 Test Metrics", 
         "🔍 Best Model Details", "🗂️ Training Progression"]
    )
    
    st.divider()
    st.subheader("📌 Model Configuration")
    st.write(f"**Best τ:** {best_tau}")
    st.write(f"**Tau Values Tested:** 0.70, 0.80, 0.90, 0.95")
    st.write(f"**Max Iterations:** 10")

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

if page == "📊 Overview":
    st.header("Overview: Baseline vs Best Self-Training")
    
    # Metrics comparison
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Baseline Accuracy",
            f"{baseline_test_acc:.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Self-Training Accuracy",
            f"{best_st_metrics['accuracy']:.4f}",
            delta=f"{best_st_metrics['accuracy'] - baseline_test_acc:+.4f}"
        )
    
    with col3:
        st.metric(
            "Baseline F1-Macro",
            f"{baseline_test_f1:.4f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Self-Training F1-Macro",
            f"{best_st_metrics['f1_macro']:.4f}",
            delta=f"{best_st_metrics['f1_macro'] - baseline_test_f1:+.4f}"
        )
    
    st.divider()
    
    # Improvement metrics already calculated at top of script
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Improvement Metrics")
        improvement_data = {
            "Metric": ["Accuracy", "F1-Macro"],
            "Absolute Change": [
                f"{best_st_metrics['accuracy'] - baseline_test_acc:+.4f}",
                f"{best_st_metrics['f1_macro'] - baseline_test_f1:+.4f}"
            ],
            "Percentage Change": [
                f"{pct_acc_improvement:+.2f}%",
                f"{pct_f1_improvement:+.2f}%"
            ]
        }
        st.dataframe(improvement_data, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Tau Comparison")
        tau_values = sorted(st_results.keys())
        tau_comparison = {
            "τ": [f"{t:.2f}" for t in tau_values],
            "Test Accuracy": [f"{st_results[t]['test_metrics']['accuracy']:.4f}" for t in tau_values],
            "F1-Macro": [f"{st_results[t]['test_metrics']['f1_macro']:.4f}" for t in tau_values]
        }
        st.dataframe(tau_comparison, use_container_width=True)
    
    st.divider()
    
    # Test accuracy comparison chart (tau_test_accuracies calculated at top)
    fig = create_tau_comparison_chart(st_results, baseline_test_acc, tau_test_accuracies)
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE: VALIDATION EVOLUTION
# ============================================================================

elif page == "📈 Validation Evolution":
    st.header("Validation Accuracy Evolution")
    
    st.subheader("Comparison Across All Tau Values")
    st.write("How validation accuracy changes during self-training iterations for different thresholds.")
    
    fig = create_validation_evolution_chart(st_results, baseline_test_acc)
    st.pyplot(fig, use_container_width=True)
    
    st.divider()
    
    # Detailed iteration history for best tau
    st.subheader(f"Detailed Training History (Best τ = {best_tau})")
    
    history_best = st_results[best_tau]["history"]
    
    # Create display dataframe
    display_data = history_best[["iter", "val_accuracy", "val_f1_macro", "new_pseudo", "unlabeled_pool"]].copy()
    display_data.columns = ["Iteration", "Val Accuracy", "Val F1-Macro", "Pseudo-Labeled Added", "Unlabeled Remaining"]
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    # Pseudo-labeled samples chart
    st.subheader("Pseudo-Labeled Samples per Iteration")
    fig = create_pseudo_labeled_chart(history_best)
    st.pyplot(fig, use_container_width=True)
    
    # F1-Macro evolution
    st.subheader("F1-Macro Score Evolution")
    fig = create_f1_evolution_chart(history_best, baseline_test_f1, best_tau)
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE: TEST METRICS
# ============================================================================

elif page == "🎯 Test Metrics":
    st.header("Test Set Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Model - Classification Report")
        baseline_report_df = format_classification_report(baseline_report)
        st.dataframe(baseline_report_df.style.format("{:.4f}"), use_container_width=True)
        
        if not y_test_filtered.empty:
            st.subheader("Baseline - Confusion Matrix")
            fig = create_confusion_matrix_chart(y_test_filtered, y_test_pred_filtered, 
                                               "Baseline Confusion Matrix", AQI_CLASSES)
            st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"Self-Training (τ={best_tau}) - Classification Report")
        best_st_report_df = format_classification_report(best_st_metrics["report"])
        st.dataframe(best_st_report_df.style.format("{:.4f}"), use_container_width=True)
        
        if not best_st_y_test_filtered.empty:
            st.subheader(f"Self-Training (τ={best_tau}) - Confusion Matrix")
            fig = create_confusion_matrix_chart(best_st_y_test_filtered, best_st_y_test_pred_filtered,
                                               f"Self-Training (τ={best_tau}) Confusion Matrix", AQI_CLASSES)
            st.pyplot(fig, use_container_width=True)

# ============================================================================
# PAGE: BEST MODEL DETAILS
# ============================================================================

elif page == "🔍 Best Model Details":
    st.header(f"Best Model Details (τ = {best_tau})")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Threshold (τ)", f"{best_tau:.2f}")
    
    with col2:
        st.metric("Test Accuracy", f"{best_st_metrics['accuracy']:.4f}")
    
    with col3:
        st.metric("F1-Macro", f"{best_st_metrics['f1_macro']:.4f}")
    
    with col4:
        st.metric("Improvement vs Baseline", f"{pct_acc_improvement:+.2f}%")
    
    st.divider()
    
    # Per-class performance
    st.subheader("Per-Class Performance Metrics")
    
    report_df = format_classification_report(best_st_metrics["report"])
    
    # Highlight classes with best/worst performance
    st.dataframe(
        report_df.style.format("{:.4f}").background_gradient(subset=['f1-score'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.divider()
    
    # Confusion matrix
    st.subheader("Confusion Matrix Analysis")
    
    if not best_st_y_test_filtered.empty:
        fig = create_confusion_matrix_chart(best_st_y_test_filtered, best_st_y_test_pred_filtered,
                                           f"Self-Training (τ={best_tau}) Confusion Matrix", AQI_CLASSES)
        st.pyplot(fig, use_container_width=True)
    
    # Full classification report
    st.subheader("Detailed Classification Report")
    with st.expander("View full report"):
        st.text(best_st_metrics.get("report_text", "No report available"))

# ============================================================================
# PAGE: TRAINING PROGRESSION
# ============================================================================

elif page == "🗂️ Training Progression":
    st.header("Training Progression Analysis")
    
    # Select which tau to analyze
    selected_tau = st.selectbox("Select τ value to analyze:", sorted(st_results.keys()), 
                                index=list(st_results.keys()).index(best_tau))
    
    st.subheader(f"Progression Details for τ = {selected_tau}")
    
    history = st_results[selected_tau]["history"]
    
    # Display table
    display_data = history[["iter", "val_accuracy", "val_f1_macro", "new_pseudo", "unlabeled_pool"]].copy()
    display_data.columns = ["Iteration", "Val Accuracy", "Val F1-Macro", "Pseudo-Labeled", "Unlabeled Remaining"]
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Validation accuracy
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["iter"], history["val_accuracy"], "o-", linewidth=2.5, markersize=10, color='blue')
        ax.axhline(y=baseline_test_acc, color='r', linestyle='--', linewidth=2, label=f"Baseline = {baseline_test_acc:.4f}")
        ax.set_xlabel("Iteration", fontsize=11, fontweight='bold')
        ax.set_ylabel("Validation Accuracy", fontsize=11, fontweight='bold')
        ax.set_title(f"Validation Accuracy (τ = {selected_tau})", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        # F1-Macro evolution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history["iter"], history["val_f1_macro"], "s-", linewidth=2.5, markersize=10, color='green')
        ax.axhline(y=baseline_test_f1, color='r', linestyle='--', linewidth=2, label=f"Baseline F1 = {baseline_test_f1:.4f}")
        ax.set_xlabel("Iteration", fontsize=11, fontweight='bold')
        ax.set_ylabel("F1-Macro Score", fontsize=11, fontweight='bold')
        ax.set_title(f"F1-Macro Evolution (τ = {selected_tau})", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    st.divider()
    
    # Pseudo-labeled samples
    st.subheader("Pseudo-Labeled Samples Added per Iteration")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(history["iter"], history["new_pseudo"], alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel("Iteration", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Samples", fontsize=11, fontweight='bold')
    ax.set_title(f"Pseudo-Labeled Samples (τ = {selected_tau})", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for iter_val, count in zip(history["iter"], history["new_pseudo"]):
        ax.text(iter_val, count + 1000, f'{int(count):,}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
    <div style="text-align: center; color: #666666; font-size: 12px; margin-top: 30px;">
    <p>Self-Training Model Evaluation Dashboard | Air Quality Index Classification</p>
    <p>Data Science Project | 2026</p>
    </div>
""", unsafe_allow_html=True)
