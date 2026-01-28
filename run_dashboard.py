#!/usr/bin/env python
"""
Quick start script for Streamlit Dashboard
Runs the notebook and launches the dashboard automatically
"""

import subprocess
import sys
from pathlib import Path
import time

def run_notebook():
    """Run the notebook using papermill"""
    print("=" * 70)
    print("STEP 1: Running notebook to generate results...")
    print("=" * 70)
    
    project_root = Path(__file__).parent.resolve()
    
    try:
        result = subprocess.run(
            [sys.executable, "run_papermill.py"],
            cwd=project_root,
            check=True
        )
        print("✓ Notebook executed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running notebook: {e}")
        return False

def launch_dashboard():
    """Launch Streamlit dashboard"""
    print("\n" + "=" * 70)
    print("STEP 2: Launching Streamlit Dashboard...")
    print("=" * 70)
    
    project_root = Path(__file__).parent.resolve()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "streamlit_dashboard.py"],
            cwd=project_root
        )
    except Exception as e:
        print(f"✗ Error launching dashboard: {e}")
        print("\nTry running manually:")
        print(f"  cd {project_root}")
        print(f"  streamlit run streamlit_dashboard.py")

def main():
    """Main entry point"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  SELF-TRAINING MODEL EVALUATION DASHBOARD - QUICK START".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Check if cache exists
    project_root = Path(__file__).parent.resolve()
    cache_file = project_root / "data/processed/st_results_cache.pkl"
    
    if cache_file.exists():
        print(f"✓ Cache found at {cache_file}")
        response = input("Cache found. Skip notebook execution? (y/n): ").strip().lower()
        
        if response == 'n':
            if not run_notebook():
                sys.exit(1)
    else:
        print(f"✗ Cache not found at {cache_file}")
        print("Need to run notebook first...")
        if not run_notebook():
            sys.exit(1)
    
    # Launch dashboard
    print("\n✓ Ready to launch dashboard!")
    time.sleep(1)
    launch_dashboard()

if __name__ == "__main__":
    main()
