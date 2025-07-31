#!/usr/bin/env python3
"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a proprietary license. Use requires written
permission from Justin K. Lietz. See LICENSE file for full terms.

FUM Mathematical Frameworks Validation Runner

Usage:
    python main.py --sie         # Run SIE stability validation
    python main.py --kgtda       # Run Knowledge Graph TDA validation
    python main.py --all         # Run both validations
"""

import os
import sys
import subprocess
import argparse
import time

def run_sie_validation():
    """Run SIE stability validation."""
    print("Running SIE Stability Validation...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sie_dir = os.path.join(script_dir, "SIE_Analysis", "Implementation")
    
    # Run simulation
    cmd = [sys.executable, "simulate_sie_stability.py", "--eta", "0.01", "--lambda_decay", "0.001", "--scaling_target", "10.0"]
    result = subprocess.run(cmd, cwd=sie_dir)
    if result.returncode != 0:
        return 1
    
    # Run analysis
    cmd = [sys.executable, "analyze_sie_stability_data.py", "--sweep"]
    result = subprocess.run(cmd, cwd=sie_dir)
    if result.returncode != 0:
        return 1
    
    # Show results location
    results_dir = os.path.join("SIE_Analysis", "results")
    if os.path.exists(results_dir):
        plot_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        data_files = [f for f in os.listdir(results_dir) if f.endswith('.npz')]
        if plot_files:
            latest_plot = max(plot_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
            print(f"SIE results: {os.path.join(results_dir, latest_plot)}")
    
    return 0

def run_kgtda_validation():
    """Run Knowledge Graph TDA validation."""
    print("Running Knowledge Graph TDA Validation...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kg_dir = os.path.join(script_dir, "Knowledge_Graph_Analysis")
    
    # Generate snapshots
    cmd = [sys.executable, "Implementation/generate_kg_snapshots_v2.py"]
    result = subprocess.run(cmd, cwd=kg_dir)
    if result.returncode != 0:
        return 1
    
    # Wait for snapshot files
    snapshot_dir = os.path.join(kg_dir, "data", "kg_snapshots")
    for _ in range(10):  # Wait up to 5 seconds
        if os.path.exists(snapshot_dir) and os.listdir(snapshot_dir):
            break
        time.sleep(0.5)
    
    # Run analysis
    cmd = [sys.executable, "Implementation/run_analysis.py"]
    result = subprocess.run(cmd, cwd=kg_dir)
    if result.returncode != 0:
        return 1
    
    # Show results location
    results_file = os.path.join("Knowledge_Graph_Analysis", "results", "tda_analysis_results.txt")
    if os.path.exists(results_file):
        print(f"TDA results: {results_file}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='FUM Mathematical Frameworks Validation Runner')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sie', action='store_true', help='Run SIE stability validation')
    group.add_argument('--kgtda', action='store_true', help='Run Knowledge Graph TDA validation')
    group.add_argument('--all', action='store_true', help='Run both validations')
    
    args = parser.parse_args()
    
    print("FUM Mathematical Frameworks Validation Runner")
    print("=" * 50)
    
    if args.sie:
        return run_sie_validation()
    elif args.kgtda:
        return run_kgtda_validation()
    elif args.all:
        print("Running complete validation suite...\n")
        
        # Run SIE first
        sie_result = run_sie_validation()
        if sie_result != 0:
            print("SIE validation failed.")
            return sie_result
        
        print("\n" + "-" * 30)
        
        # Run TDA second  
        kgtda_result = run_kgtda_validation()
        if kgtda_result != 0:
            print("TDA validation failed.")
            return kgtda_result
        
        print("\nAll validations completed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
