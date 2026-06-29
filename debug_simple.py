"""
Simple debug script - analyze data without importing heavy libraries.
"""

import os
import csv
from pathlib import Path

def analyze_data_simple():
    """Simple analysis of CSV files without heavy imports."""
    print("\n" + "="*80)
    print("DATA FILE ANALYSIS (Simple)")
    print("="*80)
    
    train_dir = "data_new_format/train"
    print(f"\nTrain data directory: {train_dir}\n")
    
    total_rows = 0
    total_files = 0
    
    for filename in sorted(os.listdir(train_dir)):
        if filename.endswith(".csv"):
            filepath = os.path.join(train_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Total lines - 9 metadata lines = data lines
            data_lines = len(lines) - 9
            total_rows += data_lines
            total_files += 1
            
            print(f"  {filename:30s}: {len(lines):6d} lines ({data_lines:5d} data rows)")
    
    print(f"\n[Summary]")
    print(f"  Total files: {total_files}")
    print(f"  Total data rows: {total_rows}")
    
    # Estimate preprocessing
    window_size = 64
    step_size = 32
    
    # Calculate number of windows per file
    avg_windows_per_file = ((total_rows // total_files - window_size) // step_size) + 1 if total_files > 0 else 0
    total_windows = avg_windows_per_file * total_files
    
    print(f"\n[Preprocessing Info]")
    print(f"  Window size: {window_size}")
    print(f"  Step size: {step_size}")
    print(f"  Avg windows per file: ~{avg_windows_per_file}")
    print(f"  Estimated total windows: ~{total_windows}")
    
    # Features info
    n_features = 6  # As per project config
    flattened_features = window_size * n_features
    
    print(f"\n[Feature Info]")
    print(f"  Features per sample: {n_features}")
    print(f"  Flattened features (window × features): {window_size} × {n_features} = {flattened_features}")
    
    # One-Class SVM complexity
    print(f"\n[One-Class SVM Complexity]")
    print(f"  Input shape: ({total_windows}, {flattened_features})")
    print(f"  Elements: {total_windows * flattened_features:,}")
    print(f"  Estimated complexity: O(n^2.5) ≈ O({total_windows}^2.5)")
    print(f"  WARNING: This can take 30+ seconds for RBF kernel!")
    
    # Check for data validation issues
    print(f"\n[Data Validation Check]")
    check_file = os.path.join(train_dir, sorted([f for f in os.listdir(train_dir) if f.endswith('.csv')])[0])
    
    with open(check_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"  Checking: {os.path.basename(check_file)}")
    print(f"  Line 10 (first data row): {lines[9].strip()[:80]}")
    
    # Try to parse one data row
    try:
        parts = lines[9].split(',')
        print(f"  Data columns: {len(parts)}")
        print(f"  First 3 values: {parts[0]}, {parts[1]}")
    except Exception as e:
        print(f"  ERROR parsing data: {e}")

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("AI-Modeling FFT | SIMPLE DEBUG ANALYSIS")
    print("█"*80)
    
    analyze_data_simple()
    
    print("\n" + "█"*80)
    print("Analysis complete!")
    print("█"*80 + "\n")
