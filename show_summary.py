"""Quick summary of the analysis tools setup."""
import os
from pathlib import Path

print("\n" + "="*80)
print("ANALYSIS TOOLS SETUP SUMMARY")
print("="*80)

# New files created
print("\n[NEW FILES CREATED]\n")
new_files = [
    'analyze_by_dataset.py',
    'compare_datasets.py', 
    'DATASET_ANALYSIS_GUIDE.md',
    'ANALYSIS_TOOLS_README.md',
    'show_summary.py'
]

for f in new_files:
    if Path(f).exists():
        size = Path(f).stat().st_size
        print(f"  [OK] {f:40s} ({size:>7,} bytes)")

# Results structure
print("\n[RESULTS STRUCTURE]\n")
results_path = Path("results/MYSTONE")
if results_path.exists():
    print(f"  results/MYSTONE/")
    for item in sorted(results_path.iterdir()):
        if item.is_file():
            size = item.stat().st_size
            print(f"    - {item.name:35s} ({size:>7,} bytes)")

# Quick stats
print("\n[CURRENT DATASET ANALYSIS]\n")
csv_path = Path("results/MYSTONE/experiments.csv")
if csv_path.exists():
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"  Dataset: MYSTONE")
    print(f"  Total Experiments: {len(df)}")
    print(f"  Strategies Tested: {', '.join(df['strategy'].unique())}")
    print(f"  Corruption Levels: {sorted(df['corrupt_ratio'].unique())}")
    
    if 'FedAgain' in df['strategy'].values and 'FedAvg' in df['strategy'].values:
        fedagain_acc = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
        fedavg_acc = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
        improvement = ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
        print(f"\n  FedAgain vs FedAvg:")
        print(f"    FedAgain Mean Accuracy: {fedagain_acc:.4f}")
        print(f"    FedAvg Mean Accuracy:   {fedavg_acc:.4f}")
        print(f"    Overall Improvement:    {improvement:+.2f}%")

print("\n" + "="*80)
print("USAGE:")
print("="*80)
print("""
1. Run experiments on a dataset:
   python test_fedagain_superiority.py --mystone

2. Analyze and organize results:
   python analyze_by_dataset.py

3. Compare multiple datasets:
   python compare_datasets.py

4. View detailed guide:
   - ANALYSIS_TOOLS_README.md (comprehensive guide)
   - DATASET_ANALYSIS_GUIDE.md (detailed analysis info)
""")

print("="*80)
print("[OK] Setup complete! Ready to analyze experimental results.")
print("="*80 + "\n")

