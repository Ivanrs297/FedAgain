# Dataset-Specific Analysis Guide

This guide explains how to analyze experimental results organized by dataset using the `analyze_by_dataset.py` script.

## Overview

The `analyze_by_dataset.py` script automatically:
1. Detects all datasets in your experiments CSV
2. Creates a separate folder for each dataset in `results/`
3. Generates dataset-specific visualizations and analysis reports
4. Organizes everything in a clean, hierarchical structure

## Directory Structure

After running the analysis, your results will be organized as follows:

```
results/
├── experiments.csv                    # Main results file (all datasets)
├── MYSTONE/
│   ├── experiments.csv               # MYSTONE-specific results
│   ├── ANALYSIS_SUMMARY.md           # Detailed analysis report
│   ├── plot1_corruption_comparison.png
│   ├── plot2_strategy_bar_30pct.png
│   ├── plot3_severity_impact.png
│   ├── plot4_improvement_percentage.png
│   └── plot5_f1_comparison.png
├── CIFAR10/
│   ├── experiments.csv
│   ├── ANALYSIS_SUMMARY.md
│   └── plot*.png
├── JONATHAN_SUR/
│   ├── experiments.csv
│   ├── ANALYSIS_SUMMARY.md
│   └── plot*.png
└── [other datasets...]/
```

## Usage

### Basic Usage

Analyze all datasets in the default experiments.csv file:

```bash
python analyze_by_dataset.py
```

### Specify Custom CSV File

Analyze a specific CSV file:

```bash
python analyze_by_dataset.py path/to/experiments.csv
```

### Examples

```bash
# Analyze current experiments
python analyze_by_dataset.py

# Analyze archived results
python analyze_by_dataset.py results/experiments_19102025.csv

# Analyze from a specific path
python analyze_by_dataset.py /path/to/my_experiments.csv
```

## What Gets Generated

For each dataset found in the CSV, the script generates:

### 1. ANALYSIS_SUMMARY.md
A comprehensive Markdown report containing:
- Executive Summary
- Strategy Performance Tables
- Test-by-Test Analysis (Tests 1-7)
- Severity Progression Analysis
- Non-IID Distribution Impact
- Client Scaling Analysis
- Key Findings and Recommendations

### 2. Visualizations (PNG files)

#### plot1_corruption_comparison.png
- Line plot showing accuracy vs corruption ratio
- Compares all strategies across corruption levels
- Helps identify which strategies degrade gracefully

#### plot2_strategy_bar_30pct.png
- Bar chart comparing all strategies at 30% corruption
- Direct visual comparison at the critical test point
- Highlights FedAgain vs other methods

#### plot3_severity_impact.png
- FedAgain vs FedAvg across severity levels (1-5)
- Shows robustness to varying corruption intensity
- Identifies severity sensitivity issues

#### plot4_improvement_percentage.png
- Bar chart showing FedAgain's percentage improvement over FedAvg
- Across different corruption levels
- Positive bars = FedAgain wins, negative = underperforms

#### plot5_f1_comparison.png
- Dual horizontal bar chart
- Compares both Accuracy and F1 Score metrics
- More robust evaluation than accuracy alone

### 3. experiments.csv
- Dataset-specific subset of the main CSV
- Contains only experiments for that dataset
- Useful for dataset-specific analysis

## Analyzing Results from Multiple Datasets

If your `experiments.csv` contains data from multiple datasets (e.g., CIFAR10, MYSTONE, JONATHAN_SUR), the script will:

1. Automatically detect all unique datasets
2. Create a folder for each dataset
3. Generate separate analysis for each dataset
4. Allow easy comparison between datasets

### Example Multi-Dataset Workflow

```bash
# Run experiments on multiple datasets
python test_fedagain_superiority.py --all-kidney

# This adds results to results/experiments.csv with mixed datasets

# Analyze all datasets at once
python analyze_by_dataset.py

# Result: Separate analysis folders for each dataset
# - results/MYSTONE/
# - results/JONATHAN_SUR/
# - results/DAUDON_MIX/
# - results/DAUDON_SEC/
# - etc.
```

## Reading the Analysis Summary

Each `ANALYSIS_SUMMARY.md` contains structured sections:

### Executive Summary
Quick overview of overall performance, improvement percentages

### Strategy Performance Summary
Table showing mean, std dev, min, max for each strategy

### Test 1-7 Analysis
Detailed breakdown of each test scenario:
- **Test 1:** Clean data baseline
- **Test 2:** 10% corruption
- **Test 3:** 30% corruption
- **Test 4:** 40% corruption
- **Test 5:** Severity progression (1-5)
- **Test 6:** Non-IID distribution impact
- **Test 7:** Client scaling (5, 10, 20 clients)

### Key Findings
Highlights best and worst scenarios for FedAgain

### Recommendations
Dataset-specific guidance on when to use each strategy

## Comparing Datasets

To compare FedAgain performance across datasets:

1. Run analysis for all datasets:
   ```bash
   python analyze_by_dataset.py
   ```

2. Check each dataset's `ANALYSIS_SUMMARY.md`:
   ```bash
   # Windows
   type results\MYSTONE\ANALYSIS_SUMMARY.md
   type results\CIFAR10\ANALYSIS_SUMMARY.md
   
   # Linux/Mac
   cat results/MYSTONE/ANALYSIS_SUMMARY.md
   cat results/CIFAR10/ANALYSIS_SUMMARY.md
   ```

3. Compare key metrics:
   - Overall improvement percentage
   - Win rate at different corruption levels
   - Performance on non-IID data
   - Scaling behavior

## Tips and Best Practices

### 1. Keep Original CSV
Always keep a backup of `results/experiments.csv`:
```bash
cp results/experiments.csv results/experiments_backup_$(date +%Y%m%d).csv
```

### 2. Run After Each Test Suite
After completing a full test suite on a dataset:
```bash
python test_fedagain_superiority.py --mystone
python analyze_by_dataset.py  # Immediately analyze
```

### 3. Compare Multiple Runs
If you run experiments multiple times, save different versions:
```bash
# After first run
python analyze_by_dataset.py
mv results/MYSTONE results/MYSTONE_run1

# After second run
python analyze_by_dataset.py
mv results/MYSTONE results/MYSTONE_run2

# Compare
diff results/MYSTONE_run1/ANALYSIS_SUMMARY.md results/MYSTONE_run2/ANALYSIS_SUMMARY.md
```

### 4. Create Comparative Reports
Combine multiple dataset summaries:
```bash
# Windows PowerShell
Get-Content results\*\ANALYSIS_SUMMARY.md > combined_analysis.md

# Linux/Mac
cat results/*/ANALYSIS_SUMMARY.md > combined_analysis.md
```

## Customizing the Analysis

The script can be modified to add custom analyses:

### Add New Plots
Edit `analyze_by_dataset.py` and add new plot functions:

```python
def plot_custom_metric(df, output_dir, dataset_name):
    # Your custom plotting code
    pass

# Then call it in analyze_dataset():
plot_custom_metric(df, output_dir, dataset_name)
```

### Add Custom Metrics
Modify the `generate_analysis_text()` function to include:
- Custom performance metrics
- Domain-specific analyses
- Statistical tests (t-tests, ANOVA)
- Confidence intervals

### Filter Specific Experiments
Pre-filter the CSV before analysis:

```python
import pandas as pd

# Load and filter
df = pd.read_csv('results/experiments.csv')
df_filtered = df[df['num_rounds'] >= 10]  # Only long runs
df_filtered.to_csv('results/filtered_experiments.csv', index=False)

# Analyze filtered data
# python analyze_by_dataset.py results/filtered_experiments.csv
```

## Troubleshooting

### Issue: "CSV file not found"
**Solution:** Ensure you're running from the project root directory:
```bash
cd path/to/fedagain-cls
python analyze_by_dataset.py
```

### Issue: "No dataset column in CSV"
**Solution:** The CSV must have a 'dataset' column. Check column names:
```python
import pandas as pd
df = pd.read_csv('results/experiments.csv')
print(df.columns)
```

### Issue: "Empty plots or missing data"
**Solution:** Ensure your experiments cover all test scenarios:
- Check that you have data for corruption levels: 0%, 10%, 30%, 40%
- Verify severity levels: 1-5
- Confirm multiple strategies are tested

### Issue: "Encoding errors on Windows"
**Solution:** The script now uses UTF-8 encoding. If issues persist:
```python
# Open files with explicit encoding
with open('file.csv', 'r', encoding='utf-8') as f:
    # ...
```

## Integration with Paper Writing

The generated files are designed for direct use in papers:

### Using Plots in LaTeX
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\textwidth]{results/MYSTONE/plot1_corruption_comparison.png}
    \caption{FedAgain performance comparison on MYSTONE dataset.}
    \label{fig:mystone_comparison}
\end{figure}
```

### Using Analysis in Papers
Copy key sections from `ANALYSIS_SUMMARY.md`:
- Tables (already in Markdown format)
- Key findings
- Performance improvements

### Converting to LaTeX Tables
Use pandoc to convert Markdown tables:
```bash
pandoc results/MYSTONE/ANALYSIS_SUMMARY.md -o analysis.tex
```

## Advanced Usage

### Batch Analysis of Multiple CSV Files
```bash
# Analyze all CSV files in results/
for csv in results/*.csv; do
    python analyze_by_dataset.py "$csv"
done
```

### Generate Summary Report
Create a master summary comparing all datasets:

```python
import pandas as pd
from pathlib import Path

datasets = ['MYSTONE', 'CIFAR10', 'JONATHAN_SUR']
summary_data = []

for dataset in datasets:
    csv_path = Path(f'results/{dataset}/experiments.csv')
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        fedagain_acc = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
        fedavg_acc = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
        summary_data.append({
            'Dataset': dataset,
            'FedAgain': fedagain_acc,
            'FedAvg': fedavg_acc,
            'Improvement': ((fedagain_acc - fedavg_acc) / fedavg_acc) * 100
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_markdown(index=False))
```

## Related Scripts

- `test_fedagain_superiority.py` - Run experiments
- `visualize_fedagain_results.py` - Original visualization script (deprecated)
- `run_experiments.py` - Batch experiment runner
- `analyze_by_dataset.py` - **This script** - Dataset-organized analysis

## Future Enhancements

Planned features:
- [ ] Statistical significance testing (t-tests, p-values)
- [ ] Confidence interval plots
- [ ] Interactive HTML reports
- [ ] Automated LaTeX table generation
- [ ] Cross-dataset comparison plots
- [ ] Performance dashboard (web-based)

---

*Last Updated: October 24, 2025*  
*For questions or issues, refer to the main README.md*

