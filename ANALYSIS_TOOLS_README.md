# Analysis Tools Documentation

Complete guide to analyzing FedAgain experimental results with organized, dataset-specific outputs.

## 📋 Overview

This project now includes sophisticated analysis tools that automatically organize experimental results by dataset, generate comprehensive reports, and create publication-ready visualizations.

### Key Features

✅ **Automatic Dataset Detection** - Detects all datasets in your experiments CSV  
✅ **Organized Folder Structure** - Each dataset gets its own results folder  
✅ **Comprehensive Analysis** - 7 test scenarios analyzed automatically  
✅ **Publication-Ready Plots** - 5 high-quality visualizations per dataset  
✅ **Cross-Dataset Comparison** - Compare FedAgain performance across datasets  
✅ **Markdown Reports** - Detailed analysis summaries in Markdown format  

---

## 🚀 Quick Start

### 1. Run Experiments

First, run experiments on one or more datasets:

```bash
# Single dataset
python test_fedagain_superiority.py --mystone

# Multiple datasets
python test_fedagain_superiority.py --all-kidney

# Or just CIFAR10
python test_fedagain_superiority.py --cifar10
```

This populates `results/experiments.csv` with your experimental data.

### 2. Analyze Results

Run the analysis tool to organize and analyze all datasets:

```bash
python analyze_by_dataset.py
```

This automatically:
- Detects all unique datasets in experiments.csv
- Creates a folder for each dataset in `results/`
- Generates plots and analysis summaries
- Organizes everything neatly

### 3. Compare Datasets (Optional)

Compare FedAgain performance across all analyzed datasets:

```bash
python compare_datasets.py
```

This creates a comprehensive comparison report.

---

## 📂 Directory Structure

After running the analysis tools, your results will be organized as:

```
results/
├── experiments.csv                     # Main results (all datasets)
├── dataset_comparison.csv              # Cross-dataset comparison
│
├── MYSTONE/                            # Dataset-specific folder
│   ├── experiments.csv                 # MYSTONE experiments only
│   ├── ANALYSIS_SUMMARY.md             # Detailed analysis report
│   ├── plot1_corruption_comparison.png
│   ├── plot2_strategy_bar_30pct.png
│   ├── plot3_severity_impact.png
│   ├── plot4_improvement_percentage.png
│   └── plot5_f1_comparison.png
│
├── CIFAR10/                            # Another dataset
│   ├── experiments.csv
│   ├── ANALYSIS_SUMMARY.md
│   └── plot*.png
│
└── [other datasets...]/
```

---

## 🛠️ Tool Reference

### analyze_by_dataset.py

**Purpose:** Organize experimental results by dataset and generate comprehensive analysis.

**Usage:**
```bash
# Analyze default CSV (results/experiments.csv)
python analyze_by_dataset.py

# Analyze specific CSV file
python analyze_by_dataset.py path/to/experiments.csv

# Analyze archived results
python analyze_by_dataset.py results/experiments_19102025.csv
```

**What it does:**
1. Reads the CSV file
2. Identifies all unique datasets
3. For each dataset:
   - Creates a dedicated folder in `results/`
   - Saves dataset-specific experiments.csv
   - Generates 5 visualization plots
   - Creates a detailed ANALYSIS_SUMMARY.md report

**Output per dataset:**
- `experiments.csv` - Filtered experiments for this dataset
- `ANALYSIS_SUMMARY.md` - Complete analysis report
- `plot1_corruption_comparison.png` - Accuracy vs corruption level
- `plot2_strategy_bar_30pct.png` - Strategy comparison at 30% corruption
- `plot3_severity_impact.png` - Severity progression analysis
- `plot4_improvement_percentage.png` - FedAgain improvement percentages
- `plot5_f1_comparison.png` - Accuracy and F1 score comparison

---

### compare_datasets.py

**Purpose:** Compare FedAgain performance across multiple datasets.

**Usage:**
```bash
python compare_datasets.py
```

**What it does:**
1. Finds all analyzed datasets in `results/` directory
2. Extracts key metrics from each dataset
3. Creates comparison tables and detailed breakdowns
4. Saves cross-dataset comparison CSV

**Output:**
- Console: Detailed comparison report
- `results/dataset_comparison.csv` - Summary table for all datasets

**Sample Output:**
```
Dataset   Experiments  FedAgain Acc    FedAvg Acc    Improvement  Win Rate
MYSTONE           41  0.5583±0.0502  0.5482±0.0383      +1.83%     22.2%
CIFAR10           35  0.8234±0.0423  0.7891±0.0501      +4.35%     45.0%
```

---

### visualize_fedagain_results.py (Deprecated)

**Status:** Legacy script, now replaced by `analyze_by_dataset.py`

This script was the original visualization tool but didn't organize results by dataset. Use `analyze_by_dataset.py` instead for better organization.

---

### test_fedagain_superiority.py

**Purpose:** Run comprehensive test suites to evaluate FedAgain.

**Key Commands:**
```bash
# Full test suite (all tests + analysis)
python test_fedagain_superiority.py

# Individual tests (CIFAR10 only)
python test_fedagain_superiority.py --test1  # Baseline
python test_fedagain_superiority.py --test2  # 10% corruption
python test_fedagain_superiority.py --test3  # 30% corruption
python test_fedagain_superiority.py --test4  # 40% corruption
python test_fedagain_superiority.py --test5  # Severity progression
python test_fedagain_superiority.py --test6  # Non-IID analysis
python test_fedagain_superiority.py --test7  # Scaling analysis

# Dataset-specific test suites (all 7 tests)
python test_fedagain_superiority.py --cifar10
python test_fedagain_superiority.py --mystone
python test_fedagain_superiority.py --jonathan-sur
python test_fedagain_superiority.py --daudon-mix

# All kidney stone datasets
python test_fedagain_superiority.py --all-kidney

# Just analyze existing results
python test_fedagain_superiority.py --analyze
```

---

## 📊 Understanding the Output

### ANALYSIS_SUMMARY.md

Each dataset's analysis summary contains:

#### 1. Executive Summary
- Overall performance metrics
- FedAgain vs FedAvg comparison
- Total experiments and strategies tested

#### 2. Strategy Performance Summary
Table showing mean, std dev, min, max accuracy for each strategy.

#### 3. Test-by-Test Results

**Test 1: Baseline (Clean Data)**
- Verifies all strategies work on non-corrupted data
- Expected: Similar performance across strategies

**Test 2: Light Corruption (10%)**
- 10% of clients are corrupted
- Shows FedAgain's ability to handle minimal corruption

**Test 3: Moderate Corruption (30%)**
- 30% of clients corrupted (key test scenario)
- Expected: FedAgain should outperform significantly

**Test 4: Heavy Corruption (40%)**
- Stress test with majority corruption
- Tests robustness under extreme conditions

**Test 5: Severity Progression**
- Tests corruption severity levels 1-5
- Shows how FedAgain handles varying corruption intensity

**Test 6: Non-IID Distribution Impact**
- Tests IID, non-IID label, and non-IID Dirichlet
- Evaluates performance on realistic data distributions

**Test 7: Client Scaling**
- Tests with 5, 10, and 20 clients
- Shows scalability characteristics

#### 4. Key Findings
Highlights best and worst scenarios for FedAgain.

#### 5. Recommendations
Dataset-specific guidance on when to use FedAgain.

---

### Visualization Plots

#### Plot 1: Corruption Comparison
**Filename:** `plot1_corruption_comparison.png`

Line plot showing accuracy vs corruption ratio for all strategies.
- **X-axis:** Corrupted clients (%)
- **Y-axis:** Test accuracy
- **Lines:** One per strategy (FedAvg, FedProx, FedMedian, Bulyan, FedAgain)

**Use in paper:** Shows robustness trends as corruption increases.

---

#### Plot 2: Strategy Bar Chart (30% Corruption)
**Filename:** `plot2_strategy_bar_30pct.png`

Bar chart comparing all strategies at the critical 30% corruption point.
- **X-axis:** Strategies
- **Y-axis:** Test accuracy
- **Colors:** FedAvg=red, FedAgain=green, others=blue

**Use in paper:** Direct visual comparison at key test point.

---

#### Plot 3: Severity Impact
**Filename:** `plot3_severity_impact.png`

Line plot showing FedAgain vs FedAvg across severity levels 1-5.
- **X-axis:** Corruption severity (1-5)
- **Y-axis:** Test accuracy
- **Lines:** FedAgain and FedAvg

**Use in paper:** Shows robustness to corruption intensity.

---

#### Plot 4: Improvement Percentage
**Filename:** `plot4_improvement_percentage.png`

Bar chart showing FedAgain's percentage improvement over FedAvg.
- **X-axis:** Corruption level (%)
- **Y-axis:** Improvement over FedAvg (%)
- **Bars:** Green (positive = better), red (negative = worse)

**Use in paper:** Quantifies FedAgain's advantage.

---

#### Plot 5: F1 Score Comparison
**Filename:** `plot5_f1_comparison.png`

Dual horizontal bar chart showing both accuracy and F1 score.
- **Left panel:** Accuracy comparison
- **Right panel:** F1 score comparison
- **Bars:** Green for FedAgain, blue for others

**Use in paper:** Multi-metric evaluation beyond accuracy.

---

## 💡 Usage Examples

### Example 1: Analyze MYSTONE Results

```bash
# Run experiments
python test_fedagain_superiority.py --mystone

# Analyze results
python analyze_by_dataset.py

# Check output
cd results/MYSTONE
cat ANALYSIS_SUMMARY.md  # or 'type' on Windows
```

### Example 2: Compare Multiple Datasets

```bash
# Run experiments on multiple datasets
python test_fedagain_superiority.py --mystone
python test_fedagain_superiority.py --cifar10
python test_fedagain_superiority.py --jonathan-sur

# All results go to results/experiments.csv

# Analyze all datasets at once
python analyze_by_dataset.py

# Compare across datasets
python compare_datasets.py

# Check comparison
cat results/dataset_comparison.csv
```

### Example 3: Re-analyze Old Results

```bash
# You have old results in results/experiments_19102025.csv
python analyze_by_dataset.py results/experiments_19102025.csv

# Organized output goes to results/[DATASET]/ folders
```

### Example 4: Generate Paper Figures

```bash
# Run full analysis
python analyze_by_dataset.py

# Plots are ready for LaTeX
# results/MYSTONE/plot1_corruption_comparison.png
# results/MYSTONE/plot2_strategy_bar_30pct.png
# etc.

# Use in LaTeX:
# \includegraphics[width=0.8\textwidth]{results/MYSTONE/plot1_corruption_comparison.png}
```

---

## 🔧 Customization

### Adding Custom Plots

Edit `analyze_by_dataset.py`:

```python
def plot_custom_metric(df, output_dir, dataset_name):
    """Your custom plot function."""
    fig, ax = plt.subplots(figsize=(10, 6))
    # Your plotting code here
    plt.savefig(output_dir / 'plot_custom.png', dpi=300)
    plt.close()

# Add to analyze_dataset() function:
plot_custom_metric(df, output_dir, dataset_name)
```

### Adding Custom Analysis

Edit `generate_analysis_text()` in `analyze_by_dataset.py`:

```python
# Add new section
analysis.append("## My Custom Analysis\n")
# Your analysis code
analysis.append(f"Custom metric: {my_value:.4f}\n")
```

### Filtering Results

Pre-filter before analysis:

```python
import pandas as pd

df = pd.read_csv('results/experiments.csv')
# Filter only successful runs
df_filtered = df[df['final_accuracy'] > 0.5]
df_filtered.to_csv('results/filtered.csv', index=False)

# Then analyze
# python analyze_by_dataset.py results/filtered.csv
```

---

## 📝 Best Practices

### 1. Regular Backups
```bash
# Backup before new experiments
cp results/experiments.csv results/backup_$(date +%Y%m%d).csv
```

### 2. Incremental Analysis
```bash
# After each dataset's experiments complete
python analyze_by_dataset.py
```

### 3. Version Control
```bash
# Commit results periodically
git add results/MYSTONE/
git commit -m "Add MYSTONE analysis results"
```

### 4. Documentation
Keep notes on experiment parameters in commit messages or separate notes file.

---

## 🐛 Troubleshooting

### Issue: "No dataset folders found"
**Cause:** Haven't run `analyze_by_dataset.py` yet.  
**Fix:** 
```bash
python analyze_by_dataset.py
```

### Issue: "CSV file not found"
**Cause:** Wrong path or no experiments run yet.  
**Fix:** 
```bash
# Check if experiments.csv exists
ls results/experiments.csv

# Run experiments first
python test_fedagain_superiority.py --mystone
```

### Issue: "Empty plots or missing data"
**Cause:** Experiments don't cover all test scenarios.  
**Fix:** Run complete test suite:
```bash
python test_fedagain_superiority.py --mystone
```

### Issue: Unicode/Encoding errors on Windows
**Cause:** Console encoding issues.  
**Fix:** Use UTF-8:
```powershell
# PowerShell
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

---

## 📚 Related Documentation

- `DATASET_ANALYSIS_GUIDE.md` - Comprehensive analysis guide
- `EXPERIMENTS_GUIDE.md` - How to run experiments
- `FEDAGAIN_TESTING_GUIDE.md` - FedAgain testing protocols
- `README.md` - Main project documentation

---

## 🎓 For Paper Writing

### Using Tables in LaTeX

Copy tables from ANALYSIS_SUMMARY.md (they're in Markdown format):

```latex
\begin{table}[h]
\centering
\caption{Strategy Performance on MYSTONE Dataset}
\begin{tabular}{lcccc}
\hline
Strategy & Mean Acc & Std Dev & Min & Max \\
\hline
FedProx & 0.5944 & 0.0631 & 0.5217 & 0.6342 \\
Bulyan & 0.5756 & 0.1075 & 0.4175 & 0.6450 \\
FedMedian & 0.5629 & 0.0894 & 0.4525 & 0.6708 \\
FedAgain & 0.5583 & 0.0502 & 0.4942 & 0.6525 \\
FedAvg & 0.5482 & 0.0383 & 0.4950 & 0.6317 \\
\hline
\end{tabular}
\end{table}
```

### Including Figures

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{results/MYSTONE/plot1_corruption_comparison.png}
    \caption{Federated learning strategy comparison under varying corruption levels on MYSTONE dataset.}
    \label{fig:mystone_corruption}
\end{figure}
```

### Citing Results

Reference specific findings from ANALYSIS_SUMMARY.md:

"On the MYSTONE kidney stone classification dataset, FedAgain demonstrated a +1.83% average improvement over FedAvg across 41 experiments. At 10% corruption, FedAgain achieved a +10.33% improvement, validating its robustness under light adversarial conditions."

---

## 🚀 Quick Reference Card

```bash
# 1. Run experiments
python test_fedagain_superiority.py --mystone

# 2. Analyze and organize results
python analyze_by_dataset.py

# 3. Compare datasets
python compare_datasets.py

# 4. Check outputs
ls results/MYSTONE/

# 5. View analysis
cat results/MYSTONE/ANALYSIS_SUMMARY.md

# 6. View comparison
cat results/dataset_comparison.csv
```

---

**Last Updated:** October 24, 2025  
**Version:** 2.0  
**Contact:** See main README.md for contact information

