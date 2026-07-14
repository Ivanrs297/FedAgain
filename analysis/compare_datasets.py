"""
Cross-Dataset Comparison Script
Compares FedAgain performance across multiple datasets.
"""

import pandas as pd
from pathlib import Path
import sys


def compare_all_datasets():
    """Compare FedAgain performance across all analyzed datasets."""
    csv_path = Path("results/experiments.csv")
    
    if not csv_path.exists():
        print("[ERROR] results/experiments.csv not found!")
        return
    
    # Read consolidated CSV
    full_df = pd.read_csv(csv_path)
    
    if 'dataset' not in full_df.columns:
        print("[ERROR] 'dataset' column not found in experiments.csv")
        return
    
    datasets = full_df['dataset'].unique()
    
    print("\n" + "="*80)
    print("CROSS-DATASET COMPARISON")
    print("="*80)
    print(f"\nFound {len(datasets)} dataset(s):\n")
    
    comparison_data = []
    
    for dataset_name in sorted(datasets):
        df = full_df[full_df['dataset'] == dataset_name].copy()
        
        try:
            
            # Overall stats
            total_exp = len(df)
            strategies = df['strategy'].unique()
            
            # FedAgain vs FedAvg comparison
            if 'FedAgain' in strategies and 'FedAvg' in strategies:
                fedagain_mean = df[df['strategy'] == 'FedAgain']['final_accuracy'].mean()
                fedagain_std = df[df['strategy'] == 'FedAgain']['final_accuracy'].std()
                fedavg_mean = df[df['strategy'] == 'FedAvg']['final_accuracy'].mean()
                fedavg_std = df[df['strategy'] == 'FedAvg']['final_accuracy'].std()
                improvement = ((fedagain_mean - fedavg_mean) / fedavg_mean) * 100
                
                # Best strategy
                best_strategy = df.groupby('strategy')['final_accuracy'].mean().idxmax()
                best_accuracy = df.groupby('strategy')['final_accuracy'].mean().max()
                
                # Count wins (scenarios where FedAgain performs best)
                conditions = ['corrupt_ratio', 'corruption_severity', 'data_distribution']
                available_conditions = [c for c in conditions if c in df.columns]
                
                wins = 0
                total_scenarios = 0
                if len(available_conditions) > 0:
                    grouped = df.groupby(available_conditions)
                    for name, group in grouped:
                        if len(group['strategy'].unique()) > 1:
                            best = group.loc[group['final_accuracy'].idxmax(), 'strategy']
                            total_scenarios += 1
                            if best == 'FedAgain':
                                wins += 1
                
                win_rate = (wins / total_scenarios * 100) if total_scenarios > 0 else 0
                
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Experiments': total_exp,
                    'Strategies': len(strategies),
                    'FedAgain Acc': f"{fedagain_mean:.4f}±{fedagain_std:.4f}",
                    'FedAvg Acc': f"{fedavg_mean:.4f}±{fedavg_std:.4f}",
                    'Improvement': f"{improvement:+.2f}%",
                    'Win Rate': f"{win_rate:.1f}%",
                    'Best Strategy': best_strategy,
                    'Best Acc': f"{best_accuracy:.4f}"
                })
            else:
                comparison_data.append({
                    'Dataset': dataset_name,
                    'Experiments': total_exp,
                    'Strategies': len(strategies),
                    'FedAgain Acc': 'N/A',
                    'FedAvg Acc': 'N/A',
                    'Improvement': 'N/A',
                    'Win Rate': 'N/A',
                    'Best Strategy': 'N/A',
                    'Best Acc': 'N/A'
                })
                
        except Exception as e:
            print(f"[ERROR] Failed to analyze {dataset_name}: {e}")
            continue
    
    if len(comparison_data) == 0:
        print("[ERROR] No data to compare!")
        return
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display results
    print("\n" + "="*80)
    print("OVERALL COMPARISON TABLE")
    print("="*80 + "\n")
    print(comparison_df.to_string(index=False))
    
    # Detailed comparison
    print("\n\n" + "="*80)
    print("DETAILED ANALYSIS BY DATASET")
    print("="*80)
    
    for idx, row in comparison_df.iterrows():
        dataset_name = row['Dataset']
        
        print(f"\n{'-'*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'-'*80}")
        
        try:
            # Get data for this dataset from the already loaded full_df
            df = full_df[full_df['dataset'] == dataset_name].copy()
            
            # Test-specific results
            print("\n  Test Results:")
            
            # Test 1: Clean data
            clean = df[df['corrupt_clients'] == False]
            if len(clean) > 0:
                print(f"    Clean Data (0% corruption):")
                if 'FedAgain' in clean['strategy'].values and 'FedAvg' in clean['strategy'].values:
                    fa = clean[clean['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fv = clean[clean['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    print(f"      FedAgain: {fa:.4f} | FedAvg: {fv:.4f} | Diff: {(fa-fv):+.4f}")
            
            # Test 2: 10% corruption
            test2 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.1)]
            if len(test2) > 0:
                print(f"    10% Corruption:")
                if 'FedAgain' in test2['strategy'].values and 'FedAvg' in test2['strategy'].values:
                    fa = test2[test2['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fv = test2[test2['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    imp = ((fa - fv) / fv * 100) if fv > 0 else 0
                    print(f"      FedAgain: {fa:.4f} | FedAvg: {fv:.4f} | Improvement: {imp:+.2f}%")
            
            # Test 3: 30% corruption
            test3 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.3)]
            if len(test3) > 0:
                print(f"    30% Corruption:")
                if 'FedAgain' in test3['strategy'].values and 'FedAvg' in test3['strategy'].values:
                    fa = test3[test3['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fv = test3[test3['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    imp = ((fa - fv) / fv * 100) if fv > 0 else 0
                    print(f"      FedAgain: {fa:.4f} | FedAvg: {fv:.4f} | Improvement: {imp:+.2f}%")
            
            # Test 4: 40% corruption
            test4 = df[(df['corrupt_clients'] == True) & (df['corrupt_ratio'] == 0.4)]
            if len(test4) > 0:
                print(f"    40% Corruption:")
                if 'FedAgain' in test4['strategy'].values and 'FedAvg' in test4['strategy'].values:
                    fa = test4[test4['strategy'] == 'FedAgain']['final_accuracy'].mean()
                    fv = test4[test4['strategy'] == 'FedAvg']['final_accuracy'].mean()
                    imp = ((fa - fv) / fv * 100) if fv > 0 else 0
                    print(f"      FedAgain: {fa:.4f} | FedAvg: {fv:.4f} | Improvement: {imp:+.2f}%")
            
            # Strategy rankings
            print("\n  Strategy Rankings (by mean accuracy):")
            rankings = df.groupby('strategy')['final_accuracy'].mean().sort_values(ascending=False)
            for i, (strategy, acc) in enumerate(rankings.items(), 1):
                marker = "*" if strategy == "FedAgain" else " "
                print(f"    {i}. {strategy:12s} {marker} {acc:.4f}")
                
        except Exception as e:
            print(f"    [ERROR] {e}")
    
    # Summary statistics
    print("\n\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    valid_improvements = []
    for item in comparison_data:
        try:
            imp_str = item['Improvement'].replace('%', '').replace('+', '')
            if imp_str != 'N/A':
                valid_improvements.append(float(imp_str))
        except:
            pass
    
    if len(valid_improvements) > 0:
        print(f"\nFedAgain vs FedAvg Across All Datasets:")
        print(f"  Average Improvement: {sum(valid_improvements)/len(valid_improvements):+.2f}%")
        print(f"  Best Improvement: {max(valid_improvements):+.2f}%")
        print(f"  Worst Improvement: {min(valid_improvements):+.2f}%")
        print(f"  Datasets where FedAgain improves: {sum(1 for x in valid_improvements if x > 0)}/{len(valid_improvements)}")
        print(f"  Datasets where FedAgain underperforms: {sum(1 for x in valid_improvements if x < 0)}/{len(valid_improvements)}")
    
    # Save comparison to CSV
    output_path = Path("results") / "dataset_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\n[OK] Comparison saved to: {output_path}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)


def main():
    """Main function."""
    compare_all_datasets()


if __name__ == "__main__":
    main()

