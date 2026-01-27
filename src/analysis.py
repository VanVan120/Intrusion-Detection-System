import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_analysis(show_plots=True):
    print("\n--- Running Final Performance Analysis ---")
    sns.set_style("whitegrid")
    
    # Define file paths
    result_files = {
        'Baseline (DT)': 'results/baseline_metrics.json',
        'Genetic Algo (GA)': 'results/ga_metrics.json',
        'PSO': 'results/pso_metrics.json',
        'ABC': 'results/abc_metrics.json',
        'Hybrid PSO-GA': 'results/hybrid_metrics.json',
        'Joint Opt': 'results/joint_metrics.json'
    }

    data_list = []

    print("Loading results from JSON files...")
    for method_name, file_path in result_files.items():
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    metrics = json.load(f)
                    
                entry = {
                    'Method': method_name,
                    'Accuracy': metrics.get('Accuracy', 0),
                    'Precision': metrics.get('Precision', 0),
                    'Recall': metrics.get('Recall', 0),
                    'F1-Score': metrics.get('F1-Score', 0),
                    'Detection Rate': metrics.get('Detection Rate (TPR)', 0),
                    'FPR': metrics.get('False Positive Rate (FPR)', 0),
                    'Feature_Count': metrics.get('Feature Count', 0),
                    'Training_Time_Sec': metrics.get('Runtime (s)', 0)
                }
                data_list.append(entry)
                print(f"Loaded: {file_path}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        else:
            print(f"Warning: {file_path} not found. Skipping.")

    if not data_list:
        print("No result files found! Please run training modes first.")
        return

    # Create DataFrame
    df_results = pd.DataFrame(data_list)
    df_results.set_index('Method', inplace=True)
    
    if 'Baseline (DT)' in df_results.index:
        baseline_feats = df_results.loc['Baseline (DT)', 'Feature_Count']
        df_results['Reduction (%)'] = ((baseline_feats - df_results['Feature_Count']) / baseline_feats) * 100
    else:
        df_results['Reduction (%)'] = 0

    print("\n=== Consolidated Results Table ===")
    print(df_results)
    
    # Ensure directory exists for saving plots
    os.makedirs('results/plots', exist_ok=True)

    # 1. General Metrics
    print("\nDisplaying General Metrics Plot...")
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    ax = df_results[metrics_to_plot].plot(kind='bar', figsize=(15, 8), width=0.85, colormap='viridis', edgecolor='black')
    plt.title('General Classification Metrics Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Score (0.0 - 1.0)', fontsize=12)
    plt.xlabel('Optimization Method', fontsize=12)
    min_val = df_results[metrics_to_plot].min().min()
    plt.ylim(min_val * 0.999, 1.0002) 
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=True, fontsize=11)
    plt.xticks(rotation=15, ha='right')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=4, rotation=90, fontsize=9)
    plt.tight_layout()
    plt.savefig('results/plots/1_general_metrics.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 2. Security Metrics
    print("Displaying Security Metrics Plot...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df_results))
    width = 0.35
    rects1 = ax1.bar(x - width/2, df_results['Detection Rate'], width, label='Detection Rate (TPR)', color='#2ca02c', edgecolor='black')
    ax1.set_ylabel('Detection Rate (Higher is Better)', fontsize=12, color='#2ca02c', fontweight='bold')
    ax1.set_title('Security Metrics: Detection Rate vs False Positive Rate', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_results.index, rotation=15, ha='right')
    ax1.set_ylim(df_results['Detection Rate'].min() * 0.99, 1.001)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df_results['FPR'], width, label='False Positive Rate (FPR)', color='#d62728', edgecolor='black')
    ax2.set_ylabel('False Positive Rate (Lower is Better)', fontsize=12, color='#d62728', fontweight='bold')
    ax2.set_ylim(0, df_results['FPR'].max() * 1.5)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=11)
    ax1.bar_label(rects1, fmt='%.4f', padding=3, fontsize=10)
    ax2.bar_label(rects2, fmt='%.4f', padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig('results/plots/2_security_metrics.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 3. Feature Reduction
    print("Displaying Feature Reduction Analysis...")
    fig, ax1 = plt.subplots(figsize=(14, 7))
    bar_color = '#a6cee3'
    x_indexes = range(len(df_results))
    bars = ax1.bar(x_indexes, df_results['Feature_Count'], color=bar_color, edgecolor='black', width=0.6, alpha=0.9, label='Feature Count')
    ax1.set_ylabel('Number of Selected Features', fontsize=12, fontweight='bold')
    ax1.set_title('Feature Reduction by Method', fontsize=16, fontweight='bold')
    short_labels = [name.replace(' ', '\n') for name in df_results.index]
    ax1.set_xticks(x_indexes)
    ax1.set_xticklabels(short_labels, fontsize=11, fontweight='bold', rotation=0)
    ax1.bar_label(bars, padding=3, fontsize=11, fontweight='bold')
    ax2 = ax1.twinx()
    ax2.plot(x_indexes, df_results['Reduction (%)'], color='#e31a1c', marker='D', markersize=8, linewidth=2.5, label='Reduction %')
    ax2.set_ylabel('Reduction Percentage (%)', fontsize=12, color='#e31a1c', fontweight='bold')
    ax2.set_ylim(0, 110)
    # Annotate percentages
    for i, txt in enumerate(df_results['Reduction (%)']):
        ax2.annotate(f"{txt:.1f}%", (i, txt), textcoords="offset points", xytext=(0, 12), ha='center', color='#e31a1c', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#e31a1c', lw=1))
    plt.tight_layout()
    plt.savefig('results/plots/3_feature_reduction.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 4. Computational Cost
    print("Displaying Computational Cost Analysis...")
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("muted", len(df_results))
    bars = plt.bar(df_results.index, df_results['Training_Time_Sec'], color=colors, edgecolor='black')
    plt.title('Execution Time Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Runtime (Seconds)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/plots/4_runtime_comparison.png')
    if show_plots:
        plt.show()
    else:
        plt.close()

    # 5. Pareto Front
    print("Displaying Pareto Frontier...")
    plt.figure(figsize=(14, 9))
    sns.scatterplot(data=df_results, x='Feature_Count', y='Accuracy', hue='Method', style='Method', s=600, palette='bright', edgecolor='black', linewidth=1.5, alpha=0.85)
    
    label_offsets = {
        'Baseline (DT)':       {'xytext': (-15, 0),  'ha': 'right', 'va': 'center'},
        'Genetic Algo (GA)':   {'xytext': (15, -10), 'ha': 'left',  'va': 'top'},
        'PSO':                 {'xytext': (15, 10),  'ha': 'left',  'va': 'bottom'},
        'ABC':                 {'xytext': (-15, -15), 'ha': 'right', 'va': 'top'},
        'Hybrid PSO-GA':       {'xytext': (15, 0),   'ha': 'left',  'va': 'center'},
        'Joint Opt':           {'xytext': (0, 25),   'ha': 'center','va': 'bottom'}
    }

    for i, method in enumerate(df_results.index):
        x_val = df_results['Feature_Count'].iloc[i]
        y_val = df_results['Accuracy'].iloc[i]
        config = label_offsets.get(method, {'xytext': (0, 15), 'ha': 'center', 'va': 'bottom'})
        plt.annotate(method, xy=(x_val, y_val), xytext=config['xytext'], textcoords='offset points', ha=config['ha'], va=config['va'], fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9), arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.1", color='gray'))

    plt.title('Pareto Frontier: Accuracy vs. Complexity', fontsize=22, fontweight='bold', pad=25)
    plt.xlabel('Number of Features (Complexity) \n← Simpler (Better) | Complex (Worse) →', fontsize=14)
    plt.ylabel('Test Accuracy \n← Worse | Better →', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    target_x = x_min + (x_max - x_min) * 0.05
    target_y = y_max - (y_max - y_min) * 0.05
    text_x = x_min + (x_max - x_min) * 0.15
    text_y = y_min + (y_max - y_min) * 0.7
    
    plt.annotate('IDEAL ZONE\n(Max Accuracy, Min Complexity)', xy=(target_x, target_y), xytext=(text_x, text_y), arrowprops=dict(facecolor='#d62728', shrink=0.05, width=3, headwidth=10, connectionstyle="arc3,rad=-0.2"), fontsize=13, fontweight='bold', color='#ce1141', bbox=dict(boxstyle="round,pad=0.3", fc="#ffeaea", ec="#ce1141", lw=2))
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Optimization Method', fontsize=12, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('results/plots/5_pareto_frontier.png')
    if show_plots:
        plt.show()
    else:
        plt.close()
