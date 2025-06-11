# File: code/BART_plot_training_logs.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 300})  # High resolution

def plot_dual_graph(df, x_col, y_cols, titles, ylabels, filename, colors):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, y_col in enumerate(y_cols):
        sub_df = df.dropna(subset=[y_col])
        axes[i].plot(sub_df[x_col], sub_df[y_col], marker='o', color=colors[i], linewidth=2)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel(ylabels[i])
        axes[i].grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f" Saved: {filename}")

def main(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    if 'epoch' not in df.columns:
        raise ValueError("CSV must contain 'epoch' column.")

    # Plot 1: Training & Evaluation Loss
    plot_dual_graph(
        df,
        'epoch',
        ['loss', 'eval_loss'],
        ['Training Loss vs. Epochs', 'Evaluation Loss vs. Epochs'],
        ['Training Loss', 'Evaluation Loss'],
        os.path.join(output_dir, 'bart_loss_vs_epochs.png'),
        colors=['blue', 'blue']  # Same color
    )

    # Plot 2: Gradient Norm & Learning Rate
    plot_dual_graph(
        df,
        'epoch',
        ['grad_norm', 'learning_rate'],
        ['Gradient Norm vs. Epochs', 'Learning Rate vs. Epochs'],
        ['Gradient Norm', 'Learning Rate'],
        os.path.join(output_dir, 'bart_grad_lr_vs_epochs.png'),
        colors=['lightgreen', 'lightgreen']  # Light green for both
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot BART Training Logs by Epoch")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to training_log.csv file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)
