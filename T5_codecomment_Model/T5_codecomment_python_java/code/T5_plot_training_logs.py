# File: code/T5_plot_training_logs.py

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_dual_graph(df, x_col, y_cols, titles, ylabels, filename, clip_grad_norm=None, color=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for i, y_col in enumerate(y_cols):
        sub_df = df.dropna(subset=[y_col])

        if clip_grad_norm and y_col == 'grad_norm':
            sub_df[y_col] = sub_df[y_col].clip(upper=clip_grad_norm)

        plot_color = color[i] if color else None
        axes[i].plot(sub_df[x_col], sub_df[y_col], marker='o', color=plot_color)
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

    plot_dual_graph(
        df,
        'epoch',
        ['loss', 'eval_loss'],
        ['Training Loss vs. Epochs', 'Evaluation Loss vs. Epochs'],
        ['Training Loss', 'Evaluation Loss'],
        os.path.join(output_dir, 'loss_vs_epochs.png'),
        color=['#1f77b4', '#1f77b4']  #  blue color
    )

    plot_dual_graph(
        df,
        'epoch',
        ['grad_norm', 'learning_rate'],
        ['Gradient Norm vs. Epochs', 'Learning Rate vs. Epochs'],
        ['Gradient Norm', 'Learning Rate'],
        os.path.join(output_dir, 'grad_lr_vs_epochs.png'),
        clip_grad_norm=1.5,
        color=['#66cc99', '#66cc99']  # Light green
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot T5 Training Logs by Epoch")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to training_log.csv file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    args = parser.parse_args()
    main(args.csv_path, args.output_dir)
