import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from cycler import cycler

def load_actual_data():
    data = {
        "Epoch": [0.00, 0.35, 0.70, 1.05, 1.41, 1.76, 2.11, 2.46, 2.81, 3.16, 3.52, 3.87, 4.22, 4.57, 4.92, 5.27, 5.63, 5.98, 6.00],
        "Loss": [None, 2.7988, 0.3347, 0.2823, 0.2491, 0.2488, 0.2398, 0.2351, 0.2126, 0.2252, 0.2189, 0.2019, 0.2103, 0.2071, 0.2124, 0.2037, 0.2083, 0.2027, 0.3812],
        "Eval Loss": [None, None, 0.2592, None, 0.2287, None, 0.2151, None, 0.2071, None, 0.2030, None, 0.1994, None, 0.1976, None, 0.1967, None, None],
        "Grad Norm (min)": [67643.90, 0.2602, 59.248, 27.815, 9.001, 29.118, 57.398, 21.243, 21.187, 7.227, 22.244, 5.356, 12.528, 40.140, 18.479, 18.477, 28.146, 37.012, None],
        "Grad Norm (max)": [444065.0, 94.5149, 81.657, 40.385, 56.536, 88.970, 88.420, 106.253, 40.529, 11.187, 26.868, 20.417, 62.240, 56.668, 49.065, 39.957, 37.874, 64.558, None],
        "Learning Rate": [None, 1.479e-05, 1.408e-05, 1.222e-05, 1.128e-05, 1.034e-05, 9.412e-06, 8.480e-06, 7.547e-06, 6.613e-06, 5.679e-06, 4.745e-06, 3.812e-06, 2.878e-06, 1.944e-06, 1.012e-06, 7.844e-08, 7.844e-08, None]
    }
    df = pd.DataFrame(data)
    df = df.dropna(subset=["Epoch"])
    df["Clipped Min"] = df["Grad Norm (min)"].clip(upper=1.0)
    df["Clipped Max"] = df["Grad Norm (max)"].clip(upper=1.0)
    return df

def plot_training_metrics(df):
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.size'] = 10
    colors = sns.color_palette("Set2", 4)
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(df["Epoch"], df["Grad Norm (min)"], label="Original Min", marker='o', markersize=4)
    axs[0].plot(df["Epoch"], df["Clipped Min"], label="Clipped Min", marker='o', markersize=4)
    axs[0].plot(df["Epoch"], df["Grad Norm (max)"], label="Original Max", marker='o', markersize=4)
    axs[0].plot(df["Epoch"], df["Clipped Max"], label="Clipped Max", marker='o', markersize=4)
    axs[0].set_title('Gradient Norms (Original vs Clipped)', fontsize=16, fontweight='bold')
    axs[0].set_xlabel('Epochs', fontsize=12)
    axs[0].set_ylabel('Gradient Norm', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    axs[1].plot(df["Epoch"], df["Learning Rate"], label="Learning Rate", marker='o', markersize=4)
    axs[1].set_title('Learning Rate vs. Epochs', fontsize=16, fontweight='bold')
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].set_ylabel('Learning Rate', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    plt.tight_layout()
    return fig

def save_plot(fig, filename='t5_training_metrics_plot.png'):
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")

def main():
    try:
        df = load_actual_data()
        fig = plot_training_metrics(df)
        save_plot(fig)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
