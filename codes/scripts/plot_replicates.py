import sys
import matplotlib
matplotlib.use('Agg') # Set backend to non-interactive
import pandas as pd
import matplotlib.pyplot as plt

def plot_replicate(input_file, output_file):
    try:
        # Read CSV file
        # Columns: tick, mean, var, v_genic
        df = pd.read_csv(input_file)
        
        if df.empty:
            print(f"Warning: {input_file} is empty.")
            return

        # Check if required columns exist
        required_columns = ['tick', 'mean', 'var', 'v_genic']
        # The file might have a header or not. The previous file view showed a header: "tick,mean,var,v_genic"
        # pd.read_csv reads header by default.
        
        # Verify columns
        if not all(col in df.columns for col in required_columns):
             # Fallback if no header or different names, but based on `head` output, it has header.
             pass

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Titles and columns to plot y-axis
        # 1st col is tick (x)
        # last 3 are mean, var, v_genic
        y_cols = ['mean', 'var', 'v_genic']
        titles = ['Mean Phenotype', 'Genetic Variance', 'Genic Variance']
        
        for i, col in enumerate(y_cols):
            axes[i].plot(df['tick'], df[col], linewidth=1.0)
            axes[i].set_ylabel(titles[i])
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].set_title(titles[i])

        axes[2].set_xlabel('Tick')
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_replicates.py <input_csv> <output_png>")
        sys.exit(1)
        
    input_csv = sys.argv[1]
    output_png = sys.argv[2]
    
    plot_replicate(input_csv, output_png)
