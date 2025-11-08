import matplotlib.pyplot as plt
import numpy as np

def plot_axis_errors(mse_axis, mae_axis, out_path):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.title('MSE per axis'); plt.bar(['x','y','z'], mse_axis)
    plt.subplot(1,2,2); plt.title('MAE per axis'); plt.bar(['x','y','z'], mae_axis)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def save_summary_csv(rows, path):
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
