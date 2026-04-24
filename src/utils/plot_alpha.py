import matplotlib.pyplot as plt
import numpy as np

def plot_alpha(alpha_list, alpha_time, save_path=None):
    alpha_arr = np.array(alpha_list)  # shape: [N, 3]
    t = np.array(alpha_time)

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    labels = [r"$\alpha_{Q1}$", r"$\alpha_{Q2}$", r"$\alpha_R$"]

    for i in range(3):
        axs[i].plot(t, alpha_arr[:, i], linewidth=1.8)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
        axs[i].axvline(2.0, linestyle=":", color="gray")  # 爆胎时刻

    axs[-1].set_xlabel("Time (s)")

    fig.suptitle("Adaptive MPC Parameters (RL Output)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()