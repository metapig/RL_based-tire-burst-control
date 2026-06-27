import numpy as np
data = np.load("runs/encoder_data/encoder_dataset.npz")
labels = data["labels"]

import matplotlib.pyplot as plt
plt.hist(labels, bins=50)
plt.xlabel("mu")
plt.ylabel("count")
plt.show()

print(f"μ < 0.4: {(labels < 0.4).mean():.2%}")
print(f"0.4 ≤ μ ≤ 0.7: {((labels >= 0.4) & (labels <= 0.7)).mean():.2%}")
print(f"μ > 0.7: {(labels > 0.7).mean():.2%}")
