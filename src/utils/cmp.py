import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 使用系统内置中文字体，避免中文乱码
#font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
#font = FontProperties(fname=font_path)


plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

# 横坐标：无控制、LPV-MPC控、RL-based MPC
x_labels = ['无控制', 'LPV-MPC', 'RL-based MPC']
x = np.arange(len(x_labels))

# 数据顺序：无控制, LPV-MPC控, RL-based MPC
data = {
    'FR': [-10544.187, -2018.881, -698.526],
    'FL': [-4543.614, -1406.439, -660.786],
    'RL': [-1154.401, -202.971, -65.317],
    'RR': [np.nan, -585.692, -79.498],  # RR 无控制不画
}

plt.figure(figsize=(8, 5))

for cond, y in data.items():
    y = np.array(y, dtype=float)
    if cond == 'RR':
        plt.plot(x[1:], y[1:], marker='o', linewidth=2, label=cond)
    else:
        plt.plot(x, y, marker='o', linewidth=2, label=cond)

plt.xticks(x, x_labels)
plt.xlabel('控制形式')
plt.ylabel('回合奖励')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()