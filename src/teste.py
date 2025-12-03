import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig_files = [
    "models/rn_daily_caseonly_comparative_pred_vs_real.png",
    "models/rn_daily_full_comparative_pred_vs_real.png",
    "models/rn_weekly_casesonly_comparative_pred_vs_real.png",
    "models/rn_weekly_full_comparative_pred_vs_real.png",
    "models/rj_daily_caseonly_comparative_pred_vs_real.png",
    "models/rj_daily_full_comparative_pred_vs_real.png",
    "models/rj_weekly_casesonly_comparative_pred_vs_real.png",
    "models/rj_weekly_full_comparative_pred_vs_real.png",
]

titles = [
    "RN · Diário · Apenas Casos",
    "RN · Diário · Todas as Features",
    "RN · Semanal · Apenas Casos",
    "RN · Semanal · Todas as Features",
    "RJ · Diário · Apenas Casos",
    "RJ · Diário · Todas as Features",
    "RJ · Semanal · Apenas Casos",
    "RJ · Semanal · Todas as Features",
]

fig, axes = plt.subplots(4, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, fname, title in zip(axes, fig_files, titles):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("panel_series_pred_vs_real.png", dpi=300)
plt.close()
