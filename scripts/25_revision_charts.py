"""Charts for the revised report.
chart3_tehran : 75 m2 Tehran price / income of young (25-34) households in TEHRAN county (95% bootstrap band),
                national urban young households for comparison.
chart9        : living arrangements at age 30-34 by 5-year birth cohort (stacked 100%).
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *
GRAY, LGRAY = "#8a8984", "#c9c8c2"

# ---------- chart 3 (Tehran) ----------
R = pd.read_csv("Data/indicator_pti_young_tehran.csv")
t = R[R.group == "tehran_county"].set_index("year"); n = R[R.group == "iran_urban"].set_index("year")
fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURF); style(ax)
ax.fill_between(t.index, t.pti_lo, t.pti_hi, color=BLUE, alpha=0.13, lw=0)
ax.plot(n.index, n.pti, color=GRAY, lw=1.6, ls=(0, (4, 3)), marker="o", ms=5, markeredgecolor=SURF)
ax.plot(t.index, t.pti, color=BLUE, lw=2.6, marker="o", ms=9, markeredgecolor=SURF, markeredgewidth=2)
for y, v in t.pti.items():
    ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, -20), ha="center", fontsize=11, color=INK, weight="bold")
ax.text(1395, 3.2, fa("خانوار جوان تهرانی"), color=BLUE, fontsize=11, weight="bold", ha="left")
ax.annotate(fa(f"خانوار جوان شهری کل کشور ({n.pti[1402]:.0f} سال در ۱۴۰۲)"), (1399, n.pti[1399]), textcoords="offset points",
            xytext=(-6, 8), ha="right", color=INK2, fontsize=9.5)
ax.set_xticks(t.index); ax.set_xticklabels([fa(y) for y in t.index]); ax.set_ylim(0, 32)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))
ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
title(fig, "در هفت سال، خانه‌ی ۷۵ متری تهران دو برابر سال‌های درآمد یک خانوار جوان تهرانی شد",
      "قیمت یک آپارتمان ۷۵ متری ÷ درآمد سالانه‌ی خانوارهای شهرستان تهران با سرپرست ۲۵ تا ۳۴ ساله؛ نوار = بازه‌ی اطمینان ۹۵٪")
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart3_pti_young_tehran.png", dpi=170, facecolor=SURF); plt.close()

# ---------- chart 9 ----------
L = pd.read_csv("Data/living_arrangements_by_cohort.csv")
L = L[(L.cohort_type == "5yr") & (L.age_group == "30-34")].sort_values("cohort")
CAT = [("parents_home", ORANGE, "در خانه‌ی پدر و مادر"), ("renter", AQUA, "مستقل، مستأجر"),
       ("owner", BLUE, "مستقل، در خانه‌ی ملکی"), ("other_tenure", LGRAY, "مستقل، سایر (رایگان، سازمانی)"),
       ("other_relation", "#e4e3df", "سایر")]
lab = {r.cohort: fa(f"{int(r.cohort)} تا {int(r.cohort) + 4}") for r in L.itertuples()}
lab[1370] = fa("۱۳۷۰ تا ۱۳۷۲"); lab[1335] = fa("۱۳۳۵ تا ۱۳۳۹")
fig, ax = plt.subplots(figsize=(10.5, 6.8), facecolor=SURF); style(ax); ax.grid(False)
y = np.arange(len(L))[::-1]
left = np.zeros(len(L))
for key, c, name in CAT:
    v = L[key].values
    ax.barh(y, v, left=left, color=c, height=0.66, edgecolor=SURF, lw=2)
    for yi, l, vi in zip(y, left, v):
        if vi >= 0.07 and key in ("parents_home", "renter", "owner"):
            ax.text(l + vi / 2, yi, pct(vi), ha="center", va="center", fontsize=10, color="#ffffff", weight="bold")
    left += v
ax.set_yticks(y); ax.set_yticklabels([lab[c] for c in L.cohort]); ax.set_xlim(0, 1); ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.set_ylabel(fa("متولدین"), color=INK2)
for i, (key, c, name) in enumerate(CAT[:4]):
    ax.add_patch(matplotlib.patches.Rectangle((0.0 + i * 0.25, len(L) - 0.47), 0.018, 0.24, color=c, clip_on=False))
    ax.text(0.026 + i * 0.25, len(L) - 0.35, fa(name), color=INK, fontsize=9.5, ha="left", va="center")
ax.set_ylim(-0.6, len(L) + 0.1)
title(fig, "سی‌ساله‌ها کجا زندگی می‌کنند؟ هر نسل کمتر مستقل شده است",
      "وضعیت زندگی افراد شهری در ۳۰ تا ۳۴ سالگی، به تفکیک سال تولد")
plt.subplots_adjust(left=0.13, right=0.97, top=0.84, bottom=0.04)
plt.savefig("Data/final_chart9_living_arrangements.png", dpi=170, facecolor=SURF); plt.close()
print("ok")
