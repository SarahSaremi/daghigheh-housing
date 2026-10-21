"""Generational charts.
A) yearly owner share at age 25-34: (i) among household HEADS 25-34, (ii) among ALL urban persons 25-34
   (= head or spouse of an owner-occupied home). Sample-frame renewals marked.
B) birth-cohort profiles: owner (head/spouse) share of persons by age group, one line per decade cohort."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal: older cohort = lighter
FRAMES = [1379, 1392, 1397]

def style(ax):
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True); ax.set_facecolor(SURF)

# ---------- A ----------
Y = pd.read_csv("Data/owner_share_25_34.csv")
fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURF); style(ax)
for f in FRAMES:
    ax.axvline(f - 0.5, color=GRID, lw=1.5, ls=(0, (4, 3)))
    ax.text(f - 0.4, 0.585, "new sample frame", color=INK2, fontsize=8, rotation=90, va="top")
ax.fill_between(Y.year, Y.owner_share_head_25_34 - 1.96 * Y.se, Y.owner_share_head_25_34 + 1.96 * Y.se,
                color=BLUE, alpha=0.15, lw=0)
series = [("owner_share_head_25_34", BLUE, "Households with a 25–34 head"),
          ("share_owner_head_or_spouse", ORANGE, "All people aged 25–34 (owner as head/spouse)")]
for col, c, lab in series:
    ax.plot(Y.year, Y[col], color=c, lw=2, marker="o", ms=5, markeredgecolor=SURF, markeredgewidth=1.5, label=lab)
    for i in [0, len(Y) - 1]:
        ax.annotate(f"{Y[col].iloc[i]:.0%}", (Y.year.iloc[i], Y[col].iloc[i]), textcoords="offset points",
                    xytext=(-14 if i == 0 else 10, 0), ha="right" if i == 0 else "left", va="center", color=INK, fontsize=10)
ax.set_ylim(0, 0.6); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlim(Y.year.min() - 1.5, Y.year.max() + 1.5); ax.set_xticks(range(1369, 1403, 3))
ax.legend(loc="lower left", frameon=False, labelcolor=INK, fontsize=9.5)
ax.set_title("Homeownership at age 25–34, urban Iran", loc="left", color=INK, fontsize=14, pad=12)
fig.text(0.01, 0.015, "Source: SCI Household Budget Survey microdata (HBSIR), urban, weighted. Owner = owns building (with or without land). "
         "Band = 95% CI.\nBetween frame renewals the survey sample ages and under-covers newly formed young households; "
         "compare levels mainly at renewal years.", color=INK2, fontsize=8, va="bottom")
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("Data/chart_ownership_25_34.png", dpi=150, facecolor=SURF); plt.close()

# ---------- B ----------
D = pd.read_csv("Data/cohort_decades_same_age.csv")
D = D[D.cohort.isin(["1330s", "1340s", "1350s", "1360s", "1370s"])]
ages = ["25-29", "30-34", "35-39", "40-44"]; xi = {a: i for i, a in enumerate(ages)}
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURF); style(ax)
for (coh, d), c in zip(D.groupby("cohort"), RAMP):
    d = d.sort_values("age"); x = d.age.map(xi)
    ax.plot(x, d.persons_owner_head_or_spouse, color=c, lw=2, marker="o", ms=8, markeredgecolor=SURF,
            markeredgewidth=2, label=f"born {coh}")
    last = d.iloc[-1]
    ax.annotate(f"born {coh}", (xi[last.age], last.persons_owner_head_or_spouse), textcoords="offset points",
                xytext=(10, 0), va="center", color=INK, fontsize=10)
ax.set_xticks(range(len(ages))); ax.set_xticklabels([f"age {a}" for a in ages])
ax.set_xlim(-0.3, len(ages) - 0.3); ax.set_ylim(0, 0.9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.legend(loc="upper left", frameon=False, labelcolor=INK, fontsize=9.5)
ax.set_title("Each generation owns less at the same age\nShare of urban people who own their home (as head or spouse)",
             loc="left", color=INK, fontsize=13, pad=12)
fig.text(0.01, 0.015, "Source: SCI HBS microdata 1369–1402 (HBSIR), urban, weighted. Birth year = survey year − age (±1).\n"
         "Partial cohorts: 1330s at 30–34 = born 1335–39; 1360s at 35–39 = born 1360–67, at 40–44 = born 1360–62; 1370s at 30–34 = born 1370–72.",
         color=INK2, fontsize=8, va="bottom")
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("Data/chart_cohort_ownership.png", dpi=150, facecolor=SURF); plt.close()
print("ok")
