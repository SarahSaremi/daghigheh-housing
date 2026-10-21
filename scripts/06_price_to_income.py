"""Indicator 1: price-to-income ratio.
reference home = 75 m2 Tehran apartment; price = annual mean price/m2 x 75;
ratio = price / average annual URBAN household income (both current toman, same year)."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
REF_M2 = 75
m = pd.read_csv("Data/master_annual.csv")
m["ref_home_price"] = m.tehran_price_per_m2 * REF_M2
m["price_to_income_years"] = m.ref_home_price / m.household_income
out = m.dropna(subset=["price_to_income_years"])[["year", "tehran_price_per_m2", "ref_home_price",
                                                    "household_income", "price_to_income_years"]]
out.to_csv("Data/indicator_price_to_income.csv", index=False, encoding="utf-8-sig")
print(out.round(2).to_string())

INK, INK2, GRID, SURF, S1 = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURF); ax.set_facecolor(SURF)
ax.plot(out.year, out.price_to_income_years, color=S1, lw=2, marker="o", ms=8,
        markeredgecolor=SURF, markeredgewidth=2, solid_capstyle="round")
for _, r in out.iloc[[0, out.price_to_income_years.idxmax() - out.index[0], -1]].iterrows():
    ax.annotate(f"{r.price_to_income_years:.1f} years", (r.year, r.price_to_income_years),
                textcoords="offset points", xytext=(0, 12), ha="center", color=INK, fontsize=10)
ax.set_ylim(0, out.price_to_income_years.max() * 1.2)
ax.set_xticks(out.year)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=INK2, length=0)
ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
ax.set_title(f"Years of total urban household income needed to buy a {REF_M2} m² apartment in Tehran",
             loc="left", color=INK, fontsize=13, pad=14)
plt.tight_layout(rect=(0, 0.07, 1, 1))
NOTE = ("Price = CBI annual mean of 12 monthly avg prices/m² (Tehran) × 75 m².\n"
        "Income = SCI HBS, weighted mean annual income of urban households (HBSIR microdata). Both nominal, same year.")
fig.text(0.01, 0.015, NOTE, color=INK2, fontsize=8.5, va="bottom")
plt.savefig("Data/chart_price_to_income.png", dpi=150, facecolor=SURF)
