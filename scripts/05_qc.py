"""Quality control: one time-series chart per variable + automatic flags for
missing years, abnormal jumps, rial/toman (x10) breaks and definitional breaks."""
import os, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("Data/qc", exist_ok=True)
ext = pd.read_csv("Data/master_annual_extended.csv")
mon = pd.read_csv("Data/processed/cbi_tehran_price_monthly.csv")
master = pd.read_csv("Data/master_annual.csv")

vars_ = {"household_income": ("income_rial", 0.1), "household_expenditure": ("expenditure_rial", 0.1),
         "housing_expenditure": ("housing_exp_rial", 0.1), "imputed_rent": ("imputed_rent_rial", 0.1),
         "household_size": ("household_size", 1), "head_age": ("head_age", 1),
         "owner_share": ("owner_share", 1), "tenant_share": ("tenant_share", 1),
         "n_households_urban (sample)": ("n_households_urban", 1)}
flags = []
fig, axes = plt.subplots(3, 4, figsize=(20, 12)); axes = axes.ravel()
for ax, (name, (col, k)) in zip(axes, vars_.items()):
    s = ext.set_index("year")[col] * k
    ax.plot(s.index, s.values, marker="o", ms=3)
    if "rial" in col: ax.set_yscale("log"); ax.set_title(name + " (toman, log)")
    else: ax.set_title(name)
    for y in s[s.isna()].index: flags.append((name, y, "missing"))
    g = s.pct_change(fill_method=None)
    for y, v in g.items():
        if pd.isna(v): continue
        if "rial" in col and (v > 0.6 or v < -0.05): flags.append((name, y, f"jump {v:+.0%}"))
        if "rial" in col and (v > 5 or v < -0.8): flags.append((name, y, f"POSSIBLE x10 UNIT BREAK {v:+.0%}"))
        if "rial" not in col and col != "n_households_urban" and abs(v) > 0.1: flags.append((name, y, f"jump {v:+.0%}"))
        if col == "n_households_urban" and abs(v) > 0.35: flags.append((name, y, f"sample size change {v:+.0%}"))
    ax.grid(alpha=.3)
# housing share of expenditure & income/expenditure ratio (scale-free; unit errors show up here)
ax = axes[9]; r = ext.housing_exp_rial / ext.expenditure_rial
ax.plot(ext.year, r, marker="o", ms=3); ax.set_title("housing share of expenditure"); ax.grid(alpha=.3)
ax = axes[10]; r2 = ext.income_rial / ext.expenditure_rial
ax.plot(ext.year, r2, marker="o", ms=3); ax.set_title("income / expenditure"); ax.grid(alpha=.3)
for y, v in r2.items():
    if pd.notna(v) and not 0.7 < v < 1.4: flags.append(("income/expenditure", ext.year[y], f"ratio {v:.2f}"))
ax = axes[11]
mm = mon.assign(t=mon.year + (mon.month - 1) / 12)
ax.plot(mm.t, mm.price_million_rial_m2 * 1e5, marker=".", lw=1); ax.set_yscale("log")
ax.set_title("CBI Tehran price/m2, monthly (toman, log)"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig("Data/qc/qc_timeseries_all.png", dpi=110); plt.close()

# monthly price checks: gaps and month-on-month jumps
full = pd.MultiIndex.from_product([range(mon.year.min(), mon.year.max() + 1), range(1, 13)])
have = set(zip(mon.year, mon.month))
for y, m in full:
    if (y, m) not in have and (y, m) <= (1403, 5): flags.append(("tehran_price_monthly", f"{y}-{m:02d}", "missing month"))
mm = mm.sort_values("t"); mom = mm.price_million_rial_m2.pct_change()
for (y, m), v, gap in zip(zip(mm.year, mm.month), mom, mm.t.diff()):
    if pd.notna(v) and abs(gap - 1/12) < 1e-6 and (v > 0.15 or v < -0.08): flags.append(("tehran_price_monthly", f"{y}-{m:02d}", f"MoM {v:+.1%}"))
    if pd.notna(v) and (v > 5 or v < -0.8): flags.append(("tehran_price_monthly", f"{y}-{m:02d}", "POSSIBLE UNIT BREAK"))
for _, r in mon[mon.max_rel_diff_across_reports > 0.005].iterrows():
    flags.append(("tehran_price_monthly", f"{r.year}-{r.month:02d}", f"reports disagree by {r.max_rel_diff_across_reports:.1%}"))

# external anchors: SCI published urban averages (thousand rials)
anchors = [(1396, "income_rial", 366947e3, "SCI 1396 urban avg income"),
           (1400, "income_rial", 1124210e3, "SCI 1400 urban avg income (112,421 thousand toman)")]
for y, col, val, lab in anchors:
    ours = ext.loc[ext.year == y, col].iloc[0]
    flags.append(("anchor_check", y, f"{lab}: ours/published = {ours/val:.4f}"))
pd.DataFrame(flags, columns=["variable", "year", "flag"]).to_csv("Data/qc/qc_flags.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(flags, columns=["variable", "year", "flag"]).to_string())
