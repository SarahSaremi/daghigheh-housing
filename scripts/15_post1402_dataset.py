"""Independent monthly dataset (NOT spliced into HIES): 1395-01 .. 1405-07.
transaction price: CBI (to 1403-05). asking price: Kilid listings (1402-08 ..), snapshot 1405-07-15.
min wage: official base wage (monthly = daily x 30), applies to the whole Solar Hijri year.
Charts: A) overlap asking vs transaction + ratio; B) one m2 = how many months of minimum wage."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *

cbi = pd.read_csv("Data/processed/cbi_tehran_price_monthly.csv")
cbi = cbi.assign(transaction_m2_toman=cbi.price_million_rial_m2 * 1e5)[["year", "month", "transaction_m2_toman"]]
kil = pd.read_csv("Data/external/kilid_tehran_asking_price_snapshot_14050715.csv")
kil["asking_m2_toman"] = kil.asking_price_m2_million_toman * 1e6
wage = pd.read_csv("Data/external/minimum_wage.csv")[["year", "monthly_min_wage_toman"]]
grid = pd.DataFrame([(y, m) for y in range(1395, 1406) for m in range(1, 13) if (y, m) <= (1405, 7)], columns=["year", "month"])
D = grid.merge(cbi, how="left").merge(kil[["year", "month", "asking_m2_toman"]], how="left").merge(wage, how="left")
D["asking_to_transaction"] = D.asking_m2_toman / D.transaction_m2_toman
D["m2_in_months_min_wage_transaction"] = D.transaction_m2_toman / D.monthly_min_wage_toman
D["m2_in_months_min_wage_asking"] = D.asking_m2_toman / D.monthly_min_wage_toman
D["home75_in_years_min_wage_transaction"] = 75 * D.transaction_m2_toman / (12 * D.monthly_min_wage_toman)
D["home75_in_years_min_wage_asking"] = 75 * D.asking_m2_toman / (12 * D.monthly_min_wage_toman)
D["price_type_note"] = np.where(D.transaction_m2_toman.notna(), "transaction (CBI)", np.where(D.asking_m2_toman.notna(), "asking only (Kilid)", ""))
D.to_csv("Data/monthly_1395_1405_independent.csv", index=False, encoding="utf-8-sig")
D["t"] = D.year + (D.month - 1) / 12
ov = D.dropna(subset=["asking_to_transaction"])
print(ov[["year", "month", "asking_to_transaction"]].round(3).to_string())

MONTHS = "فروردین اردیبهشت خرداد تیر مرداد شهریور مهر آبان آذر دی بهمن اسفند".split()
def ticks(ax, years):
    ax.set_xticks([y for y in years]); ax.set_xticklabels([fa(y) for y in years])

# ---------- A: overlap ----------
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), facecolor=SURF, gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
style(ax); style(ax2)
w = D[(D.t >= 1402.5) & (D.t <= 1405.6)]
ax.plot(w.t, w.transaction_m2_toman / 1e6, color=BLUE, lw=2.4, marker="o", ms=5, markeredgecolor=SURF)
ax.plot(w.t, w.asking_m2_toman / 1e6, color=ORANGE, lw=2.4, marker="o", ms=5, markeredgecolor=SURF)
ax.set_ylim(55, 235)
ax.text(1402.62, 64, fa("قیمت معاملاتی (بانک مرکزی)"), color=BLUE, fontsize=10, weight="bold", ha="left")
ax.text(1403.9, 120, fa("قیمت آگهی (کیلید)"), color=ORANGE, fontsize=10, weight="bold", ha="left")
ax.axvline(1403 + 4.5 / 12, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(1403 + 4.7 / 12, 200, fa("آخرین گزارش بانک مرکزی (مرداد ۱۴۰۳)"), color=INK2, fontsize=9, ha="left")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))
ax.set_ylabel(fa("میلیون تومان / متر"), color=INK2)
ax2.plot(ov.t, ov.asking_to_transaction, color=INK, lw=2, marker="o", ms=5, markeredgecolor=SURF)
ax2.axhline(1, color=GRID, lw=1.2)
for _, r in ov.iloc[[0, -1]].iterrows():
    ax2.annotate(fa(f"{r.asking_to_transaction:.2f}"), (r.t, r.asking_to_transaction), textcoords="offset points",
                 xytext=(0, 9), ha="center", fontsize=10, color=INK, weight="bold")
ax2.set_ylim(0.85, 1.3); ax2.set_ylabel(fa("آگهی ÷ معامله"), color=INK2)
ax2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.1f}")))
ticks(ax2, [1403, 1404, 1405])
title(fig, "قیمت آگهی و قیمت معامله هم‌جهت حرکت نکردند",
      "در ۱۰ ماه همپوشانی، نسبت قیمت آگهی به معامله پیوسته از حدود ۱٫۱۶ به ۰٫۹۸ رسید؛ پس دو سری را به هم وصل نمی‌کنیم")
note(fig, ["قیمت معاملاتی: میانگین قیمت هر متر مربع واحدهای معامله‌شده در تهران، بانک مرکزی. قیمت آگهی: میانگین قیمت هر متر آگهی‌های فروش تهران در کیلید (برداشت مهر ۱۴۰۵).",
           "روش محاسبه‌ی کیلید منتشر نشده و ممکن است هموارسازی شده باشد."])
plt.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.08, hspace=0.12)
plt.savefig("Data/chart_asking_vs_transaction_overlap.png", dpi=170, facecolor=SURF); plt.close()

# ---------- B: m2 in months of minimum wage ----------
fig, ax = plt.subplots(figsize=(11, 6.3), facecolor=SURF); style(ax)
tx = D.dropna(subset=["m2_in_months_min_wage_transaction"]); ak = D[(D.t > 1403 + 4.5 / 12)].dropna(subset=["m2_in_months_min_wage_asking"])
ax.plot(tx.t, tx.m2_in_months_min_wage_transaction, color=BLUE, lw=2.4)
ax.plot(ak.t, ak.m2_in_months_min_wage_asking, color=ORANGE, lw=2.4)
cut = 1403 + 4.5 / 12
ax.axvline(cut, color=INK, lw=1.2)
ax.text(cut + 0.08, 19.2, fa("از اینجا به بعد: قیمت آگهی، نه معامله"), color=INK, fontsize=10, weight="bold", ha="left")
for y in range(1396, 1406):
    ax.axvline(y, color=GRID, lw=0.8, ls=":")
ax.text(1405.02, 4.2, fa("افزایش سالانه‌ی حداقل مزد\n(هر فروردین)"), color=INK2, fontsize=8, ha="left")
for r, c in [(tx.iloc[0], BLUE), (tx.loc[tx.m2_in_months_min_wage_transaction.idxmax()], BLUE), (ak.iloc[-1], ORANGE)]:
    ax.annotate(fa(f"{r.m2_in_months_min_wage_transaction if c == BLUE else r.m2_in_months_min_wage_asking:.1f} ماه"),
                (r.t, r.m2_in_months_min_wage_transaction if c == BLUE else r.m2_in_months_min_wage_asking),
                textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10.5, color=INK, weight="bold")
ax.set_ylim(0, 21); ax.set_yticks(range(0, 21, 4)); ticks(ax, range(1395, 1406))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))
ax.set_ylabel(fa("ماه حداقل مزد"), color=INK2)
title(fig, "یک متر خانه در تهران = چند ماه حداقل مزد؟",
      "قیمت هر متر مربع ÷ حداقل مزد پایه‌ی ماهانه‌ی همان سال (بدون حق مسکن و بن)")
note(fig, ["آبی: قیمت معاملاتی بانک مرکزی تا مرداد ۱۴۰۳. نارنجی: قیمت آگهی کیلید از شهریور ۱۴۰۳ تا مهر ۱۴۰۵ (معمولاً با قیمت معامله فرق دارد).",
           "حداقل مزد: مصوبه‌ی شورای عالی کار، مزد روزانه × ۳۰. اره‌ای‌بودن نمودار به‌خاطر ثابت‌ماندن مزد در طول سال است."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.85, bottom=0.08)
plt.savefig("Data/final_chart5_m2_in_min_wage_months.png", dpi=170, facecolor=SURF); plt.close()
print(D[D.month.isin([1, 7])][["year", "month", "m2_in_months_min_wage_transaction", "m2_in_months_min_wage_asking",
      "home75_in_years_min_wage_transaction", "home75_in_years_min_wage_asking"]].round(1).to_string())
