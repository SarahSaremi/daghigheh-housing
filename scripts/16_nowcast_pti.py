"""Nowcast of price-to-income for YOUNG urban households (head 25-34), 1403-1405. ESTIMATE, not observed income.
Observed: HIES young income through 1402; CBI transaction prices through 1403-05.
Income paths from 1402 HIES level:
  low  = 'other wage levels' rule applied to a worker earning 2x minimum wage in 1402
         (1403: x1.22 + 230,026 rial/day; 1404: x1.32 + 310,535; 1405: x1.45 + 519,549)
  mid  = minimum wage growth (official)
  high = minimum wage growth + mean gap (young HIES income growth - min wage growth), 1396-1402
Price (transaction-equivalent) = CBI where available, else Kilid asking / r, r in [min, max] of the overlap ratio,
mid r = overlap mean. 1405 = Farvardin..Mehr only (7 months)."""
import os
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *
# BASE=iran_urban (default; young urban households, whole country) or BASE=tehran_county (young urban households in Tehran)
BASE = os.environ.get("BASE", "iran_urban"); SUF = "" if BASE == "iran_urban" else f"_{BASE}"

wage = pd.read_csv("Data/external/minimum_wage.csv").set_index("year").monthly_min_wage_toman
R = pd.read_csv("Data/young_households_burden.csv"); Y = R[R.group == "head_25_34"].set_index("year").income
pi = pd.read_csv("Data/indicator_price_to_income_young.csv").set_index("year")
if BASE != "iran_urban":
    T = pd.read_csv("Data/indicator_pti_young_tehran.csv"); T = T[T.group == BASE].set_index("year")
    pi = pi.assign(pti_young_25_34=T.pti)                     # observed line = Tehran young households
    BASE_INCOME_1402 = T.income[1402]
else:
    BASE_INCOME_1402 = None
D = pd.read_csv("Data/monthly_1395_1405_independent.csv")

# --- income paths ---
wg = wage.pct_change()
gap = (Y.pct_change() - wg).loc[1396:1402]
w = 2 * 1769428.0; other = {}
for yr, (a, b) in {1403: (1.22, 230026), 1404: (1.32, 310535), 1405: (1.45, 519549)}.items():
    w2 = a * w + b; other[yr] = w2 / w - 1; w = w2
paths = {"low": other, "mid": {y: wg[y] for y in (1403, 1404, 1405)},
         "high": {y: wg[y] + gap.mean() for y in (1403, 1404, 1405)}}
inc = {}
for k, g in paths.items():
    v = Y[1402] if BASE_INCOME_1402 is None else BASE_INCOME_1402; inc[k] = {}
    for y in (1403, 1404, 1405):
        v = v * (1 + g[y]); inc[k][y] = v

# --- prices ---
ov = D.dropna(subset=["asking_to_transaction"]).asking_to_transaction
rs = {"r_low_price": ov.max(), "r_mid": ov.mean(), "r_high_price": ov.min()}   # dividing by smaller r -> higher price
price = {}
for y in (1403, 1404, 1405):
    d = D[D.year == y]
    for k, r in rs.items():
        p = np.where(d.transaction_m2_toman.notna(), d.transaction_m2_toman, d.asking_m2_toman / r)
        price.setdefault(k, {})[y] = np.nanmean(p)
    price.setdefault("n_months", {})[y] = int((d.asking_m2_toman.notna() | d.transaction_m2_toman.notna()).sum())

rows = []
for y in (1403, 1404, 1405):
    pti = {f"{ik}|{pk}": 75 * price[pk][y] / inc[ik][y] for ik in inc for pk in ("r_low_price", "r_mid", "r_high_price")}
    rows.append(dict(year=y, income_low=inc["low"][y], income_mid=inc["mid"][y], income_high=inc["high"][y],
                     growth_low=paths["low"][y], growth_mid=paths["mid"][y], growth_high=paths["high"][y],
                     price_m2_low=price["r_low_price"][y], price_m2_mid=price["r_mid"][y], price_m2_high=price["r_high_price"][y],
                     months_of_price_data=price["n_months"][y],
                     pti_min=min(pti.values()), pti_mid=pti["mid|r_mid"], pti_max=max(pti.values()),
                     pti_optimistic_income_mid_price=pti["high|r_mid"]))
# point-in-time: latest month (Mehr 1405) price vs 1405 annual income paths
last = D.dropna(subset=["asking_m2_toman"]).iloc[-1]
pit = {f"{ik}|{pk}": 75 * last.asking_m2_toman / r / inc[ik][1405] for ik in inc for pk, r in rs.items()}
PIT = dict(year=1405, month=int(last.month), pti_min=min(pit.values()), pti_mid=pit["mid|r_mid"], pti_max=max(pit.values()))
pd.DataFrame([PIT]).to_csv(f"Data/nowcast_pti_young_point_in_time_latest{SUF}.csv", index=False)
print("point-in-time", {k: round(v, 1) for k, v in PIT.items()})
N = pd.DataFrame(rows); N.to_csv(f"Data/nowcast_pti_young_1403_1405{SUF}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("mean gap young income vs min wage growth 1396-1402:", round(gap.mean(), 3), gap.round(3).to_dict())
print("overlap ratio:", {k: round(v, 3) for k, v in rs.items()})
print(N.round(3).to_string())

# --- chart ---
obs = pi.pti_young_25_34
fig, ax = plt.subplots(figsize=(10.5, 6.3), facecolor=SURF); style(ax)
ax.plot(obs.index, obs.values, color=BLUE, lw=2.6, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2)
xb = [1402] + list(N.year); lo = [obs[1402]] + list(N.pti_min); hi = [obs[1402]] + list(N.pti_max); md = [obs[1402]] + list(N.pti_mid)
ax.fill_between(xb, lo, hi, color=BLUE, alpha=0.13, lw=0)
ax.plot(xb, md, color=BLUE, lw=2, ls=(0, (4, 3)))
ax.axvline(1402.5, color=INK2, lw=1)
ax.text(1402.6, 37, fa("برآورد، نه درآمد مشاهده‌شده"), color=INK, fontsize=10.5, weight="bold", ha="left")
for y, v in obs.items():
    ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=10, color=INK)
for _, r in N.iterrows():
    ax.annotate(fa(f"{r.pti_min:.0f} تا {r.pti_max:.0f}"), (r.year, r.pti_max), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9.5, color=INK2)
ax.errorbar(1405.3, PIT["pti_mid"], yerr=[[PIT["pti_mid"] - PIT["pti_min"]], [PIT["pti_max"] - PIT["pti_mid"]]],
            fmt="D", color=ORANGE, ms=7, capsize=4, lw=1.6)
ax.annotate(fa("قیمت مهر ۱۴۰۵:") + "\n" + fa(f"{PIT['pti_min']:.0f} تا {PIT['pti_max']:.0f} سال"), (1405.3, PIT["pti_max"]),
            textcoords="offset points", xytext=(-4, 6), ha="center", fontsize=9, color=ORANGE, weight="bold")
ax.set_xlim(1394.6, 1405.7)
ax.set_xticks(list(obs.index) + list(N.year)); ax.set_xticklabels([fa(y) + (chr(10) + fa("(۷ ماه اول)") if y == 1405 else "") for y in list(obs.index) + list(N.year)])
ax.set_ylim(0, 40); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))
ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
title(fig, "خانه‌ی ۷۵ متری تهران معادل چند سال کل درآمد خانوار جوان" + ("" if BASE == "iran_urban" else " تهرانی") + "؟ برآورد تا ۱۴۰۵",
      "خط پر: مشاهده‌شده (طرح هزینه و درآمد). نوار: بازه‌ی برآورد با سه مسیر درآمد × بازه‌ی تبدیل قیمت آگهی به معامله")
note(fig, ["مسیرهای درآمد از سطح ۱۴۰۲: پایین = قاعده‌ی افزایش «سایر سطوح مزدی»؛ میانه = رشد حداقل مزد؛ بالا = رشد حداقل مزد + میانگین فاصله‌ی تاریخی ۱۳۹۶ تا ۱۴۰۲.",
           "قیمت ۱۴۰۳ تا مرداد معاملاتی است و بعد از آن قیمت آگهی کیلید تقسیم بر نسبت آگهی/معامله (۰٫۹۸ تا ۱٫۱۹). * ۱۴۰۵ فقط فروردین تا مهر."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig(f"Data/final_chart6_nowcast_pti_young{SUF}.png", dpi=170, facecolor=SURF); plt.close()
