"""Chart 8: years for a young tenant household (head 25-34) to buy a 75 m2 Tehran apartment
if it saved all cash income except essential spending. Observed 1395-1402, estimate band 1403-1405."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *
S = pd.read_csv("Data/young_savings_capacity.csv"); S = S[(S.group == "young_tenants") & S.year.between(1395, 1402)].set_index("year")
O = pd.read_csv("Data/nowcast_years_saving_young.csv"); O = O[O.group == "young_tenants"]
band = O.groupby("year").agg(lo=("years_save_wide", "min"), hi=("years_save_wide", "max"))
mid = O[(O.income_path == "mid") & (O.price_case == "chained")].set_index("year")
pit = O[O.year == 1405]; pit_mid = mid.loc[1405, "years_save_wide_mehr1405"]
GRAY = "#8a8984"
fig, ax = plt.subplots(figsize=(10.5, 6.4), facecolor=SURF); style(ax)
ax.plot(S.index, S.years_save_min, color=GRAY, lw=1.6, marker="o", ms=5, markeredgecolor=SURF)
ax.plot(S.index, S.years_save_wide, color=ORANGE, lw=2.6, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2)
xb = [1402] + list(band.index)
ax.fill_between(xb, [S.years_save_wide[1402]] + list(band.lo), [S.years_save_wide[1402]] + list(np.minimum(band.hi, 240)), color=ORANGE, alpha=0.13, lw=0)
ax.plot(xb, [S.years_save_wide[1402]] + list(mid.years_save_wide), color=ORANGE, lw=2, ls=(0, (4, 3)))
ax.plot(xb, [S.years_save_min[1402]] + list(mid.years_save_min), color=GRAY, lw=1.4, ls=(0, (4, 3)))
for y, v in S.years_save_wide.items():
    ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=INK, weight="bold")
for y, v in mid.years_save_wide.items():
    ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=INK)
ax.errorbar(1405.35, pit_mid, yerr=[[pit_mid - pit.years_save_wide_mehr1405.min()], [min(pit.years_save_wide_mehr1405.max(), 240) - pit_mid]],
            fmt="D", color=ORANGE, ms=7, capsize=4, lw=1.4)
ax.annotate(fa(f"با قیمت مهر ۱۴۰۵: {pit_mid:.0f}"), (1405.35, pit_mid), textcoords="offset points", xytext=(-10, 0),
            ha="right", va="center", fontsize=9.5, color=ORANGE, weight="bold")
ax.axvline(1402.5, color=INK2, lw=1)
ax.text(1402.6, 228, fa("برآورد"), color=INK, fontsize=10.5, weight="bold", ha="left")
ax.text(1395, 118, fa("کسر خرج‌های ضروری (گسترده)"), color=ORANGE, fontsize=10, weight="bold", ha="left")
ax.text(1395, S.years_save_min[1395] - 16, fa("کسر فقط خوراک، پوشاک، مسکن و درمان"), color=INK2, fontsize=9.5, ha="left")
ax.set_ylim(0, 240); ax.set_xlim(1394.6, 1405.7)
xs = list(S.index) + [1403, 1404, 1405]
ax.set_xticks(xs); ax.set_xticklabels([fa(y) + (chr(10) + fa("(۷ ماه اول)") if y == 1405 else "") for y in xs])
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))
ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
title(fig, "اگر مستأجر جوان فقط خرج ضروری کند، چند سال تا یک خانه‌ی ۷۵ متری در تهران؟",
      "همه‌ی درآمد به‌جز خرج‌های ضروری پس‌انداز می‌شود؛ خانوارهای مستأجر شهری با سرپرست ۲۵ تا ۳۴ ساله")
note(fig, ["ضروری گسترده = خوراک، پوشاک، مسکن (اجاره و قبوض)، درمان، حمل‌ونقل، ارتباطات و بهداشت شخصی. درآمد و هزینه نقدی (بدون اجاره‌ی منسوب). مشاهده‌شده: طرح هزینه و درآمد مرکز آمار.",
           "برآورد ۱۴۰۳ تا ۱۴۰۵: خرج ضروری با تورم سالانه‌ی مرکز آمار (۳۲٫۵، ۵۰٫۶ و ۷۳٫۶ درصد) و درآمد با سه مسیر مزدی رشد می‌کند. نوار = بازه‌ی سناریوها (بالای ۲۴۰ بریده شده).",
           "قیمت پس از مرداد ۱۴۰۳ = قیمت معاملاتی زنجیره‌شده با رشد قیمت آگهی کیلید. * ۱۴۰۵ = میانگین فروردین تا مهر. فرض ساده: پس‌انداز سود ندارد و قیمت خانه در طول پس‌انداز ثابت می‌ماند."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart8_years_saving_young_tenants.png", dpi=170, facecolor=SURF); plt.close()
print("ok")
