"""Chart 3: years of young-household income to buy a 75 m2 Tehran apartment (1395-1402), vs all urban.
Chart 4: housing cost pressure on young (head 25-34) TENANT households, 1369-1402."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *
GRAY = "#8a8984"

# ---------- chart 3 ----------
pi = pd.read_csv("Data/indicator_price_to_income_young.csv")
fig, ax = plt.subplots(figsize=(10, 6.2), facecolor=SURF); style(ax)
ax.plot(pi.year, pi.pti_all_urban, color=GRAY, lw=1.6, marker="o", ms=6, markeredgecolor=SURF, markeredgewidth=1.5)
ax.plot(pi.year, pi.pti_young_25_34, color=BLUE, lw=2.6, marker="o", ms=9, markeredgecolor=SURF, markeredgewidth=2)
for _, r in pi.iterrows():
    ax.annotate(fa(f"{r.pti_young_25_34:.1f}"), (r.year, r.pti_young_25_34), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=10, color=INK, weight="bold")
first, last = pi.iloc[0], pi.iloc[-1]
ax.text(first.year - 0.1, 27, fa("خانوار جوان (سرپرست ۲۵ تا ۳۴ ساله)"), ha="left", color=BLUE, fontsize=10.5, weight="bold")
ax.annotate(fa(f"متوسط کل خانوارهای شهری (۱۴۰۲: {last.pti_all_urban:.1f} سال)"), (first.year, first.pti_all_urban),
            textcoords="offset points", xytext=(-8, -22), ha="left", color=INK2, fontsize=9.5)
ax.set_xticks(pi.year); ax.set_xticklabels([fa(y) for y in pi.year]); ax.set_ylim(0, 34)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))
ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
title(fig, "قیمت یک آپارتمان ۷۵ متری در تهران معادل چند سال کل درآمد یک خانوار جوان بود؟",
      "این عدد مدت پس‌انداز نیست: یعنی اگر خانوار هیچ خرجی نداشت و همه‌ی درآمدش را کنار می‌گذاشت")
note(fig, ["قیمت: میانگین ۱۲ ماهه‌ی قیمت معاملاتی هر متر مربع در تهران (بانک مرکزی) × ۷۵. درآمد: میانگین وزنی درآمد سالانه‌ی خانوارهای شهری کل کشور",
           "با سرپرست ۲۵ تا ۳۴ ساله (طرح هزینه و درآمد مرکز آمار). هر دو اسمی و هم‌سال. درآمد شامل اجاره‌ی منسوب و یارانه است."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart3_pti_young.png", dpi=170, facecolor=SURF); plt.close()

# ---------- chart 4 ----------
R = pd.read_csv("Data/young_households_burden.csv"); R = R[R.group == "head_25_34"]
fig, ax = plt.subplots(figsize=(11, 6.2), facecolor=SURF); style(ax)
for f in FRAMES:
    ax.axvline(f - 0.5, color=GRID, lw=1.5, ls=(0, (4, 3)))
    ax.text(f - 0.35, 0.01, fa("چارچوب نمونه‌ی جدید"), color=INK2, fontsize=8, rotation=90, va="bottom")
ax.axvline(1382.5, color=GRID, lw=1.5, ls=":")
ax.text(1382.65, 0.01, fa("تغییر طبقه‌بندی هزینه"), color=INK2, fontsize=8, rotation=90, va="bottom")
S = [("tenants_rent_to_income_excl_imputed", ORANGE, "اجاره (شامل معادل اجاره‌ی رهن) ÷ درآمد", "-"),
     ("tenants_housing_share_exp", BLUE, "سهم مسکن از کل هزینه", "-"),
     ("tenants_cash_rent_to_income", GRAY, "فقط اجاره‌ی نقدی ÷ درآمد", (0, (4, 2)))]
for col, c, lab, ls in S:
    d = R.dropna(subset=[col])
    ax.plot(d.year, d[col], color=c, lw=2.2 if ls == "-" else 1.6, ls=ls, marker="o" if ls == "-" else None, ms=4.5,
            markeredgecolor=SURF, markeredgewidth=1)
    pos = {"tenants_rent_to_income_excl_imputed": (1369, 0.415), "tenants_housing_share_exp": (1369, 0.20),
           "tenants_cash_rent_to_income": (1384.3, 0.115)}[col]
    ax.text(*pos, fa(lab), ha="left", color=c if c != GRAY else INK2, fontsize=9.5, weight="bold")
    ax.annotate(pct(d[col].iloc[-1]), (d.year.iloc[-1], d[col].iloc[-1]), textcoords="offset points",
                xytext=(8, 0), ha="left", va="center", color=INK, fontsize=9.5)
ax.set_ylim(0, 0.6); pct_axis(ax); ax.set_xlim(1368, 1404.5)
ax.set_xticks(range(1369, 1403, 3)); ax.set_xticklabels([fa(y) for y in range(1369, 1403, 3)])
title(fig, "پیش از هر پس‌اندازی، حدود یک‌سوم تا دوپنجم درآمد مستأجر جوان صرف سقف بالای سرش می‌شود",
      "خانوارهای مستأجر شهری با سرپرست ۲۵ تا ۳۴ ساله")
note(fig, ["منبع: ریزداده‌های طرح هزینه و درآمد مرکز آمار، وزن‌دار. مستأجر = اجاره + رهن (تا ۱۳۸۲: اجاره + «سایر»).",
           "درآمد بدون اجاره‌ی منسوب (مرکز آمار معادل اجاره‌ی رهن را به درآمد هم اضافه می‌کند). اجاره‌ی نقدی جداگانه فقط از ۱۳۸۳ قابل استخراج است.",
           "بین دو چارچوب نمونه، نمونه پیر می‌شود؛ تغییرات سال‌به‌سال را با احتیاط بخوانید."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart4_young_tenant_burden.png", dpi=170, facecolor=SURF); plt.close()
print("ok")
