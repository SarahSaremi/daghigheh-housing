"""Housing loan affordability (Tehran, purchase loan only; jaaleh kept separate as sensitivity).
coverage = ceiling / (75 m2 x price/m2 in the policy month)
installment share = monthly annuity payment / monthly income of young (25-34 head) urban household
Price: CBI transaction where available; else Kilid asking / r (r = overlap mean, range [min,max]).
Income: HIES observed for the policy year (<=1402); nowcast mid (low/high range) for 1403+ (ESTIMATE)."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *
L = pd.read_csv("Data/external/housing_loan_policy_tehran.csv")
D = pd.read_csv("Data/monthly_1395_1405_independent.csv")
ov = D.dropna(subset=["asking_to_transaction"]).asking_to_transaction
r_mid, r_lo, r_hi = ov.mean(), ov.min(), ov.max()
R = pd.read_csv("Data/young_households_burden.csv"); Y = R[R.group == "head_25_34"].set_index("year").income
N = pd.read_csv("Data/nowcast_pti_young_1403_1405.csv").set_index("year")
def annuity(P, rate, years):
    r = rate / 12; n = years * 12; return P * r / (1 - (1 + r) ** -n)
rows = []
for _, p in L.iterrows():
    d = D[(D.year == p.year) & (D.month == p.month)].iloc[0]
    if pd.notna(d.transaction_m2_toman):
        pm = plo = phi = d.transaction_m2_toman; ptype = "transaction (CBI)"
    else:
        pm, plo, phi = d.asking_m2_toman / r_mid, d.asking_m2_toman / r_hi, d.asking_m2_toman / r_lo; ptype = "asking (Kilid) / ratio"
    home = 75 * pm
    row = dict(effective_date=p.effective_date, price_type=ptype, home75_toman=home,
               cov_individual=p.individual_ceiling_toman / home, cov_couple=p.couple_ceiling_toman / home,
               cov_couple_plus_jaaleh=(p.couple_ceiling_toman + p.jaaleh_ceiling_toman) / home,
               cov_couple_low=p.couple_ceiling_toman / (75 * phi), cov_couple_high=p.couple_ceiling_toman / (75 * plo))
    if pd.notna(p.interest_rate):
        inst = annuity(p.couple_ceiling_toman, p.interest_rate, p.repayment_years)
        if p.year <= 1402:
            inc_m = Y[p.year] / 12; inc_type = "HIES observed"; inc_lo = inc_hi = inc_m
        else:
            inc_m = N.loc[p.year, "income_mid"] / 12; inc_lo = N.loc[p.year, "income_low"] / 12; inc_hi = N.loc[p.year, "income_high"] / 12; inc_type = "ESTIMATE (nowcast)"
        row.update(monthly_installment_couple=inst, young_monthly_income=inc_m, income_type=inc_type,
                   installment_to_income=inst / inc_m, installment_to_income_range=f"{inst / inc_hi:.2f}-{inst / inc_lo:.2f}")
    rows.append(row)
O = pd.DataFrame(rows); O.to_csv("Data/indicator_loans_tehran.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(O.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURF); style(ax)
x = np.arange(len(O))
ax.bar(x, O.cov_couple, width=0.55, color=BLUE)
for i, r in O.iterrows():
    if r.price_type.startswith("asking"):
        ax.errorbar(i, r.cov_couple, yerr=[[r.cov_couple - r.cov_couple_low], [r.cov_couple_high - r.cov_couple]], color=INK2, capsize=4, lw=1.3)
    ax.annotate(pct(r.cov_couple), (i, r.cov_couple_high if r.price_type.startswith("asking") else r.cov_couple),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=12, weight="bold", color=INK)
    ax.text(i, 0.008, fa(f"وام فردی: {r.cov_individual * 100:.0f}%"), ha="center", fontsize=9.5, color="#ffffff", weight="bold")
MON = "فروردین اردیبهشت خرداد تیر مرداد شهریور مهر آبان آذر دی بهمن اسفند".split()
ax.set_xticks(x); ax.set_xticklabels([fa(f"{MON[int(d.split('-')[1]) - 1]} {d.split('-')[0]}") for d in O.effective_date])
ax.set_ylim(0, 0.25); pct_axis(ax)
title(fig, "وام خرید زوجین چند درصد قیمت یک آپارتمان ۷۵ متری تهران را پوشش می‌داد؟",
      "سقف وام خرید (از محل اوراق حق تقدم بانک مسکن) در ماه تصویب، بدون جعاله")
note(fig, ["قیمت: میانگین قیمت معاملاتی هر متر در تهران (بانک مرکزی) در همان ماه؛ برای ۱۴۰۴ و ۱۴۰۵ قیمت آگهی کیلید تقسیم بر نسبت آگهی/معامله (خط خطا = بازه‌ی ۰٫۹۸ تا ۱٫۱۹).",
           "سقف‌ها از اخبار مصوبات شورای پول و اعتبار/بانک مرکزی؛ سال‌های پیش از ۱۳۹۸ هنوز با منبع معتبر تأیید نشده‌اند."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart7_loan_coverage.png", dpi=170, facecolor=SURF); plt.close()
