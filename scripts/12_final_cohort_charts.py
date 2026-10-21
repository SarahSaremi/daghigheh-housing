"""Final report charts 1 & 2 (Persian).
1) Owner (head/spouse) share by age group, one line per birth decade; incomplete cohort dashed/hollow.
2) Share of 30-34-year-olds living as child of the household head, by birth decade (bar)."""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from fa_style import *

P = pd.read_parquet("Data/processed/persons_urban.parquet")
P = P[(P.age_group != "nan") & P.birth_decade.between(1330, 1370)]
def agg(d):
    ok = d.owner.notna()
    return pd.Series({"owner": np.average(d.own_head_or_spouse[ok], weights=d.w[ok]),
                      "child": np.average(d.child_of_head, weights=d.w), "n": len(d),
                      "bmin": d.birth_year.min(), "bmax": d.birth_year.max()})
T = P.groupby(["birth_decade", "age_group"]).apply(agg).reset_index()
T["complete"] = (T.bmin == T.birth_decade) & (T.bmax == T.birth_decade + 9)
T.to_csv("Data/chart_data_cohort_final.csv", index=False, encoding="utf-8-sig")
AGES = ["25-29", "30-34", "35-39", "40-44"]; xi = {a: i for i, a in enumerate(AGES)}
DEC = {1330: "۳۰", 1340: "۴۰", 1350: "۵۰", 1360: "۶۰", 1370: "۷۰"}

# ---------- chart 1 ----------
fig, ax = plt.subplots(figsize=(10.5, 6.6), facecolor=SURF); style(ax)
for (dec, d), c in zip(T.groupby("birth_decade"), RAMP5):
    d = d.sort_values("age_group"); x = d.age_group.map(xi).values; y = d.owner.values
    ls = (0, (5, 3)) if dec == 1370 else "-"
    ax.plot(x, y, color=c, lw=2.2, ls=ls, zorder=2)
    for xx, yy, comp in zip(x, y, d.complete):
        ax.plot(xx, yy, "o", ms=9, color=SURF if not comp else c, markeredgecolor=c if not comp else SURF,
                markeredgewidth=2, zorder=3)
    lab = f"متولدین دهه {DEC[dec]}" + (" (ناقص: ۱۳۷۰–۷۲)" if dec == 1370 else "")
    ax.annotate(fa(lab), (x[-1], y[-1]), textcoords="offset points", xytext=(12, 0), va="center",
                color=INK, fontsize=10)
    for xx, yy in zip(x, y):
        if xx == 1:
            ax.annotate(pct(yy), (xx, yy), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=9, color=INK2)
ax.set_xticks(range(4)); ax.set_xticklabels([fa(f"{a.replace('-', '–')} سالگی") for a in AGES])
ax.set_xlim(-0.3, 3.9); ax.set_ylim(0, 0.9); pct_axis(ax)
title(fig, "نسل‌های جدید در سنی مشابه کمتر صاحب‌خانه‌اند",
      "سهم افراد شهری که سرپرست یا همسر سرپرستِ خانواری‌اند که در ملک خودش زندگی می‌کند")
note(fig, ["منبع: ریزداده‌های طرح هزینه و درآمد خانوار مرکز آمار ۱۳۶۹–۱۴۰۲، مناطق شهری، وزن‌دار. سال تولد = سال آمارگیری − سن (±۱ سال).",
           "دایره‌ی توخالی = همه‌ی متولدین آن دهه هنوز به این سن نرسیده‌اند؛ خط‌چین = دهه‌ی ۷۰ که فقط متولدین ۱۳۷۰ تا ۱۳۷۲ را دارد."])
plt.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart1_cohort_ownership.png", dpi=170, facecolor=SURF); plt.close()

# ---------- chart 2 ----------
C = T[T.age_group == "30-34"].sort_values("birth_decade")
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURF); style(ax)
x = np.arange(len(C))
for i, (dec, v, comp, c) in enumerate(zip(C.birth_decade, C.child, C.complete, RAMP5)):
    ax.bar(i, v, width=0.62, color=c, hatch="//" if dec == 1370 else None, edgecolor=SURF if dec != 1370 else "#ffffff", lw=0)
    ax.annotate(pct(v), (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=12, color=INK, weight="bold")
ax.set_xticks(x); ax.set_xticklabels([fa(f"متولدین دهه {DEC[d]}") + (chr(10) + fa("(۱۳۷۰ تا ۱۳۷۲)") if d == 1370 else "") for d in C.birth_decade])
ax.set_ylim(0, 0.55); pct_axis(ax); ax.grid(axis="y", color=GRID)
title(fig, "در ۳۰ تا ۳۴ سالگی، هر نسل بیشتر از قبلی هنوز در خانه‌ی پدر و مادر است",
      "سهم افراد ۳۰ تا ۳۴ ساله‌ی شهری که «فرزندِ سرپرست خانوار» ثبت شده‌اند")
note(fig, ["منبع: ریزداده‌های طرح هزینه و درآمد خانوار مرکز آمار، مناطق شهری، وزن‌دار.",
           "* دهه‌ی ۷۰ فقط متولدین ۱۳۷۰ تا ۱۳۷۲ (مشاهده‌شده در ۱۴۰۰–۱۴۰۲)."])
plt.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.08)
plt.savefig("Data/final_chart2_living_with_parents.png", dpi=170, facecolor=SURF); plt.close()
print(C[["birth_decade", "child", "owner", "n", "bmin", "bmax"]].round(3).to_string())
