"""Critique 4: price-to-income with TEHRAN young households' own income (same-city numerator and denominator).
Groups: urban households with head 25-34 in (a) Tehran county, (b) Tehran province, (c) all urban Iran.
Income = total income (as in chart 3). 95% CI from 2000 weighted bootstrap resamples of households."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir
rng = np.random.default_rng(1402)
price = pd.read_csv("Data/master_annual.csv").set_index("year").tehran_price_per_m2
rows = []
for y in range(1395, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet"); h = h[h.Weight.notna() & h.head_age.between(25, 34)].copy()
    h = hbsir.add_attribute(h, "Province"); h = hbsir.add_attribute(h, "County")
    P75 = 75 * price[y]
    for grp, d in [("tehran_county", h[h.County.astype(str) == "Tehran"]), ("tehran_province", h[h.Province.astype(str) == "Tehran"]), ("iran_urban", h)]:
        inc = d.Income.values / 10; w = d.Weight.astype(float).values
        m = np.average(inc, weights=w)
        bs = [np.average(inc[i], weights=w[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(2000))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(year=y, group=grp, n=len(d), income=m, pti=P75 / m, pti_lo=P75 / hi, pti_hi=P75 / lo))
R = pd.DataFrame(rows); R.to_csv("Data/indicator_pti_young_tehran.csv", index=False, encoding="utf-8-sig")
print(R.pivot(index="year", columns="group", values="pti").round(1).to_string())
print(R.pivot(index="year", columns="group", values="n").to_string())
t = R[R.group == "tehran_county"].set_index("year"); print(t[["pti", "pti_lo", "pti_hi"]].round(1).to_string())
