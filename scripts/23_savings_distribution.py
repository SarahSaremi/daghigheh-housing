"""Critique 5: distribution behind the 'years of saving' figure for young tenant households (head 25-34).
Household-level surplus = cash income - essential spending (wide / min), same definitions as script 18.
Reports: ratio-of-means years (headline method), weighted quantiles of household surplus,
years implied by the median and 75th-percentile household surplus, share with surplus <= 0."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir
price = pd.read_csv("Data/master_annual.csv").set_index("year").tehran_price_per_m2
MIN = ["food", "cloth", "home", "medical"]; WIDE = MIN + ["transportation", "communication", "miscellaneous"]
def wq(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]; c = np.cumsum(w) / w.sum()
    return x[np.searchsorted(c, q)]
rows = []
for y in range(1395, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet"); h = h[h.Weight.notna() & h.head_age.between(25, 34)]
    h = h[h.Tenure.astype(str).isin({"Rent", "Mortgage"})].copy()
    ex = hbsir.load_table("Expenditures", y)[["ID", "Table_Name", "Commodity_Code", "Gross_Expenditure"]]
    ex = ex[ex.ID.isin(h.ID) & ~ex.Commodity_Code.fillna(0).astype(int).between(42000, 42999)]
    by = ex.pivot_table(index="ID", columns="Table_Name", values="Gross_Expenditure", aggfunc="sum", observed=True).fillna(0)
    h = h.merge(by, left_on="ID", right_index=True, how="left").fillna(0)
    for k in WIDE:
        if k not in h: h[k] = 0.0
    P75 = 75 * price[y]; w = h.Weight.astype(float)
    cash = (h.Income - h.Imputed_Rent) / 10
    for k, cols in [("wide", WIDE), ("min", MIN)]:
        s = cash - h[cols].sum(axis=1) / 10
        q = {p: wq(s, w, p) for p in (0.25, 0.5, 0.75, 0.9)}
        rows.append(dict(year=y, essentials=k, n=len(h),
                         years_ratio_of_means=P75 / np.average(s, weights=w),
                         share_surplus_le0=np.average(s <= 0, weights=w),
                         surplus_p25=q[0.25], surplus_median=q[0.5], surplus_p75=q[0.75], surplus_p90=q[0.9],
                         years_at_median=P75 / q[0.5] if q[0.5] > 0 else np.inf,
                         years_at_p75=P75 / q[0.75], years_at_p90=P75 / q[0.9],
                         surplus_share_of_income_median=wq(s / cash.replace(0, np.nan).fillna(1), w, 0.5)))
R = pd.DataFrame(rows); R.to_csv("Data/young_tenant_savings_distribution.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(R[["year", "essentials", "n", "years_ratio_of_means", "share_surplus_le0", "years_at_median", "years_at_p75", "years_at_p90",
         "surplus_share_of_income_median"]].round(2).to_string(index=False))
