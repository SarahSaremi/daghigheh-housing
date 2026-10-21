"""Young (head 25-34) urban households, 1369-1402, weighted.
- income, expenditure, housing expenditure (toman)
- housing share of expenditure (all young; young tenants)
- rent burden for young TENANTS: rent cost / income.
  rent cost (consistent): all 'rentals for housing' codes of tenant households
     <1383: 311xx (old SCI classification)   >=1383: 41xxx actual rent + 42212 rent-equivalent of deposit (رهن)
  cash rent only (>=1383): 411xx
Tenant (consistent): Rent+Other (<1383), Rent+Mortgage (>=1383)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir

def wmed(x, w):
    m = x.notna() & w.notna(); x, w = x[m].values, w[m].values
    o = np.argsort(x); c = np.cumsum(w[o]); return x[o][np.searchsorted(c, c[-1] / 2)]

rows = []
for y in range(1369, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet"); h = h[h.Weight.notna()].copy()
    h["w"] = h.Weight.astype(float); t = h.Tenure.astype(str)
    h["tenant"] = t.isin({"Rent", "Mortgage"}) if y >= 1383 else t.isin({"Rent", "Other"})
    ex = hbsir.load_table("Expenditures", y); ex = ex[ex.Table_Name == "home"]
    c = ex.Commodity_Code.astype(int)
    if y >= 1383:
        rc = ex[c.between(41000, 41999) | (c == 42212)]; cash = ex[c.between(41100, 41199)]
    else:
        rc = ex[c.between(31100, 31199)]; cash = None
    h = h.merge(rc.groupby("ID").Gross_Expenditure.sum().rename("rent_cost"), on="ID", how="left")
    h["rent_cost"] = h.rent_cost.fillna(0)
    if cash is not None:
        h = h.merge(cash.groupby("ID").Gross_Expenditure.sum().rename("cash_rent"), on="ID", how="left")
        h["cash_rent"] = h.cash_rent.fillna(0)
    for grp, d in [("all_urban", h), ("head_25_34", h[h.head_age.between(25, 34)])]:
        w = d.w; tn = d[d.tenant]
        r = dict(year=y, group=grp, n=len(d), n_tenants=len(tn),
                 income=np.average(d.Income, weights=w) / 10,
                 expenditure=np.average(d.Gross_Expenditure, weights=w) / 10,
                 housing_expenditure=np.average(d.Housing_Expenditure, weights=w) / 10,
                 housing_share_exp=(d.Housing_Expenditure * w).sum() / (d.Gross_Expenditure * w).sum(),
                 tenant_share=np.average(d.tenant, weights=w),
                 tenants_housing_share_exp=(tn.Housing_Expenditure * tn.w).sum() / (tn.Gross_Expenditure * tn.w).sum(),
                 tenants_rent_to_income=(tn.rent_cost * tn.w).sum() / (tn.Income * tn.w).sum(),
                 tenants_rent_to_income_median=wmed(tn.rent_cost / tn.Income.replace(0, np.nan), tn.w),
                 tenants_income=np.average(tn.Income, weights=tn.w) / 10,
                 # SCI adds the deposit rent-equivalent (رهن) to tenants' income too -> exclude imputed rent from income
                 tenants_rent_to_income_excl_imputed=(tn.rent_cost * tn.w).sum() / ((tn.Income - tn.Imputed_Rent) * tn.w).sum())
        if "cash_rent" in d:
            r["tenants_cash_rent_to_income"] = (tn.cash_rent * tn.w).sum() / (tn.Income * tn.w).sum()
        rows.append(r)
    print(y, flush=True)
R = pd.DataFrame(rows); R.to_csv("Data/young_households_burden.csv", index=False, encoding="utf-8-sig")

# price-to-income for young vs all (1395-1402)
m = pd.read_csv("Data/master_annual.csv")[["year", "tehran_price_per_m2"]]
pi = R.pivot(index="year", columns="group", values="income").reset_index().merge(m, on="year")
pi = pi.dropna(subset=["tehran_price_per_m2"])
pi["price_75m2"] = 75 * pi.tehran_price_per_m2
pi["pti_young_25_34"] = pi.price_75m2 / pi.head_25_34
pi["pti_all_urban"] = pi.price_75m2 / pi.all_urban
pi["young_income_vs_all"] = pi.head_25_34 / pi.all_urban
pi.to_csv("Data/indicator_price_to_income_young.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(pi.round(3).to_string())
print(R[R.group == "head_25_34"][["year", "n_tenants", "tenant_share", "housing_share_exp", "tenants_housing_share_exp",
      "tenants_rent_to_income", "tenants_rent_to_income_median", "tenants_cash_rent_to_income"]].round(3).to_string())
