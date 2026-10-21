"""Years to buy a 75 m2 Tehran apartment if a YOUNG household (head 25-34) saved everything except essentials.
Cash basis: imputed rent removed from income (Imputed_Rent) and from expenditure (codes 42xxx).
essential_min  = food + cloth + home(cash: rent, deposit-equivalent excluded, utilities) + medical
essential_wide = essential_min + transportation + communication + miscellaneous (personal care etc.)
excluded: entertainment, hotel/restaurant, durable (cars, appliances), tobacco.
years = 75 x annual mean Tehran price/m2 / (mean cash income - mean essential spending)  [ratio of means]"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir
price = pd.read_csv("Data/master_annual.csv").set_index("year").tehran_price_per_m2
MIN = ["food", "cloth", "home", "medical"]; WIDE = MIN + ["transportation", "communication", "miscellaneous"]
rows = []
for y in range(1390, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet"); h = h[h.Weight.notna() & h.head_age.between(25, 34)].copy()
    t = h.Tenure.astype(str); h["tenant"] = t.isin({"Rent", "Mortgage"})
    ex = hbsir.load_table("Expenditures", y)[["ID", "Table_Name", "Commodity_Code", "Gross_Expenditure"]]
    ex = ex[ex.ID.isin(h.ID)]
    c = ex.Commodity_Code.astype(int)
    ex = ex[~c.between(42000, 42999)]                      # drop imputed rents (owner & deposit-equivalent)
    by = ex.pivot_table(index="ID", columns="Table_Name", values="Gross_Expenditure", aggfunc="sum", observed=True).fillna(0)
    h = h.merge(by, left_on="ID", right_index=True, how="left").fillna({k: 0 for k in by.columns})
    for k in WIDE:
        if k not in h: h[k] = 0.0
    h["cash_income"] = h.Income - h.Imputed_Rent
    h["ess_min"] = h[MIN].sum(axis=1); h["ess_wide"] = h[WIDE].sum(axis=1)
    h["cash_exp_total"] = h[[k for k in by.columns]].sum(axis=1)
    for grp, d in [("young_all", h), ("young_tenants", h[h.tenant])]:
        w = d.Weight.astype(float); m = lambda s: np.average(s, weights=w) / 10    # toman
        r = dict(year=y, group=grp, n=len(d), cash_income=m(d.cash_income), ess_min=m(d.ess_min), ess_wide=m(d.ess_wide),
                 cash_exp_total=m(d.cash_exp_total))
        r["surplus_min"] = r["cash_income"] - r["ess_min"]; r["surplus_wide"] = r["cash_income"] - r["ess_wide"]
        r["surplus_actual"] = r["cash_income"] - r["cash_exp_total"]
        r["share_hh_surplus_wide_pos"] = np.average((d.cash_income - d.ess_wide) > 0, weights=w)
        r["ess_wide_share_of_income"] = r["ess_wide"] / r["cash_income"]
        if y in price.dropna().index:
            P = 75 * price[y]
            r.update(price75=P, years_gross_income=P / r["cash_income"], years_save_min=P / r["surplus_min"],
                     years_save_wide=P / r["surplus_wide"],
                     years_save_actual=P / r["surplus_actual"] if r["surplus_actual"] > 0 else np.inf)
        rows.append(r)
    print(y, flush=True)
S = pd.DataFrame(rows); S.to_csv("Data/young_savings_capacity.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
for g in ["young_all", "young_tenants"]:
    print("\n==", g)
    print(S[S.group == g][["year", "n", "cash_income", "ess_min", "ess_wide", "ess_wide_share_of_income", "share_hh_surplus_wide_pos",
         "years_gross_income", "years_save_min", "years_save_wide", "years_save_actual"]].round(2).to_string(index=False))
