"""Build the annual master table (one row per year, 1363-1402).
Units: CURRENT TOMAN (nominal, NOT deflated). HBS microdata are in rials -> /10.
CBI price is million rials/m2 -> *1e6/10 toman/m2. Annual price = simple mean of 12 monthly values."""
import pandas as pd, numpy as np

hbs = pd.read_csv("Data/processed/hbs_urban_aggregates_1369_1402.csv")
years = pd.DataFrame({"year": range(1363, 1403)})

# --- master (exactly the requested initial columns) ---
h = hbs.assign(household_income=hbs.income_rial / 10,
               household_expenditure=hbs.expenditure_rial / 10,
               housing_expenditure=hbs.housing_exp_rial / 10)
master = years.merge(h[["year", "household_income", "household_expenditure", "housing_expenditure"]],
                     on="year", how="left")

# --- CBI Tehran price: annual mean of monthly values (only if all 12 months exist) ---
m = pd.read_csv("Data/processed/cbi_tehran_price_monthly.csv")
ann = m.groupby("year").agg(n_months=("month", "nunique"),
                            price_m2_mrial_mean=("price_million_rial_m2", "mean")).reset_index()
ann["tehran_price_per_m2"] = np.where(ann.n_months == 12, ann.price_m2_mrial_mean * 1e6 / 10, np.nan)
master = master.merge(ann[["year", "tehran_price_per_m2"]], on="year", how="left")
master.to_csv("Data/master_annual.csv", index=False, encoding="utf-8-sig")

# --- auxiliary table: extra HBS variables + provenance flags ---
aux = years.merge(hbs, on="year", how="left")
aux["owner_share"] = aux[["tenure_share__Owned_Estate_Land", "tenure_share__Owned_Estate"]].sum(axis=1, min_count=1)
aux["tenant_share"] = aux[["tenure_share__Rent", "tenure_share__Mortgage"]].sum(axis=1, min_count=1)
aux["imputed_rent_toman"] = aux.imputed_rent_rial / 10
aux["income_median_toman"] = aux.income_median_rial / 10
aux = aux.merge(ann.rename(columns={"n_months": "price_n_months"}), on="year", how="left")
aux["hbs_source"] = np.where(aux.income_rial.notna(), "HBSIR cleaned microdata, weighted urban mean",
                             "MISSING: HBSIR has no normalized income/weights before 1369")
aux.to_csv("Data/master_annual_extended.csv", index=False, encoding="utf-8-sig")
print(master.to_string())
