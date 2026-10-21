"""Build weighted URBAN household averages per year from HBSIR cleaned microdata.
Raw/cleaned microdata stay untouched in Data/HBSIR/. Output values are in RIAL (as in source);
conversion to toman happens in 03_build_master.py.
"""
import sys, warnings, traceback
import pandas as pd, numpy as np
import hbsir
warnings.filterwarnings("ignore")

years = [int(a) for a in sys.argv[1:]] or list(range(1369, 1403))
rows = []

def wmean(s, w):
    m = s.notna() & w.notna()
    return float(np.average(s[m], weights=w[m])) if m.any() else np.nan

def province_share_weight(hh, y):
    """Replicates hbsir.schema_functions.weight.calculate_weight (buggy for 1371, undefined <1371)
    using HBSIR's own province_weights_1371.csv: weight = province urban share / province sample count."""
    shares = pd.read_csv(hbsir.api.defaults.package_dir / "internal_data" / "province_weights_1371.csv",
                         index_col=0, dtype={"Province": str})
    shares.index = shares.index.astype(int)
    hh = hbsir.add_attribute(hh, "Province", aspects="code")
    prov = hh["Province"].astype(int)
    cnt = hh.groupby([prov, hh["Urban_Rural"]])["ID"].transform("count")
    share = [shares.loc[p, ur] if p in shares.index else np.nan for p, ur in zip(prov, hh["Urban_Rural"].astype(str))]
    hh["Weight"] = np.array(share, dtype=float) / cnt.values * 1e6
    return hh

for y in years:
    r = {"year": y}
    try:
        mp = hbsir.load_table("members_properties", y)
        heads = mp.loc[mp["Relationship"] == "Head", ["ID", "Age"]].drop_duplicates("ID")
        hh = heads.rename(columns={"Age": "head_age"})
        hh["Year"] = y
        hh = hbsir.add_attribute(hh, "Urban_Rural")
        try:
            hh = hbsir.add_weight(hh); r["weight_source"] = "hbsir"
        except Exception as e:
            hh = province_share_weight(hh, y); r["weight_source"] = "province_share_1371 (HBSIR bug/no weight for <=1371)"
        if hh["Weight"].isna().all():
            hh["Weight"] = 1.0; r["weight_source"] = "UNWEIGHTED"
        hh = hh[hh["Urban_Rural"] == "Urban"].copy()
        r["n_households_urban"] = len(hh)

        def merge(df, cols):
            global hh
            hh = hh.merge(df[["ID"] + cols], on="ID", how="left")
        try:
            merge(hbsir.load_table("Total_Income", y), ["Income"]); hh["Income"] = hh["Income"].fillna(0)
        except Exception as e: print(y, "income ERR", repr(e)[:150], flush=True)
        try:
            merge(hbsir.load_table("Total_Expenditure", y), ["Gross_Expenditure"]); hh["Gross_Expenditure"] = hh["Gross_Expenditure"].fillna(0)
        except Exception as e: print(y, "exp ERR", repr(e)[:150], flush=True)
        try:
            ex = hbsir.load_table("Expenditures", y)
            home = ex[ex["Table_Name"] == "home"].groupby("ID", as_index=False)["Gross_Expenditure"].sum() \
                     .rename(columns={"Gross_Expenditure": "Housing_Expenditure"})
            merge(home, ["Housing_Expenditure"]); hh["Housing_Expenditure"] = hh["Housing_Expenditure"].fillna(0)
            del ex
        except Exception as e: print(y, "home ERR", repr(e)[:150], flush=True)
        try:
            ir = hbsir.load_table("Imputed_Rent", y).groupby("ID", as_index=False)["Income"].sum().rename(columns={"Income": "Imputed_Rent"})
            merge(ir, ["Imputed_Rent"]); hh["Imputed_Rent"] = hh["Imputed_Rent"].fillna(0)
        except Exception as e: print(y, "imprent ERR", repr(e)[:150], flush=True)
        try:
            merge(hbsir.load_table("Number_of_Members", y), ["Members"])
        except Exception as e: print(y, "members ERR", repr(e)[:150], flush=True)
        try:
            hs = hbsir.load_table("house_specifications", y)
            merge(hs.drop_duplicates("ID"), ["Tenure"])
        except Exception as e: print(y, "tenure ERR", repr(e)[:150], flush=True)

        w = hh["Weight"].astype(float)
        for col, name in [("Income", "income_rial"), ("Gross_Expenditure", "expenditure_rial"),
                          ("Housing_Expenditure", "housing_exp_rial"), ("Imputed_Rent", "imputed_rent_rial"),
                          ("Members", "household_size"), ("head_age", "head_age")]:
            if col in hh: r[name] = wmean(hh[col].astype(float), w)
        if "Income" in hh:
            r["income_median_rial"] = float(hh["Income"].median())
        if "Tenure" in hh:
            t = hh.dropna(subset=["Tenure"]); tw = t["Weight"].astype(float)
            shares = tw.groupby(t["Tenure"].astype(str)).sum() / tw.sum()
            for k, v in shares.items(): r[f"tenure_share__{k}"] = v
        hh.to_parquet(f"Data/processed/hh_urban_{y}.parquet")
        print(y, "OK", {k: (round(v) if isinstance(v, float) and v > 100 else v) for k, v in r.items() if not k.startswith("tenure")}, flush=True)
    except Exception as e:
        print(y, "FATAL", repr(e)[:300], flush=True); traceback.print_exc()
    rows.append(r)
    pd.DataFrame(rows).to_csv(f"Data/processed/hbs_urban_aggregates_{years[0]}_{years[-1]}.csv", index=False)
