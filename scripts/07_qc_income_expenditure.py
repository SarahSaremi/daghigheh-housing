"""QC3: has the definition of income/expenditure changed over the series?
Weighted URBAN composition of income (by Income_Type) and expenditure (by table) per year."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir
inc_rows, exp_rows = [], []
for y in range(1369, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet")[["ID", "Weight"]]
    ib = hbsir.load_table("Income_Breakdown", y).merge(h, on="ID")
    s = ib.assign(wI=ib.Income * ib.Weight.astype(float)).groupby("Income_Type", observed=True).wI.sum()
    inc_rows.append((s / s.sum()).rename(y))
    ex = hbsir.load_table("Expenditures", y)[["ID", "Table_Name", "Gross_Expenditure"]].merge(h, on="ID")
    e = ex.assign(wE=ex.Gross_Expenditure * ex.Weight.astype(float)).groupby("Table_Name", observed=True).wE.sum()
    exp_rows.append((e / e.sum()).rename(y))
    print(y, "done", flush=True)
inc = pd.DataFrame(inc_rows); ex = pd.DataFrame(exp_rows)
inc.to_csv("Data/qc/income_composition_urban.csv"); ex.to_csv("Data/qc/expenditure_composition_urban.csv")
pd.set_option("display.width", 250)
print((inc * 100).round(1).to_string()); print((ex * 100).round(1).to_string())
