"""Years to buy a 75 m2 Tehran apartment for a YOUNG household saving all income except essentials.
Observed 1395-1402 (HIES, cash basis). ESTIMATE 1403-1405:
  income paths (low/mid/high) as in 16_nowcast_pti (applied to cash income);
  essentials grow with SCI annual CPI (1405 = 12-month avg to Shahrivar 1405);
  'strict' 1405 variant: food at food inflation, rest at non-food inflation (weights = 1402 young budget);
  price: transaction chained from Mordad 1403 with Kilid growth (= asking / 0.979), low-price bound asking / 1.186.
  1405 price = mean Farvardin..Mehr; also point-in-time Mehr 1405."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir, matplotlib.pyplot as plt
from fa_style import *
S = pd.read_csv("Data/young_savings_capacity.csv"); cpi = pd.read_csv("Data/external/cpi_sci.csv").set_index("year")
N = pd.read_csv("Data/nowcast_pti_young_1403_1405.csv").set_index("year")
D = pd.read_csv("Data/monthly_1395_1405_independent.csv")
ov = D.dropna(subset=["asking_to_transaction"]).asking_to_transaction; r_chain, r_lowp = ov.iloc[-1], ov.max()

# food share of wide essentials, young households 1402 (for the 'strict' 1405 variant)
h = pd.read_parquet("Data/processed/hh_urban_1402.parquet"); h = h[h.head_age.between(25, 34) & h.Weight.notna()]
ex = hbsir.load_table("Expenditures", 1402); ex = ex[ex.ID.isin(h.ID) & ~ex.Commodity_Code.fillna(0).astype(int).between(42000, 42999)]
ex = ex.merge(h[["ID", "Weight"]], on="ID"); ex["wE"] = ex.Gross_Expenditure * ex.Weight.astype(float)
W = ex.groupby("Table_Name", observed=True).wE.sum()
ess = W[["food", "cloth", "home", "medical", "transportation", "communication", "miscellaneous"]]
food_share = W["food"] / ess.sum(); print("food share of essentials (young, 1402):", round(food_share, 3))

def price_year(y, r):
    d = D[D.year == y]
    p = np.where(d.transaction_m2_toman.notna(), d.transaction_m2_toman, d.asking_m2_toman / r)
    return np.nanmean(p)
last = D.dropna(subset=["asking_m2_toman"]).iloc[-1]

rows = []
for grp in ["young_all", "young_tenants"]:
    b = S[(S.group == grp) & (S.year == 1402)].iloc[0]
    for path in ["low", "mid", "high"]:
        inc, e_min, e_wide, e_strict = b.cash_income, b.ess_min, b.ess_wide, b.ess_wide
        for y in (1403, 1404, 1405):
            g = N.loc[y, f"growth_{path}"]; inc *= 1 + g
            pi_all = cpi.loc[y, "annual_inflation_all"]
            e_min *= 1 + pi_all; e_wide *= 1 + pi_all
            e_strict *= (1 + (food_share * cpi.loc[y, "annual_inflation_food"] + (1 - food_share) * cpi.loc[y, "annual_inflation_nonfood"])
                         if y == 1405 else 1 + pi_all)
            for pcase, r in [("chained", r_chain), ("low_price", r_lowp)]:
                P = 75 * price_year(y, r)
                rec = dict(group=grp, year=y, income_path=path, price_case=pcase, cash_income=inc, ess_min=e_min,
                           ess_wide=e_wide, ess_strict=e_strict, price75=P)
                for k in ["min", "wide", "strict"]:
                    s = inc - rec[f"ess_{k}"]; rec[f"years_save_{k}"] = P / s if s > 0 else np.inf
                if y == 1405:
                    Pp = 75 * last.asking_m2_toman / r
                    for k in ["min", "wide", "strict"]:
                        s = inc - rec[f"ess_{k}"]; rec[f"years_save_{k}_mehr1405"] = Pp / s if s > 0 else np.inf
                rows.append(rec)
O = pd.DataFrame(rows); O.to_csv("Data/nowcast_years_saving_young.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
summ = O.groupby(["group", "year"]).agg(min_lo=("years_save_min", "min"), min_hi=("years_save_min", "max"),
        wide_lo=("years_save_wide", "min"), wide_hi=("years_save_wide", "max"),
        strict_lo=("years_save_strict", "min"), strict_hi=("years_save_strict", "max"))
mid = O[(O.income_path == "mid") & (O.price_case == "chained")].set_index(["group", "year"])[["years_save_min", "years_save_wide", "years_save_strict"]]
summ = summ.join(mid.add_prefix("MID_")); summ.to_csv("Data/nowcast_years_saving_young_summary.csv", encoding="utf-8-sig")
print(summ.round(1).to_string())
pit = O[O.year == 1405].groupby("group")[["years_save_min_mehr1405", "years_save_wide_mehr1405"]].agg(["min", "max"])
print(pit.round(1).to_string())
print(O[(O.year == 1405) & (O.income_path == "mid") & (O.price_case == "chained")][["group", "years_save_min_mehr1405", "years_save_wide_mehr1405"]].round(1).to_string())
