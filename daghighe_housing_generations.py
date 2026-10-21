"""Housing and generations in urban Iran: data pipeline, indicators and report charts.

Inputs
  - SCI Household Income & Expenditure Survey microdata, 1369-1402 (via the HBSIR package)
  - CBI monthly reports "Tehran housing market developments" (PDF), 1396-1403
  - data/external: minimum wage, housing-loan policy, Kilid asking prices, SCI inflation
Run
  python daghighe_housing_generations.py
"""
import glob
import re
import warnings
from pathlib import Path

import hbsir
import matplotlib
import numpy as np
import pandas as pd
import pdfplumber

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

DATA = Path("Data")
EXT = DATA / "external"
CBI_PDF = DATA / "CBI_housing_raw" / "pdf"
CBI_MANUAL = DATA / "CBI_housing_raw" / "manual_transcriptions.csv"
FONTS = Path("fonts")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

YEARS = range(1369, 1403)
OWN = {"Owned_Estate", "Owned_Estate_Land"}
ESS_MIN = ["food", "cloth", "home_cash", "medical"]
ESS_WIDE = ESS_MIN + ["transportation", "communication", "miscellaneous"]
FRAMES = [1379, 1392, 1397]


# ---------- helpers ----------

def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


def wquantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return x[o][np.searchsorted(c, q)]


def is_tenant(tenure, year):
    # deposit-only renters (رهن) were coded "Other" before 1383
    return tenure.isin({"Rent", "Mortgage"}) if year >= 1383 else tenure.isin({"Rent", "Other"})


def annuity(principal, rate, years):
    r, n = rate / 12, years * 12
    return principal * r / (1 - (1 + r) ** -n)


# ---------- 1. household panel (urban, weighted) ----------

def province_share_weight(hh):
    # HBSIR has no weights for 1369-1371: province urban shares / province sample size
    path = hbsir.api.defaults.package_dir / "internal_data" / "province_weights_1371.csv"
    shares = pd.read_csv(path, index_col=0, dtype={"Province": str})
    shares.index = shares.index.astype(int)
    hh = hbsir.add_attribute(hh, "Province", aspects="code")
    prov = hh["Province"].astype(int)
    count = hh.groupby([prov, hh["Urban_Rural"]])["ID"].transform("count")
    share = [shares.loc[p, u] if p in shares.index else np.nan for p, u in zip(prov, hh["Urban_Rural"].astype(str))]
    hh["Weight"] = np.array(share, float) / count.values * 1e6
    return hh.drop(columns="Province")


def household_year(y):
    mp = hbsir.load_table("members_properties", y)
    hh = (mp.loc[mp["Relationship"] == "Head", ["ID", "Age"]].drop_duplicates("ID")
          .rename(columns={"Age": "head_age"}))
    hh["Year"] = y
    hh = hbsir.add_attribute(hh, "Urban_Rural")
    try:
        hh = hbsir.add_weight(hh)
    except Exception:
        hh = province_share_weight(hh)
    hh = hh[hh["Urban_Rural"] == "Urban"].copy()

    inc = hbsir.load_table("Total_Income", y)[["ID", "Income"]]
    exp = hbsir.load_table("Total_Expenditure", y)[["ID", "Gross_Expenditure"]]
    imp = hbsir.load_table("Imputed_Rent", y).groupby("ID", as_index=False)["Income"].sum()
    imp = imp.rename(columns={"Income": "Imputed_Rent"})
    mem = hbsir.load_table("Number_of_Members", y)[["ID", "Members"]]
    ten = hbsir.load_table("house_specifications", y).drop_duplicates("ID")[["ID", "Tenure"]]
    for t in (inc, exp, imp, mem, ten):
        hh = hh.merge(t, on="ID", how="left")
    hh[["Income", "Gross_Expenditure", "Imputed_Rent"]] = hh[["Income", "Gross_Expenditure", "Imputed_Rent"]].fillna(0)

    ex = hbsir.load_table("Expenditures", y)[["ID", "Table_Name", "Commodity_Code", "Gross_Expenditure"]]
    ex = ex[ex["ID"].isin(hh["ID"])]
    code = ex["Commodity_Code"].fillna(0).astype(int)
    imputed = code.between(42000, 42999)
    home = ex[ex["Table_Name"] == "home"]
    hcode = home["Commodity_Code"].fillna(0).astype(int)
    parts = {
        "Housing_Expenditure": home.groupby("ID")["Gross_Expenditure"].sum(),
        "home_cash": home[~hcode.between(42000, 42999)].groupby("ID")["Gross_Expenditure"].sum(),
        # rent of tenants: actual rent + rent-equivalent of deposit (old SCI codes before 1383)
        "rent_cost": (home[hcode.between(41000, 41999) | (hcode == 42212)] if y >= 1383
                      else home[hcode.between(31100, 31199)]).groupby("ID")["Gross_Expenditure"].sum(),
        "cash_rent": home[hcode.between(41100, 41199)].groupby("ID")["Gross_Expenditure"].sum() if y >= 1383 else None,
    }
    cash_tables = ex[~imputed].pivot_table(index="ID", columns="Table_Name", values="Gross_Expenditure",
                                           aggfunc="sum", observed=True)
    hh = hh.merge(cash_tables, left_on="ID", right_index=True, how="left")
    for name, s in parts.items():
        if s is not None:
            hh = hh.merge(s.rename(name), left_on="ID", right_index=True, how="left")
    for c in ESS_WIDE + ["Housing_Expenditure", "rent_cost"]:
        hh[c] = hh[c].fillna(0) if c in hh else 0.0
    if "cash_rent" in hh:
        hh["cash_rent"] = hh["cash_rent"].fillna(0)

    hh = hbsir.add_attribute(hh, "Province")
    try:
        hh = hbsir.add_attribute(hh, "County")
    except Exception:
        hh["County"] = None                     # county codes not available for early years
    hh = hh[hh["Weight"].notna()].copy()
    t = hh["Tenure"].astype(str)
    hh["w"] = hh["Weight"].astype(float)
    hh["owner"] = np.where(hh["Tenure"].notna(), t.isin(OWN), np.nan)
    hh["tenant"] = np.where(hh["Tenure"].notna(), is_tenant(t, y), np.nan)
    hh["cash_income"] = hh["Income"] - hh["Imputed_Rent"]
    return hh


def build_households():
    frames = []
    for y in YEARS:
        frames.append(household_year(y))
        print("households", y)
    H = pd.concat(frames, ignore_index=True)
    H["Province"] = H["Province"].astype(str)
    H["County"] = H["County"].astype(str)
    H["Tenure"] = H["Tenure"].astype(str)
    H.to_parquet(OUT / "households_urban.parquet")
    return H


# ---------- 2. person panel ----------

def build_persons(H):
    parts = []
    for y in YEARS:
        mp = hbsir.load_table("members_properties", y)[["ID", "Relationship", "Age", "Sex"]]
        mp = mp.merge(H.loc[H["Year"] == y, ["ID", "w", "owner", "tenant"]], on="ID")
        mp["year"] = y
        parts.append(mp)
    P = pd.concat(parts, ignore_index=True)
    P = P[P["Age"].notna()].copy()
    P["Age"] = P["Age"].astype(int)
    P["rel"] = P["Relationship"].astype(str)
    P["Sex"] = P["Sex"].astype(str)
    P["head_or_spouse"] = P["rel"].isin({"Head", "Spouse"}).astype(float)
    P["own_head_or_spouse"] = P["head_or_spouse"] * P["owner"]
    P["child_of_head"] = (P["rel"] == "Child").astype(float)
    P["birth_year"] = P["year"] - P["Age"]
    P["birth_decade"] = (P["birth_year"] // 10 * 10).astype(int)
    P["birth_5yr"] = (P["birth_year"] // 5 * 5).astype(int)
    P["age_group"] = pd.cut(P["Age"], [24, 29, 34, 39, 44], labels=["25-29", "30-34", "35-39", "40-44"]).astype(str)
    P = P.drop(columns="Relationship")
    P.to_parquet(OUT / "persons_urban.parquet")
    return P


# ---------- 3. CBI Tehran prices ----------

def shift_month(y, m, k):
    t = y * 12 + (m - 1) + k
    return t // 12, t % 12 + 1


def parse_cbi_reports():
    row_key = "عبرم رتم ره تميق"            # 'قیمت هر متر مربع' as extracted (reversed)
    num = re.compile(r"-?\d+(?:\.\d+)?")
    obs = []
    for f in sorted(glob.glob(str(CBI_PDF / "cbi_tehran_*.pdf"))):
        y, m = map(int, re.search(r"_(\d{4})_(\d{2})_", f).groups())
        with pdfplumber.open(f) as pdf:
            lines = "\n".join((pg.extract_text() or "") for pg in pdf.pages).split("\n")
        for i, line in enumerate(lines):
            if row_key not in line:
                continue
            cand = num.findall(line)
            if len(cand) < 3 and i + 1 < len(lines):
                cand = num.findall(lines[i + 1])
            vals = list(map(float, cand))
            if len(vals) >= 5:                  # yoy%, mom%, current, previous month, same month last year
                yoy, mom, cur, prev, ly = vals[:5]
                mom_ok = prev > 0 and abs((cur / prev - 1) * 100 - mom) < 0.6
                yoy_ok = ly > 0 and abs((cur / ly - 1) * 100 - yoy) < 0.6
            elif len(vals) == 3 and min(vals) > 1000:
                cur, prev, ly = vals
                mom_ok = yoy_ok = True
            else:
                continue
            if not (mom_ok or yoy_ok):
                continue
            k = 1000 if cur > 5000 else 1       # thousand rials (older) -> million rials
            src = Path(f).name
            obs.append((y, m, cur / k, src))
            if mom_ok:
                obs.append((*shift_month(y, m, -1), prev / k, src))
            if yoy_ok:
                obs.append((y - 1, m, ly / k, src))
            break
    o = pd.DataFrame(obs, columns=["year", "month", "price_mrial_m2", "source"])
    manual = pd.read_csv(CBI_MANUAL)[["year", "month", "price_mrial_m2", "source"]]
    o = pd.concat([o, manual], ignore_index=True)
    # latest published report wins (CBI revises earlier months)
    o["report_t"] = o["source"].str.extract(r"_(\d{4})_(\d{2})_").astype(int).pipe(lambda d: d[0] * 12 + d[1])
    monthly = (o.sort_values("report_t").groupby(["year", "month"]).tail(1)
               .sort_values(["year", "month"])[["year", "month", "price_mrial_m2"]])
    monthly = monthly.rename(columns={"price_mrial_m2": "price_million_rial_m2"})
    monthly.to_csv(OUT / "cbi_tehran_price_monthly.csv", index=False)
    return monthly


def annual_price(monthly):
    a = monthly.groupby("year").agg(n=("month", "nunique"), p=("price_million_rial_m2", "mean"))
    a = a[a["n"] == 12]
    return (a["p"] * 1e5).rename("tehran_price_per_m2")     # toman per m2, mean of 12 months


# ---------- 4. master annual table ----------

def build_master(H, price):
    g = H.groupby("Year")
    m = pd.DataFrame({
        "household_income": g.apply(lambda d: wmean(d["Income"], d["w"])) / 10,
        "household_expenditure": g.apply(lambda d: wmean(d["Gross_Expenditure"], d["w"])) / 10,
        "housing_expenditure": g.apply(lambda d: wmean(d["Housing_Expenditure"], d["w"])) / 10,
    })
    m = m.reindex(range(1363, 1403)).join(price)
    m.index.name = "year"
    m.to_csv(OUT / "master_annual.csv")
    return m


# ---------- 5. generations ----------

def cohort_tables(P):
    P = P[P["age_group"] != "nan"]

    def cell(d):
        ok = d["owner"].notna()
        by_age = d[ok].groupby("Age").apply(lambda g: wmean(g["own_head_or_spouse"], g["w"]))
        return pd.Series({
            "owner": wmean(d.loc[ok, "own_head_or_spouse"], d.loc[ok, "w"]),
            "owner_age_std": by_age.mean(),
            "child_of_head": wmean(d["child_of_head"], d["w"]),
            "head_or_spouse": wmean(d["head_or_spouse"], d["w"]),
            "n": len(d),
            "birth_min": d["birth_year"].min(), "birth_max": d["birth_year"].max(),
            "survey_years": f"{d['year'].min()}-{d['year'].max()}",
        })

    dec = P[P["birth_decade"].between(1330, 1370)].groupby(["birth_decade", "age_group"]).apply(cell).reset_index()
    dec["complete"] = (dec["birth_min"] == dec["birth_decade"]) & (dec["birth_max"] == dec["birth_decade"] + 9)
    five = P[P["birth_5yr"].between(1330, 1370)].groupby(["birth_5yr", "age_group"]).apply(cell).reset_index()
    sex = (P[P["birth_decade"].between(1330, 1370)]
           .groupby(["birth_decade", "age_group", "Sex"]).apply(cell).reset_index())
    dec.to_csv(OUT / "cohort_decades.csv", index=False)
    five.to_csv(OUT / "cohort_5yr.csv", index=False)
    sex.to_csv(OUT / "cohort_by_sex.csv", index=False)
    return dec, five, sex


def living_arrangements(P):
    P = P[(P["age_group"] == "30-34") & P["owner"].notna() & P["tenant"].notna()].copy()
    hs = P["head_or_spouse"] == 1
    P["cat"] = np.select([P["child_of_head"] == 1, hs & (P["owner"] == 1), hs & (P["tenant"] == 1), hs],
                         ["parents_home", "owner", "renter", "other_tenure"], "other_relation")
    cats = ["parents_home", "renter", "owner", "other_tenure", "other_relation"]

    def comp(d):
        s = (d.groupby("cat")["w"].sum() / d["w"].sum()).reindex(cats).fillna(0)
        indep = s["owner"] + s["renter"] + s["other_tenure"]
        return pd.concat([s, pd.Series({"independent": indep, "owner_given_independent": s["owner"] / indep, "n": len(d)})])

    L = P[P["birth_5yr"].between(1335, 1370)].groupby("birth_5yr").apply(comp).reset_index()
    a, b = L.set_index("birth_5yr").loc[1340], L.set_index("birth_5yr").loc[1365]
    total = np.log(b["owner"] / a["owner"])
    L.attrs["share_from_independence"] = np.log(b["independent"] / a["independent"]) / total
    L.to_csv(OUT / "living_arrangements_30_34.csv", index=False)
    return L


# ---------- 6. price-to-income, young households ----------

def price_to_income_young(H, price, n_boot=2000):
    rng = np.random.default_rng(1402)
    rows = []
    for y in range(1395, 1403):
        h = H[(H["Year"] == y) & H["head_age"].between(25, 34)]
        p75 = 75 * price[y]
        groups = [("tehran_county", h[h["County"] == "Tehran"]),
                  ("tehran_province", h[h["Province"] == "Tehran"]),
                  ("iran_urban", h)]
        for name, d in groups:
            inc, w = d["Income"].values / 10, d["w"].values
            mean = np.average(inc, weights=w)
            boot = [np.average(inc[i], weights=w[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(year=y, group=name, n=len(d), income=mean, pti=p75 / mean, pti_lo=p75 / hi, pti_hi=p75 / lo))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "price_to_income_young.csv", index=False)
    return R


# ---------- 7. housing cost burden, young tenants ----------

def tenant_burden(H):
    rows = []
    for y, h in H[H["head_age"].between(25, 34)].groupby("Year"):
        tn = h[h["tenant"] == 1]
        w = tn["w"]
        r = dict(year=y, n_tenants=len(tn),
                 tenant_share=wmean(h["tenant"], h["w"]),
                 housing_share_exp=(tn["Housing_Expenditure"] * w).sum() / (tn["Gross_Expenditure"] * w).sum(),
                 rent_to_cash_income=(tn["rent_cost"] * w).sum() / (tn["cash_income"] * w).sum())
        if y >= 1383:
            r["cash_rent_to_income"] = (tn["cash_rent"] * w).sum() / (tn["Income"] * w).sum()
        rows.append(r)
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "young_tenant_burden.csv", index=False)
    return B


# ---------- 8. years of saving after essentials ----------

def savings_capacity(H, price):
    rows = []
    for y in range(1395, 1403):
        tn = H[(H["Year"] == y) & H["head_age"].between(25, 34) & (H["tenant"] == 1)]
        w, cash = tn["w"].values, tn["cash_income"].values / 10
        p75 = 75 * price[y]
        for name, cols in [("wide", ESS_WIDE), ("min", ESS_MIN)]:
            surplus = cash - tn[cols].sum(axis=1).values / 10
            rows.append(dict(
                year=y, essentials=name, n=len(tn),
                years_ratio_of_means=p75 / np.average(surplus, weights=w),
                share_no_surplus=np.average(surplus <= 0, weights=w),
                years_median=p75 / wquantile(surplus, w, 0.5),
                years_p75=p75 / wquantile(surplus, w, 0.75),
                years_p90=p75 / wquantile(surplus, w, 0.9),
                cash_income=np.average(cash, weights=w),
                essentials_mean=np.average(cash - surplus, weights=w)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "young_tenant_savings.csv", index=False)
    return S


# ---------- 9. 1395-1405 monthly series (independent of HIES) ----------

def monthly_series(monthly):
    kil = pd.read_csv(EXT / "kilid_tehran_asking_price_snapshot_14050715.csv")
    wage = pd.read_csv(EXT / "minimum_wage.csv")[["year", "monthly_min_wage_toman"]]
    grid = pd.DataFrame([(y, m) for y in range(1395, 1406) for m in range(1, 13) if (y, m) <= (1405, 7)],
                        columns=["year", "month"])
    D = grid.merge(monthly, how="left").merge(kil, how="left").merge(wage, how="left")
    D["transaction_m2"] = D["price_million_rial_m2"] * 1e5
    D["asking_m2"] = D["asking_price_m2_million_toman"] * 1e6
    D["asking_to_transaction"] = D["asking_m2"] / D["transaction_m2"]
    D["months_wage_transaction"] = D["transaction_m2"] / D["monthly_min_wage_toman"]
    D["months_wage_asking"] = D["asking_m2"] / D["monthly_min_wage_toman"]
    D["t"] = D["year"] + (D["month"] - 1) / 12
    D.to_csv(OUT / "monthly_1395_1405.csv", index=False)
    return D


# ---------- 10. housing loans ----------

def loan_indicators(D, R):
    L = pd.read_csv(EXT / "housing_loan_policy_tehran.csv")
    ratio = D["asking_to_transaction"].dropna()
    inc_tehran = R[R["group"] == "tehran_county"].set_index("year")["income"]
    inc_iran = R[R["group"] == "iran_urban"].set_index("year")["income"]
    rows = []
    for _, p in L.iterrows():
        d = D[(D["year"] == p["year"]) & (D["month"] == p["month"])].iloc[0]
        if pd.notna(d["transaction_m2"]):
            mid = lo = hi = d["transaction_m2"]
        else:
            mid, lo, hi = d["asking_m2"] / ratio.mean(), d["asking_m2"] / ratio.max(), d["asking_m2"] / ratio.min()
        r = dict(effective_date=p["effective_date"], asking_based=pd.isna(d["transaction_m2"]),
                 cov_individual=p["individual_ceiling_toman"] / (75 * mid),
                 cov_couple=p["couple_ceiling_toman"] / (75 * mid),
                 cov_couple_low=p["couple_ceiling_toman"] / (75 * hi),
                 cov_couple_high=p["couple_ceiling_toman"] / (75 * lo))
        if pd.notna(p["interest_rate"]) and p["year"] <= 1402:
            inst = annuity(p["couple_ceiling_toman"], p["interest_rate"], p["repayment_years"])
            r["installment_to_income_tehran"] = inst / (inc_tehran[p["year"]] / 12)
            r["installment_to_income_iran"] = inst / (inc_iran[p["year"]] / 12)
        rows.append(r)
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "loan_indicators.csv", index=False)
    return O


# ---------- 11. nowcast 1403-1405 (estimate, not observed income) ----------

def nowcast(R, D, base="tehran_county"):
    wage = pd.read_csv(EXT / "minimum_wage.csv").set_index("year")["monthly_min_wage_toman"]
    wg = wage.pct_change()
    young = R[R["group"] == "iran_urban"].set_index("year")["income"]
    gap = (young.pct_change() - wg).loc[1396:1402].mean()
    w, other = 2 * 1769428.0, {}                       # 'other wage levels' rule, worker at 2x minimum
    for yr, (a, b) in {1403: (1.22, 230026), 1404: (1.32, 310535), 1405: (1.45, 519549)}.items():
        w2 = a * w + b
        other[yr], w = w2 / w - 1, w2
    paths = {"low": other,
             "mid": {y: wg[y] for y in (1403, 1404, 1405)},
             "high": {y: wg[y] + gap for y in (1403, 1404, 1405)}}
    base_income = R[(R["group"] == base) & (R["year"] == 1402)]["income"].iloc[0]
    income = {}
    for k, g in paths.items():
        v, income[k] = base_income, {}
        for y in (1403, 1404, 1405):
            v *= 1 + g[y]
            income[k][y] = v
    ratio = D["asking_to_transaction"].dropna()
    rs = {"lo": ratio.max(), "mid": ratio.mean(), "hi": ratio.min()}
    rows = []
    for y in (1403, 1404, 1405):
        d = D[D["year"] == y]
        prices = {k: np.nanmean(np.where(d["transaction_m2"].notna(), d["transaction_m2"], d["asking_m2"] / r))
                  for k, r in rs.items()}
        pti = {(ik, pk): 75 * prices[pk] / income[ik][y] for ik in income for pk in prices}
        rows.append(dict(year=y, pti_min=min(pti.values()), pti_mid=pti[("mid", "mid")], pti_max=max(pti.values())))
    last = D.dropna(subset=["asking_m2"]).iloc[-1]
    pit = [75 * last["asking_m2"] / r / income[ik][1405] for ik in income for r in rs.values()]
    N = pd.DataFrame(rows)
    N.attrs["point_in_time"] = (min(pit), 75 * last["asking_m2"] / rs["mid"] / income["mid"][1405], max(pit))
    N.to_csv(OUT / f"nowcast_pti_{base}.csv", index=False)
    return N


def nowcast_savings(S, R, D):
    cpi = pd.read_csv(EXT / "cpi_sci.csv").set_index("year")
    wage = pd.read_csv(EXT / "minimum_wage.csv").set_index("year")["monthly_min_wage_toman"]
    wg = wage.pct_change()
    young = R[R["group"] == "iran_urban"].set_index("year")["income"]
    gap = (young.pct_change() - wg).loc[1396:1402].mean()
    w, other = 2 * 1769428.0, {}
    for yr, (a, b) in {1403: (1.22, 230026), 1404: (1.32, 310535), 1405: (1.45, 519549)}.items():
        w2 = a * w + b
        other[yr], w = w2 / w - 1, w2
    paths = {"low": other, "mid": {y: wg[y] for y in (1403, 1404, 1405)},
             "high": {y: wg[y] + gap for y in (1403, 1404, 1405)}}
    ratio = D["asking_to_transaction"].dropna()
    r_chain, r_lowp = ratio.iloc[-1], ratio.max()
    last = D.dropna(subset=["asking_m2"]).iloc[-1]
    rows = []
    for ess in ("wide", "min"):
        b = S[(S["year"] == 1402) & (S["essentials"] == ess)].iloc[0]
        for path, g in paths.items():
            inc, e = b["cash_income"], b["essentials_mean"]
            for y in (1403, 1404, 1405):
                inc *= 1 + g[y]
                e *= 1 + cpi.loc[y, "annual_inflation_all"]
                d = D[D["year"] == y]
                for pcase, r in [("chained", r_chain), ("low_price", r_lowp)]:
                    p75 = 75 * np.nanmean(np.where(d["transaction_m2"].notna(), d["transaction_m2"], d["asking_m2"] / r))
                    s = inc - e
                    rec = dict(essentials=ess, year=y, path=path, price=pcase, years=p75 / s if s > 0 else np.inf)
                    if y == 1405:
                        rec["years_latest_month"] = 75 * last["asking_m2"] / r / s if s > 0 else np.inf
                    rows.append(rec)
    O = pd.DataFrame(rows)
    O.to_csv(OUT / "nowcast_years_saving.csv", index=False)
    return O


# ---------- 12. charts ----------

for f in ("Vazirmatn-Regular.ttf", "Vazirmatn-Bold.ttf"):
    font_manager.fontManager.addfont(str(FONTS / f))
matplotlib.rcParams["font.family"] = "Vazirmatn"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY, LGRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#c9c8c2"
RAMP5 = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
FA_DIGITS = str.maketrans("0123456789%", "۰۱۲۳۴۵۶۷۸۹٪")
MONTHS_FA = "فروردین اردیبهشت خرداد تیر مرداد شهریور مهر آبان آذر دی بهمن اسفند".split()


def fa(text):
    return re.sub(r"(\d+)\s*[–-]\s*(\d+)", r"\1 تا \2", str(text)).translate(FA_DIGITS)


def pct(x):
    return fa(f"{x * 100:.0f}%")


def new_chart(w=10.5, h=6.3):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURF)
    return fig, ax


def finish(fig, name, main, sub, left=0.07):
    fig.text(0.985, 0.965, fa(main), ha="right", va="top", fontsize=15, color=INK, weight="bold")
    fig.text(0.985, 0.915, fa(sub), ha="right", va="top", fontsize=10.5, color=INK2)
    plt.subplots_adjust(left=left, right=0.97, top=0.84, bottom=0.08)
    fig.savefig(OUT / name, dpi=170, facecolor=SURF)
    plt.close(fig)


def pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: pct(v)))


def num_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fa(f"{v:.0f}")))


def chart_cohorts(dec):
    ages = ["25-29", "30-34", "35-39", "40-44"]
    xi = {a: i for i, a in enumerate(ages)}
    dname = {1330: "۳۰", 1340: "۴۰", 1350: "۵۰", 1360: "۶۰", 1370: "۷۰"}
    fig, ax = new_chart(10.5, 6.6)
    for (decade, d), c in zip(dec.groupby("birth_decade"), RAMP5):
        d = d.sort_values("age_group")
        x, y = d["age_group"].map(xi).values, d["owner"].values
        ax.plot(x, y, color=c, lw=2.2, ls=(0, (5, 3)) if decade == 1370 else "-", zorder=2)
        for xx, yy, comp in zip(x, y, d["complete"]):
            ax.plot(xx, yy, "o", ms=9, color=c if comp else SURF, markeredgecolor=SURF if comp else c,
                    markeredgewidth=2, zorder=3)
            if xx == 1:
                ax.annotate(pct(yy), (xx, yy), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=9, color=INK2)
        label = f"متولدین دهه {dname[decade]}" + (" (ناقص: ۱۳۷۰ تا ۱۳۷۲)" if decade == 1370 else "")
        ax.annotate(fa(label), (x[-1], y[-1]), textcoords="offset points", xytext=(12, 0), va="center", color=INK, fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels([fa(f"{a} سالگی") for a in ages])
    ax.set_xlim(-0.3, 3.9)
    ax.set_ylim(0, 0.9)
    pct_axis(ax)
    finish(fig, "chart1_cohort_ownership.png", "نسل‌های جدید در سنی مشابه کمتر صاحب‌خانه‌اند",
           "سهم افراد شهری که سرپرست یا همسر سرپرستِ خانواری‌اند که در ملک خودش زندگی می‌کند")

    c = dec[dec["age_group"] == "30-34"].sort_values("birth_decade")
    fig, ax = new_chart(9, 6)
    for i, (decade, v, col) in enumerate(zip(c["birth_decade"], c["child_of_head"], RAMP5)):
        ax.bar(i, v, width=0.62, color=col, hatch="//" if decade == 1370 else None,
               edgecolor="#ffffff" if decade == 1370 else SURF, lw=0)
        ax.annotate(pct(v), (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=12, color=INK, weight="bold")
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels([fa(f"متولدین دهه {dname[d]}") + ("\n" + fa("(۱۳۷۰ تا ۱۳۷۲)") if d == 1370 else "")
                        for d in c["birth_decade"]])
    ax.set_ylim(0, 0.55)
    pct_axis(ax)
    finish(fig, "chart2_living_with_parents.png", "در ۳۰ تا ۳۴ سالگی، هر نسل بیشتر از قبلی هنوز در خانه‌ی پدر و مادر است",
           "سهم افراد ۳۰ تا ۳۴ ساله‌ی شهری که «فرزندِ سرپرست خانوار» ثبت شده‌اند", left=0.08)


def chart_pti(R):
    t = R[R["group"] == "tehran_county"].set_index("year")
    n = R[R["group"] == "iran_urban"].set_index("year")
    fig, ax = new_chart(10, 6.2)
    ax.fill_between(t.index, t["pti_lo"], t["pti_hi"], color=BLUE, alpha=0.13, lw=0)
    ax.plot(n.index, n["pti"], color=GRAY, lw=1.6, ls=(0, (4, 3)), marker="o", ms=5, markeredgecolor=SURF)
    ax.plot(t.index, t["pti"], color=BLUE, lw=2.6, marker="o", ms=9, markeredgecolor=SURF, markeredgewidth=2)
    for y, v in t["pti"].items():
        ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, -20), ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(1395, 3.2, fa("خانوار جوان تهرانی"), color=BLUE, fontsize=11, weight="bold", ha="left")
    ax.annotate(fa(f"خانوار جوان شهری کل کشور ({n['pti'][1402]:.0f} سال در ۱۴۰۲)"), (1399, n["pti"][1399]),
                textcoords="offset points", xytext=(-6, 8), ha="right", color=INK2, fontsize=9.5)
    ax.set_xticks(t.index)
    ax.set_xticklabels([fa(y) for y in t.index])
    ax.set_ylim(0, 32)
    num_axis(ax)
    ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
    finish(fig, "chart3_price_to_income_tehran.png",
           "در هفت سال، خانه‌ی ۷۵ متری تهران دو برابر سال‌های درآمد یک خانوار جوان تهرانی شد",
           "قیمت یک آپارتمان ۷۵ متری ÷ درآمد سالانه‌ی خانوارهای شهرستان تهران با سرپرست ۲۵ تا ۳۴ ساله؛ نوار = بازه‌ی اطمینان ۹۵٪")


def chart_burden(B):
    fig, ax = new_chart(11, 6.2)
    for f in FRAMES:
        ax.axvline(f - 0.5, color=GRID, lw=1.5, ls=(0, (4, 3)))
        ax.text(f - 0.35, 0.01, fa("چارچوب نمونه‌ی جدید"), color=INK2, fontsize=8, rotation=90, va="bottom")
    ax.axvline(1382.5, color=GRID, lw=1.5, ls=":")
    ax.text(1382.65, 0.01, fa("تغییر طبقه‌بندی هزینه"), color=INK2, fontsize=8, rotation=90, va="bottom")
    series = [("rent_to_cash_income", ORANGE, "اجاره (شامل معادل اجاره‌ی رهن) ÷ درآمد", "-", (1369, 0.415)),
              ("housing_share_exp", BLUE, "سهم مسکن از کل هزینه", "-", (1369, 0.20)),
              ("cash_rent_to_income", GRAY, "فقط اجاره‌ی نقدی ÷ درآمد", (0, (4, 2)), (1384.3, 0.115))]
    for col, c, label, ls, pos in series:
        d = B.dropna(subset=[col])
        ax.plot(d["year"], d[col], color=c, lw=2.2 if ls == "-" else 1.6, ls=ls, marker="o" if ls == "-" else None,
                ms=4.5, markeredgecolor=SURF, markeredgewidth=1)
        ax.text(*pos, fa(label), ha="left", color=c if c != GRAY else INK2, fontsize=9.5, weight="bold")
        ax.annotate(pct(d[col].iloc[-1]), (d["year"].iloc[-1], d[col].iloc[-1]), textcoords="offset points",
                    xytext=(8, 0), ha="left", va="center", color=INK, fontsize=9.5)
    ax.set_ylim(0, 0.6)
    pct_axis(ax)
    ax.set_xlim(1368, 1404.5)
    ax.set_xticks(range(1369, 1403, 3))
    ax.set_xticklabels([fa(y) for y in range(1369, 1403, 3)])
    finish(fig, "chart4_young_tenant_burden.png",
           "پیش از هر پس‌اندازی، حدود یک‌سوم تا دوپنجم درآمد مستأجر جوان صرف سقف بالای سرش می‌شود",
           "خانوارهای مستأجر شهری با سرپرست ۲۵ تا ۳۴ ساله")


def chart_min_wage(D):
    cut = 1403 + 4.5 / 12
    tx = D.dropna(subset=["months_wage_transaction"])
    ak = D[D["t"] > cut].dropna(subset=["months_wage_asking"])
    fig, ax = new_chart(11, 6.3)
    ax.plot(tx["t"], tx["months_wage_transaction"], color=BLUE, lw=2.4)
    ax.plot(ak["t"], ak["months_wage_asking"], color=ORANGE, lw=2.4)
    ax.axvline(cut, color=INK, lw=1.2)
    ax.text(cut + 0.08, 19.2, fa("از اینجا به بعد: قیمت آگهی، نه معامله"), color=INK, fontsize=10, weight="bold", ha="left")
    for y in range(1396, 1406):
        ax.axvline(y, color=GRID, lw=0.8, ls=":")
    marks = [(tx.iloc[0], "months_wage_transaction"),
             (tx.loc[tx["months_wage_transaction"].idxmax()], "months_wage_transaction"),
             (ak.iloc[-1], "months_wage_asking")]
    for r, col in marks:
        ax.annotate(fa(f"{r[col]:.1f} ماه"), (r["t"], r[col]), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10.5, color=INK, weight="bold")
    ax.set_ylim(0, 21)
    ax.set_yticks(range(0, 21, 4))
    ax.set_xticks(range(1395, 1406))
    ax.set_xticklabels([fa(y) for y in range(1395, 1406)])
    num_axis(ax)
    ax.set_ylabel(fa("ماه حداقل مزد"), color=INK2)
    finish(fig, "chart5_m2_in_min_wage_months.png", "یک متر خانه در تهران = چند ماه حداقل مزد؟",
           "قیمت هر متر مربع ÷ حداقل مزد پایه‌ی ماهانه‌ی همان سال (بدون حق مسکن و بن)")


def chart_nowcast(R, N):
    obs = R[R["group"] == "tehran_county"].set_index("year")["pti"]
    lo_pit, mid_pit, hi_pit = N.attrs["point_in_time"]
    fig, ax = new_chart(10.5, 6.3)
    ax.plot(obs.index, obs.values, color=BLUE, lw=2.6, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2)
    xb = [1402] + list(N["year"])
    ax.fill_between(xb, [obs[1402]] + list(N["pti_min"]), [obs[1402]] + list(N["pti_max"]), color=BLUE, alpha=0.13, lw=0)
    ax.plot(xb, [obs[1402]] + list(N["pti_mid"]), color=BLUE, lw=2, ls=(0, (4, 3)))
    ax.axvline(1402.5, color=INK2, lw=1)
    ax.text(1402.6, 37, fa("برآورد، نه درآمد مشاهده‌شده"), color=INK, fontsize=10.5, weight="bold", ha="left")
    for y, v in obs.items():
        ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=10, color=INK)
    for _, r in N.iterrows():
        ax.annotate(fa(f"{r['pti_min']:.0f} تا {r['pti_max']:.0f}"), (r["year"], r["pti_max"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9.5, color=INK2)
    ax.errorbar(1405.3, mid_pit, yerr=[[mid_pit - lo_pit], [hi_pit - mid_pit]], fmt="D", color=ORANGE, ms=7, capsize=4, lw=1.6)
    ax.annotate(fa("قیمت مهر ۱۴۰۵:") + "\n" + fa(f"{lo_pit:.0f} تا {hi_pit:.0f} سال"), (1405.3, hi_pit),
                textcoords="offset points", xytext=(-4, 6), ha="center", fontsize=9, color=ORANGE, weight="bold")
    xs = list(obs.index) + list(N["year"])
    ax.set_xlim(1394.6, 1405.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([fa(y) + ("\n" + fa("(۷ ماه اول)") if y == 1405 else "") for y in xs])
    ax.set_ylim(0, 40)
    num_axis(ax)
    ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
    finish(fig, "chart6_nowcast_price_to_income.png",
           "خانه‌ی ۷۵ متری تهران معادل چند سال کل درآمد خانوار جوان تهرانی؟ برآورد تا ۱۴۰۵",
           "خط پر: مشاهده‌شده (طرح هزینه و درآمد). نوار: بازه‌ی برآورد با سه مسیر درآمد × بازه‌ی تبدیل قیمت آگهی به معامله")


def chart_loans(O):
    fig, ax = new_chart(10, 6)
    x = np.arange(len(O))
    ax.bar(x, O["cov_couple"], width=0.55, color=BLUE)
    for i, r in O.iterrows():
        top = r["cov_couple"]
        if r["asking_based"]:
            ax.errorbar(i, top, yerr=[[top - r["cov_couple_low"]], [r["cov_couple_high"] - top]], color=INK2, capsize=4, lw=1.3)
            top = r["cov_couple_high"]
        ax.annotate(pct(r["cov_couple"]), (i, top), textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=12, weight="bold", color=INK)
        ax.text(i, 0.008, fa(f"وام فردی: {r['cov_individual'] * 100:.0f}%"), ha="center", fontsize=9.5, color="#ffffff", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([fa(f"{MONTHS_FA[int(d.split('-')[1]) - 1]} {d.split('-')[0]}") for d in O["effective_date"]])
    ax.set_ylim(0, 0.25)
    pct_axis(ax)
    finish(fig, "chart7_loan_coverage.png", "وام خرید زوجین چند درصد قیمت یک آپارتمان ۷۵ متری تهران را پوشش می‌داد؟",
           "سقف وام خرید (از محل اوراق حق تقدم بانک مسکن) در ماه تصویب، بدون جعاله")


def chart_savings(S, NS):
    obs = {k: S[S["essentials"] == k].set_index("year")["years_ratio_of_means"] for k in ("wide", "min")}
    wide = NS[NS["essentials"] == "wide"]
    band = wide.groupby("year")["years"].agg(["min", "max"])
    mid = {k: NS[(NS["essentials"] == k) & (NS["path"] == "mid") & (NS["price"] == "chained")].set_index("year")
           for k in ("wide", "min")}
    pit = wide[wide["year"] == 1405]["years_latest_month"]
    pit_mid = mid["wide"].loc[1405, "years_latest_month"]
    fig, ax = new_chart(10.5, 6.4)
    ax.plot(obs["min"].index, obs["min"], color=GRAY, lw=1.6, marker="o", ms=5, markeredgecolor=SURF)
    ax.plot(obs["wide"].index, obs["wide"], color=ORANGE, lw=2.6, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2)
    xb = [1402] + list(band.index)
    ax.fill_between(xb, [obs["wide"][1402]] + list(band["min"]), [obs["wide"][1402]] + list(np.minimum(band["max"], 240)),
                    color=ORANGE, alpha=0.13, lw=0)
    ax.plot(xb, [obs["wide"][1402]] + list(mid["wide"]["years"]), color=ORANGE, lw=2, ls=(0, (4, 3)))
    ax.plot(xb, [obs["min"][1402]] + list(mid["min"]["years"]), color=GRAY, lw=1.4, ls=(0, (4, 3)))
    for y, v in obs["wide"].items():
        ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=INK, weight="bold")
    for y, v in mid["wide"]["years"].items():
        ax.annotate(fa(f"{v:.0f}"), (y, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10, color=INK)
    ax.errorbar(1405.35, pit_mid, yerr=[[pit_mid - pit.min()], [min(pit.max(), 240) - pit_mid]], fmt="D", color=ORANGE,
                ms=7, capsize=4, lw=1.4)
    ax.annotate(fa(f"با قیمت مهر ۱۴۰۵: {pit_mid:.0f}"), (1405.35, pit_mid), textcoords="offset points", xytext=(-10, 0),
                ha="right", va="center", fontsize=9.5, color=ORANGE, weight="bold")
    ax.axvline(1402.5, color=INK2, lw=1)
    ax.text(1402.6, 228, fa("برآورد"), color=INK, fontsize=10.5, weight="bold", ha="left")
    ax.text(1395, 118, fa("کسر خرج‌های ضروری (گسترده)"), color=ORANGE, fontsize=10, weight="bold", ha="left")
    ax.text(1395, obs["min"][1395] - 16, fa("کسر فقط خوراک، پوشاک، مسکن و درمان"), color=INK2, fontsize=9.5, ha="left")
    xs = list(obs["wide"].index) + [1403, 1404, 1405]
    ax.set_xlim(1394.6, 1405.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([fa(y) + ("\n" + fa("(۷ ماه اول)") if y == 1405 else "") for y in xs])
    ax.set_ylim(0, 240)
    num_axis(ax)
    ax.set_ylabel(fa("سال"), color=INK2, rotation=0, labelpad=14)
    finish(fig, "chart8_years_of_saving.png", "اگر مستأجر جوان فقط خرج ضروری کند، چند سال تا یک خانه‌ی ۷۵ متری در تهران؟",
           "همه‌ی درآمد به‌جز خرج‌های ضروری پس‌انداز می‌شود؛ خانوارهای مستأجر شهری با سرپرست ۲۵ تا ۳۴ ساله")


def chart_living(L):
    cats = [("parents_home", ORANGE, "در خانه‌ی پدر و مادر"), ("renter", AQUA, "مستقل، مستأجر"),
            ("owner", BLUE, "مستقل، در خانه‌ی ملکی"), ("other_tenure", LGRAY, "مستقل، سایر (رایگان، سازمانی)"),
            ("other_relation", "#e4e3df", "سایر")]
    L = L.sort_values("birth_5yr")
    labels = {c: fa(f"{c} تا {c + 4}") for c in L["birth_5yr"]}
    labels[1370] = fa("۱۳۷۰ تا ۱۳۷۲")
    fig, ax = new_chart(10.5, 6.8)
    ax.grid(False)
    y = np.arange(len(L))[::-1]
    left = np.zeros(len(L))
    for key, c, _ in cats:
        v = L[key].values
        ax.barh(y, v, left=left, color=c, height=0.66, edgecolor=SURF, lw=2)
        if key in ("parents_home", "renter", "owner"):
            for yi, l, vi in zip(y, left, v):
                if vi >= 0.07:
                    ax.text(l + vi / 2, yi, pct(vi), ha="center", va="center", fontsize=10, color="#ffffff", weight="bold")
        left += v
    for i, (_, c, name) in enumerate(cats[:4]):
        ax.add_patch(matplotlib.patches.Rectangle((i * 0.25, len(L) - 0.47), 0.018, 0.24, color=c, clip_on=False))
        ax.text(0.026 + i * 0.25, len(L) - 0.35, fa(name), color=INK, fontsize=9.5, ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[c] for c in L["birth_5yr"]])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_ylabel(fa("متولدین"), color=INK2)
    ax.set_ylim(-0.6, len(L) + 0.1)
    finish(fig, "chart9_living_arrangements.png", "سی‌ساله‌ها کجا زندگی می‌کنند؟ هر نسل کمتر مستقل شده است",
           "وضعیت زندگی افراد شهری در ۳۰ تا ۳۴ سالگی، به تفکیک سال تولد", left=0.13)


# ---------- run ----------

def main():
    H = build_households()
    P = build_persons(H)
    monthly = parse_cbi_reports()
    price = annual_price(monthly)
    build_master(H, price)

    dec, five, sex = cohort_tables(P)
    L = living_arrangements(P)
    R = price_to_income_young(H, price)
    B = tenant_burden(H)
    S = savings_capacity(H, price)
    D = monthly_series(monthly)
    O = loan_indicators(D, R)
    N = nowcast(R, D, base="tehran_county")
    NS = nowcast_savings(S, R, D)

    chart_cohorts(dec)
    chart_pti(R)
    chart_burden(B)
    chart_min_wage(D)
    chart_nowcast(R, N)
    chart_loans(O)
    chart_savings(S, NS)
    chart_living(L)

    print("owner share at 30-34, 5-yr cohorts:", five[five["age_group"] == "30-34"].set_index("birth_5yr")["owner"].round(3).to_dict())
    print("share of decline from lower independence:", round(L.attrs["share_from_independence"], 2))
    print("PTI Tehran county:", R[R["group"] == "tehran_county"].set_index("year")["pti"].round(1).to_dict())
    print("done ->", OUT.resolve())


if __name__ == "__main__":
    main()
