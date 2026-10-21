"""Generational tables from HBS urban microdata (weighted).
1) year x head-age-group: income, expenditure, housing exp, owner/tenant shares, hh size, n.
2) year: owner share among households with head aged 25-34.
3) person-level (all urban individuals 25-34): share heading (or spouse of head of) a household,
   share living as head/spouse in an OWNED home, share living as child of the head.
   -> corrects the selection bias of head-based rates (young adults who never left home).
4) birth cohorts: birth_year = survey_year - age (Solar Hijri, +/-1 year)."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir
OWN = {"Owned_Estate", "Owned_Estate_Land"}
def tenant_mask(t, y):
    # 'Mortgage' (رهن) code exists only from 1383; before that full-deposit renters were coded 'Other'
    return t.isin({"Rent", "Mortgage"}) if y >= 1383 else t.isin({"Rent", "Other"})
def age_group(a):
    return pd.cut(a, [0, 29, 34, 39, 44, 49, 200], labels=["<30", "30-34", "35-39", "40-44", "45-49", "50+"])
def wmean(x, w):
    m = x.notna() & w.notna(); return np.average(x[m], weights=w[m]) if m.any() else np.nan
def n_eff(w): w = w.dropna(); return w.sum() ** 2 / (w ** 2).sum()

grp_rows, y2534, person_rows, hh_all, pers_all = [], [], [], [], []
for y in range(1369, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet")
    h = h[h.Weight.notna()].copy(); h["w"] = h.Weight.astype(float)
    t = h.Tenure.astype(str)
    h["owner"] = np.where(h.Tenure.notna(), t.isin(OWN), np.nan).astype(float)
    h["tenant"] = np.where(h.Tenure.notna(), tenant_mask(t, y), np.nan).astype(float)
    h["age_group"] = age_group(h.head_age)
    for g, d in h.groupby("age_group", observed=True):
        grp_rows.append(dict(year=y, age_group=g, n_obs=len(d), n_eff=round(n_eff(d.w)),
            income=wmean(d.Income, d.w) / 10, expenditure=wmean(d.Gross_Expenditure, d.w) / 10,
            housing_expenditure=wmean(d.Housing_Expenditure, d.w) / 10,
            owner_share=wmean(d.owner, d.w), tenant_share=wmean(d.tenant, d.w),
            household_size=wmean(d.Members.astype(float), d.w)))
    d = h[h.head_age.between(25, 34)]
    p = wmean(d.owner, d.w); ne = n_eff(d.w[d.owner.notna()])
    y2534.append(dict(year=y, owner_share_head_25_34=p, se=np.sqrt(p * (1 - p) / ne), n_obs=len(d), n_eff=round(ne)))
    hh_all.append(h[["ID", "head_age", "w", "owner", "tenant", "Income", "Gross_Expenditure"]].assign(year=y))

    # ---- person level ----
    mp = hbsir.load_table("members_properties", y)[["ID", "Relationship", "Age", "Sex"]]
    mp = mp.merge(h[["ID", "w", "owner"]], on="ID")          # urban households with weights
    mp["rel"] = mp.Relationship.astype(str)
    mp["head_or_spouse"] = mp.rel.isin({"Head", "Spouse"}).astype(float)
    mp["own_head_or_spouse"] = mp.head_or_spouse * mp.owner      # NaN if tenure unknown
    mp["child_of_head"] = (mp.rel == "Child").astype(float)
    pers_all.append(mp[["Age", "Sex", "w", "head_or_spouse", "own_head_or_spouse", "child_of_head"]].assign(year=y))
    q = mp[mp.Age.between(25, 34)]
    person_rows.append(dict(year=y, n_persons_25_34=len(q),
        share_head_or_spouse=wmean(q.head_or_spouse, q.w),
        share_owner_head_or_spouse=wmean(q.own_head_or_spouse, q.w),
        share_child_of_head=wmean(q.child_of_head, q.w)))
    print(y, "ok", flush=True)

G = pd.DataFrame(grp_rows); G.to_csv("Data/hbs_urban_by_year_agegroup.csv", index=False, encoding="utf-8-sig")
Y = pd.DataFrame(y2534).merge(pd.DataFrame(person_rows), on="year")
Y.to_csv("Data/owner_share_25_34.csv", index=False, encoding="utf-8-sig")

# ---- birth cohorts (head-based and person-based) ----
H = pd.concat(hh_all); H["birth_year"] = H.year - H.head_age
P = pd.concat(pers_all); print("persons with missing age:", P.Age.isna().sum(), "of", len(P)); P = P[P.Age.notna()].copy(); P["Age"] = P.Age.astype(int); P["birth_year"] = P.year - P.Age
def cohort(b): return (b // 5 * 5).astype(int)
H["cohort5"] = cohort(H.birth_year); P["cohort5"] = cohort(P.birth_year)
H["age5"] = (H.head_age // 5 * 5).astype(int); P["age5"] = (P.Age // 5 * 5).astype(int)
def agg(df, val):
    return df.dropna(subset=[val]).groupby(["cohort5", "age5"]).apply(
        lambda d: pd.Series({"rate": np.average(d[val], weights=d.w), "n_obs": len(d),
                             "years": f"{d.year.min()}-{d.year.max()}"})).reset_index()
ch = agg(H[H.head_age.between(20, 69)], "owner").rename(columns={"rate": "owner_share_heads"})
cp = agg(P[P.Age.between(20, 69)], "own_head_or_spouse").rename(columns={"rate": "owner_head_or_spouse_share_persons"})
C = ch.merge(cp, on=["cohort5", "age5"], how="outer", suffixes=("_heads", "_persons"))
C.to_csv("Data/cohort_ownership.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(Y.round(3).to_string())
print(C[C.age5.isin([25, 30, 35])].pivot(index="cohort5", columns="age5",
      values=["owner_share_heads", "owner_head_or_spouse_share_persons", "n_obs_heads"]).round(3).to_string())

# ---- decade cohorts at the same age (the generational comparison) ----
FRAME_RENEWAL = [1379, 1392, 1397]   # mean-age resets = new census sampling frame (see qc/sample_frame_aging.csv)
def decade(b): return (b // 10 * 10).astype(int)
H["cohort10"] = decade(H.birth_year); P["cohort10"] = decade(P.birth_year)
H["agegrp"] = pd.cut(H.head_age, [24, 29, 34, 39, 44], labels=["25-29", "30-34", "35-39", "40-44"])
P["agegrp"] = pd.cut(P.Age, [24, 29, 34, 39, 44], labels=["25-29", "30-34", "35-39", "40-44"])
rows = []
for (c, a), d in P.dropna(subset=["agegrp", "own_head_or_spouse"]).groupby(["cohort10", "agegrp"], observed=True):
    dh = H[(H.cohort10 == c) & (H.agegrp == a)].dropna(subset=["owner"])
    rows.append(dict(cohort=f"{c}s", age=a,
        persons_owner_head_or_spouse=np.average(d.own_head_or_spouse, weights=d.w),
        persons_head_or_spouse=np.average(d.head_or_spouse, weights=d.w),
        persons_child_of_head=np.average(d.child_of_head, weights=d.w),
        heads_owner_share=np.average(dh.owner, weights=dh.w) if len(dh) else np.nan,
        n_persons=len(d), n_heads=len(dh), survey_years=f"{d.year.min()}-{d.year.max()}",
        birth_years_covered=f"{d.birth_year.min()}-{d.birth_year.max()}"))
D = pd.DataFrame(rows); D.to_csv("Data/cohort_decades_same_age.csv", index=False, encoding="utf-8-sig")
print(D.round(3).to_string())
