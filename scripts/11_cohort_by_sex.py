"""Robustness: is the cohort decline present for both sexes, or driven by marriage/'spouse' role?
birth_decade x age_group x sex: owner as head/spouse, owner as HEAD only, head-or-spouse,
married-or-head proxy, child of head. Weighted (household weight), urban."""
import pandas as pd, numpy as np
P = pd.read_parquet("Data/processed/persons_urban.parquet")
P = P[P.age_group != "nan"]
P["is_head"] = (P.rel == "Head").astype(float)
P["own_head"] = P.is_head * P.owner
def agg(d):
    w = d.w; ok = d.owner.notna()
    return pd.Series({
        "owner_head_or_spouse": np.average(d.own_head_or_spouse[ok], weights=w[ok]),
        "owner_as_head": np.average(d.own_head[ok], weights=w[ok]),
        "head_or_spouse": np.average(d.head_or_spouse, weights=w),
        "head": np.average(d.is_head, weights=w),
        "child_of_head": np.average(d.child_of_head, weights=w),
        "n": len(d), "years": f"{d.year.min()}-{d.year.max()}",
        "birth_years": f"{d.birth_year.min()}-{d.birth_year.max()}"})
T = P.groupby(["birth_decade", "age_group", "Sex"]).apply(agg).reset_index()
T = T[T.birth_decade.between(1330, 1370)]
T.to_csv("Data/cohort_by_sex.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
for v in ["owner_head_or_spouse", "child_of_head", "head_or_spouse", "owner_as_head"]:
    print("\n==", v); print(T.pivot_table(index="birth_decade", columns=["age_group", "Sex"], values=v).round(3).to_string())
print("\n== n"); print(T.pivot_table(index="birth_decade", columns=["age_group", "Sex"], values="n").to_string())
