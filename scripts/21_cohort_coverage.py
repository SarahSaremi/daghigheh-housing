"""Critique 1: coverage of each birth cohort x age cell, and robustness with 5-year cohorts.
Also: age-standardised check (equal weight to each single age inside the age group), so cells
that only contain the older/younger end of an age group do not drive the comparison."""
import pandas as pd, numpy as np
P = pd.read_parquet("Data/processed/persons_urban.parquet")
P = P[(P.age_group != "nan")]
P["b5"] = (P.birth_year // 5 * 5).astype(int)
def cov(d):
    ok = d.owner.notna()
    by_age = d[ok].groupby("Age").apply(lambda g: np.average(g.own_head_or_spouse, weights=g.w))
    return pd.Series({"n": len(d), "birth_years": f"{d.birth_year.min()}-{d.birth_year.max()}",
                      "n_birth_years": d.birth_year.nunique(), "survey_years": f"{d.year.min()}-{d.year.max()}",
                      "ages_observed": f"{d.Age.min()}-{d.Age.max()}",
                      "owner": np.average(d.own_head_or_spouse[ok], weights=d.w[ok]),
                      "owner_age_std": by_age.mean(),
                      "child": np.average(d.child_of_head, weights=d.w)})
C10 = P[P.birth_decade.between(1330, 1370)].groupby(["birth_decade", "age_group"]).apply(cov).reset_index()
C5 = P[P.b5.between(1330, 1370)].groupby(["b5", "age_group"]).apply(cov).reset_index()
C10.to_csv("Data/cohort_coverage_decades.csv", index=False, encoding="utf-8-sig")
C5.to_csv("Data/cohort_coverage_5yr.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(C10[C10.age_group == "30-34"].round(3).to_string(index=False))
print(C5.pivot(index="b5", columns="age_group", values="owner").round(3).to_string())
print(C5.pivot(index="b5", columns="age_group", values="owner_age_std").round(3).to_string())
print(C5[C5.age_group == "30-34"][["b5", "n", "birth_years", "survey_years", "ages_observed"]].to_string(index=False))
