"""New analysis A+B: living arrangements of urban people at the same age, by birth cohort (repeated cross-sections,
'synthetic cohorts'), and decomposition  P(owner & independent) = P(independent) x P(owner | independent).
independent = head or spouse of head. Categories (mutually exclusive):
  parents' home   = child of head
  owner           = head/spouse in an owner-occupied home (household tenure, not legal title)
  renter          = head/spouse, tenure rent or deposit (رهن) [<1383: rent or 'other']
  other tenure    = head/spouse, rent-free / employer-provided / other
  other relation  = child-in-law, grandchild, sibling, parent, non-family, etc."""
import pandas as pd, numpy as np
P = pd.read_parquet("Data/processed/persons_urban.parquet")
P = P[(P.age_group != "nan") & P.owner.notna() & P.tenant.notna()].copy()
P["b5"] = (P.birth_year // 5 * 5).astype(int)
hs = P.head_or_spouse == 1
P["cat"] = np.select([P.child_of_head == 1, hs & (P.owner == 1), hs & (P.tenant == 1), hs],
                     ["parents_home", "owner", "renter", "other_tenure"], "other_relation")
CATS = ["parents_home", "renter", "owner", "other_tenure", "other_relation"]
def comp(d):
    s = d.groupby("cat").w.sum() / d.w.sum(); s = s.reindex(CATS).fillna(0)
    indep = s.owner + s.renter + s.other_tenure
    return pd.concat([s, pd.Series({"independent": indep, "owner_given_independent": s.owner / indep, "n": len(d)})])
out = []
for key, lab in [("birth_decade", "decade"), ("b5", "5yr")]:
    T = P[P[key].between(1330, 1370)].groupby([key, "age_group"]).apply(comp).reset_index().rename(columns={key: "cohort"})
    T["cohort_type"] = lab; out.append(T)
T = pd.concat(out); T.to_csv("Data/living_arrangements_by_cohort.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
d = T[(T.cohort_type == "5yr") & (T.age_group == "30-34")]
print(d[["cohort"] + CATS + ["independent", "owner_given_independent", "n"]].round(3).to_string(index=False))
# decomposition of the change in owner share between two fully-covered 5-yr cohorts (log shares split exactly)
a, b = d.set_index("cohort").loc[1340], d.set_index("cohort").loc[1365]
dl = np.log(b.owner / a.owner); di = np.log(b.independent / a.independent); dc = np.log(b.owner_given_independent / a.owner_given_independent)
print(f"\n1340-44 -> 1365-69 at 30-34: owner {a.owner:.3f} -> {b.owner:.3f}; independent {a.independent:.3f} -> {b.independent:.3f}; "
      f"owner|indep {a.owner_given_independent:.3f} -> {b.owner_given_independent:.3f}")
print(f"share of log-decline from lower independence: {di / dl:.0%}; from lower ownership among independent: {dc / dl:.0%}")
