"""Person-level urban panel 1369-1402 (one row per person) -> Data/processed/persons_urban.parquet.
Columns: year, ID, Age, Sex, rel, w (household weight), owner (household tenure), tenant,
head_or_spouse, own_head_or_spouse, child_of_head, birth_year, birth_decade, age_group."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, hbsir
OWN = {"Owned_Estate", "Owned_Estate_Land"}
parts = []
for y in range(1369, 1403):
    h = pd.read_parquet(f"Data/processed/hh_urban_{y}.parquet")
    h = h[h.Weight.notna()]
    t = h.Tenure.astype(str)
    tenant = t.isin({"Rent", "Mortgage"}) if y >= 1383 else t.isin({"Rent", "Other"})
    hh = pd.DataFrame({"ID": h.ID, "w": h.Weight.astype(float),
                       "owner": np.where(h.Tenure.notna(), t.isin(OWN), np.nan),
                       "tenant": np.where(h.Tenure.notna(), tenant, np.nan)})
    mp = hbsir.load_table("members_properties", y)[["ID", "Relationship", "Age", "Sex"]].merge(hh, on="ID")
    mp["year"] = y
    parts.append(mp)
    print(y, len(mp), flush=True)
P = pd.concat(parts, ignore_index=True)
P["rel"] = P.Relationship.astype(str); P = P.drop(columns="Relationship")
P["Sex"] = P.Sex.astype(str)
P = P[P.Age.notna()].copy(); P["Age"] = P.Age.astype(int)
P["head_or_spouse"] = P.rel.isin({"Head", "Spouse"}).astype(float)
P["own_head_or_spouse"] = P.head_or_spouse * P.owner
P["child_of_head"] = (P.rel == "Child").astype(float)
P["birth_year"] = P.year - P.Age
P["birth_decade"] = (P.birth_year // 10 * 10).astype(int)
P["age_group"] = pd.cut(P.Age, [24, 29, 34, 39, 44], labels=["25-29", "30-34", "35-39", "40-44"]).astype(str)
P.to_parquet("Data/processed/persons_urban.parquet")
print(P.shape, P.Sex.value_counts().to_dict())
