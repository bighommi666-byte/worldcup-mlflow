"""
Append verified 2018 (Russia) and 2022 (Qatar) World Cup matches to
WorldCupMatches.csv, matching the dataset's existing format and conventions:
  - penalty-decided knockout matches are recorded as draws (score after ET),
    with the shootout noted in 'Win conditions'
  - team names use the same forms the cleaner normalises (IR Iran -> Iran,
    Korea Republic -> South Korea, etc.)
Each tuple: (Year, Stage, HomeTeam, HomeGoals, AwayGoals, AwayTeam, WinCond)
"""
import pandas as pd

P = ""  # empty win-condition

m = []  # (year, stage, home, hg, ag, away, wincond)

# ============================ 2018 — GROUP STAGE ============================
g18 = [
    ("Group A","Russia",5,0,"Saudi Arabia"), ("Group A","Egypt",0,1,"Uruguay"),
    ("Group B","Morocco",0,1,"IR Iran"), ("Group B","Portugal",3,3,"Spain"),
    ("Group C","France",2,1,"Australia"), ("Group D","Argentina",1,1,"Iceland"),
    ("Group C","Peru",0,1,"Denmark"), ("Group D","Croatia",2,0,"Nigeria"),
    ("Group E","Costa Rica",0,1,"Serbia"), ("Group F","Germany",0,1,"Mexico"),
    ("Group E","Brazil",1,1,"Switzerland"), ("Group F","Sweden",1,0,"Korea Republic"),
    ("Group G","Belgium",3,0,"Panama"), ("Group G","Tunisia",1,2,"England"),
    ("Group H","Colombia",1,2,"Japan"), ("Group H","Poland",1,2,"Senegal"),
    ("Group A","Russia",3,1,"Egypt"), ("Group B","Portugal",1,0,"Morocco"),
    ("Group A","Uruguay",1,0,"Saudi Arabia"), ("Group B","IR Iran",0,1,"Spain"),
    ("Group C","Denmark",1,1,"Australia"), ("Group C","France",1,0,"Peru"),
    ("Group D","Argentina",0,3,"Croatia"), ("Group E","Brazil",2,0,"Costa Rica"),
    ("Group D","Nigeria",2,0,"Iceland"), ("Group E","Serbia",1,2,"Switzerland"),
    ("Group G","Belgium",5,2,"Tunisia"), ("Group F","Korea Republic",1,2,"Mexico"),
    ("Group F","Germany",2,1,"Sweden"), ("Group G","England",6,1,"Panama"),
    ("Group H","Japan",2,2,"Senegal"), ("Group H","Poland",0,3,"Colombia"),
    ("Group A","Saudi Arabia",2,1,"Egypt"), ("Group A","Uruguay",3,0,"Russia"),
    ("Group B","IR Iran",1,1,"Portugal"), ("Group B","Spain",2,2,"Morocco"),
    ("Group C","Australia",0,2,"Peru"), ("Group C","Denmark",0,0,"France"),
    ("Group D","Nigeria",1,2,"Argentina"), ("Group D","Iceland",1,2,"Croatia"),
    ("Group F","Korea Republic",2,0,"Germany"), ("Group F","Mexico",0,3,"Sweden"),
    ("Group E","Switzerland",2,2,"Costa Rica"), ("Group E","Serbia",0,2,"Brazil"),
    ("Group H","Senegal",0,1,"Colombia"), ("Group H","Japan",0,1,"Poland"),
    ("Group G","England",0,1,"Belgium"), ("Group G","Panama",1,2,"Tunisia"),
]
for stage,h,hg,ag,a in g18:
    m.append((2018,stage,h,hg,ag,a,P))

# ============================ 2018 — KNOCKOUT =============================
k18 = [
    ("Round of 16","France",4,3,"Argentina",P),
    ("Round of 16","Uruguay",2,1,"Portugal",P),
    ("Round of 16","Spain",1,1,"Russia","Russia win on penalties (4 - 3)"),
    ("Round of 16","Croatia",1,1,"Denmark","Croatia win on penalties (3 - 2)"),
    ("Round of 16","Brazil",2,0,"Mexico",P),
    ("Round of 16","Belgium",3,2,"Japan",P),
    ("Round of 16","Sweden",1,0,"Switzerland",P),
    ("Round of 16","Colombia",1,1,"England","England win on penalties (4 - 3)"),
    ("Quarter-finals","Uruguay",0,2,"France",P),
    ("Quarter-finals","Brazil",1,2,"Belgium",P),
    ("Quarter-finals","Sweden",0,2,"England",P),
    ("Quarter-finals","Russia",2,2,"Croatia","Croatia win on penalties (4 - 3)"),
    ("Semi-finals","France",1,0,"Belgium",P),
    ("Semi-finals","Croatia",2,1,"England","Croatia win after extra time"),
    ("Play-off for third place","Belgium",2,0,"England",P),
    ("Final","France",4,2,"Croatia",P),
]
for stage,h,hg,ag,a,wc in k18:
    m.append((2018,stage,h,hg,ag,a,wc))

# ============================ 2022 — GROUP STAGE ============================
g22 = [
    ("Group A","Qatar",0,2,"Ecuador"), ("Group B","England",6,2,"IR Iran"),
    ("Group A","Senegal",0,2,"Netherlands"), ("Group B","USA",1,1,"Wales"),
    ("Group C","Argentina",1,2,"Saudi Arabia"), ("Group D","Denmark",0,0,"Tunisia"),
    ("Group C","Mexico",0,0,"Poland"), ("Group D","France",4,1,"Australia"),
    ("Group F","Morocco",0,0,"Croatia"), ("Group E","Germany",1,2,"Japan"),
    ("Group F","Belgium",1,0,"Canada"), ("Group E","Spain",7,0,"Costa Rica"),
    ("Group G","Switzerland",1,0,"Cameroon"), ("Group H","Uruguay",0,0,"Korea Republic"),
    ("Group G","Brazil",2,0,"Serbia"), ("Group H","Portugal",3,2,"Ghana"),
    ("Group A","Qatar",1,3,"Senegal"), ("Group A","Netherlands",1,1,"Ecuador"),
    ("Group B","Wales",0,2,"IR Iran"), ("Group B","England",0,0,"USA"),
    ("Group C","Poland",2,0,"Saudi Arabia"), ("Group C","Argentina",2,0,"Mexico"),
    ("Group D","Tunisia",0,1,"Australia"), ("Group D","France",2,1,"Denmark"),
    ("Group E","Japan",0,1,"Costa Rica"), ("Group E","Spain",1,1,"Germany"),
    ("Group F","Belgium",0,2,"Morocco"), ("Group F","Croatia",4,1,"Canada"),
    ("Group G","Cameroon",3,3,"Serbia"), ("Group G","Brazil",1,0,"Switzerland"),
    ("Group H","Korea Republic",2,3,"Ghana"), ("Group H","Portugal",2,0,"Uruguay"),
    ("Group A","Ecuador",1,2,"Senegal"), ("Group A","Netherlands",2,0,"Qatar"),
    ("Group B","Wales",0,3,"England"), ("Group B","IR Iran",0,1,"USA"),
    ("Group C","Poland",0,2,"Argentina"), ("Group C","Saudi Arabia",1,2,"Mexico"),
    ("Group D","Australia",1,0,"Denmark"), ("Group D","Tunisia",1,0,"France"),
    ("Group E","Japan",2,1,"Spain"), ("Group E","Costa Rica",2,4,"Germany"),
    ("Group F","Croatia",0,0,"Belgium"), ("Group F","Canada",1,2,"Morocco"),
    ("Group G","Serbia",2,3,"Switzerland"), ("Group G","Cameroon",1,0,"Brazil"),
    ("Group H","Korea Republic",2,1,"Portugal"), ("Group H","Ghana",0,2,"Uruguay"),
]
for stage,h,hg,ag,a in g22:
    m.append((2022,stage,h,hg,ag,a,P))

# ============================ 2022 — KNOCKOUT =============================
k22 = [
    ("Round of 16","Netherlands",3,1,"USA",P),
    ("Round of 16","Argentina",2,1,"Australia",P),
    ("Round of 16","France",3,1,"Poland",P),
    ("Round of 16","England",3,0,"Senegal",P),
    ("Round of 16","Japan",1,1,"Croatia","Croatia win on penalties (3 - 1)"),
    ("Round of 16","Brazil",4,1,"Korea Republic",P),
    ("Round of 16","Morocco",0,0,"Spain","Morocco win on penalties (3 - 0)"),
    ("Round of 16","Portugal",6,1,"Switzerland",P),
    ("Quarter-finals","Croatia",1,1,"Brazil","Croatia win on penalties (4 - 2)"),
    ("Quarter-finals","Netherlands",2,2,"Argentina","Argentina win on penalties (4 - 3)"),
    ("Quarter-finals","Morocco",1,0,"Portugal",P),
    ("Quarter-finals","England",1,2,"France",P),
    ("Semi-finals","Argentina",3,0,"Croatia",P),
    ("Semi-finals","France",2,0,"Morocco",P),
    ("Play-off for third place","Croatia",2,1,"Morocco",P),
    ("Final","Argentina",3,3,"France","Argentina win on penalties (4 - 2)"),
]
for stage,h,hg,ag,a,wc in k22:
    m.append((2022,stage,h,hg,ag,a,wc))

print(f"Total new matches built: {len(m)}  (expected 128)")
n2018 = sum(1 for r in m if r[0]==2018); n2022 = sum(1 for r in m if r[0]==2022)
print(f"  2018: {n2018}   2022: {n2022}")

# ---- Build rows matching the existing CSV columns ----
orig = pd.read_csv("data/WorldCupMatches.csv")
cols = list(orig.columns)

new_rows = []
for (yr, stage, home, hg, ag, away, wc) in m:
    row = {c: "" for c in cols}
    row["Year"] = yr
    row["Stage"] = stage
    row["Home Team Name"] = home
    row["Home Team Goals"] = hg
    row["Away Team Goals"] = ag
    row["Away Team Name"] = away
    row["Win conditions"] = wc
    new_rows.append(row)

new_df = pd.DataFrame(new_rows, columns=cols)
combined = pd.concat([orig, new_df], ignore_index=True)
combined.to_csv("data/WorldCupMatches.csv", index=False)
print(f"\nCSV updated. Total rows now: {len(combined)}")
