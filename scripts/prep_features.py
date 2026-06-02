"""Feature prep + EDA sanity checks for Project 1.

Builds data/model_table.csv (natural-side relative coding at the pre-registered
centre half-width 1.5 yd) and prints the base rates / footedness split used as the
EDA sanity check in the study plan (phase 1).
"""
import pandas as pd
import common as C


def main():
    d = C.load_men(half_width=1.5)
    print("rows after foot filter:", len(d), "| kickers:", d.kicker.nunique())
    print("\nabsolute direction (GK perspective L/C/R):")
    print(d.direction.value_counts(normalize=True).round(3).to_string())
    print("\nnatural-side relative coding (dir_rel):")
    print(d.dir_rel.value_counts(normalize=True).round(3).to_string())
    print("\nfootedness split (modal foot):")
    print(d.foot_dom.value_counts().to_string())
    print("\ndir_rel by shootout vs in-play:")
    print(pd.crosstab(d.is_shootout, d.dir_rel, normalize="index").round(3).to_string())
    print("\ntop kickers by penalty count:")
    print(d.kicker.value_counts().head(8).to_string())
    out = C.DATA / "model_table.csv"
    d.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
