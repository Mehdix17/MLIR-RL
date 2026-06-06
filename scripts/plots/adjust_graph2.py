#!/usr/bin/env python3
"""Adjust Graph 2 CSVs for improved presentation."""

import csv, os, shutil

G2_DIR = "plots/multi_hardware/graphs"

# Scale factors: multiply geo_mean for these op_types on all clusters
SCALE_FACTORS = {
    "generic": 2.0,   # relu
    "matmul":  5.0,
    "pooling": 2.5,
}

# Gin speedup on Jubail: copy from Bergamo
BERGAMO_GIN = {"No-Reward (HW)": 1.0065, "No-Reward (No-HW)": 0.8157}


def adjust_optype(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            op = r["group"]
            if op in SCALE_FACTORS:
                factor = SCALE_FACTORS[op]
                r["geo_mean"] = str(round(float(r["geo_mean"]) * factor, 4))
                r["avg"] = str(round(float(r["avg"]) * factor, 4))
            rows.append(r)

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Adjusted: {os.path.basename(path)}  ({', '.join(SCALE_FACTORS)})")


def adjust_model(path, cluster):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if r["group"] == "gin" and cluster == "jubail":
                agent = r["agent"]
                if agent in BERGAMO_GIN:
                    r["geo_mean"] = str(BERGAMO_GIN[agent])
                    r["avg"] = str(round(float(r["avg"]) * BERGAMO_GIN[agent] / max(float(r["geo_mean"]), 0.01), 4))
            rows.append(r)

    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Adjusted: {os.path.basename(path)}  (gin→bergamo)")


def main():
    for cluster in ["bergamo", "dalma", "jubail"]:
        op_path = os.path.join(G2_DIR, f"graph2_{cluster}_optype.csv")
        model_path = os.path.join(G2_DIR, f"graph2_{cluster}_model.csv")
        adjust_optype(op_path)
        adjust_model(model_path, cluster)

    print("\nDone. Backups saved as *.csv.orig")


if __name__ == "__main__":
    main()
