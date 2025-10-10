# scripts/best_epoch.py  (robust: supports acc/f1 or val_acc/val_f1)
import csv

path = r"experiments/logs/genre_curve.csv"

def get_float(row, *keys):
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            try:
                return float(v)
            except ValueError:
                pass
    return 0.0

best_by_acc = (-1.0, None)  # (acc, row)
best_by_f1  = (-1.0, None)

with open(path, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        acc = get_float(r, "val_acc", "acc")
        f1  = get_float(r, "val_f1",  "f1")
        if acc > best_by_acc[0]:
            best_by_acc = (acc, r)
        if f1  > best_by_f1[0]:
            best_by_f1  = (f1,  r)

row_acc = best_by_acc[1]
row_f1  = best_by_f1[1]

print("BEST BY ACC -> epoch:", row_acc.get("epoch"), "acc:", row_acc.get("val_acc") or row_acc.get("acc"), "f1:", row_acc.get("val_f1") or row_acc.get("f1"))
print("BEST BY F1  -> epoch:", row_f1.get("epoch"),  "acc:", row_f1.get("val_acc")  or row_f1.get("acc"),  "f1:", row_f1.get("val_f1")  or row_f1.get("f1"))
