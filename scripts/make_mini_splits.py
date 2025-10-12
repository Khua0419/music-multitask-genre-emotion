# scripts/make_mini_splits.py
import os, json, random

random.seed(0)

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def dump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

train = load("data/lists/gtzan_train.json")
val   = load("data/lists/gtzan_val.json")

# Only retain files containing the wav key, where the file exists and the genre is 0..9.
def ok(x):
    return isinstance(x.get("wav"), str) and os.path.exists(x["wav"]) \
        and isinstance(x.get("genre"), int) and 0 <= x["genre"] < 10

train = [x for x in train if ok(x)]
val   = [x for x in val   if ok(x)]

random.shuffle(train); random.shuffle(val)
mini_tr = train[:32]
mini_va = val[:8]

dump(mini_tr, "data/lists/gtzan_train_mini.json")
dump(mini_va, "data/lists/gtzan_val_mini.json")

print("mini sizes:", len(mini_tr), len(mini_va))
print("first train item keys:", list(mini_tr[0].keys()) if mini_tr else [])
