# scripts/make_mini_splits_balanced.py
import json, os, random, collections

random.seed(0)

def load_list(p):
    with open(p, "r", encoding="utf-8") as f:
        L = json.load(f)
    # 兼容 'path' 或 'wav' 两种键名，统一转为 'wav'
    out = []
    for x in L:
        wav = x.get("wav") or x.get("path")
        g   = x.get("genre")
        if isinstance(g, int) and 0 <= g < 10 and wav and os.path.exists(wav):
            out.append({"wav": wav, "genre": g, "emotion": x.get("emotion", [0.0, 0.0])})
    return out

train = load_list("data/lists/gtzan_train.json")
val   = load_list("data/lists/gtzan_val.json")

print("loaded:", len(train), len(val))

# 分别按类别聚合
by_g_tr = collections.defaultdict(list)
by_g_va = collections.defaultdict(list)
for x in train: by_g_tr[x["genre"]].append(x)
for x in val:   by_g_va[x["genre"]].append(x)

mini_tr, mini_va = [], []

# 每类取 4 个做训练，1 个做验证（如果不够就尽量取）
for g in range(10):
    random.shuffle(by_g_tr[g])
    random.shuffle(by_g_va[g])
    mini_tr.extend(by_g_tr[g][:4])
    if by_g_va[g]:
        mini_va.append(by_g_va[g][0])
    elif by_g_tr[g][4:5]:
        # 若 val 这类空，就从 train 剩余里“借”1 个做验证
        mini_va.append(by_g_tr[g][4])

print("mini sizes:", len(mini_tr), len(mini_va))

os.makedirs("data/lists", exist_ok=True)
with open("data/lists/gtzan_train_mini.json", "w", encoding="utf-8") as f:
    json.dump(mini_tr, f, ensure_ascii=False, indent=2)
with open("data/lists/gtzan_val_mini.json", "w", encoding="utf-8") as f:
    json.dump(mini_va, f, ensure_ascii=False, indent=2)

print("saved: data/lists/gtzan_train_mini.json , data/lists/gtzan_val_mini.json")

# 打印类别覆盖
def hist(L):
    c = collections.Counter([x["genre"] for x in L])
    return {k:int(v) for k,v in sorted(c.items())}

print("train_hist:", hist(mini_tr))
print("val_hist:",   hist(mini_va))