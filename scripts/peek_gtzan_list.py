# scripts/peek_gtzan_list.py
import json, os, collections

def peek(path):
    data = json.load(open(path, "r", encoding="utf-8"))
    print(f"\n== {path} ==")
    print("total:", len(data))
    if not data:
        return
    print("sample item:", data[0])

    # 统计键名
    keys = set()
    for x in data[:20]:
        keys |= set(x.keys())
    print("keys (first 20 items):", sorted(keys))

    # 统计路径字段是否存在
    def get_path(x):
        return x.get("path") or x.get("wav") or x.get("file") or x.get("audio")
    missing = sum(1 for x in data if not get_path(x))
    print("items WITHOUT path-like key:", missing)

    # 统计 genre 的类型和示例
    gtypes = collections.Counter(type(x.get("genre")).__name__ for x in data)
    print("genre types:", gtypes)

    # 如果 genre 是字符串，看看都有哪些取值
    glabels = collections.Counter(x.get("genre") for x in data if isinstance(x.get("genre"), str))
    print("top genre labels:", glabels.most_common(10))

peek("data/lists/gtzan_train.json")
peek("data/lists/gtzan_val.json")