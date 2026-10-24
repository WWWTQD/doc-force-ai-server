import json

def main():
    with open("outputs/ppstructure_anh1_results/sample.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    res = data.get("res", {})
    print("Keys in res:", list(res.keys()))
    for k, v in res.items():
        if isinstance(v, list):
            print(f"Key '{k}': list of length {len(v)}")
            if len(v) > 0:
                print(f"  Sample item 0: {repr(v[0])[:120]}")
        elif isinstance(v, dict):
            print(f"Key '{k}': dict with keys {list(v.keys())}")
        else:
            print(f"Key '{k}': {v}")

if __name__ == "__main__":
    main()
