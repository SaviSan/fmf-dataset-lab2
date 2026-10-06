import os, json, random
from sklearn.model_selection import train_test_split

def get_all_paths(root="fmfdataset"):
    class_names = json.load(open(f"{root}/config.json"))["class_names"]
    paths, labels = [], []
    for idx, c in enumerate(class_names):
        folder = f"{root}/raw/{c}"
        for fn in os.listdir(folder):
            if fn.lower().endswith(".jpg"):
                paths.append(os.path.join(folder, fn))
                labels.append(idx)
    return paths, labels, class_names

def split_data(paths, labels):
    random.seed(42)
    data = list(zip(paths, labels))
    random.shuffle(data)
    paths, labels = zip(*data)

    tr_p, va_p, tr_l, va_l = train_test_split(
        list(paths), list(labels),
        test_size=0.2, stratify=labels, random_state=42)
    return tr_p, tr_l, va_p, va_l

if __name__ == "__main__":
    p, l, names = get_all_paths()
    tr_p, tr_l, va_p, va_l = split_data(p, l)
    print("train:", len(tr_p), "val:", len(va_p))