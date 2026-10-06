import os, json

friends = ["dinithi", "savindi"]   
class_names = friends + ["unknown"]

for c in class_names:
    os.makedirs(f"fmfdataset/raw/{c}", exist_ok=True)
os.makedirs("fmfdataset/processed", exist_ok=True)

with open("fmfdataset/config.json", "w") as f:
    json.dump({"class_names": class_names}, f, indent=2)

print("done", class_names)