import json

with open("data/raw/PKLot/versions/1/train/_annotations.coco.json", "r") as file:
    data = json.load(file)

print(data["categories"])

print("-" * 10)

for i in range(4):
    print(data["images"][i])

print("-" * 10)

for j in range(4):
    print(data["annotations"][j])
