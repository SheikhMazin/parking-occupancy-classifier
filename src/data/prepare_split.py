import json
import os
from PIL import Image

def process_split(split):
    annotations_path = f"data/raw/PKLot/versions/1/{split}/_annotations.coco.json"
    images_dir = f"data/raw/PKLot/versions/1/{split}"

    with open(annotations_path, "r") as file:
        data = json.load(file)

    image_lookup = {}
    category_lookup = {}
    for image in data["images"]:
        image_lookup[image["id"]] = image["file_name"]
    for cat in data["categories"]:
        if cat["id"] != 0:
            category_lookup[cat["id"]] = cat["name"]

    os.makedirs(f"data/processed/{split}/Empty", exist_ok=True)
    os.makedirs(f"data/processed/{split}/Occupied", exist_ok=True)

    for entry in data["annotations"]:
        image_id = entry["image_id"]
        category_id = entry["category_id"]
        bbox = entry["bbox"]
        box = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        box = [int(num) for num in box]
        ann_id = entry["id"]

        filename = image_lookup[image_id]
        category = category_lookup[category_id]

        if category == "space-empty":
            output_path = f"data/processed/{split}/Empty/{ann_id}.jpg"
        else:
            output_path = f"data/processed/{split}/Occupied/{ann_id}.jpg"

        path = f"{images_dir}/{filename}"
        with Image.open(path) as img:
            cropped_image = img.crop(box)
            cropped_image.save(output_path)

    print(f"Finished processing '{split}' split.")

for split in ["train","valid", "test"]:
    process_split(split)
