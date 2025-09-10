import os
import json
import numpy as np
from tqdm import tqdm

def load_pred_boxes(pred_path):
    with open(pred_path, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        fields = line.strip().split()
        if len(fields) < 15:
            continue
        x1, y1, x2, y2 = map(float, fields[4:8])
        box2d = [x1, y1, x2, y2]
        h, w, l = map(float, fields[8:11])
        x, y, z = map(float, fields[11:14])
        ry = float(fields[14])
        box3d = [x, y, z, w, h, l, ry]
        boxes.append({"bbox2d": box2d, "bbox3d": box3d})
    return boxes

def boxes_overlap(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(xi2 - xi1, 0)
    ih = max(yi2 - yi1, 0)
    return iw * ih > 0

def merge_json_with_boxes(flat_json_path, pred_dir, output_path):
    with open(flat_json_path, 'r') as f:
        data = json.load(f)

    grouped = {}
    for obj in data:
        image_id = os.path.splitext(obj["image_id"])[0]
        grouped.setdefault(image_id, []).append(obj)

    new_data = []

    for image_id, objects in tqdm(grouped.items()):
        pred_path = os.path.join(pred_dir, image_id + ".txt")
        if not os.path.exists(pred_path):
            
            continue
        print(pred_path)
        pred_boxes = load_pred_boxes(pred_path)
        print(pred_boxes,'1')

        for pred in pred_boxes:
            matched = False
            for obj in objects:
                if "crop_bbox" in obj and boxes_overlap(pred["bbox2d"], obj["crop_bbox"]):
                    obj["bbox3d"] = pred["bbox3d"]
                    new_data.append(obj)
                    matched = True
                    break
            # skip unmatched box

    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)



# 예시 사용
if __name__ == "__main__":
    pred_dir = "../Mono/MonoDGP_ours/output_ffc_key_finetune/monodgp/outputs/data"

    splits = ["train", "val"]
    for split in splits:
        input_json = f"keypoints_with_theta_{split}.json"
        output_json = f"RGR_input_{split}.json"
        merge_json_with_boxes(input_json, pred_dir, output_json)

