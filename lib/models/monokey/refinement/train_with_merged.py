import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Import the existing model components
import sys
sys.path.append('../model')
from box_merge import load_kitti_pred_file

# --------------- KITTI GT 로드 함수 ---------------
def parse_kitti_label_file(label_path):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts[0] != 'Car':
                continue
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            boxes.append([x, y, z, w, h, l, ry])
    return boxes

# --------------- Dataset 정의 ---------------
class GraphRefineDataset(Dataset):
    def __init__(self, json_path, label_dir, merge_output_dir, train_list_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.label_dir = label_dir
        self.merge_output_dir = merge_output_dir
        
        # Load train.txt to filter by image IDs
        with open(train_list_path, 'r') as f:
            self.train_ids = set(line.strip() for line in f)
        
        # Filter data to only include train images
        self.data = [obj for obj in self.data if obj["image_id"].replace(".png", "") in self.train_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obj = self.data[idx]
        image_id = obj["image_id"].replace(".png", "")
        
        # Load merged prediction from merge_output
        merge_path = os.path.join(self.merge_output_dir, f"{image_id}.txt")
        if os.path.exists(merge_path):
            merged_detections = load_kitti_pred_file(merge_path)
            if merged_detections:
                # Use the first merged detection as bbox3d
                bbox3d = torch.tensor(merged_detections[0]["box3d"], dtype=torch.float32)
            else:
                # Fallback to original bbox3d if no merged detection
                bbox3d = torch.tensor(obj["bbox3d"], dtype=torch.float32)
        else:
            # Fallback to original bbox3d if no merged file
            bbox3d = torch.tensor(obj["bbox3d"], dtype=torch.float32)
        
        keypoints = torch.tensor(obj["keypoints"], dtype=torch.float32).flatten()
        theta = torch.tensor([obj["theta"]], dtype=torch.float32)
        crop_bbox = torch.tensor(obj["crop_bbox"], dtype=torch.float32)

        feature = torch.cat([bbox3d, keypoints, theta], dim=0)  # (32,)
        center = bbox3d[:3]  # x, y, z

        # Load GT box from label_2
        label_path = os.path.join(self.label_dir, f"{image_id}.txt")
        gt_boxes = parse_kitti_label_file(label_path)

        # Match by nearest center
        min_dist = float('inf')
        matched_gt = None
        for gt in gt_boxes:
            gt_center = np.array(gt[:3])
            dist = np.linalg.norm(gt_center - center.numpy())
            if dist < min_dist:
                min_dist = dist
                matched_gt = gt

        if matched_gt is None:
            # If no GT found, use a default box
            matched_gt = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        
        gt_bbox3d = torch.tensor(matched_gt, dtype=torch.float32)

        return {
            "feature": feature,
            "center": center,
            "gt_bbox3d": gt_bbox3d,
            "init_bbox3d": bbox3d,
            "crop_bbox": crop_bbox,
            "image_id": obj["image_id"]
        }

# --------------- Graph Utility 함수 ---------------
def build_distance_adj_matrix(pos, threshold=3.0):
    N = pos.size(0)
    dists = torch.cdist(pos, pos)
    adj = (dists < threshold).float()
    adj.fill_diagonal_(0)
    deg = adj.sum(1, keepdim=True)
    adj = adj / (deg + 1e-6)
    return adj

# --------------- GCN Layer 정의 ---------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        agg = torch.matmul(adj, x)
        return F.relu(self.linear(agg))

# --------------- GNN 모델 정의 ---------------
class GraphRefinementModule(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, out_dim=7):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, out_dim)

    def forward(self, feat, center):
        adj = build_distance_adj_matrix(center, threshold=3.0)
        x = F.relu(self.linear1(feat))
        x = F.relu(self.linear2(x))
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        delta = self.regressor(x)
        return delta

# --------------- KITTI 저장 함수 ---------------
def save_kitti_format(refined_bbox, crop_bbox, image_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    obj_class = "Car"
    truncated, occluded, alpha = 0.00, 0, -1.67
    x1, y1, x2, y2 = crop_bbox.tolist()
    x, y, z, w, h, l, ry = refined_bbox.tolist()
    line = f"{obj_class} {truncated:.2f} {occluded} {alpha:.2f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f}"

    save_path = os.path.join(save_dir, image_id.replace(".png", ".txt"))
    with open(save_path, 'a') as f:
        f.write(line + "\n")

# --------------- Collate 함수 ---------------
def collate_fn(batch):
    return {
        "feature": torch.stack([b["feature"] for b in batch]),
        "center": torch.stack([b["center"] for b in batch]),
        "gt_bbox3d": torch.stack([b["gt_bbox3d"] for b in batch]),
        "init_bbox3d": torch.stack([b["init_bbox3d"] for b in batch]),
        "crop_bbox": [b["crop_bbox"] for b in batch],
        "image_id": [b["image_id"] for b in batch]
    }

# --------------- 학습 + 저장 파이프라인 ---------------
if __name__ == "__main__":
    # Paths
    json_path = "keypoints_with_theta_train.json"
    label_dir = "label_2"
    merge_output_dir = "merge_output"
    train_list_path = "ImageSets/train.txt"
    
    # Create logs directory
    log_dir = "../logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training_log_merged.txt")

    # 로그 파일 헤더 작성
    with open(log_path, "w") as f:
        f.write(f"Training log (merged outputs) - {datetime.now()}\n")
        f.write("epoch,avg_loss,reg_loss,ang_loss\n")

    def angle_diff_wrap(pred, gt):
        return torch.atan2(torch.sin(pred - gt), torch.cos(pred - gt))

    # Dataset and DataLoader
    dataset = GraphRefineDataset(json_path, label_dir, merge_output_dir, train_list_path)
    print(f"Dataset size: {len(dataset)} samples")
    
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    # Model and optimizer
    model = GraphRefinementModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(500):
        model.train()
        running_loss = 0.0
        running_reg_loss = 0.0
        running_ang_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            delta = model(batch["feature"], batch["center"])
            refined = batch["init_bbox3d"] + delta
            target = batch["gt_bbox3d"]

            resid_lin = refined[:, :6] - target[:, :6]
            resid_ang = angle_diff_wrap(refined[:, 6], target[:, 6])

            reg_loss = F.smooth_l1_loss(resid_lin, torch.zeros_like(resid_lin))
            ang_loss = F.smooth_l1_loss(resid_ang, torch.zeros_like(resid_ang))
            loss = reg_loss + 2.0 * ang_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_reg_loss += reg_loss.item()
            running_ang_loss += ang_loss.item()

        avg_loss = running_loss / len(loader)
        avg_reg_loss = running_reg_loss / len(loader)
        avg_ang_loss = running_ang_loss / len(loader)

        # 콘솔 출력
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f} "
            f"(Reg: {avg_reg_loss:.4f}, Ang: {avg_ang_loss:.4f})")

        # 로그 파일 저장
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_reg_loss:.6f},{avg_ang_loss:.6f}\n")

        # 모델 저장
        torch.save(model.state_dict(), f"../graph_refine_merged_epoch{epoch+1}.pth")

    # Inference and save predictions
    model.eval()
    save_dir = "../output_labels_merged"
    with torch.no_grad():
        for sample in tqdm(dataset, desc="Saving Predictions"):
            delta = model(sample["feature"].unsqueeze(0), sample["center"].unsqueeze(0))
            refined = sample["init_bbox3d"] + delta.squeeze(0)
            save_kitti_format(refined, sample["crop_bbox"], sample["image_id"], save_dir)
