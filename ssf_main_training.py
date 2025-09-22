# deployment.py
import os
import time
import math
import argparse
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Import your existing helpers (must be available in PYTHONPATH)
from utils import *

# ---------------------------
# Command line arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Deployment/Online retraining for IDS")
parser.add_argument("--dataset", type=str, default="modbus", choices=["nsl", "modbus", "unsw"])
parser.add_argument("--mode", type=str, default="deploy", choices=["train_teacher", "deploy"])
parser.add_argument("--seed", type=int, default=5011)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--teacher_epochs", type=int, default=4)
parser.add_argument("--student_epochs_per_cycle", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--buffer_threshold", type=int, default=20000, help="Number of samples to collect before checking drift / retrain")
parser.add_argument("--sample_interval", type=int, default=20000, help="Used for drift detection sample size (kept for parity)")
parser.add_argument("--percent", type=float, default=0.8)
parser.add_argument("--num_labeled_sample", type=int, default=200)
parser.add_argument("--opt_new_lr", type=float, default=50.0)
parser.add_argument("--opt_old_lr", type=float, default=1.0)
parser.add_argument("--new_sample_weight", type=float, default=100.0)
parser.add_argument("--lwf_lambda", type=float, default=0.5)
parser.add_argument("--drift_threshold", type=float, default=0.05)
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--update_teacher", action="store_true", help="If set, teacher gets updated to student after each retrain cycle")
args = parser.parse_args()

# ---------------------------
# Setup
# ---------------------------
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
os.makedirs(args.save_dir, exist_ok=True)
setup_seed(args.seed)

# Input dim selection 
if args.dataset == "nsl":
    input_dim = 121
elif args.dataset == "modbus":
    # to be updated after processing modbus file
    input_dim = 1
else:
    input_dim = 196

# Loss and other helpers
criterion = InfoNCELoss(device, tem=0.02)
classification_criterion = None
if args.dataset != "nsl":
    classification_criterion = nn.BCELoss(reduction="none")

# ---------------------------
# Utility: load / preprocess train data (offline)
# ---------------------------
def load_offline_data(dataset_name: str):
    """
    Loads training/test csv data using your SplitData and load_data functions.
    Returns (x_train, y_train), (x_test, y_test) as torch tensors on CPU.
    """
    if dataset_name == "nsl":
        KDDTrain_dataset_path = "NSL_pre_data/PKDDTrain+.csv"
        KDDTest_dataset_path = "NSL_pre_data/PKDDTest+.csv"
        KDDTrain = load_data(KDDTrain_dataset_path)
        KDDTest = load_data(KDDTest_dataset_path)
        splitter = SplitData(dataset="nsl")
        x_train, y_train = splitter.transform(KDDTrain, labels="labels2")
        x_test, y_test = splitter.transform(KDDTest, labels="labels2")
    elif dataset_name == "modbus":
        ModbusTrain_dataset_path = "Modbus_pre_data/ModbusTrain.csv"
        ModbusTest_dataset_path = "Modbus_pre_data/ModbusTest.csv"
        ModbusTrain = load_data(ModbusTrain_dataset_path)
        ModbusTest = load_data(ModbusTest_dataset_path)
        splitter = SplitData(dataset="modbus")
        x_train, y_train = splitter.transform(ModbusTrain, labels="label")
        x_test, y_test = splitter.transform(ModbusTest, labels="label")
    else:
        UNSWTrain_dataset_path = "UNSW_pre_data/UNSWTrain.csv"
        UNSWTest_dataset_path = "UNSW_pre_data/UNSWTest.csv"
        UNSWTrain = load_data(UNSWTrain_dataset_path)
        UNSWTest = load_data(UNSWTest_dataset_path)
        splitter = SplitData(dataset="unsw")
        x_train, y_train = splitter.transform(UNSWTrain, labels="label")
        x_test, y_test = splitter.transform(UNSWTest, labels="label")

    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    x_test = torch.FloatTensor(x_test)
    y_test = torch.LongTensor(y_test)
    return (x_train, y_train), (x_test, y_test)

# ---------------------------
# Offline teacher training
# ---------------------------
def train_teacher_and_save(save_path: str):
    """Train teacher once offline on historical clean data, save its weights."""
    (x_train, y_train), (x_test, y_test) = load_offline_data(args.dataset)

    # Create an initial split: simulate prior 'online_x_train' used in your experiment.
    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(x_train, y_train, test_size=args.percent)
    train_ds = TensorDataset(online_x_train, online_y_train)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)

    if args.dataset == "nsl":
        model = AE(input_dim).to(device)
    else:
        model = AE_classifier(input_dim).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(args.teacher_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if args.dataset == "nsl":
                features, recon_vec = model(inputs)
                con_loss = criterion(recon_vec, labels)
                loss = con_loss.mean()
            else:
                features, recon_vec, classifications = model(inputs)
                con_loss = criterion(recon_vec, labels)
                classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                loss = con_loss.mean() + classification_loss.mean()
            loss.backward()
            optimizer.step()

    # Save teacher weights
    torch.save(model.state_dict(), save_path)
    print(f"Teacher model saved to: {save_path}")

    # Optional TorchScript export
    try:
        model.eval()
        example = torch.randn(1, input_dim).to(device)
        scripted = torch.jit.trace(model, example)
        scripted.save(save_path.replace(".pth", ".pt"))
        print("TorchScript model saved.")
    except Exception as e:
        print("TorchScript export failed:", e)

    return model

# ---------------------------
# Real-time data interface (placeholder)
# ---------------------------
def get_realtime_flow_batch() -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Placeholder for real-time data ingestion.
    Replace this to return:
      - features: numpy array of shape (N, input_dim)
      - labels: optional numpy array of shape (N,) if ground truth available (can be None)
    For demonstration, this returns random data (DO NOT use in production).
    """
    # ---- REPLACE THIS WITH YOUR REAL INGESTION/PREPROCESSING ----
    N = 1024  # number of flows captured in this small batch
    features = np.random.randn(N, input_dim).astype(np.float32)
    labels = None  # in real world you often don't have labels immediately
    return features, labels

# ---------------------------
# Retraining routine (student update)
# ---------------------------
def retrain_student(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    x_train_this_epoch: torch.Tensor,
    y_train_this_epoch: torch.Tensor,
    memory: int,
    opt_old_lr: float,
    opt_new_lr: float,
    new_sample_weight: float,
    lwf_lambda: float,
    epoch_1: int,
    detect_drift_flag: bool,
):
    """
    Retrains student_model in-place using either drift-branch or no-drift branch.
    Uses same logic and weighting you used in the experiment.
    """
    # Prepare train_ds with mask computed outside (we will compute mask here)
    # For mask optimization we need control_res, treatment_res first (handled prior to calling this)
    # But to align with your code, retrain_student will accept x_train_this_epoch already containing chosen representative samples
    train_ds = TensorDataset(x_train_this_epoch, y_train_this_epoch)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.SGD(student_model.parameters(), lr=0.001)
    student_model.train()

    for epoch in range(epoch_1):
        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if args.dataset == "nsl":
                features, recon_vec = student_model(inputs)
                con_loss = criterion(recon_vec, labels)
                weighted_loss = con_loss.mean()
                # Distillation with teacher recon
                with torch.no_grad():
                    teacher_features, teacher_recon_vec = teacher_model(inputs)
                distillation_loss = F.mse_loss(recon_vec, teacher_recon_vec)
                total_loss = weighted_loss + lwf_lambda * distillation_loss
            else:
                features, recon_vec, classifications = student_model(inputs)
                con_loss = criterion(recon_vec, labels)
                classification_loss = classification_criterion(classifications.squeeze(), labels.float())
                weighted_loss = con_loss.mean() + classification_loss.mean()
                with torch.no_grad():
                    teacher_features, teacher_recon_vec, teacher_logits = teacher_model(inputs)
                distillation_loss = F.mse_loss(classifications, teacher_logits)
                total_loss = weighted_loss + lwf_lambda * distillation_loss

            total_loss.backward()
            optimizer.step()

    return student_model

# ---------------------------
# Deployment main loop
# ---------------------------
def run_deploy_loop(teacher_path: str):
    # Load offline reference data to compute initial train distribution if needed
    (x_train_offline, y_train_offline), (x_test_offline, y_test_offline) = load_offline_data(args.dataset)

    # Initialize teacher and student
    if args.dataset == "nsl":
        teacher_model = AE(input_dim).to(device)
        student_model = AE(input_dim).to(device)
    else:
        teacher_model = AE_classifier(input_dim).to(device)
        student_model = AE_classifier(input_dim).to(device)

    # Load teacher weights
    if os.path.exists(teacher_path):
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        student_model.load_state_dict(teacher_model.state_dict())
        print("Loaded teacher weights and initialized student from teacher.")
    else:
        raise FileNotFoundError(f"Teacher weights not found at {teacher_path}. Run --mode train_teacher first.")

    # Buffer for incoming unlabeled data (features) and optional labels
    feature_buffer = deque()
    label_buffer = deque()  # may remain empty if no ground truth available

    # We'll store a sliding-window cumulative online train set (x_train_this_epoch) and labels
    # Initially use a small base set from offline training to represent 'normal'
    online_x_train = x_train_offline[: int((1 - args.percent) * len(x_train_offline))].to(device)
    online_y_train = y_train_offline[: int((1 - args.percent) * len(y_train_offline))].to(device)
    memory = int(math.floor(len(x_train_offline) * (1 - args.percent)))
    print(f"Initial memory size set to: {memory}; online_x_train shape: {online_x_train.shape}")

    collected_since_last_retrain = 0

    # Precompute any starting 'normal' reference for NSL branch
    if args.dataset == "nsl":
        with torch.no_grad():
            normal_recon_temp = torch.mean(F.normalize(teacher_model(online_x_train[(online_y_train == 0).squeeze()])[1], p=2, dim=1), dim=0)

    # Start the "real-time" loop
    cycle = 0
    try:
        while True:
            cycle += 1
            # Get an incoming batch from the environment
            features_np, labels_np = get_realtime_flow_batch()  # replace with real ingestion

            # store into buffer
            for i in range(features_np.shape[0]):
                feature_buffer.append(features_np[i])
                label_buffer.append(labels_np[i] if labels_np is not None else -1)  # -1 denotes unknown
            collected_since_last_retrain += features_np.shape[0]

            # If buffer reached threshold -> perform drift detection and possibly retrain
            if collected_since_last_retrain >= args.buffer_threshold:
                print(f"[Cycle {cycle}] Buffer threshold reached ({collected_since_last_retrain}). Performing drift detection...")
                # Prepare test chunk for detection
                # Convert to tensor batch (we'll detect drift comparing test logits vs train logits)
                x_test_chunk = torch.FloatTensor(np.array(feature_buffer)).to(device)
                # If labels are present in buffer use them else set dummy
                y_test_chunk = None
                if any(l >= 0 for l in label_buffer):
                    y_test_chunk = torch.LongTensor(np.array(list(label_buffer))).to(device)

                # Compute logits / recon vectors for drift detection
                if args.dataset == "nsl":
                    # compute recon vectors and PDFs similar to your experiment
                    with torch.no_grad():
                        pdf1, pdf2, _ = evaluate(normal_recon_temp, online_x_train, online_y_train, online_x_train, online_y_train, teacher_model, get_probs=True)
                        pdf11, pdf22, _ = evaluate(normal_recon_temp, online_x_train, online_y_train, x_test_chunk, 0, teacher_model, get_probs=True)
                        pdf1_probe = pdf1 / (pdf1 + pdf2)
                        pdf11_probe = pdf11 / (pdf11 + pdf22)
                        drift_flag = detect_drift(pdf11_probe, pdf1_probe, args.sample_interval, args.drift_threshold)
                        control_res = pdf1_probe.cpu().numpy()
                        treatment_res = pdf11_probe.cpu().numpy()
                else:
                    with torch.no_grad():
                        _, _, test_logits = teacher_model(x_test_chunk)
                        test_logits = test_logits.squeeze()
                    with torch.no_grad():
                        _, _, train_logits = teacher_model(online_x_train)
                        train_logits = train_logits.squeeze()
                    drift_flag = detect_drift(test_logits, train_logits, args.sample_interval, args.drift_threshold)
                    control_res = train_logits.cpu().numpy()
                    treatment_res = test_logits.cpu().numpy()

                print("Drift? ->", drift_flag)

                # Optimize masks (old/new) as in your experiment
                M_c = optimize_old_mask(control_res, treatment_res, device, initialization="0.5-1", lr=args.opt_old_lr)
                M_t = optimize_new_mask(control_res, treatment_res, M_c, device, initialization="0-0.5", lr=args.opt_new_lr)

                # Prepare train candidates (representative selection)
                if drift_flag:
                    x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples_when_drift(
                        online_x_train, online_y_train, x_test_chunk, y_test_chunk if y_test_chunk is not None else torch.zeros(x_test_chunk.shape[0], dtype=torch.long, device=device),
                        M_c, M_t, args.num_labeled_sample, device, memory, student_model, normal_recon_temp if args.dataset == "nsl" else None
                    )
                else:
                    x_train_this_epoch, y_train_this_epoch, labeled_indices_current, new_mask = select_and_update_representative_samples(
                        online_x_train, online_y_train, x_test_chunk, y_test_chunk if y_test_chunk is not None else torch.zeros(x_test_chunk.shape[0], dtype=torch.long, device=device),
                        M_c, M_t, args.num_labeled_sample, device
                    )

                # Retrain student using distillation from teacher
                student_model = retrain_student(
                    student_model,
                    teacher_model,
                    x_train_this_epoch.to(device),
                    y_train_this_epoch.to(device),
                    memory,
                    args.opt_old_lr,
                    args.opt_new_lr,
                    args.new_sample_weight,
                    args.lwf_lambda,
                    args.student_epochs_per_cycle,
                    detect_drift_flag=drift_flag,
                )

                # Optionally update teacher
                if args.update_teacher:
                    teacher_model.load_state_dict(student_model.state_dict())
                    torch.save(teacher_model.state_dict(), os.path.join(args.save_dir, f"teacher_{int(time.time())}.pth"))
                    print("Teacher updated from student and saved.")

                # Save student checkpoint for inference
                torch.save(student_model.state_dict(), os.path.join(args.save_dir, f"student_latest.pth"))
                print("Student model checkpoint saved.")

                # Reset buffer and update online_x_train to include selected representatives
                feature_buffer.clear()
                label_buffer.clear()
                collected_since_last_retrain = 0

                # Update online_x_train/online_y_train to the new selected set (simulate memory sliding window)
                online_x_train = x_train_this_epoch.to(device)
                online_y_train = y_train_this_epoch.to(device)

                # Update normal_recon_temp (for NSL)
                if args.dataset == "nsl":
                    with torch.no_grad():
                        normal_recon_temp = torch.mean(F.normalize(student_model(online_x_train[(online_y_train == 0).squeeze()])[1], p=2, dim=1), dim=0)

            # End of cycle, sleep briefly (in real-time you'd wait for more data)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Deployment loop interrupted by user. Exiting.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    # If training teacher offline
    teacher_path = os.path.join(args.save_dir, f"teacher_{args.dataset}.pth")
    if args.mode == "train_teacher":
        print("Training teacher offline...")
        teacher_model = train_teacher_and_save(teacher_path)
        print("Done.")
    elif args.mode == "deploy":
        print("Starting deploy mode...")
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher model not found at {teacher_path}. Run with --mode train_teacher first.")
        run_deploy_loop(teacher_path)
