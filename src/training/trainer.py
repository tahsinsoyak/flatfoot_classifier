"""Training engine with Automatic Mixed Precision (AMP) and evaluation callbacks."""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_medical_metrics


class Trainer:
    """Trainer for binary foot radiograph classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir: Path | str = "experiments/checkpoints",
        use_amp: bool = True,
        early_stopping_patience: int = 10,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and (device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.early_stopping_patience = early_stopping_patience

        self.history = []
        self.best_metric = -1.0
        self.best_epoch = 0

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        y_true_list, y_pred_list, y_prob_list = [], [], []

        for images, labels in tqdm(self.train_loader, desc="Train Epoch", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds)
            y_prob_list.extend(probs)

        avg_loss = total_loss / len(self.train_loader.dataset)
        metrics = compute_medical_metrics(
            np.array(y_true_list), np.array(y_pred_list), np.array(y_prob_list)
        )
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        y_true_list, y_pred_list, y_prob_list = [], [], []

        for images, labels in tqdm(loader, desc="Evaluate", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds)
            y_prob_list.extend(probs)

        avg_loss = total_loss / len(loader.dataset)
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        y_prob = np.array(y_prob_list)
        metrics = compute_medical_metrics(y_true, y_pred, y_prob)

        return avg_loss, metrics, y_true, y_pred, y_prob

    def fit(self, num_epochs: int = 30) -> Dict[str, Any]:
        print(f"Starting training for {num_epochs} epochs on {self.device} (AMP: {self.use_amp})...")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start_t = time.time()
            train_loss, train_metrics = self.train_epoch()
            val_loss, val_metrics, _, _, _ = self.evaluate(self.val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - start_t
            # Monitor validation ROC-AUC or F1
            val_target_metric = val_metrics["roc_auc"]

            log_entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_acc": train_metrics["accuracy"],
                "val_acc": val_metrics["accuracy"],
                "train_auc": train_metrics["roc_auc"],
                "val_auc": val_metrics["roc_auc"],
                "val_sens": val_metrics["sensitivity"],
                "val_spec": val_metrics["specificity"],
                "val_f1": val_metrics["f1_score"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_sec": round(elapsed, 1),
            }
            self.history.append(log_entry)

            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_metrics['accuracy']:.4f} AUC: {train_metrics['roc_auc']:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['roc_auc']:.4f} | "
                f"Sens: {val_metrics['sensitivity']:.4f} Spec: {val_metrics['specificity']:.4f} | "
                f"{elapsed:.1f}s"
            )

            # Checkpoint best model
            if val_target_metric > self.best_metric:
                self.best_metric = val_target_metric
                self.best_epoch = epoch
                patience_counter = 0
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                }, best_path)
                print(f"  --> Saved new best model checkpoint (Val AUC: {val_target_metric:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs (Best epoch: {self.best_epoch})")
                    break

        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.checkpoint_dir / "training_history.csv", index=False)

        return {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "history": self.history,
        }
