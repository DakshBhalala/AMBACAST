"""Production training loop for CompactTS with early stopping, AMP, and metrics tracking."""

from typing import Optional, Dict, Any, List, Tuple
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compactts.models.compact_ts import CompactTS
from compactts.training.loss import get_loss_function, PinballLoss, GaussianNLLLoss
from compactts.utils.hardware import HardwareMonitor


class CompactTSTrainer:
    """Trainer for CompactTS forecasting models."""

    def __init__(
        self,
        model: CompactTS,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        clip_grad: float = 1.0,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model.to(self.device)

        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.clip_grad = clip_grad
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.loss_fn = get_loss_function(
            head_type=model.config.head_type,
            quantiles=model.config.quantiles,
        ).to(self.device)

        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                if self.model.config.head_type == "distribution":
                    mean, std = self.model(batch_x)
                    loss = self.loss_fn(mean, std, batch_y)
                else:
                    preds = self.model(batch_x)
                    loss = self.loss_fn(preds, batch_y)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation dataloader."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                if self.model.config.head_type == "distribution":
                    mean, std = self.model(batch_x)
                    loss = self.loss_fn(mean, std, batch_y)
                else:
                    preds = self.model(batch_x)
                    loss = self.loss_fn(preds, batch_y)

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 15,
        model_name: str = "compact_ts_best.pt",
    ) -> Dict[str, Any]:
        """Fit model with early stopping and checkpointing.

        Returns:
            Dictionary with training summary and metrics history.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_checkpoint_path = os.path.join(self.checkpoint_dir, model_name)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        HardwareMonitor.reset_peak_memory()
        start_time = time.perf_counter()

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self.train_epoch(train_loader)
            scheduler.step()

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
            else:
                val_loss = train_loss

            epoch_time = time.perf_counter() - t0

            # Early stopping check
            improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_checkpoint_path)
                improved = True
            else:
                patience_counter += 1

            epoch_stat = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "best_val_loss": round(best_val_loss, 5),
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_sec": round(epoch_time, 2),
                "improved": improved,
            }
            self.history.append(epoch_stat)

            if patience_counter >= self.patience:
                break

        total_time = time.perf_counter() - start_time
        peak_mem = HardwareMonitor.get_peak_memory()

        # Reload best model weights if saved
        if os.path.exists(best_checkpoint_path):
            self.model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device, weights_only=True))

        return {
            "total_time_sec": round(total_time, 2),
            "best_val_loss": round(best_val_loss, 5),
            "epochs_completed": len(self.history),
            "checkpoint_path": best_checkpoint_path,
            "peak_vram_mb": peak_mem["peak_vram_mb"],
            "process_ram_mb": peak_mem["process_ram_mb"],
            "history": self.history,
        }
