#!/usr/bin/env python3
import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def key_from_name(name: str, prefix: str) -> str:
    if name.startswith(prefix):
        return name[len(prefix) :]
    # Fallback: keep only trailing digits for pairing safety.
    m = re.search(r"(\d+)$", name)
    return m.group(1) if m else name


def list_videos(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out[p.stem] = p
    return out


@dataclass
class Triplet:
    key: str
    left: Path
    right: Path
    ground: Path
    frames: int


def discover_triplets(left_dir: Path, right_dir: Path, ground_dir: Path) -> List[Triplet]:
    lefts = list_videos(left_dir)
    rights = list_videos(right_dir)
    grounds = list_videos(ground_dir)

    left_by_key = {key_from_name(stem, "cam_L"): path for stem, path in lefts.items()}
    right_by_key = {key_from_name(stem, "cam_R"): path for stem, path in rights.items()}
    ground_by_key = {key_from_name(stem, "cam_REF"): path for stem, path in grounds.items()}

    keys = sorted(set(left_by_key) & set(right_by_key) & set(ground_by_key))
    triplets: List[Triplet] = []

    for key in keys:
        l, r, g = left_by_key[key], right_by_key[key], ground_by_key[key]
        cap_l = cv2.VideoCapture(str(l))
        cap_r = cv2.VideoCapture(str(r))
        cap_g = cv2.VideoCapture(str(g))
        n_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
        n_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
        n_g = int(cap_g.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_l.release()
        cap_r.release()
        cap_g.release()
        frames = min(n_l, n_r, n_g)
        if frames > 2:
            triplets.append(Triplet(key=key, left=l, right=r, ground=g, frames=frames))

    return triplets


class StereoVideoDataset(Dataset):
    def __init__(
        self,
        triplets: List[Triplet],
        size: Tuple[int, int],
        alpha: int,
        frame_stride: int,
        max_frames_per_video: int,
        augment: bool = False,
    ) -> None:
        self.triplets = triplets
        self.width, self.height = size
        self.alpha = alpha
        self.frame_stride = frame_stride
        self.augment = augment
        self.index_map: List[Tuple[int, int]] = []
        self.captures: Dict[Tuple[int, str], cv2.VideoCapture] = {}

        for t_idx, t in enumerate(triplets):
            start = max(alpha, 1)
            stop = max(start, t.frames - alpha)
            frame_indices = list(range(start, stop, frame_stride))
            if max_frames_per_video > 0:
                frame_indices = frame_indices[:max_frames_per_video]
            self.index_map.extend((t_idx, f_idx) for f_idx in frame_indices)

    def __len__(self) -> int:
        return len(self.index_map)

    def _get_cap(self, t_idx: int, which: str) -> cv2.VideoCapture:
        key = (t_idx, which)
        if key in self.captures:
            return self.captures[key]

        triplet = self.triplets[t_idx]
        path = triplet.left if which == "left" else triplet.right if which == "right" else triplet.ground
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {path}")
        self.captures[key] = cap
        return cap

    def _read_frame(self, t_idx: int, which: str, frame_idx: int) -> torch.Tensor:
        cap = self._get_cap(t_idx, which)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed reading frame {frame_idx} from {which} video")

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        return torch.from_numpy(frame).permute(2, 0, 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_idx, f = self.index_map[idx]
        a = self.alpha

        x1 = self._read_frame(t_idx, "left", max(0, f - a))
        x2 = self._read_frame(t_idx, "left", max(0, f - 1))
        x3 = self._read_frame(t_idx, "left", f)
        x4 = self._read_frame(t_idx, "left", min(self.triplets[t_idx].frames - 1, f + 1))
        x5 = self._read_frame(t_idx, "left", min(self.triplets[t_idx].frames - 1, f + a))

        # Teacher forcing on temporal state: previous right frame as x0.
        x0 = self._read_frame(t_idx, "right", max(0, f - 1))

        target_right = self._read_frame(t_idx, "right", f)
        target_ground = self._read_frame(t_idx, "ground", f)

        if self.augment:
            # Color jitter applied consistently to all left input frames only.
            # x0/target_right/target_ground are NOT jittered to preserve target accuracy.
            brightness = random.uniform(0.7, 1.3)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)

            def _jitter(t: torch.Tensor) -> torch.Tensor:
                t = TF.adjust_brightness(t, brightness)
                t = TF.adjust_contrast(t, contrast)
                t = TF.adjust_saturation(t, saturation)
                t = TF.adjust_hue(t, hue)
                return t

            x1, x2, x3, x4, x5 = _jitter(x1), _jitter(x2), _jitter(x3), _jitter(x4), _jitter(x5)

        model_input = torch.cat((x1, x2, x0, x3, x4, x5), dim=0)
        return {
            "input": model_input,
            "right": target_right,
            "ground": target_ground,
        }


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    best_val_l1: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_l1": best_val_l1,
        "args": vars(args),
    }
    if scaler is not None:
        ckpt["scaler_state"] = scaler.state_dict()
    torch.save(ckpt, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Deep3D TorchScript model on stereo triplets")
    parser.add_argument("--data-left", type=Path, required=True)
    parser.add_argument("--data-right", type=Path, required=True)
    parser.add_argument("--data-ground", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=str, default="auto", help="auto | path | none")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--alpha", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--train-width", type=int, default=640)
    parser.add_argument("--train-height", type=int, default=360)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--ground-loss-weight", type=float, default=0.0)
    parser.add_argument("--parallax-loss-weight", type=float, default=0.1,
                        help="weight for anti-collapse loss: penalizes pred being identical to left input")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="enable color jitter augmentation on left input frames")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    meta_file = args.save_dir / "dataset_summary.json"

    triplets = discover_triplets(args.data_left, args.data_right, args.data_ground)
    if not triplets:
        raise RuntimeError("No matched left/right/ground video triplets found.")

    dataset = StereoVideoDataset(
        triplets=triplets,
        size=(args.train_width, args.train_height),
        alpha=args.alpha,
        frame_stride=args.frame_stride,
        max_frames_per_video=args.max_frames_per_video,
        augment=args.augment,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after frame sampling settings.")

    n_train = max(1, int(len(dataset) * args.train_split))
    n_train = min(n_train, len(dataset) - 1) if len(dataset) > 1 else 1
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:] if len(dataset) > 1 else indices[:1]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    meta = {
        "triplets": [
            {
                "key": t.key,
                "left": str(t.left),
                "right": str(t.right),
                "ground": str(t.ground),
                "frames": t.frames,
            }
            for t in triplets
        ],
        "total_samples": len(dataset),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
    }
    meta_file.write_text(json.dumps(meta, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] total_samples={len(dataset)} train={len(train_ds)} val={len(val_ds)}")

    model = torch.jit.load(str(args.pretrained), map_location=device)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "Loaded model has no trainable parameters. Use a trainable (non-frozen) model artifact."
        )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    last_ckpt = args.save_dir / "last.pt"
    start_epoch = 1
    best_val_l1 = float("inf")

    resume_path: Optional[Path] = None
    if args.resume.lower() == "auto" and last_ckpt.exists():
        resume_path = last_ckpt
    elif args.resume.lower() != "none" and args.resume.lower() != "auto":
        resume_path = Path(args.resume)

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_l1 = float(ckpt.get("best_val_l1", best_val_l1))
        print(f"[INFO] Resumed from {resume_path} at epoch {start_epoch}")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_l1 = 0.0
        train_count = 0

        for batch in train_loader:
            x = batch["input"].to(device, non_blocking=True)
            y_right = batch["right"].to(device, non_blocking=True)
            y_ground = batch["ground"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)
                loss_right = F.l1_loss(pred, y_right)
                # x3 is the center left frame: channels 9:12 in the 6-frame×3ch input.
                # Negative L1 against left input forces pred away from identity mapping.
                x3_left = x[:, 9:12, :, :]
                loss_parallax = -F.l1_loss(pred, x3_left)
                loss = loss_right + args.parallax_loss_weight * loss_parallax
                if args.ground_loss_weight > 0:
                    loss_ground = F.l1_loss(pred, y_ground)
                    loss = loss + args.ground_loss_weight * loss_ground

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if global_step <= 50:
                print(
                    f"[step {global_step:03d}] loss_right={loss_right.item():.6f} "
                    f"loss_parallax(raw)={-loss_parallax.item():.6f} "
                    f"ratio={(-loss_parallax.item() / max(loss_right.item(), 1e-9)):.2f}x",
                    flush=True,
                )

            bs = x.size(0)
            train_l1 += loss_right.item() * bs
            train_count += bs

        train_l1 /= max(1, train_count)

        model.eval()
        val_l1 = 0.0
        val_psnr = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device, non_blocking=True)
                y_right = batch["right"].to(device, non_blocking=True)
                pred = model(x)
                l1 = F.l1_loss(pred, y_right)
                bs = x.size(0)
                val_l1 += l1.item() * bs
                val_psnr += psnr(pred, y_right) * bs
                val_count += bs

        val_l1 /= max(1, val_count)
        val_psnr /= max(1, val_count)
        print(
            f"[Epoch {epoch:03d}] train_l1={train_l1:.6f} val_l1={val_l1:.6f} val_psnr={val_psnr:.2f}"
        )

        save_checkpoint(last_ckpt, epoch, model, optimizer, scaler, best_val_l1, args)
        if val_l1 < best_val_l1:
            best_val_l1 = val_l1
            save_checkpoint(args.save_dir / "best.pt", epoch, model, optimizer, scaler, best_val_l1, args)

    print("[INFO] Training complete")


if __name__ == "__main__":
    main()
