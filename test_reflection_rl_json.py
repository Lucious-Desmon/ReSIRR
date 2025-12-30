"""
python test_reflection_rl_json.py \
  --json_base_dir /sirr2/datasets/DRR/jsonl \
  --json_files real=real/metadata_test_real.json SIR2=SIR2/SIR2.json \
  --output_dir ./rl_benchmark_outputs \
  --model_dir ./model/reflect_30000 \
  --episode_len 6 \
  --n_actions 9 \
  --move_step 0.02 \
  --batch_size 1 \
  --compute_lpips \
  --deterministic

"""

import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import MyFCN_el
import pixelwise_a3c_el
from State_reflect import StateReflect
from custom_dataset import JsonDataset
from evaluation_utils import tensor_to_numpy, calculate_metrics, MetricsTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL dereflection benchmark on json datasets.")
    parser.add_argument("--json_base_dir", type=str, default="./test")
    parser.add_argument(
        "--json_files",
        nargs="+",
        help="Override dataset json files, format: name=relative/or/abs/path.json",
    )
    parser.add_argument("--output_dir", type=str, default="./rl_benchmark_outputs")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved RL snapshot directory.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--episode_len", type=int, default=6)
    parser.add_argument("--n_actions", type=int, default=9)
    parser.add_argument("--move_step", type=float, default=0.02)
    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_json_map(args: argparse.Namespace) -> Dict[str, str]:
    default_map = {
        "nature": "nature/metadata_test_nature.json",
        "real": "real/metadata_test_real.json",
        "SIR2": "SIR2/SIR2.json",
    }
    if args.json_files:
        default_map = {}
        for item in args.json_files:
            if "=" not in item:
                raise ValueError(f"Invalid json_files entry '{item}', expected name=path.json")
            name, path = item.split("=", 1)
            default_map[name] = path
    resolved = {}
    for name, path in default_map.items():
        resolved[name] = path if os.path.isabs(path) else os.path.join(args.json_base_dir, path)
    return resolved


def load_agent(args: argparse.Namespace):
    model = MyFCN_el.MyFcn(args.n_actions)
    optimizer = pixelwise_a3c_el.chainer.optimizers.Adam(alpha=1e-4)
    optimizer.setup(model)
    agent = pixelwise_a3c_el.PixelWiseA3C(model, optimizer, args.episode_len, 0.95)
    if args.device.startswith("cuda"):
        pixelwise_a3c_el.chainer.cuda.get_device_from_id(args.gpu_id).use()
        agent.model.to_gpu()
    agent.load(args.model_dir)
    if args.deterministic:
        agent.act_deterministically = True
    return agent


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_to_track: List[str] = ["psnr", "ssim"]
    lpips_model = None
    if args.compute_lpips:
        try:
            import lpips
        except ImportError:
            lpips = None  # type: ignore
        if lpips is not None:
            lpips_model = lpips.LPIPS(net="alex").to(args.device)
            metrics_to_track.append("lpips")
        else:
            print("LPIPS not available; skip LPIPS.")

    agent = load_agent(args)
    json_map = resolve_json_map(args)
    transform = transforms.ToTensor()

    for dataset_name, json_path in json_map.items():
        if not os.path.exists(json_path):
            print(f"[Skip] {dataset_name}: json not found at {json_path}")
            continue

        save_dir = os.path.join(args.output_dir, dataset_name, "rl")
        os.makedirs(save_dir, exist_ok=True)

        dataset = JsonDataset(json_path=json_path, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model_tracker = MetricsTracker(metrics=metrics_to_track)
        input_tracker = MetricsTracker(metrics=metrics_to_track)

        for batch in dataloader:
            if not batch:
                continue
            input_tensor = batch["I"].to(args.device)
            gt_tensor = batch["T"].to(args.device)
            gt_path = batch["T_paths"][0]

            raw_x = input_tensor.cpu().numpy()
            current_state = StateReflect(
                raw_x.shape,
                args.n_actions,
                args.move_step,
            )
            current_state.reset(raw_x)

            for _ in range(args.episode_len):
                action = agent.act(current_state.image)
                current_state.step(action)
            agent.stop_episode()

            out_img = np.clip(current_state.image, 0, 1)
            pred_tensor = torch.from_numpy(out_img).to(args.device)

            pred_np = tensor_to_numpy(pred_tensor)
            gt_np = tensor_to_numpy(gt_tensor)

            model_tracker.update(
                calculate_metrics(
                    pred_np,
                    gt_np,
                    pred_tensor,
                    gt_tensor,
                    lpips_model,
                )
            )

            input_tracker.update(
                calculate_metrics(
                    tensor_to_numpy(input_tensor),
                    gt_np,
                    input_tensor,
                    gt_tensor,
                    lpips_model,
                )
            )

            base_name = os.path.splitext(os.path.basename(gt_path))[0] + ".png"
            rel_path = base_name
            if dataset_name == "SIR2":
                path_parts = gt_path.split(os.path.sep)
                if "SIR2" in path_parts:
                    idx = path_parts.index("SIR2")
                    if idx + 1 < len(path_parts):
                        rel_path = os.path.join(path_parts[idx + 1], base_name)
            save_path = os.path.join(save_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(pred_np).save(save_path)

        print(f"\n== {dataset_name} (RL) ==")
        print("[Baseline] Input vs GT")
        input_tracker.report(dataset_name)
        print("[Model] RL output vs GT")
        model_tracker.report(dataset_name)


if __name__ == "__main__":
    main()
