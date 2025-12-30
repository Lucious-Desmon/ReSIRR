"""
Benchmark script for Soft-B gating + refinement (NAFNet).

Prior path is derived from GT path by replacing the parent folder with "prior":
.../transmission_layer/3_134.jpg -> .../prior/3_134.npy
"""
'''
CUDA_VISIBLE_DEVICES=1 python benchmark_controlnet_unet_softb_refine.py \
  --json_base_dir /inspire/hdd/global_user/zhangchaoyang-240108100048/sirr2/datasets/test \
  --output_dir ./dai_benchmark_conunet2.3-1600ckpt-refinev1.1seed42-1227 \
  --model_id /inspire/hdd/global_user/zhangchaoyang-240108100048/sirr2/HF_CACHE/DAI_weights \
  --controlnet_dir /inspire/hdd/global_user/zhangchaoyang-240108100048/sirr2/DAI/xxxxxtloutput/v2.3/ckpt/checkpoint-1600/controlnet \
  --unet_dir /inspire/hdd/global_user/zhangchaoyang-240108100048/sirr2/DAI/xxxxxtloutput/v2.3/ckpt/checkpoint-1600/unet \
  --refine_net_path /inspire/hdd/global_user/zhangchaoyang-240108100048/sirr2/DAI/xxxxxtloutput/refinev1.1seed42/nafnet_refine_final.pth \
  --refine_head_path /inspire/hdd/global_user/zhangchaoyang-240108100048/sirr2/DAI/xxxxxtloutput/refinev1.1seed42/nafnet_refine_head_final.pth \
  --nafnet_width 64 \
  --prompt "remove glass reflection" \
  --beta 0.25 \
  --batch_size 1 \
  --num_workers 2 \
  --compute_lpips

'''
import argparse
import os
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from basicsr.models.archs.NAFNet_arch import NAFNet
from DAI.controlnetvae import ControlNetVAEModel
from DAI.pipeline_onestep import OneStepPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, AutoTokenizer

from custom_dataset import JsonDataset
from evaluation_utils import tensor_to_numpy, calculate_metrics, MetricsTracker
from wavelet_color_fix import wavelet_decomposition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark with custom ControlNet/UNet weights using Soft-B gating + refinement."
    )
    parser.add_argument("--json_base_dir", type=str, default="./test", help="Base dir of dataset json files.")
    parser.add_argument(
        "--json_files",
        nargs="+",
        help="Override dataset json files, format: name=relative/or/abs/path.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./custom_benchmark_outputs",
        help="Directory to save predictions.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="sjtu-deepvision/dereflection-any-image-v0",
        help="Base model id/path for VAE, tokenizer, text encoder.",
    )
    parser.add_argument(
        "--controlnet_dir",
        type=str,
        required=True,
        help="Path to trained controlnet weights (folder containing config.json).",
    )
    parser.add_argument(
        "--unet_dir",
        type=str,
        required=True,
        help="Path to trained unet weights (folder containing config.json).",
    )
    parser.add_argument(
        "--refine_net_path",
        type=str,
        required=True,
        help="Path to NAFNet refinement weights.",
    )
    parser.add_argument(
        "--refine_head_path",
        type=str,
        required=True,
        help="Path to refinement head weights.",
    )
    parser.add_argument("--nafnet_width", type=int, default=64, help="NAFNet width for refinement.")
    parser.add_argument("--prompt", type=str, default="remove glass reflection", help="Prompt text.")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Soft-B gating strength.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep 1 for safety).")
    parser.add_argument("--num_workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--compute_lpips",
        action="store_true",
        help="Compute LPIPS if lpips is installed.",
    )
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


def load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> OneStepPipeline:
    controlnet = ControlNetVAEModel.from_pretrained(args.controlnet_dir, torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(args.unet_dir, torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae", torch_dtype=dtype).to(device)
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder", torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", use_fast=False)

    pipe = OneStepPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        scheduler=None,
        feature_extractor=None,
        t_start=0,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    for p in controlnet.parameters():
        p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    controlnet.eval()
    unet.eval()
    vae.eval()
    text_encoder.eval()
    return pipe


def load_refine_models(args: argparse.Namespace, device: torch.device) -> tuple[NAFNet, torch.nn.Module]:
    in_ch = 10
    refine_net = NAFNet(
        img_channel=in_ch,
        width=args.nafnet_width,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
    ).to(device)
    refine_head = torch.nn.Conv2d(in_ch, 3, kernel_size=1, bias=True).to(device)

    refine_net.load_state_dict(torch.load(args.refine_net_path, map_location="cpu"))
    refine_head.load_state_dict(torch.load(args.refine_head_path, map_location="cpu"))
    refine_net.eval()
    refine_head.eval()
    return refine_net, refine_head


def prior_path_from_gt(gt_path: str) -> str:
    base_name = os.path.splitext(os.path.basename(gt_path))[0] + ".npy"
    gt_dir = os.path.dirname(gt_path)
    parent_dir = os.path.dirname(gt_dir)
    prior_dir = os.path.join(parent_dir, "prior")
    return os.path.join(prior_dir, base_name)


def load_prior_tensor(prior_path: str, cond_size: tuple[int, int]) -> torch.Tensor:
    prior = np.load(prior_path)
    if prior.ndim > 2:
        prior = prior.squeeze()
    prior = np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)
    prior = np.clip(prior.astype(np.float32), 0.0, 1.0)
    prior_img = Image.fromarray((prior * 255.0).astype(np.uint8), mode="L")
    if prior_img.size != cond_size:
        prior_img = prior_img.resize(cond_size, Image.Resampling.LANCZOS)
    prior_tensor = TF.to_tensor(prior_img).unsqueeze(0)  # [1,1,H,W]
    return prior_tensor


def _normalize_to_01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = (1, 2, 3)
    x_min = x.amin(dim=dims, keepdim=True)
    x_max = x.amax(dim=dims, keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def compute_hf_image(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf = _normalize_to_01(high_freq)
    return hf


def compute_hf_mag(image: torch.Tensor) -> torch.Tensor:
    high_freq, _ = wavelet_decomposition(image)
    hf_mag = high_freq.abs().mean(dim=1, keepdim=True)
    mean = hf_mag.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)
    hf_mag = (hf_mag / mean).clamp(0.0, 1.0)
    return hf_mag


def infer_with_softb(
    pipeline: OneStepPipeline,
    image_tensor: torch.Tensor,
    prompt: str,
    prior_tensor: torch.Tensor,
    beta: float,
) -> np.ndarray:
    device = pipeline._execution_device
    dtype = pipeline.dtype

    if pipeline.empty_text_embedding is None:
        text_inputs = pipeline.tokenizer(
            "",
            padding="do_not_pad",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        pipeline.empty_text_embedding = pipeline.text_encoder(text_input_ids)[0]

    if pipeline.prompt_embeds is None or pipeline.prompt != prompt:
        pipeline.prompt = prompt
        pipeline.prompt_embeds = None

    if pipeline.prompt_embeds is None:
        prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
            pipeline.prompt,
            device,
            1,
            False,
            None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )
        pipeline.prompt_embeds = prompt_embeds
        pipeline.negative_prompt_embeds = negative_prompt_embeds

    image, padding, original_resolution = pipeline.image_processor.preprocess(
        image_tensor, pipeline.default_processing_resolution, "bilinear", device, dtype
    )
    image_latent, pred_latent = pipeline.prepare_latents(
        image, None, None, 1, 1
    )

    prior = prior_tensor.to(device=device, dtype=dtype)
    prior = F.interpolate(prior, size=image.shape[-2:], mode="bilinear", align_corners=False)
    hf_mag = compute_hf_mag(image)

    down_block_res_samples, mid_block_res_sample = pipeline.controlnet(
        image_latent.detach(),
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        controlnet_cond=image,
        conditioning_scale=1.0,
        guess_mode=False,
        return_dict=False,
    )

    gated_down = []
    prior_l = prior.clamp(0.0, 1.0)
    hf_l = hf_mag.clamp(0.0, 1.0)
    for res in down_block_res_samples:
        prior_res = F.interpolate(prior_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        hf_res = F.interpolate(hf_l, size=res.shape[-2:], mode="area").clamp(0.0, 1.0)
        gate = 1.0 + beta * prior_res * hf_res
        gate = gate.clamp(1.0, 1.0 + beta)
        gated_down.append(res * gate)
    down_block_res_samples = gated_down

    prior_mid = F.interpolate(prior_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    hf_mid = F.interpolate(hf_l, size=mid_block_res_sample.shape[-2:], mode="area").clamp(0.0, 1.0)
    gate_mid = 1.0 + beta * prior_mid * hf_mid
    gate_mid = gate_mid.clamp(1.0, 1.0 + beta)
    mid_block_res_sample = mid_block_res_sample * gate_mid

    latent_x_t = pipeline.unet(
        pred_latent,
        pipeline.t_start,
        encoder_hidden_states=pipeline.prompt_embeds,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        return_dict=False,
    )[0]

    prediction = pipeline.decode_prediction(latent_x_t)
    prediction = pipeline.image_processor.unpad_image(prediction, padding)
    prediction = pipeline.image_processor.resize_antialias(
        prediction, original_resolution, "bilinear", is_aa=False
    )
    prediction = pipeline.image_processor.pt_to_numpy(prediction)[0]
    return prediction


def refine_image(
    refine_net: NAFNet,
    refine_head: torch.nn.Module,
    cond_tensor: torch.Tensor,
    prelim_tensor: torch.Tensor,
    prior_tensor: torch.Tensor,
) -> torch.Tensor:
    hf = compute_hf_image(cond_tensor)
    x = torch.cat([prelim_tensor, hf, prior_tensor, cond_tensor], dim=1)
    feat = refine_net(x)
    refined = refine_head(feat).clamp(0.0, 1.0)
    return refined


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    pipeline = load_pipeline(args, device, dtype)
    refine_net, refine_head = load_refine_models(args, device)

    metrics_to_track: List[str] = ["psnr", "ssim"]
    lpips_model = None
    if args.compute_lpips:
        try:
            import lpips
        except ImportError:
            lpips = None  # type: ignore
        if lpips is not None:
            lpips_model = lpips.LPIPS(net="alex").to(device)
            metrics_to_track.append("lpips")
        else:
            print("LPIPS not available; skip LPIPS.")

    json_map = resolve_json_map(args)
    transform = transforms.ToTensor()

    for dataset_name, json_path in json_map.items():
        if not os.path.exists(json_path):
            print(f"[Skip] {dataset_name}: json not found at {json_path}")
            continue

        save_dir_softb = os.path.join(args.output_dir, dataset_name, "softb")
        save_dir_refine = os.path.join(args.output_dir, dataset_name, "refine")
        os.makedirs(save_dir_softb, exist_ok=True)
        os.makedirs(save_dir_refine, exist_ok=True)

        dataset = JsonDataset(json_path=json_path, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model_tracker = MetricsTracker(metrics=metrics_to_track)
        refine_tracker = MetricsTracker(metrics=metrics_to_track)
        input_tracker = MetricsTracker(metrics=metrics_to_track)

        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            if not batch:
                continue

            input_tensor = batch["I"].to(device)
            gt_tensor = batch["T"].to(device)
            gt_path = batch["T_paths"][0]

            prior_path = prior_path_from_gt(gt_path)
            if not os.path.exists(prior_path):
                print(f"[Warn] prior not found: {prior_path}")
                continue

            _, _, h, w = input_tensor.shape
            prior_tensor = load_prior_tensor(prior_path, (w, h)).to(device)

            with torch.no_grad():
                output = infer_with_softb(
                    pipeline,
                    input_tensor,
                    args.prompt,
                    prior_tensor,
                    args.beta,
                )

            pred_tensor = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0)
            pred_tensor = ((pred_tensor + 1) / 2).clamp(0, 1).to(device)

            with torch.no_grad():
                refined_tensor = refine_image(
                    refine_net,
                    refine_head,
                    input_tensor,
                    pred_tensor,
                    prior_tensor,
                )

            pred_np = tensor_to_numpy(pred_tensor)
            refine_np = tensor_to_numpy(refined_tensor)
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

            refine_tracker.update(
                calculate_metrics(
                    refine_np,
                    gt_np,
                    refined_tensor,
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
            save_softb = os.path.join(save_dir_softb, rel_path)
            save_refine = os.path.join(save_dir_refine, rel_path)
            os.makedirs(os.path.dirname(save_softb), exist_ok=True)
            os.makedirs(os.path.dirname(save_refine), exist_ok=True)
            Image.fromarray(pred_np).save(save_softb)
            Image.fromarray(refine_np).save(save_refine)

        print(f"\n== {dataset_name} (Soft-B) ==")
        print("[Baseline] Input vs GT")
        input_tracker.report(dataset_name)
        print("[Model] Soft-B vs GT")
        model_tracker.report(dataset_name)
        print("[Model] Soft-B + Refine vs GT")
        refine_tracker.report(dataset_name)


if __name__ == "__main__":
    main()
