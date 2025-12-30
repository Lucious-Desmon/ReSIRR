# evaluation_utils.py (已集成 LPIPS)
import numpy as np
import cv2
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from typing import Dict, List, Tuple

# <--- 新增：导入 lpips 库 ---
try:
    import lpips
except ImportError:
    print("LPIPS库未安装。如需计算LPIPS指标，请运行: pip install lpips")
    lpips = None

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将PyTorch张量转换为适用于图像评估的NumPy数组。
    (H, W, C), uint8, [0, 255]
    
    注意：您原版的注释中提到了[-1, 1]的逆归一化，
    但实际代码 `img.permute(1, 2, 0).numpy() * 255.0`
    表明输入张量的范围是 [0, 1]。我们将基于 [0, 1] 继续。
    """
    img = tensor.squeeze(0).cpu().detach()
    
    # 将 [0, 1] 范围的张量转换为 [0, 255] 的 NumPy 数组
    img_np = img.permute(1, 2, 0).numpy() * 255.0
    
    img_np = np.clip(img_np, 0, 255)
    
    return img_np.astype(np.uint8)

# <--- 新增：一个辅助函数，用于将 [0, 1] 的张量归一化到 [-1, 1] ---
def normalize_tensor_for_lpips(tensor: torch.Tensor) -> torch.Tensor:
    """
    将 [0, 1] 范围的张量转换为 LPIPS 期望的 [-1, 1] 范围。
    """
    return (tensor * 2) - 1

def calculate_metrics(
    pred_np: np.ndarray,
    gt_np: np.ndarray,
    pred_tensor: torch.Tensor = None,
    gt_tensor: torch.Tensor = None,
    lpips_model: 'lpips.LPIPS' = None
) -> Dict[str, float]:
    """
    计算图像对之间的 PSNR, SSIM 和 LPIPS。

    Args:
        pred_np (np.ndarray): 预测图 (NumPy, [0, 255], uint8)
        gt_np (np.ndarray): 真值图 (NumPy, [0, 255], uint8)
        pred_tensor (torch.Tensor, optional): 预测图 (Tensor, [0, 1], float)。计算LPIPS时需要。
        gt_tensor (torch.Tensor, optional): 真值图 (Tensor, [0, 1], float)。计算LPIPS时需要。
        lpips_model (lpips.LPIPS, optional): 预先加载的LPIPS模型实例。

    Returns:
        Dict[str, float]: 包含 'psnr', 'ssim', 和 (可选的) 'lpips' 得分的字典。
    """
    # 确保图像尺寸一致，以GT为准
    if pred_np.shape != gt_np.shape:
        h, w = gt_np.shape[:2]
        pred_np = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_AREA)

    # --- 1. 计算 PSNR 和 SSIM (使用 NumPy) ---
    psnr_score = psnr(gt_np, pred_np, data_range=255)
    ssim_score = ssim(gt_np, pred_np, multichannel=True, data_range=255, channel_axis=2)

    results = {'psnr': psnr_score, 'ssim': ssim_score}

    # --- 2. 计算 LPIPS (使用 Tensor) ---
    if lpips is not None and lpips_model is not None and \
       pred_tensor is not None and gt_tensor is not None:
        
        # 将张量从 [0, 1] 转换为 [-1, 1]
        pred_tensor_norm = normalize_tensor_for_lpips(pred_tensor)
        gt_tensor_norm = normalize_tensor_for_lpips(gt_tensor)
        
        with torch.no_grad():
            lpips_score = lpips_model(pred_tensor_norm, gt_tensor_norm).item()
        
        results['lpips'] = lpips_score

    return results


class MetricsTracker:
    """
    一个用于跟踪、累加和计算多个评估指标平均值的类。
    (此类无需修改，它会自动处理传入的指标名称)
    """
    def __init__(self, metrics: List[str] = ['psnr', 'ssim']):
        """
        初始化一个空的跟踪器。
        Args:
            metrics (List[str]): 需要跟踪的指标名称列表。
                                (例如: ['psnr', 'ssim', 'lpips'])
        """
        self.metrics_to_track = metrics
        self.scores = {metric: [] for metric in self.metrics_to_track}
        self.count = 0

    def update(self, scores_dict: Dict[str, float]):
        """
        用新的一组成绩更新跟踪器。

        Args:
            scores_dict (Dict[str, float]): 从 calculate_metrics 返回的字典。
        """
        for metric in self.metrics_to_track:
            if metric in scores_dict:
                self.scores[metric].append(scores_dict[metric])
        self.count += 1

    def get_average_results(self) -> Dict[str, float]:
        """
        计算所有已累加分数的平均值。
        """
        if self.count == 0:
            return {metric: 0.0 for metric in self.metrics_to_track}

        avg_results = {metric: np.mean(values) for metric, values in self.scores.items()}
        avg_results['count'] = self.count
        return avg_results
        
    def report(self, dataset_name: str) -> None:
        """
        以美观的格式打印最终的平均结果报告。
        """
        results = self.get_average_results()
        print(f"\n--- Evaluation Report for: {dataset_name} ---")
        if results['count'] > 0:
            report_str = f"  Images Evaluated: {results['count']}"
            for metric in self.metrics_to_track:
                # LPIPS (越低越好) 和其他指标 (越高越好) 的格式相同
                report_str += f" | Avg {metric.upper()}: {results[metric]:.4f}"
            print(report_str)
        else:
            print("  No images were evaluated.")
        print("-" * (29 + len(dataset_name)))