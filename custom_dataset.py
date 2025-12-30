# custom_dataset.py
import json
import os
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

class JsonDataset(data.Dataset):
    """
    一个用于加载由JSON文件索引的数据集。
    它会同时加载输入（blended）和目标（gt）图像，并对它们应用相同的变换。
    """
    def __init__(self, json_path: str, transform=None):
        """
        Args:
            json_path (str): JSON索引文件的路径。
            transform (callable, optional): 应用于样本的转换操作。
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON index file not found at: {json_path}")

        with open(json_path, 'r') as f:
            self.index = json.load(f)

        self.transform = transform
        print(f"Loaded dataset from {os.path.basename(json_path)} with {len(self.index)} images.")

    def __len__(self) -> int:
        """返回数据集中样本的总数。"""
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        """
        获取数据集中的第 i 个样本。
        """
        entry = self.index[i]
        blended_path = entry['blended']
        gt_path = entry['gt']

        try:
            blended_img = Image.open(blended_path).convert('RGB')
            gt_img = Image.open(gt_path).convert('RGB') # <--- 新增: 同时加载GT图像
        except FileNotFoundError as e:
            print(f"FATAL ERROR: Image not found at {e.filename}. Please check your JSON file.")
            return {}

        # --- 动态等比缩放逻辑 (同时应用于输入和GT) ---
        width, height = blended_img.size
        max_dim = 1024

        if width > max_dim or height > max_dim:
            if width > height:
                ratio = max_dim / width
                new_width, new_height = max_dim, int(height * ratio)
            else:
                ratio = max_dim / height
                new_height, new_width = max_dim, int(width * ratio)
            
            # 关键修正：对输入图和GT图应用完全相同的缩放
            blended_img = blended_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            gt_img = gt_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # --- 缩放逻辑结束 ---

        # 对两张图应用后续的transform
        if self.transform:
            blended_tensor = self.transform(blended_img)
            gt_tensor = self.transform(gt_img) # <--- 新增: 对GT图也进行transform

        # 返回包含了输入张量、GT张量和路径的字典
        return {'I': blended_tensor, 'T': gt_tensor, 'T_paths': gt_path}
        # return {'I': blended_tensor, 'T': gt_tensor, 'I_paths': blended_path}