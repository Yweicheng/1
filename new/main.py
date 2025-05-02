# your_script_name.py

import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns # 保留以备将来绘图使用
from tqdm import tqdm
import warnings
import traceback
from types import SimpleNamespace # 用于创建配置对象

# ===== 导入新模型 =====
try:
    from iTransformer_model import Model as iTransformerModel
    print("成功导入 iTransformer 模型。")
except ImportError:
    print("错误：导入 iTransformer 模型失败。请确保 iTransformer_model.py 在 Python 路径中。")
    exit()
except Exception as e:
    print(f"错误：导入 iTransformer 时发生错误：{e}")
    exit()

# 忽略 UserWarning 和 PIL DecompressionBombWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Possibly corrupt EXIF data.*")


# ===== 1. 数据处理器 (基本不变 - 用于目标阀门值) =====
class DataProcessor:
    """
    用于处理数值序列（如高度或阀门开度）的标准化和逆转换。
    现在主要用于目标阀门变量的归一化/逆转换。
    """
    def __init__(self, method='standardize'):
        if method not in ['standardize', 'normalize']:
            raise ValueError("方法必须是 'standardize' 或 'normalize'")
        self.method = method
        self.scaler = None

    def fit(self, data_list):
        all_values = [item for sublist in data_list for item in sublist if isinstance(item, (int, float))]
        if not all_values:
             print("警告：未找到用于拟合缩放器的数值数据。缩放器将不会被拟合。")
             self.scaler = None
             return
        all_values = np.array(all_values).reshape(-1, 1)
        if self.method == 'standardize':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        try:
            self.scaler.fit(all_values)
            print(f"缩放器 ({self.method}) 拟合成功。")
            if hasattr(self.scaler, 'mean_'): print(f"  均值: {self.scaler.mean_[0]:.4f}")
            if hasattr(self.scaler, 'scale_'): print(f"  尺度 (标准差): {self.scaler.scale_[0]:.4f}")
            if hasattr(self.scaler, 'min_'): print(f"  最小值: {self.scaler.min_[0]:.4f}")
            if hasattr(self.scaler, 'data_max_'): print(f"  最大值: {self.scaler.data_max_[0]:.4f}")
        except Exception as e:
            print(f"拟合缩放器时出错: {e}。缩放器可能无法使用。")
            self.scaler = None

    def transform(self, sequence, target_len):
        """ 转换单个序列（现在主要用于目标阀门值）。 """
        if self.scaler is None or not (hasattr(self.scaler, 'mean_') or hasattr(self.scaler, 'min_')):
            # print(f"警告：缩放器 ({self.method}) 未拟合或无效。返回零。") # 减少干扰
             # 根据 target_len 确定形状以保持一致性
            return torch.zeros((target_len, 1), dtype=torch.float32)

        numeric_sequence = [item for item in sequence if isinstance(item, (int, float))]
        if not numeric_sequence:
             # 如果没有数值数据，则返回正确形状的零
             return torch.zeros((target_len, 1), dtype=torch.float32)

        sequence_np = np.array(numeric_sequence).reshape(-1, 1)
        try:
            processed_sequence = self.scaler.transform(sequence_np)
        except Exception as e:
             print(f"转换序列时出错: {e}。使用零。")
             processed_sequence = np.zeros_like(sequence_np, dtype=float)

        # 填充/截断 (现在主要用于单个目标值的情况)
        current_len = processed_sequence.shape[0]
        if current_len >= target_len:
             # 取最后 'target_len' 个元素 (如果 target_len > 1 则相关)
             padded_sequence = processed_sequence[current_len - target_len:]
        else:
            # 在开头用第一个值（或 0）填充
            pad_value = processed_sequence[0, 0] if current_len > 0 else 0.0
            padding = np.full((target_len - current_len, 1), pad_value)
            padded_sequence = np.vstack((padding, processed_sequence))

        return torch.tensor(padded_sequence, dtype=torch.float32)

    # --- 新增：仅进行填充/截断而不进行缩放的方法 ---
    def pad_truncate_sequence(self, sequence, target_len):
        """
        将数值序列填充（在开头）或截断（从开头）到 target_len，而不应用缩放。
        返回一个 NumPy 数组。
        """
        numeric_sequence = [item for item in sequence if isinstance(item, (int, float))]

        if not numeric_sequence:
             sequence_np = np.zeros((0, 1)) # 如果没有数字，则从空数组开始
        else:
             sequence_np = np.array(numeric_sequence).reshape(-1, 1)

        current_len = sequence_np.shape[0]
        if current_len >= target_len:
            # 截断（取最后 target_len 个元素）
            padded_sequence = sequence_np[current_len - target_len:]
        else:
            # 在开头填充（使用第一个值或 0）
            pad_value = sequence_np[0, 0] if current_len > 0 else 0.0
            padding = np.full((target_len - current_len, 1), pad_value)
            padded_sequence = np.vstack((padding, sequence_np))

        return padded_sequence # 作为 numpy 数组返回

    def inverse_transform(self, data_tensor):
        """ 对数据进行逆转换（用于预测和目标）。 """
        if self.scaler is None or not (hasattr(self.scaler, 'mean_') or hasattr(self.scaler, 'min_')):
            # print(f"警告：缩放器 ({self.method}) 未拟合用于逆转换。返回原始数据。") # 减少干扰
            if isinstance(data_tensor, torch.Tensor):
                 return data_tensor.detach().cpu().numpy()
            return data_tensor # 已经是 numpy 或其他类型

        if isinstance(data_tensor, torch.Tensor):
            data_np = data_tensor.detach().cpu().numpy()
        else:
            data_np = data_tensor

        # 根据预期的缩放器输入（通常是 N, 1）重塑形状
        original_shape = data_np.shape
        if data_np.ndim == 0:
             data_np = data_np.reshape(1, 1)
        elif data_np.ndim == 1:
             data_np = data_np.reshape(-1, 1)
        # 处理模型可能的的多维输出（例如 B, L, F）-> (B*L, F)
        elif data_np.ndim > 2:
            data_np = data_np.reshape(-1, data_np.shape[-1])

        # 在逆转换之前检查兼容性
        if data_np.shape[1] != self.scaler.n_features_in_:
             print(f"警告：逆转换形状不匹配。输入：{data_np.shape}，缩放器预期：(*, {self.scaler.n_features_in_})。返回输入。")
             return data_np.reshape(original_shape) # 重塑回去

        try:
             data_np = np.nan_to_num(data_np.astype(float))
             original_scale_data = self.scaler.inverse_transform(data_np)
             # 重塑回原始形状结构（如果特征维度为 1 则忽略）
             if original_shape != data_np.shape:
                  try:
                      return original_scale_data.reshape(original_shape)
                  except ValueError:
                      print(f"警告：无法将逆转换后的数据重塑回 {original_shape}。返回展平后的数据。")
                      return original_scale_data.flatten()

             return original_scale_data

        except ValueError as ve:
             print(f"逆转换期间出现 ValueError：{ve}。输入形状：{data_np.shape}。返回输入数组。")
             return data_np.reshape(original_shape)
        except Exception as e:
             print(f"逆转换期间出现意外错误：{e}。返回输入数组。")
             return data_np.reshape(original_shape)


# ===== 2. 数据集类 (修改了 __getitem__ 中对高度的处理) =====
class ValveDataset(Dataset):
    """
    用于图像、高度（未归一化）和阀门目标（已归一化）的自定义数据集。
    """
    def __init__(self, data, image_dir, transform=None, seq_len=10, is_train=False, valve_proc_method='normalize'): # 移除了 height_proc_method
        """
        Args:
            data (list): 包含样本信息的列表，每个样本是一个字典。
            image_dir (str): 图像的基础目录。
            transform (callable, optional): 应用于图像的转换。
            seq_len (int): 输入序列的长度。
            is_train (bool): 指示是否为训练数据集（用于拟合处理器）。
            valve_proc_method (str): 目标阀门值的处理方法。
        """
        self.samples = data
        self.image_dir = image_dir
        self.transform = transform
        self.seq_len = seq_len
        # 仅用于目标阀门开度值的处理器
        self.valve_processor = DataProcessor(method=valve_proc_method)
        # 高度处理器实例（用于 pad_truncate 方法）- 此处无需拟合
        self.height_helper = DataProcessor() # 只需要 pad_truncate 方法

        if is_train:
            print("在训练数据上拟合目标阀门处理器...")
            # 仅提取每个序列中最后的阀门开度用于拟合
            all_valves = [sample.get('valve_opening', []) for sample in self.samples if isinstance(sample.get('valve_opening'), list)]
            last_valves = [[seq[-1]] for seq in all_valves if seq and isinstance(seq[-1], (int, float))]
            print(f"找到 {len(last_valves)} 个目标阀门值用于拟合。")
            self.valve_processor.fit(last_valves)
            print("目标阀门处理器已拟合。")
        else:
            self.valve_processor.scaler = None # 将由 set_processors 设置

        self.image_paths_per_sample = self._precompute_image_paths()

    def _precompute_image_paths(self):
        # (此处无需更改)
        all_paths = []
        for sample in tqdm(self.samples, desc="预计算图像路径"):
            image_files = sample.get('image_paths', [])
            if not isinstance(image_files, list): image_files = []
            # 确保路径分隔符一致，并处理 None 或非字符串路径
            paths = [p.replace('\\', '/') for p in image_files if p and isinstance(p, str)]
            current_len = len(paths)
            if current_len >= self.seq_len:
                # 截断（取最后的 seq_len 个）
                final_paths = paths[current_len - self.seq_len:]
            else:
                # 填充（在开头重复第一个有效路径，如果存在）
                pad_path = paths[0] if paths else None
                final_paths = [pad_path] * (self.seq_len - current_len) + paths
            all_paths.append(final_paths)
        return all_paths

    # --- 修改：仅设置阀门处理器 ---
    def set_processors(self, valve_processor):
        """ 为测试/验证集设置预先拟合的阀门处理器。 """
        if valve_processor is None:
            raise ValueError("必须提供阀门处理器")
        if valve_processor.scaler is None or not (hasattr(valve_processor.scaler, 'mean_') or hasattr(valve_processor.scaler, 'min_')):
             print("警告：提供的阀门处理器可能未被拟合。")
        self.valve_processor = valve_processor

    def get_original_data(self, idx):
        # (此处无需更改)
        sample = self.samples[idx]
        original_height_sequence = [item for item in sample.get('height', []) if isinstance(item, (int, float))]
        original_valve_sequence = [item for item in sample.get('valve_opening', []) if isinstance(item, (int, float))]
        # 获取序列中最后一个有效的阀门开度值作为目标
        original_valve_target = original_valve_sequence[-1] if original_valve_sequence else 0.0
        return original_height_sequence, original_valve_target

    def __len__(self):
        return len(self.samples)

    # --- 修改：返回未归一化的高度，归一化的阀门目标 ---
    def __getitem__(self, idx):
        """
        返回:
            tuple: (images_tensor, height_tensor_unnormalized, valve_target_normalized)
                   - images_tensor: [seq_len, 3, 224, 224]
                   - height_tensor_unnormalized: [seq_len, 1] (原始尺度, 已填充/截断)
                   - valve_target_normalized: 标量张量 (已归一化)
        """
        # 检查阀门处理器是否就绪（对目标值很重要）
        if self.valve_processor is None or self.valve_processor.scaler is None:
            # 如果处理器未就绪，transform 会处理（通常返回零）
            pass # transform handles this by returning zeros

        sample = self.samples[idx]
        image_paths = self.image_paths_per_sample[idx]

        # --- 图像加载 (无需更改) ---
        images = []
        default_img = None # 缓存默认图像以提高效率
        for i, path in enumerate(image_paths):
            img = None
            if path and os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGB')
                except Image.DecompressionBombError:
                    # print(f"Warning: DecompressionBombError for image {path}. Using default.") # 可能过于冗余
                    pass
                except Exception as e:
                    # print(f"Warning: Could not load image {path}: {e}. Using default.") # 可能过于冗余
                    pass

            # 如果图像加载失败或路径无效，则使用默认图像
            if img is None:
                if default_img is None:
                    default_img = Image.new('RGB', (224, 224), color=(128, 128, 128)) # 灰色图像
                img = default_img

            # 应用图像转换
            if self.transform:
                try:
                    img = self.transform(img)
                except Exception as e:
                    # print(f"Warning: Error applying transform to image (path: {path}): {e}. Using zeros tensor.") # 可能过于冗余
                    img = torch.zeros((3, 224, 224), dtype=torch.float32) # 使用零张量作为后备
            else: # 如果没有提供 transform，使用基本的转换
                 temp_transform = transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor()
                 ])
                 img = temp_transform(img)

            images.append(img)

        # 将图像列表堆叠成一个张量
        try:
            images_tensor = torch.stack(images) # 预期形状: [seq_len, 3, 224, 224]
        except Exception as e: # 捕获潜在的堆叠错误（例如，如果某个图像转换失败导致形状不一致）
            # print(f"Error stacking images at index {idx}: {e}. Using zeros tensor.") # 可能过于冗余
            images_tensor = torch.zeros((self.seq_len, 3, 224, 224), dtype=torch.float32)


        # --- 处理高度序列 (获取未归一化的, 已填充/截断) ---
        height_sequence = sample.get('height', [])
        if not isinstance(height_sequence, list): height_sequence = []
        # 使用辅助处理器进行填充/截断，不进行缩放
        height_np_unnormalized = self.height_helper.pad_truncate_sequence(height_sequence, self.seq_len) # 返回 NumPy [seq_len, 1]
        height_tensor_unnormalized = torch.tensor(height_np_unnormalized, dtype=torch.float32) # 转换为张量 [seq_len, 1]


        # --- 处理阀门目标 (获取归一化的标量) ---
        valve_opening_sequence = sample.get('valve_opening', [])
        if not isinstance(valve_opening_sequence, list): valve_opening_sequence = []
        # 获取序列中最后一个有效的数值作为目标值
        last_valve_value = None
        if valve_opening_sequence:
            for val in reversed(valve_opening_sequence): # 从后往前找第一个数值
                 if isinstance(val, (int, float)):
                     last_valve_value = val
                     break
        if last_valve_value is None: last_valve_value = 0.0 # 如果序列为空或无效，默认为 0

        # 使用 valve_processor 转换单个目标值
        # valve_processor.transform 期望一个列表, 返回 [1, 1] 张量
        # Squeeze 两次以获得标量张量 []
        valve_target_normalized = self.valve_processor.transform([last_valve_value], 1).squeeze(0).squeeze(0)

        return images_tensor, height_tensor_unnormalized, valve_target_normalized


# ===== 3. 模型架构 (已移除旧模型) =====
# ImageSequenceRegressionModel 类现已移除。
# 我们稍后将实例化 iTransformerModel。


# ===== 4. 训练和评估工具 (根据新模型的输入/输出进行了修改) =====

def setup_device(visible_devices="0"):
    """设置 GPU/CPU 设备。"""
    # (无需更改)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_count = torch.cuda.device_count()
        print(f"使用 {gpu_count} 个 GPU: {', '.join([torch.cuda.get_device_name(i) for i in range(gpu_count)])}")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    return device

def setup_data_parallel(model, device):
    """如果存在多个 GPU，则设置 DataParallel。"""
    # (无需更改)
    model = model.to(device)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"跨 {torch.cuda.device_count()} 个 GPU 使用 DataParallel。")
        model = nn.DataParallel(model)
    return model

# --- 修改：train_epoch 处理新的输入/输出 ---
def train_epoch(cnn_backbone, itransformer_model, loader, criterion, optimizer, device, seq_len, configs):
    """使用 iTransformer 模型训练一个周期。"""
    cnn_backbone.eval() # 主干网络用于特征提取，此处不训练
    itransformer_model.train() # 将 iTransformer 设置为训练模式
    total_loss = 0.0
    batches_processed = 0

    for images, height_unnormalized, valve_target_normalized in tqdm(loader, desc="训练中", leave=False):
        try:
            # 将必要的张量移动到设备
            images = images.to(device) # [B, L, C, H, W]
            height_unnormalized = height_unnormalized.to(device) # [B, L, 1]
            valve_target_normalized = valve_target_normalized.to(device) # [B]
        except Exception as e:
            print(f"将批次数据移动到设备 {device} 时出错：{e}。跳过此批次。")
            continue

        optimizer.zero_grad()

        # --- 准备 iTransformer 的输入 ---
        batch_size, _, c, h, w = images.shape
        img_features = None
        try:
            with torch.no_grad(): # 提取特征时不跟踪梯度
                # 为 CNN 重塑: [B*L, C, H, W]
                cnn_input = images.view(batch_size * seq_len, c, h, w)
                img_features = cnn_backbone(cnn_input) # 输出: [B*L, num_ftrs]
                # 重塑回序列格式: [B, L, num_ftrs]
                img_features = img_features.view(batch_size, seq_len, -1)
        except Exception as e:
            print(f"CNN 特征提取期间出错：{e}。跳过此批次。")
            continue

        # 沿特征维度连接图像特征和未归一化的高度
        # img_features: [B, L, num_ftrs], height_unnormalized: [B, L, 1]
        try:
            x_enc = torch.cat((img_features, height_unnormalized), dim=-1) # 形状: [B, L, num_ftrs + 1]
             # 验证形状是否与模型预期的输入 (configs.enc_in) 匹配
            if x_enc.shape[-1] != configs.enc_in:
                 print(f"致命错误：输入特征维度不匹配。x_enc 形状：{x_enc.shape}，预期的最后一个维度：{configs.enc_in}。请检查 enc_in 计算。")
                 # 你可能想在这里引发错误或退出
                 continue # 如果发生致命错误，则跳过批次

        except Exception as e:
            print(f"连接特征时出错：{e}。跳过此批次。")
            continue

        # --- iTransformer 前向传播 ---
        # iTransformer 期望输入 x_enc, x_mark_enc, x_dec, x_mark_dec
        # 对于这种仅编码器的回归设置（预测长度 pred_len=1 的预测任务）：
        # - 如果不使用时间特征，x_mark_enc 可以是 None。
        # - x_dec 和 x_mark_dec 通常不由编码器部分使用。
        pred = None
        try:
            # 为未使用的参数传递 None
            # 输出形状取决于任务，对于预测：[B, pred_len, D]
            # 由于 pred_len=1, D=1 (1 个目标特征), 预期形状：[B, 1, 1]
            dec_out = itransformer_model(x_enc, None, None, None)
            # 选择预测结果（对于某些任务，它可能返回整个序列）
            # 对于 'forecast' 任务，它应该已经只返回 [B, pred_len, D]
            pred = dec_out # 形状: [B, 1, 1]

        except Exception as e:
            print(f"训练中 iTransformer 前向传播期间出错：{e}")
            # traceback.print_exc() # 可选：打印完整的追溯信息
            continue # 跳过此批次

        # --- 损失计算 ---
        if pred is None: continue # 如果前向传播失败则跳过

        # 检查形状：pred 应为 [B, 1, 1]，valve_target_normalized 为 [B]
        if pred.shape[0] != valve_target_normalized.shape[0] or \
           pred.shape[1] != 1 or pred.shape[2] != 1 :
             print(f"模型输出后形状不匹配：pred {pred.shape}，target {valve_target_normalized.shape}。预期 pred [B, 1, 1]。跳过此批次。")
             continue

        try:
            # 压缩预测 [B, 1, 1] -> [B] 以匹配 MSELoss 的目标 [B]
            loss = criterion(pred.squeeze(1).squeeze(1), valve_target_normalized.float())
        except Exception as e:
            print(f"计算损失时出错：{e}。压缩后 Pred 形状：{pred.squeeze(1).squeeze(1).shape}，目标形状：{valve_target_normalized.shape}。跳过此批次。")
            continue

        # 反向传播和优化器步骤 (此处无需更改)
        if not torch.isfinite(loss):
            print(f"警告：检测到非有限损失 ({loss.item()})。跳过反向传播/步骤。")
            continue
        try:
            loss.backward()
        except Exception as e:
            print(f"反向传播期间出错：{e}。跳过优化器步骤。")
            optimizer.zero_grad() # 清除可能存在的无效梯度
            continue
        try:
            # 可选：梯度裁剪
            # torch.nn.utils.clip_grad_norm_(itransformer_model.parameters(), max_norm=1.0)
            optimizer.step()
        except Exception as e:
            print(f"优化器步骤期间出错：{e}。")
            continue

        total_loss += loss.item()
        batches_processed += 1

    return (total_loss / batches_processed) if batches_processed > 0 else 0.0

# --- 修改：validate 函数 ---
def validate(cnn_backbone, itransformer_model, loader, criterion, device, seq_len, configs):
    """验证 iTransformer 模型。"""
    cnn_backbone.eval()
    itransformer_model.eval()
    total_loss = 0.0
    batches_processed = 0
    with torch.no_grad():
        for images, height_unnormalized, valve_target_normalized in tqdm(loader, desc="验证中", leave=False):
            try:
                images = images.to(device)
                height_unnormalized = height_unnormalized.to(device)
                valve_target_normalized = valve_target_normalized.to(device)
            except Exception as e:
                 print(f"验证期间将批次数据移动到设备 {device} 时出错：{e}。跳过此批次。")
                 continue

            # --- 准备输入 ---
            batch_size, _, c, h, w = images.shape
            img_features = None
            try:
                cnn_input = images.view(batch_size * seq_len, c, h, w)
                img_features = cnn_backbone(cnn_input)
                img_features = img_features.view(batch_size, seq_len, -1)
            except Exception as e:
                print(f"CNN 特征提取期间出错（验证）：{e}。跳过。")
                continue

            try:
                x_enc = torch.cat((img_features, height_unnormalized), dim=-1)
                if x_enc.shape[-1] != configs.enc_in:
                     print(f"致命验证错误：输入特征维度不匹配。x_enc 形状：{x_enc.shape}，预期的最后一个维度：{configs.enc_in}。")
                     continue
            except Exception as e:
                 print(f"连接特征时出错（验证）：{e}。跳过。")
                 continue

            # --- 前向传播 ---
            pred = None
            try:
                dec_out = itransformer_model(x_enc, None, None, None)
                pred = dec_out # 形状: [B, 1, 1]
            except Exception as e:
                 print(f"验证中 iTransformer 前向传播期间出错：{e}。跳过此批次。")
                 continue

            # --- 损失计算 ---
            if pred is None: continue

            if pred.shape[0] != valve_target_normalized.shape[0] or \
               pred.shape[1] != 1 or pred.shape[2] != 1:
                 print(f"模型输出后形状不匹配（验证）：pred {pred.shape}，target {valve_target_normalized.shape}。跳过。")
                 continue

            try:
                # 压缩预测 [B, 1, 1] -> [B]
                loss = criterion(pred.squeeze(1).squeeze(1), valve_target_normalized.float())
                if torch.isfinite(loss):
                    total_loss += loss.item()
                    batches_processed += 1
                else:
                    print(f"警告：检测到非有限验证损失 ({loss.item()})。")
            except Exception as e:
                print(f"计算验证损失时出错：{e}。跳过此批次。")
                continue

    return (total_loss / batches_processed) if batches_processed > 0 else float('inf')


# --- 修改：evaluate_model 函数 (移除了注意力绘图) ---
def evaluate_model(cnn_backbone, itransformer_model, loader, device, valve_processor, seq_len, configs):
    """
    在测试集上评估模型，计算指标。
    valve_processor 用于对目标和预测进行逆转换。
    """
    cnn_backbone.eval()
    itransformer_model.eval()
    actuals_original = []
    predictions_original = []

    # 检查阀门处理器的有效性（对指标至关重要）
    if valve_processor is None or valve_processor.scaler is None or not (hasattr(valve_processor.scaler, 'mean_') or hasattr(valve_processor.scaler, 'min_')):
         print("错误：评估需要一个有效且已拟合的阀门处理器来进行逆转换。")
         return {'std': np.nan, 'r2': np.nan, 'pearson_r': np.nan, 'mae': np.nan,
                 'actuals_original': [], 'predictions_original': []}

    with torch.no_grad():
        for images, height_unnormalized, valve_target_normalized in tqdm(loader, desc="评估中", leave=False):
            try:
                # 输入只需要图像和高度在设备上
                images = images.to(device)
                height_unnormalized = height_unnormalized.to(device)
                # valve_target_normalized 保持在 CPU 上（它是加载的真实标签）
            except Exception as e:
                 print(f"评估期间将批次数据移动到设备 {device} 时出错：{e}。跳过此批次。")
                 continue

            # --- 准备输入 ---
            batch_size, _, c, h, w = images.shape
            img_features = None
            try:
                 cnn_input = images.view(batch_size * seq_len, c, h, w)
                 img_features = cnn_backbone(cnn_input) # [B*L, F]
                 img_features = img_features.view(batch_size, seq_len, -1) # [B, L, F]
            except Exception as e:
                 print(f"CNN 特征提取期间出错（评估）：{e}。跳过此批次。")
                 continue

            try:
                x_enc = torch.cat((img_features, height_unnormalized), dim=-1) # [B, L, F+1]
                if x_enc.shape[-1] != configs.enc_in:
                     print(f"致命评估错误：输入特征维度不匹配。x_enc 形状：{x_enc.shape}，预期的最后一个维度：{configs.enc_in}。")
                     continue
            except Exception as e:
                print(f"连接特征时出错（评估）：{e}。跳过此批次。")
                continue

            # --- 前向传播 ---
            pred_normalized = None
            try:
                # 归一化尺度上的预测，预期形状 [B, 1, 1]
                dec_out = itransformer_model(x_enc, None, None, None)
                pred_normalized = dec_out.cpu() # 移动到 CPU 进行逆转换
            except Exception as e:
                 print(f"评估中 iTransformer 前向传播期间出错：{e}。跳过此批次。")
                 continue

            # --- 逆转换 ---
            if pred_normalized is None: continue

            # 逆转换前检查形状
            # pred_normalized: [B, 1, 1], valve_target_normalized: [B]
            if pred_normalized.shape[0] != valve_target_normalized.shape[0] or \
               pred_normalized.shape[1] != 1 or pred_normalized.shape[2] != 1:
                 print(f"逆转换前形状不匹配（评估）：pred {pred_normalized.shape}，target {valve_target_normalized.shape}。跳过。")
                 continue

            try:
                # pred_normalized 需要形状 [B, 1] 以用于 valve_processor 的 inverse_transform
                pred_original = valve_processor.inverse_transform(pred_normalized.squeeze(-1)) # 压缩最后一个维度 -> [B, 1]
                # valve_target_normalized 需要形状 [B, 1]
                actual_original = valve_processor.inverse_transform(valve_target_normalized.cpu().numpy().reshape(-1, 1))
            except Exception as e:
                 print(f"评估中逆转换期间出错：{e}。跳过此批次。")
                 traceback.print_exc() # 打印详细错误信息
                 continue

            # 附加结果（逆转换输出 [B, 1] 后展平）
            if pred_original.ndim == 2 and actual_original.ndim == 2 and pred_original.shape[1] == 1 and actual_original.shape[1] == 1:
                actuals_original.extend(actual_original.flatten().tolist())
                predictions_original.extend(pred_original.flatten().tolist())
            else:
                print(f"警告：逆转换后出现意外形状。实际：{actual_original.shape}，预测：{pred_original.shape}。跳过此批次。")


    # --- 计算指标 (此处无需更改) ---
    if not actuals_original or not predictions_original:
        print("警告：没有收集到用于评估的有效数据点。")
        return {'std': np.nan, 'r2': np.nan, 'pearson_r': np.nan, 'mae': np.nan, 'actuals_original': [], 'predictions_original': []}

    actuals_original = np.array(actuals_original)
    predictions_original = np.array(predictions_original)

    # 处理可能存在的 NaN 或 Inf 值
    valid_indices = np.isfinite(actuals_original) & np.isfinite(predictions_original)
    if not np.all(valid_indices):
        num_invalid = np.sum(~valid_indices)
        print(f"警告：在评估结果中发现 {num_invalid} 个非有限值。正在移除它们。")
        actuals_original = actuals_original[valid_indices]
        predictions_original = predictions_original[valid_indices]

    if len(actuals_original) < 2: # R2 和 Pearson 需要至少两个点
        print("警告：没有足够的有效数据点（< 2）来计算指标。")
        mae = np.nan
        if len(actuals_original) > 0:
             try: mae = mean_absolute_error(actuals_original, predictions_original)
             except ValueError: mae = np.nan
        return {'std': np.nan, 'r2': np.nan, 'pearson_r': np.nan, 'mae': mae, 'actuals_original': actuals_original.tolist(), 'predictions_original': predictions_original.tolist()}

    error = actuals_original - predictions_original
    std_dev = np.std(error)
    try: r2 = r2_score(actuals_original, predictions_original)
    except ValueError: r2 = np.nan
    try: mae = mean_absolute_error(actuals_original, predictions_original)
    except ValueError: mae = np.nan

    pearson_r, pearson_p = np.nan, np.nan
    # 检查是否有足够的变异性来计算相关性
    if len(np.unique(actuals_original)) > 1 and len(np.unique(predictions_original)) > 1:
         try:
             pearson_r, pearson_p = pearsonr(actuals_original.flatten(), predictions_original.flatten())
         except ValueError:
             print("警告：计算皮尔逊相关性时出现 ValueError。")
    else:
         print("警告：无法计算皮尔逊相关性（常数值或数据不足）。")

    return {'std': std_dev, 'r2': r2, 'pearson_r': pearson_r, 'mae': mae,
            'actuals_original': actuals_original.tolist(), 'predictions_original': predictions_original.tolist()}


# ===== 5. 绘图工具 (无需更改, 移除了 plot_attention_weights) =====
def plot_predictions(actual, predicted, filename="prediction_comparison.png"):
    # (无需更改)
    if not actual or not predicted or len(actual) != len(predicted):
        print("警告: 实际值或预测值列表为空或长度不匹配，无法绘制散点图。")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, alpha=0.5, label=f'样本数 N={len(actual)}')
    try:
        # 仅使用有限值确定绘图范围和 y=x 线
        finite_actual = [x for x in actual if np.isfinite(x)]
        finite_predicted = [x for x in predicted if np.isfinite(x)]
        if not finite_actual or not finite_predicted: raise ValueError("没有有限值用于绘图范围")
        min_val = min(np.min(finite_actual), np.min(finite_predicted))
        max_val = max(np.max(finite_actual), np.max(finite_predicted))
        buffer = (max_val - min_val) * 0.05 + 1e-6 # 添加缓冲以避免点落在边缘，加上小值防止 max=min
        min_val -= buffer
        max_val += buffer
        if min_val >= max_val: # 如果缓冲导致范围无效，使用后备方案
             min_val = min(finite_actual + finite_predicted) - 1
             max_val = max(finite_actual + finite_predicted) + 1
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想情况 (y=x)')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
    except Exception as e:
        print(f"警告：无法绘制 y=x 线或设置坐标轴范围: {e}")
    plt.xlabel("实际阀门开度 (原始尺度)")
    plt.ylabel("预测阀门开度 (原始尺度)")
    plt.title("预测值 vs 实际值 (散点图)")
    plt.legend()
    plt.grid(True)
    try:
        plt.savefig(filename)
        print(f"预测散点图已保存：{filename}")
    except Exception as e:
        print(f"保存预测散点图时出错：{e}")
    plt.close() # 关闭图形以释放内存

def plot_prediction_lines(actual, predicted, filename="prediction_lines.png"):
    # (无需更改)
    if not actual or not predicted or len(actual) != len(predicted):
        print("警告: 实际值或预测值列表为空或长度不匹配，无法绘制折线图。")
        return
    plt.figure(figsize=(15, 7))
    num_samples = len(actual)
    indices = range(num_samples)

    # 仅绘制有限值
    actual_np, predicted_np = np.array(actual), np.array(predicted)
    valid_mask = np.isfinite(actual_np) & np.isfinite(predicted_np)
    if not np.any(valid_mask):
        print("警告：没有有限数据用于绘制折线图。")
        plt.close()
        return

    indices_valid = np.array(indices)[valid_mask]
    actual_valid = actual_np[valid_mask]
    predicted_valid = predicted_np[valid_mask]

    plt.plot(indices_valid, actual_valid, 'b-', label=f'实际值 (N={len(actual_valid)})', linewidth=1.5)
    plt.plot(indices_valid, predicted_valid, 'r--', label=f'预测值 (N={len(predicted_valid)})', linewidth=1.5, alpha=0.8)

    plt.xlabel("样本索引 (测试集)")
    plt.ylabel("阀门开度 (原始尺度)")
    plt.title("实际 vs. 预测阀门开度 (折线图)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, num_samples - 1 if num_samples > 1 else 1) # 设置 x 轴范围为原始样本数

    try:
        plt.savefig(filename)
        print(f"预测折线图已保存：{filename}")
    except Exception as e:
        print(f"保存预测折线图时出错：{e}")
    plt.close()

def plot_loss_curves(train_losses, val_losses, filename="loss_curves.png"):
    # (无需更改)
    # 过滤掉非有限值
    valid_train_losses = [(i, l) for i, l in enumerate(train_losses) if isinstance(l, (int, float)) and np.isfinite(l)]
    valid_val_losses = [(i, l) for i, l in enumerate(val_losses) if isinstance(l, (int, float)) and np.isfinite(l)]

    if not valid_train_losses and not valid_val_losses:
        print("警告：没有有效的损失数据可供绘制。")
        return

    plt.figure(figsize=(10, 6))
    if valid_train_losses:
        epochs_train, losses_train = zip(*valid_train_losses)
        plt.plot([e + 1 for e in epochs_train], losses_train, 'bo-', label='训练损失') # 周期从 1 开始计数
    if valid_val_losses:
        epochs_val, losses_val = zip(*valid_val_losses)
        plt.plot([e + 1 for e in epochs_val], losses_val, 'ro-', label='验证损失') # 周期从 1 开始计数

    plt.xlabel("周期 (Epoch)")
    plt.ylabel("平均损失 (MSE)")
    plt.title("训练和验证损失曲线")
    if valid_train_losses or valid_val_losses: plt.legend()
    plt.grid(True)
    # 动态设置 x 轴范围
    max_epoch = max(len(train_losses), len(val_losses))
    plt.xlim(1, max_epoch if max_epoch > 0 else 1)

    try:
        plt.savefig(filename)
        print(f"损失曲线图已保存：{filename}")
    except Exception as e:
        print(f"保存损失曲线图时出错：{e}")
    plt.close()

# --- 移除：plot_attention_weights 函数 ---
# 注意力绘图函数已被移除，因为它特定于旧模型架构。
# iTransformer 使用不同的自注意力机制，可视化方式不同。


# ===== 6. 主函数 (已修改) =====
def main():
    # --- 参数解析器 (为 iTransformer 修改) ---
    parser = argparse.ArgumentParser(description="训练一个基于 iTransformer 的阀门开度回归模型")
    # 数据和路径
    parser.add_argument('--train_data', type=str, default="/home/temp03/ywc/veiw_json/new_data/train_data.json", help='训练 JSON 数据路径')
    parser.add_argument('--test_data', type=str, default="/home/temp03/ywc/veiw_json/new_data/test_data.json", help='测试 JSON 数据路径')
    parser.add_argument('--image_dir', type=str, default="/home/temp03/ywc/veiw_json/valve_img/", help='图像的基础目录（如果 JSON 中的路径是相对的）')
    parser.add_argument('--output_dir', type=str, default="./results_iTransformer/", help='输出目录')
    # iTransformer 超参数
    parser.add_argument('--seq_len', type=int, default=10, help='输入序列长度')
    # enc_in 将根据 CNN 特征 + 1 (高度) 计算得出
    # 对于此回归任务，pred_len 固定为 1
    parser.add_argument('--d_model', type=int, default=512, help='模型嵌入的维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头的数量')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')
    parser.add_argument('--activation', type=str, default='gelu', choices=['relu', 'gelu'], help='激活函数')
    parser.add_argument('--embed', type=str, default='timeF', help='时间嵌入类型 (例如, timeF, fixed, learned)') # 来自 iTransformer 代码库
    parser.add_argument('--freq', type=str, default='h', help='时间特征的频率 (仅在使用时间特征时相关)') # 来自 iTransformer 代码库
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=50, help='训练周期数') # 调整了默认值
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小') # 可能调整了默认值
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    # 数据加载和预处理
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 工作进程数')
    # valve_proc 应用于目标变量
    parser.add_argument('--valve_proc', type=str, default='normalize', choices=['normalize', 'standardize'], help='目标阀门值的处理方法')
    # 高度输入不再由 DataProcessor 归一化
    # 执行控制
    parser.add_argument('--pretrained', type=str, default=None, help='加载预训练 iTransformer 权重（.pth 文件）的路径')
    parser.add_argument('--visible_devices', type=str, default="0", help='可见的 CUDA 设备')
    parser.add_argument('--eval_only', action='store_true', help='仅使用 --pretrained 模型进行评估')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    # --- 可复现性 ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # 可选：为了完全可复现，但这可能降低性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # --- 设置输出目录和文件名 (已修改) ---
    os.makedirs(args.output_dir, exist_ok=True)
    proc_suffix = f"vTarget_{args.valve_proc}" # 从后缀中移除高度处理方法
    model_config_suffix = f"seq{args.seq_len}_d{args.d_model}_h{args.n_heads}_e{args.e_layers}_dff{args.d_ff}_drop{args.dropout}"
    base_filename = f"iTransformer_{model_config_suffix}_{proc_suffix}"

    model_save_path = os.path.join(args.output_dir, f'{base_filename}_best.pth')
    pred_plot_path = os.path.join(args.output_dir, f'{base_filename}_predictions.png')
    pred_line_plot_path = os.path.join(args.output_dir, f'{base_filename}_prediction_lines.png')
    loss_plot_path = os.path.join(args.output_dir, f'{base_filename}_loss_curves.png')
    # 移除注意力图路径

    print("--- 配置信息 ---")
    # 在设置 configs.enc_in 之前确定 ResNet18 的特征数量
    print("确定 ResNet 特征数量...")
    try:
        # 使用推荐的权重加载方式 (如果 torchvision 版本支持)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if hasattr(models, 'ResNet18_Weights') else None
        temp_cnn = models.resnet18(weights=weights, progress=False) # progress=False 避免下载时打印信息
        if weights is None: # 兼容旧版本
            temp_cnn = models.resnet18(pretrained=True, progress=False)
        num_ftrs = temp_cnn.fc.in_features
        del temp_cnn # 释放内存
        print(f"ResNet18 特征维度: {num_ftrs}")
        args.enc_in = num_ftrs + 1 # 图像特征 + 1 (高度序列)
    except Exception as e:
        print(f"确定 ResNet 特征数时出错: {e}。请检查 torchvision 安装。")
        print("假设 ResNet18 特征数为 512。")
        num_ftrs = 512 # 使用默认值作为后备
        args.enc_in = num_ftrs + 1

    for key, value in vars(args).items(): print(f"{key}: {value}")
    print("---------------------")
    print(f"模型输入特征维度 (enc_in): {args.enc_in}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型保存路径: {model_save_path}")
    print(f"预测散点图路径: {pred_plot_path}")
    print(f"预测折线图路径: {pred_line_plot_path}")
    print(f"损失曲线图路径: {loss_plot_path}")

    # --- 设置设备 ---
    device = setup_device(visible_devices=args.visible_devices)

    # --- 加载数据 ---
    def load_json_data(filepath):
        # (无需更改)
        print(f"从以下路径加载数据: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            if isinstance(data, dict): # 处理字典格式（可能是按 ID 存储）
                processed_data = list(data.values())
            elif isinstance(data, list): # 处理列表格式
                processed_data = data
            else:
                raise TypeError(f"意外的 JSON 数据类型: {type(data)}")
            print(f"已加载 {len(processed_data)} 条目。")
            # 检查数据有效性
            if not processed_data:
                print("警告：加载的数据为空列表。")
            elif not all(isinstance(item, dict) for item in processed_data):
                print("警告：数据列表中的并非所有项都是字典。")
            return processed_data
        except FileNotFoundError:
            print(f"错误：文件未找到 - {filepath}")
            return None
        except json.JSONDecodeError as e:
            print(f"错误：解析 JSON 文件时出错 ({filepath}): {e}")
            return None
        except Exception as e:
            print(f"从 {filepath} 加载数据时发生未知错误: {e}")
            return None

    train_data = load_json_data(args.train_data)
    test_data = load_json_data(args.test_data)
    if train_data is None or test_data is None or not train_data or not test_data:
        print("由于数据加载错误或数据为空，正在退出。")
        return

    # --- 图像转换 ---
    # 使用 ImageNet 的均值和标准差进行归一化
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("图像转换已定义。")

    # --- 创建数据集和数据加载器 (已修改) ---
    print('正在创建数据集...')
    train_dataset, test_dataset = None, None
    try:
        # 创建训练数据集（拟合阀门处理器）
        train_dataset = ValveDataset(
            train_data, args.image_dir, transform, seq_len=args.seq_len, is_train=True,
            valve_proc_method=args.valve_proc # 现在只需要阀门处理方法
        )
        # 检查阀门处理器是否正确拟合
        train_valve_proc_valid = (train_dataset.valve_processor and
                                  train_dataset.valve_processor.scaler and
                                  (hasattr(train_dataset.valve_processor.scaler, 'mean_') or
                                   hasattr(train_dataset.valve_processor.scaler, 'min_')))
        if not train_valve_proc_valid:
             print("错误：训练数据的阀门处理器未能成功拟合。请检查目标阀门值是否有效且非空。")
             return

        # 创建测试数据集
        test_dataset = ValveDataset(
            test_data, args.image_dir, transform, seq_len=args.seq_len, is_train=False,
            valve_proc_method=args.valve_proc
        )
        # 为测试集设置阀门处理器
        test_dataset.set_processors(train_dataset.valve_processor) # 只传递阀门处理器
        print("来自训练集的阀门处理器已应用于测试数据集。")

    except Exception as e:
        print(f"创建数据集时出错: {e}")
        traceback.print_exc()
        return

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("错误：训练集或测试集为空。无法继续。")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'), # 如果使用 GPU，启用 pin_memory 可能加速数据传输
        drop_last=True # 丢弃最后一个不完整的批次，确保批次大小一致
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # 测试集不需要打乱
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
        # 测试集通常不 drop_last，以便评估所有样本
    )
    print('数据加载器已创建。')


    # --- 初始化模型 (CNN 主干网络 + iTransformer) ---
    print('正在初始化模型...')
    # 1. CNN 主干网络 (ResNet18 用于特征提取)
    try:
        # 再次尝试使用推荐的权重加载方式
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if hasattr(models, 'ResNet18_Weights') else None
        cnn_backbone = models.resnet18(weights=weights, progress=False)
        if weights is None: cnn_backbone = models.resnet18(pretrained=True, progress=False)
        cnn_backbone.fc = nn.Identity() # 移除最终的分类层
        cnn_backbone = setup_data_parallel(cnn_backbone, device) # 将主干网络移动到设备
        print("CNN 主干网络 (ResNet18) 已初始化。")
    except Exception as e:
        print(f"初始化 ResNet18 时出错: {e}。请检查 torchvision。")
        return

    # 2. iTransformer 模型
    # 为 iTransformer 模型创建一个配置对象 (Namespace)
    configs = SimpleNamespace(
        # --- 任务相关 ---
        task_name='short_term_forecast', # 或 'long_term_forecast' - 对于 pred_len=1，输出层相同
        seq_len=args.seq_len,            # 输入序列长度
        pred_len=1,                      # 预测长度（回归单个值）
        output_attention=False,          # 通常在推理中不需要输出注意力权重

        # --- 模型维度 ---
        enc_in=args.enc_in,              # 编码器输入维度 (图像特征 + 高度)
        dec_in=0,                        # 解码器输入维度（此任务中未使用）
        d_model=args.d_model,            # 模型隐藏维度
        n_heads=args.n_heads,            # 注意力头数
        e_layers=args.e_layers,          # 编码器层数
        d_layers=0,                      # 解码器层数（此任务中未使用）
        d_ff=args.d_ff,                  # 前馈网络维度
        c_out=1,                         # 输出维度（预测 1 个值）

        # --- 嵌入和激活 ---
        dropout=args.dropout,            # Dropout 率
        embed=args.embed,                # 时间嵌入类型 (timeF, fixed, learned)
        activation=args.activation,      # 激活函数 (relu, gelu)
        freq=args.freq,                  # 时间特征频率 (h, d, m, etc.) - 如果 embed='timeF' 则需要

        # --- 其他 (可能未使用或使用默认值) ---
        factor=5,                        # ProbSparse Attention 因子 (如果使用)
        num_class=0                      # 分类任务类别数（回归任务未使用）
        # distil=True,                   # 是否使用蒸馏 (Informer/Autoformer 特有)
        # ... 其他 iTransformer 可能有的参数 ...
    )
    print("iTransformer 配置:", configs)

    # 实例化 iTransformer
    try:
        itransformer_model = iTransformerModel(configs)
        print("iTransformer 模型结构已创建。") # 打印模型结构可能非常冗长，先注释掉
        # print(itransformer_model)
    except Exception as e:
        print(f"实例化 iTransformer 模型时出错: {e}")
        print("请确保 iTransformer_model.py 中的 Model 类及其依赖项正确无误，并且 configs 对象包含所有必需的参数。")
        traceback.print_exc()
        return

    total_params_itransformer = sum(p.numel() for p in itransformer_model.parameters() if p.requires_grad)
    print(f"iTransformer 可训练参数数量: {total_params_itransformer:,}")


    # --- 加载预训练的 iTransformer 权重 (如果指定) ---
    if args.pretrained:
        if os.path.exists(args.pretrained):
            print(f"尝试从以下路径加载预训练的 iTransformer 权重: {args.pretrained}")
            print("警告：预训练权重必须与当前的 iTransformer 架构完全匹配 (d_model, n_heads, e_layers 等)。")
            try:
                state_dict = torch.load(args.pretrained, map_location=device) # 加载到目标设备
                # 处理 'module.' 前缀（如果权重是从 DataParallel 模型保存的，而当前模型不是）
                # 同时处理当前是 DataParallel 而权重不是的情况（虽然不太常见）
                is_current_parallel = isinstance(itransformer_model, nn.DataParallel)
                is_saved_parallel = list(state_dict.keys())[0].startswith('module.')

                if is_saved_parallel and not is_current_parallel:
                     print("从状态字典中移除 'module.' 前缀...")
                     state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                elif not is_saved_parallel and is_current_parallel:
                     print("为状态字典添加 'module.' 前缀...")
                     state_dict = {'module.' + k: v for k, v in state_dict.items()}

                # 使用 strict=False 加载以查看不匹配项，这有助于调试
                missing_keys, unexpected_keys = itransformer_model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"警告：加载权重时发现缺失的键: {missing_keys}")
                    # 如果只是输出层或嵌入层缺失，可能是在微调，可以接受
                if unexpected_keys:
                    print(f"警告：加载权重时发现意外的键: {unexpected_keys}")
                    # 这通常表示架构不匹配，需要仔细检查

                if not missing_keys and not unexpected_keys:
                    print("预训练权重加载成功 (严格匹配)。")
                elif not unexpected_keys: # 允许缺失键（例如，微调时只加载部分层）
                    print("预训练权重已加载 (部分键缺失，可能是正常的)。")
                else: # 存在意外键，通常是问题
                    print("警告：预训练权重已加载，但存在意外键，请仔细检查架构匹配！")

            except FileNotFoundError:
                 print(f"错误：预训练模型文件未找到：{args.pretrained}。")
                 if args.eval_only: print("无法在 --eval_only 模式下继续。"); return
                 else: print("将从头开始训练。")
            except Exception as e:
                 print(f"加载预训练 iTransformer 模型时出错：{e}。将从头开始训练。")
                 traceback.print_exc()
                 if args.eval_only: print("无法在 --eval_only 模式下继续。"); return
        else:
            print(f"警告：预训练模型文件未找到：{args.pretrained}。")
            if args.eval_only:
                print("错误：--eval_only 模式需要一个有效的 --pretrained 文件。正在退出。")
                return
            else:
                print("将从头开始训练。")

    # --- 为 iTransformer 设置多 GPU ---
    # 注意：如果在加载权重前应用 DataParallel，加载时需要处理 'module.' 前缀
    # 如果在加载权重后应用 DataParallel (如下)，则加载时不需要处理 'module.' (除非保存的模型本身是 Parallel 的)
    itransformer_model = setup_data_parallel(itransformer_model, device)


    # --- 仅评估模式 ---
    if args.eval_only:
        print("\n=== 在仅评估模式下运行 ===")
        if not args.pretrained:
            print("错误：--eval_only 模式需要指定 --pretrained 模型路径。")
            return
        # 再次检查测试数据集的处理器是否有效
        test_valve_proc_valid = (test_dataset.valve_processor and
                                 test_dataset.valve_processor.scaler and
                                 (hasattr(test_dataset.valve_processor.scaler, 'mean_') or
                                  hasattr(test_dataset.valve_processor.scaler, 'min_')))
        if not test_valve_proc_valid:
            print("错误：测试数据集的阀门处理器对于 eval_only 模式无效或未正确设置。")
            print("请确保训练阶段已正确运行或手动提供了有效的处理器。")
            return

        print("评估加载的模型...")
        start_eval_time = time.time()
        eval_metrics = evaluate_model(
            cnn_backbone,                   # 传递主干网络
            itransformer_model,             # 传递 iTransformer
            test_loader,                    # 测试数据加载器
            device,                         # 设备
            test_dataset.valve_processor,   # 传递用于逆转换的阀门处理器
            args.seq_len,                   # 传递序列长度
            configs                         # 传递配置对象
        )
        end_eval_time = time.time()
        print(f"评估在 {end_eval_time - start_eval_time:.2f} 秒内完成。")
        print(f"评估结果 (原始尺度):")
        print(f"  预测误差标准差: {eval_metrics.get('std', 'N/A'):.4f}")
        print(f"  R² 系数:           {eval_metrics.get('r2', 'N/A'):.4f}")
        print(f"  皮尔逊相关系数 (R): {eval_metrics.get('pearson_r', 'N/A'):.4f}")
        print(f"  平均绝对误差 (MAE): {eval_metrics.get('mae', 'N/A'):.4f}")

        # 如果有结果，绘制评估图
        if eval_metrics.get('actuals_original') and eval_metrics.get('predictions_original'):
            # 修改文件名以区分仅评估模式的图
            eval_pred_plot_path = pred_plot_path.replace(".png", "_eval_only.png")
            eval_pred_line_plot_path = pred_line_plot_path.replace(".png", "_eval_only.png")
            plot_predictions(eval_metrics['actuals_original'], eval_metrics['predictions_original'], filename=eval_pred_plot_path)
            plot_prediction_lines(eval_metrics['actuals_original'], eval_metrics['predictions_original'], filename=eval_pred_line_plot_path)
        else:
            print("无法生成评估预测图（缺少有效的评估结果）。")
        print("评估完成（--eval_only 模式）。正在退出。")
        return


    # --- 训练设置 ---
    criterion = nn.MSELoss() # 使用均方误差损失
    # 优化器仅训练 iTransformer 的参数（假设 CNN 主干网络是预训练且固定的）
    optimizer = torch.optim.Adam(itransformer_model.parameters(), lr=args.lr)
    # 学习率调度器：当验证损失停止改善时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-7)


    # --- 训练循环 ---
    print("\n=== 开始训练 ===")
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    start_train_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n--- 周期 {epoch + 1}/{args.epochs} ---")

        # 训练阶段
        avg_train_loss = train_epoch(cnn_backbone, itransformer_model, train_loader, criterion, optimizer, device, args.seq_len, configs)
        train_losses.append(avg_train_loss)

        # 验证阶段
        # 验证前再次确保测试阀门处理器有效
        test_valve_proc_valid = (test_dataset.valve_processor and
                                 test_dataset.valve_processor.scaler and
                                 (hasattr(test_dataset.valve_processor.scaler, 'mean_') or
                                  hasattr(test_dataset.valve_processor.scaler, 'min_')))
        if not test_valve_proc_valid:
             print("错误：测试数据集阀门处理器无效。跳过验证。")
             avg_val_loss = float('inf') # 设为无穷大以避免保存模型或调整学习率
        else:
             avg_val_loss = validate(cnn_backbone, itransformer_model, test_loader, criterion, device, args.seq_len, configs)
        val_losses.append(avg_val_loss)

        # 学习率调度
        if avg_val_loss != float('inf'): # 只有在验证损失有效时才调整学习率
            scheduler.step(avg_val_loss)
        else:
            print("跳过学习率调度器步骤（无效的验证损失）。")

        epoch_end_time = time.time()
        print(f"周期 {epoch + 1} 总结:")
        print(f"  平均训练损失: {avg_train_loss:.6f}")
        print(f"  平均验证损失: {avg_val_loss:.6f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6e}") # 打印当前学习率
        print(f"  周期时间: {epoch_end_time - epoch_start_time:.2f} 秒")

        # 保存最佳模型 (基于验证损失)
        if avg_val_loss < best_val_loss and avg_val_loss != float('inf'):
            best_val_loss = avg_val_loss
            print(f"✅ 发现新的最佳模型 (验证损失: {best_val_loss:.6f})，保存到 {model_save_path}...")
            # 使用 DataParallel 时，保存基础模型的 state_dict
            model_to_save = itransformer_model.module if isinstance(itransformer_model, nn.DataParallel) else itransformer_model
            try:
                torch.save(model_to_save.state_dict(), model_save_path)
                print("模型已保存。")
            except Exception as e:
                print(f"保存模型时出错：{e}")
        # 可选的早停逻辑:
        # if epoch - np.argmin(val_losses) > args.early_stopping_patience:
        #     print("验证损失长时间未改善，触发早停。")
        #     break

    # --- 训练完成 ---
    end_train_time = time.time()
    print(f"\n=== 训练完成 ===")
    total_training_time_hours = (end_train_time - start_train_time) / 3600
    print(f"总训练时间: {total_training_time_hours:.2f} 小时")
    print(f"最佳验证损失: {best_val_loss:.6f}")


    # --- 使用最佳模型进行最终评估 ---
    print("\n=== 使用最佳模型进行最终评估 ===")
    if os.path.exists(model_save_path):
        print(f"从 {model_save_path} 加载最佳 iTransformer 模型...")
        # 重新初始化模型结构（确保与保存时一致）
        final_configs = configs # 使用训练时的相同配置
        final_itransformer = iTransformerModel(final_configs)

        try:
            # 加载状态字典到 CPU，以防设备不匹配
            state_dict = torch.load(model_save_path, map_location='cpu')
            # 不需要手动处理 'module.' 前缀，因为我们加载到基础模型 final_itransformer
            final_itransformer.load_state_dict(state_dict, strict=True) # 最终加载使用 strict=True
            print("最佳模型权重加载成功。")

            # 将模型移动到目标设备并设置 DataParallel（如果需要）
            final_itransformer = setup_data_parallel(final_itransformer, device)
            final_itransformer.eval() # 设置为评估模式

            # 再次确认测试数据处理器有效
            test_valve_proc_valid = (test_dataset.valve_processor and
                                     test_dataset.valve_processor.scaler and
                                     (hasattr(test_dataset.valve_processor.scaler, 'mean_') or
                                      hasattr(test_dataset.valve_processor.scaler, 'min_')))
            if not test_valve_proc_valid:
                 print("错误：测试数据集阀门处理器对于最终评估无效。")
            else:
                 # 使用训练结束时的 CNN 主干网络（它应该已经在设备上并处于 eval 模式）
                 final_eval_start = time.time()
                 final_metrics = evaluate_model(
                    cnn_backbone,               # 已设置好的主干网络
                    final_itransformer,         # 加载的最佳 iTransformer
                    test_loader,                # 测试数据加载器
                    device,                     # 设备
                    test_dataset.valve_processor, # 阀门处理器
                    args.seq_len,               # 序列长度
                    final_configs               # 配置对象
                 )
                 final_eval_end = time.time()
                 print(f"最终评估在 {final_eval_end - final_eval_start:.2f} 秒内完成。")
                 print(f"最佳模型评估结果 (原始尺度):")
                 print(f"  预测误差标准差: {final_metrics.get('std', 'N/A'):.4f}")
                 print(f"  R² 系数:           {final_metrics.get('r2', 'N/A'):.4f}")
                 print(f"  皮尔逊相关系数 (R): {final_metrics.get('pearson_r', 'N/A'):.4f}")
                 print(f"  平均绝对误差 (MAE): {final_metrics.get('mae', 'N/A'):.4f}")

                 # 生成最终的预测图
                 if final_metrics.get('actuals_original') and final_metrics.get('predictions_original'):
                      plot_predictions(final_metrics['actuals_original'], final_metrics['predictions_original'], filename=pred_plot_path)
                      plot_prediction_lines(final_metrics['actuals_original'], final_metrics['predictions_original'], filename=pred_line_plot_path)
                 else:
                      print("无法生成最终预测图（缺少评估结果）。")

                 # 绘制训练过程中的损失曲线
                 if train_losses and val_losses:
                     plot_loss_curves(train_losses, val_losses, filename=loss_plot_path)
                 else:
                     print("无法生成损失曲线图（缺少损失数据）。")

        except FileNotFoundError:
            print(f"错误：未找到最佳模型文件 {model_save_path}。")
        except Exception as e:
            print(f"加载或评估最佳模型时出错：{e}")
            traceback.print_exc()
    else:
        print(f"未找到最佳模型文件 ({model_save_path})。跳过最终评估。")

    print("\n脚本执行完成。")


# ===== 程序入口点 =====
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 捕获主函数中的任何未处理异常
        print(f"\n主函数发生致命错误: {e}")
        traceback.print_exc() # 打印完整的错误堆栈信息
