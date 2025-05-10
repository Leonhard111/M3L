import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from vit_pytorch.vit import Transformer

encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to('cuda')
transformer=Transformer(384, 1, 4, 64, 384, 0.1).to('cuda')
# 设置随机种子以便结果可复现
torch.manual_seed(42)

# 定义batch size和图片尺寸
batch_size = 4
img_size = 70  # DINO v2 ViT-S/14需要224x224的输入

# 创建一个随机batch的图片 (batch_size, 3, height, width)
random_images = torch.rand(batch_size*4, 3, img_size, 3*img_size, device='cuda')
print(f"输入图片维度: {random_images.shape}")

# 确保模型处于评估模式
encoder.eval()
transformer.eval()

# 前向传播，提取特征
with torch.no_grad():
    features = encoder(random_images).reshape(batch_size,4,-1)
    output=transformer(features)

# 输出特征维度
print(f"DINO特征维度: {features.shape}")  # b,384
print(f"transformer output:{output.shape}")