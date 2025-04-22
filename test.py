# import torch
# from dinov2.models.vision_transformer import vit_small

# # 构建ViT-S/14，有register tokens的模型
# model = vit_small(patch_size=14)

# # 加载你下载的权重路径
# state_dict = torch.load("/home/leonhard/workshop/presentation/dinov2/dinov2_vits14_reg4_pretrain.pth", map_location="cpu")

# # 有些权重是包含在 'model' key 下的
# if 'model' in state_dict:
#     state_dict = state_dict['model']
# else:
#     print("没有") 
import torch
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
model.eval()    

# 加载权重
# model.load_state_dict(state_dict)  # 如果是finetune或少部分key不匹配，设置 strict=False

# model.eval()



import torchvision.transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

img = Image.open("/home/leonhard/workshop/presentation/dinov2/jojo.png").convert("RGB")
x = transform(img).unsqueeze(0)  # shape [1, 3, 224, 224]

with torch.no_grad():
    output = model(x)  # 输出 shape: [1, num_tokens, dim]

# 提取前4个register token（通常）
register_tokens = output  # [B, 4, dim]
print(register_tokens.shape)
embedding = register_tokens.mean(dim=1)  # 聚合为 [B, dim]
print(embedding.shape)


# import torch
# dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
# dinov2_vits14_reg.eval()