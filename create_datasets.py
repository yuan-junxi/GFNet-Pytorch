# generate_pretrained.py
import torch
import torchvision.models as models
import os

print("正在加载PyTorch官方的ImageNet预训练ResNet18...")
# 创建模型并加载预训练权重
model = models.resnet18(pretrained=True)
# 保存纯state_dict
torch.save(model.state_dict(), 'checkpoints/resnet18_pretrained.pth')
print(f"✅ 预训练权重已保存到: checkpoints/resnet18_pretrained.pth")
print(f"文件大小: {os.path.getsize('checkpoints/resnet18_pretrained.pth') / 1024**2:.2f} MB")

# 验证文件内容
loaded = torch.load('checkpoints/resnet18_pretrained.pth')
print(f"键数量: {len(loaded)}")
print("前5个键:", list(loaded.keys())[:5])