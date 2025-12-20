import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
import math

# ==================== 配置 ====================
parser = argparse.ArgumentParser(description='Dual Model Fine-tuning on ImageNet10 with Robustness Testing')

parser.add_argument('--data_dir', default='./imagenet10', type=str,
                    help='path to the ImageNet10 dataset')
parser.add_argument('--output_dir', default='./output', type=str,
                    help='path to save checkpoints and logs')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of epochs to train')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size for training and testing')
parser.add_argument('--lr', default=0.01, type=float,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')

# GFNet相关参数
parser.add_argument('--patch_size', default=96, type=int,
                    help='size of local patches')
parser.add_argument('--T', default=3, type=int,
                    help='number of glance+focus steps')
parser.add_argument('--model_arch', default='resnet50', type=str,
                    help='model architecture: resnet50, resnet101, densenet121')

# 鲁棒测试参数
parser.add_argument('--test_robustness', action='store_true', default=True,
                    help='test robustness after training')
parser.add_argument('--pgd_eps', default=8 / 255, type=float,
                    help='PGD epsilon')
parser.add_argument('--pgd_alpha', default=2 / 255, type=float,
                    help='PGD alpha')
parser.add_argument('--pgd_steps', default=20, type=int,
                    help='PGD steps')
parser.add_argument('--test_samples', default=500, type=int,
                    help='number of samples for testing')

args = parser.parse_args()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)
log_file = os.path.join(args.output_dir, 'training_log.txt')


# ==================== 辅助函数 ====================
def write_log(message, log_file):
    """写入日志"""
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')


# ==================== GFNet相关组件 ====================
class FullLayer(nn.Module):
    """全连接层，用于融合全局和局部特征"""

    def __init__(self, input_dim, hidden_dim, use_rnn=False):
        super(FullLayer, self).__init__()
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim

        if use_rnn:
            self.rnn = nn.LSTMCell(input_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # RNN状态
        self.hx = None
        self.cx = None

    def forward(self, x, restart=False):
        if self.use_rnn:
            if restart:
                self.hx = None
                self.cx = None

            if self.hx is None:
                batch_size = x.size(0)
                self.hx = torch.zeros(batch_size, self.hidden_dim, device=x.device)
                self.cx = torch.zeros(batch_size, self.hidden_dim, device=x.device)

            self.hx, self.cx = self.rnn(x, (self.hx, self.cx))
            x = self.fc(self.hx)
        else:
            x = self.fc1(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.fc2(x)

        return x


def get_prime(inputs, patch_size):
    """获取全局视角的输入（下采样）"""
    # 简单的下采样方法，使用平均池化
    B, C, H, W = inputs.shape
    scale_factor = patch_size / 224.0
    new_size = int(H * scale_factor)

    if new_size >= H:
        return inputs

    # 使用自适应平均池化
    prime_input = F.adaptive_avg_pool2d(inputs, (new_size, new_size))
    # 再上采样回原大小
    prime_input = F.interpolate(prime_input, size=(H, W), mode='bilinear', align_corners=False)

    return prime_input


def get_patch(inputs, action, patch_size):
    """根据动作选择局部区域"""
    B, C, H, W = inputs.shape

    patches = []
    for i in range(B):
        # 确保action在[0,1]范围内
        x_center = torch.clamp(action[i, 0], 0, 1) * (W - patch_size)
        y_center = torch.clamp(action[i, 1], 0, 1) * (H - patch_size)

        x_center = int(x_center.item())
        y_center = int(y_center.item())

        # 确保边界
        x_start = max(0, x_center)
        y_start = max(0, y_center)
        x_end = min(W, x_start + patch_size)
        y_end = min(H, y_start + patch_size)

        # 调整大小以确保patch_size
        if x_end - x_start < patch_size:
            if x_start > 0:
                x_start = max(0, x_end - patch_size)
            else:
                x_end = min(W, x_start + patch_size)

        if y_end - y_start < patch_size:
            if y_start > 0:
                y_start = max(0, y_end - patch_size)
            else:
                y_end = min(H, y_start + patch_size)

        patch = inputs[i:i + 1, :, y_start:y_end, x_start:x_end]

        # 调整大小到224x224
        patch = F.interpolate(patch, size=(224, 224), mode='bilinear', align_corners=False)
        patches.append(patch)

    return torch.cat(patches, dim=0)


# ==================== 简化版的强化学习策略 ====================
class SimplePolicy(nn.Module):
    """简化的策略网络，用于选择关注区域"""

    def __init__(self, state_dim, hidden_dim=256):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # 输出x,y坐标

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # 使用tanh将输出限制在[-1, 1]，再映射到[0, 1]
        x = (self.tanh(self.fc3(x)) + 1) / 2
        return x


# ==================== 数据加载 ====================
def prepare_data(data_dir, batch_size=32):
    """准备ImageNet10数据集"""

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载数据集
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 保存类别信息
    num_classes = len(train_dataset.classes)

    # 限制样本数量（为了快速测试）
    if len(train_dataset) > 1000:
        indices = torch.randperm(len(train_dataset))[:1000]
        train_dataset = Subset(train_dataset, indices)

    if len(val_dataset) > 500:
        indices = torch.randperm(len(val_dataset))[:500]
        val_dataset = Subset(val_dataset, indices)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, num_classes


# ==================== PGD攻击类 ====================
class PGDAttack:
    """PGD攻击实现"""

    def __init__(self, model_prime, fc_fusion, patch_size, eps=8 / 255, alpha=2 / 255, steps=10):
        self.model_prime = model_prime
        self.fc_fusion = fc_fusion
        self.patch_size = patch_size
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_fn = nn.CrossEntropyLoss()

    def attack(self, images, labels):
        """执行PGD攻击"""
        images = images.clone().detach()
        labels = labels.clone().detach()

        # 随机初始化扰动
        delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
        adv_images = images + delta
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            # 使用模型进行前向传播
            with torch.enable_grad():
                # 全局特征
                input_prime = get_prime(adv_images, self.patch_size)
                features = self.model_prime(input_prime)

                # 通过fc层
                output = self.fc_fusion(features, restart=True)

                # 计算损失
                loss = self.loss_fn(output, labels)

            # 计算梯度
            grad = torch.autograd.grad(loss, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # 更新对抗样本
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, -self.eps, self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images


# ==================== 双模型训练 ====================
class DualModelTrainer:
    """双模型训练器"""

    def __init__(self, model_arch, num_classes, device, patch_size=96):
        self.device = device
        self.patch_size = patch_size

        # 创建两个模型：全局编码器和局部编码器
        if 'resnet' in model_arch:
            self.model_prime = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

            # 修改分类头
            feature_dim = self.model_prime.fc.in_features
            self.model_prime.fc = nn.Identity()  # 移除原始分类头
            self.model.fc = nn.Identity()

            self.fc_prime = nn.Linear(feature_dim, num_classes)  # 全局分类头
            self.fc_local = nn.Linear(feature_dim, num_classes)  # 局部分类头
            self.fc_fusion = FullLayer(feature_dim, 512, use_rnn=False)  # 融合层
            self.fc_final = nn.Linear(512, num_classes)  # 最终分类头

        else:
            raise ValueError(f"Unsupported model architecture: {model_arch}")

        # 策略网络（简化版）
        state_dim = feature_dim
        self.policy = SimplePolicy(state_dim)

        # 移动到设备
        self.model_prime = self.model_prime.to(device)
        self.model = self.model.to(device)
        self.fc_prime = self.fc_prime.to(device)
        self.fc_local = self.fc_local.to(device)
        self.fc_fusion = self.fc_fusion.to(device)
        self.fc_final = self.fc_final.to(device)
        self.policy = self.policy.to(device)

    def forward_global(self, x):
        """全局视角前向传播"""
        features = self.model_prime(x)
        output = self.fc_prime(features)
        return features, output

    def forward_local(self, x, state):
        """局部视角前向传播"""
        # 使用策略网络选择关注区域
        with torch.no_grad():
            action = self.policy(state)

        # 提取局部区域
        patches = get_patch(x, action, self.patch_size)

        # 局部特征提取
        local_features = self.model(patches)
        local_output = self.fc_local(local_features)

        return local_features, local_output, action

    def forward_fusion(self, global_features, local_features):
        """融合全局和局部特征"""
        # 简单拼接融合
        combined = global_features + local_features  # 特征相加
        fused = self.fc_fusion(combined, restart=True)
        final_output = self.fc_final(fused)
        return final_output

    def forward_single(self, x):
        """简化的前向传播（用于PGD攻击）"""
        # 全局特征
        global_features, _ = self.forward_global(x)

        # 局部特征
        local_features, _, _ = self.forward_local(x, global_features.detach())

        # 融合特征
        final_output = self.forward_fusion(global_features, local_features)
        return final_output


# ==================== 训练函数 ====================
def train_epoch(trainer, dataloader, optimizer, criterion, device, epoch, T=3):
    """训练一个epoch"""
    trainer.model_prime.train()
    trainer.model.train()
    trainer.fc_prime.train()
    trainer.fc_local.train()
    trainer.fc_fusion.train()
    trainer.fc_final.train()
    trainer.policy.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Training]', ncols=120)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        losses = []

        # 全局视角
        global_features, global_output = trainer.forward_global(inputs)
        loss_global = criterion(global_output, targets)
        losses.append(loss_global)

        # 局部视角（多步）
        state = global_features.detach()
        for t in range(T):
            local_features, local_output, _ = trainer.forward_local(inputs, state)
            loss_local = criterion(local_output, targets)
            losses.append(loss_local)

            # 更新状态（使用局部特征）
            state = local_features.detach()

            # 融合特征
            if t == T - 1:  # 最后一步融合
                final_output = trainer.forward_fusion(global_features, local_features)
                loss_fusion = criterion(final_output, targets)
                losses.append(loss_fusion)

                # 用于计算准确率
                outputs = final_output

        # 总损失
        total_loss = sum(losses) / len(losses)

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 统计
        running_loss += total_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        avg_loss = running_loss / (batch_idx + 1)
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{avg_loss:.3f}',
            'Acc': f'{accuracy:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def test_natural_accuracy(trainer, dataloader, device, num_samples=None, T=3):
    """测试自然精度"""
    trainer.model_prime.eval()
    trainer.model.eval()
    trainer.fc_prime.eval()
    trainer.fc_local.eval()
    trainer.fc_fusion.eval()
    trainer.fc_final.eval()
    trainer.policy.eval()

    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Testing Natural Accuracy', ncols=100)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            if num_samples is not None and total >= num_samples:
                break

            inputs, targets = inputs.to(device), targets.to(device)

            if num_samples is not None and total + inputs.size(0) > num_samples:
                inputs = inputs[:num_samples - total]
                targets = targets[:num_samples - total]

            # 使用简化的前向传播
            outputs = trainer.forward_single(inputs)

            # 计算准确率
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            accuracy = 100. * correct / total
            pbar.set_postfix({'Accuracy': f'{accuracy:.2f}%'})

    accuracy = 100. * correct / total
    return accuracy


def test_robust_accuracy(trainer, dataloader, device,
                         patch_size=96,
                         eps=8 / 255, alpha=2 / 255, steps=20,
                         num_samples=None):
    """测试PGD攻击下的鲁棒精度"""
    trainer.model_prime.eval()
    trainer.model.eval()
    trainer.fc_prime.eval()
    trainer.fc_local.eval()
    trainer.fc_fusion.eval()
    trainer.fc_final.eval()
    trainer.policy.eval()

    # 创建攻击器 - 使用一个简化的模型进行攻击
    class SimpleModelForAttack(nn.Module):
        def __init__(self, trainer):
            super().__init__()
            self.trainer = trainer

        def forward(self, x):
            return self.trainer.forward_single(x)

    attack_model = SimpleModelForAttack(trainer)
    attack_model.eval()

    # 创建攻击器
    attacker = PGDAttack(trainer.model_prime, trainer.fc_fusion, patch_size,
                         eps=eps, alpha=alpha, steps=steps)

    natural_correct = 0
    robust_correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Testing Robust Accuracy (PGD)', ncols=120)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        if num_samples is not None and total >= num_samples:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        if num_samples is not None and total + inputs.size(0) > num_samples:
            inputs = inputs[:num_samples - total]
            targets = targets[:num_samples - total]

        batch_size = inputs.size(0)

        # 测试自然精度
        with torch.no_grad():
            outputs = trainer.forward_single(inputs)
            _, predicted = outputs.max(1)
            natural_correct += predicted.eq(targets).sum().item()

        # 生成对抗样本
        with torch.enable_grad():
            adv_inputs = attacker.attack(inputs, targets)

        # 测试鲁棒精度
        with torch.no_grad():
            outputs = trainer.forward_single(adv_inputs)
            _, predicted = outputs.max(1)
            robust_correct += predicted.eq(targets).sum().item()

        total += batch_size

        # 更新进度条
        natural_acc = 100. * natural_correct / total
        robust_acc = 100. * robust_correct / total
        pbar.set_postfix({
            'Natural': f'{natural_acc:.2f}%',
            'Robust': f'{robust_acc:.2f}%'
        })

    natural_accuracy = 100. * natural_correct / total
    robust_accuracy = 100. * robust_correct / total

    return natural_accuracy, robust_accuracy


# ==================== 主函数 ====================
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 准备数据
    print('Loading ImageNet10 dataset...')
    train_loader, val_loader, num_classes = prepare_data(args.data_dir, args.batch_size)
    print(f'Number of classes: {num_classes}')
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')

    # 创建双模型训练器
    print(f'Creating dual model with architecture: {args.model_arch}')
    trainer = DualModelTrainer(args.model_arch, num_classes, device, args.patch_size)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 收集所有可训练参数
    params = []
    params += list(trainer.model_prime.parameters())
    params += list(trainer.model.parameters())
    params += list(trainer.fc_prime.parameters())
    params += list(trainer.fc_local.parameters())
    params += list(trainer.fc_fusion.parameters())
    params += list(trainer.fc_final.parameters())
    params += list(trainer.policy.parameters())

    optimizer = optim.SGD(params, lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    print(f'\nStarting training for {args.epochs} epochs...')
    print(f'Dual model configuration:')
    print(f'  Global Model: {args.model_arch}')
    print(f'  Local Model: {args.model_arch}')
    print(f'  Patch Size: {args.patch_size}')
    print(f'  T (steps): {args.T}')
    print('=' * 80)

    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'robust_acc': []
    }

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print('-' * 80)

        # 训练
        train_loss, train_acc = train_epoch(
            trainer, train_loader, optimizer, criterion,
            device, epoch, T=args.T
        )

        # 测试自然精度
        val_acc = test_natural_accuracy(
            trainer, val_loader, device,
            num_samples=args.test_samples, T=args.T
        )

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_prime_state_dict': trainer.model_prime.state_dict(),
                'model_state_dict': trainer.model.state_dict(),
                'fc_prime_state_dict': trainer.fc_prime.state_dict(),
                'fc_local_state_dict': trainer.fc_local.state_dict(),
                'fc_fusion_state_dict': trainer.fc_fusion.state_dict(),
                'fc_final_state_dict': trainer.fc_final.state_dict(),
                'policy_state_dict': trainer.policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
            }, os.path.join(args.output_dir, 'best_dual_model.pth'))

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # 打印epoch总结
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
        print(f'  Validation - Accuracy: {val_acc:.2f}%, Best: {best_acc:.2f}%')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # 记录到日志文件
        write_log(f'Epoch {epoch + 1}: Train Loss={train_loss:.4f}, '
                  f'Train Acc={train_acc:.2f}%, '
                  f'Val Acc={val_acc:.2f}%', log_file)

        # 每5个epoch测试一次鲁棒性
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            try:
                natural_acc, robust_acc = test_robust_accuracy(
                    trainer, val_loader, device,
                    patch_size=args.patch_size,
                    eps=args.pgd_eps,
                    alpha=args.pgd_alpha,
                    steps=args.pgd_steps // 2,  # 训练时使用较少的步数
                    num_samples=min(args.test_samples, 200)
                )

                history['robust_acc'].append(robust_acc)
                write_log(f'  Robustness - Natural: {natural_acc:.2f}%, '
                          f'Robust: {robust_acc:.2f}%, '
                          f'Drop: {natural_acc - robust_acc:.2f}%', log_file)
            except Exception as e:
                print(f'  Robustness testing failed: {e}')
                write_log(f'  Robustness testing failed: {e}', log_file)

    print('\n' + '=' * 80)
    print('Training Complete!')
    print('=' * 80)

    # 加载最佳模型
    print('\nLoading best model for final testing...')
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_dual_model.pth'))
    trainer.model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.fc_prime.load_state_dict(checkpoint['fc_prime_state_dict'])
    trainer.fc_local.load_state_dict(checkpoint['fc_local_state_dict'])
    trainer.fc_fusion.load_state_dict(checkpoint['fc_fusion_state_dict'])
    trainer.fc_final.load_state_dict(checkpoint['fc_final_state_dict'])
    trainer.policy.load_state_dict(checkpoint['policy_state_dict'])

    # 最终测试
    print('\n' + '=' * 80)
    print('Final Evaluation')
    print('=' * 80)

    # 测试自然精度
    print('\nTesting natural accuracy on validation set...')
    final_natural_acc = test_natural_accuracy(
        trainer, val_loader, device,
        num_samples=None, T=args.T
    )
    print(f'Final Natural Accuracy: {final_natural_acc:.2f}%')

    # 测试鲁棒精度
    if args.test_robustness:
        print('\n' + '=' * 80)
        print('Testing Robust Accuracy with PGD Attack')
        print('=' * 80)
        print(f'PGD Parameters: ε={args.pgd_eps:.3f}, α={args.pgd_alpha:.3f}, steps={args.pgd_steps}')

        try:
            natural_acc, robust_acc = test_robust_accuracy(
                trainer, val_loader, device,
                patch_size=args.patch_size,
                eps=args.pgd_eps,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps,
                num_samples=args.test_samples
            )

            print('\n' + '=' * 80)
            print('Final Results:')
            print('=' * 80)
            print(f'Natural Accuracy: {natural_acc:.2f}%')
            print(f'Robust Accuracy (PGD): {robust_acc:.2f}%')
            print(f'Robustness Drop: {natural_acc - robust_acc:.2f}%')
            print('=' * 80)

            # 记录结果到文件
            write_log('\n\nFinal Evaluation Results:', log_file)
            write_log(f'Natural Accuracy: {natural_acc:.2f}%', log_file)
            write_log(f'Robust Accuracy (PGD): {robust_acc:.2f}%', log_file)
            write_log(f'Robustness Drop: {natural_acc - robust_acc:.2f}%', log_file)
            write_log(f'PGD Parameters: ε={args.pgd_eps}, α={args.pgd_alpha}, steps={args.pgd_steps}', log_file)
        except Exception as e:
            print(f'\nError in robustness testing: {e}')
            write_log(f'\nError in final robustness testing: {e}', log_file)

    # 保存训练历史
    history_file = os.path.join(args.output_dir, 'training_history.pth')
    torch.save(history, history_file)
    print(f'\nTraining history saved to: {history_file}')

    print('\nAll done! Results saved to:', log_file)


# ==================== 运行 ====================
if __name__ == '__main__':
    main()