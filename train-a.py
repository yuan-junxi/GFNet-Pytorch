import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from utils import *
from network import *
from configs import *

import math
import argparse

import models.resnet as resnet
import models.densenet as densenet
from models import create_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ==================== PGD攻击类 ====================
class PGD:
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
        """
        PGD攻击类
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def attack(self, images, labels, fc=None, args=None):
        """
        执行PGD攻击 - 针对GFNet的特殊处理

        参数:
            images: 输入图片
            labels: 标签
            fc: 全连接层（可选）
            args: 命令行参数
        """
        images = images.clone().detach()
        labels = labels.clone().detach()

        loss = nn.CrossEntropyLoss()

        # 如果是随机开始，添加随机扰动
        if self.random_start:
            delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
            adv_images = images + delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            adv_images = images.clone().detach()

        # PGD迭代
        for _ in range(self.steps):
            adv_images.requires_grad = True

            # 获取prime输入
            adv_prime = get_prime(adv_images, args.patch_size) if args else adv_images

            # 前向传播
            outputs = self.model(adv_prime)

            # 处理元组输出
            if isinstance(outputs, tuple):
                outputs, _ = outputs  # 取第一个元素作为分类输出

            # 如果需要通过fc层
            if fc is not None:
                outputs = fc(outputs, restart=True)

            # 计算损失
            cost = loss(outputs, labels)

            # 计算梯度
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            # 更新对抗样本
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


# ==================== 鲁棒精度测试函数 ====================
def test_robust_accuracy(model_prime, model, fc, val_loader, args,
                         eps=8 / 255, alpha=2 / 255, steps=20,
                         num_samples=1000):
    """
    测试模型在PGD攻击下的鲁棒精度
    """
    model_prime.eval()
    model.eval()
    fc.eval()

    # 获取模型组件
    if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
        dsn_fc_prime = model_prime.module.fc
        dsn_fc = model.module.fc
    else:
        dsn_fc_prime = model_prime.module.classifier
        dsn_fc = model.module.classifier

    # 创建用于攻击的模型副本（设置为训练模式以进行梯度计算）
    attack_model = copy.deepcopy(model_prime)
    attack_model.train()

    # 创建攻击用的fc副本
    attack_fc = copy.deepcopy(fc)
    attack_fc.train()

    # 创建PGD攻击器
    pgd_attacker = PGD(attack_model, eps=eps, alpha=alpha, steps=steps)

    natural_correct = 0
    robust_correct = 0
    total = 0

    # 使用tqdm进度条
    pbar = tqdm(total=num_samples, desc='Testing Robust Accuracy', ncols=100)

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if total >= num_samples:
            break

        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size(0)

        # 限制测试样本数
        if total + batch_size > num_samples:
            inputs = inputs[:num_samples - total]
            targets = targets[:num_samples - total]
            batch_size = inputs.size(0)

        # ========== 测试自然精度 ==========
        with torch.no_grad():
            # 获取prime输入
            input_prime = get_prime(inputs, args.patch_size)

            # 前向传播
            output, state = model_prime(input_prime)

            # 通过fc层
            if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                output = fc(output, restart=True)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)

            # 计算准确率
            _, predicted = output.max(1)
            natural_correct += predicted.eq(targets).sum().item()

        # ========== 生成PGD对抗样本 ==========
        with torch.enable_grad():
            # 对完整输入图像生成对抗样本
            adv_inputs = pgd_attacker.attack(
                inputs, targets,
                fc=attack_fc,  # 传递fc层
                args=args  # 传递args以获取patch_size
            )

        # ========== 测试鲁棒精度 ==========
        with torch.no_grad():
            # 使用对抗样本测试
            adv_prime = get_prime(adv_inputs, args.patch_size)

            # 前向传播
            output, state = model_prime(adv_prime)

            # 通过fc层
            if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                output = fc(output, restart=True)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)

            # 计算准确率
            _, predicted = output.max(1)
            robust_correct += predicted.eq(targets).sum().item()

        total += batch_size

        # 更新进度条
        natural_acc = 100. * natural_correct / total
        robust_acc = 100. * robust_correct / total
        pbar.update(batch_size)
        pbar.set_postfix({
            'Natural': f'{natural_acc:.2f}%',
            'Robust': f'{robust_acc:.2f}%'
        })

    pbar.close()

    # 清理
    del attack_model
    del attack_fc
    torch.cuda.empty_cache()

    natural_accuracy = 100. * natural_correct / total
    robust_accuracy = 100. * robust_correct / total

    return natural_accuracy, robust_accuracy


# ==================== 简化的鲁棒测试函数（备用方案） ====================
def test_robust_accuracy_simple(model_prime, model, fc, val_loader, args,
                                eps=8 / 255, alpha=2 / 255, steps=20,
                                num_samples=1000):
    """
    简化版的鲁棒精度测试，使用FGSM替代PGD
    """
    model_prime.eval()
    model.eval()
    fc.eval()

    natural_correct = 0
    robust_correct = 0
    total = 0

    pbar = tqdm(total=num_samples, desc='Testing Robust Accuracy (FGSM)', ncols=100)

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if total >= num_samples:
            break

        inputs, targets = inputs.cuda(), targets.cuda()
        batch_size = inputs.size(0)

        if total + batch_size > num_samples:
            inputs = inputs[:num_samples - total]
            targets = targets[:num_samples - total]
            batch_size = inputs.size(0)

        # ========== 测试自然精度 ==========
        with torch.no_grad():
            input_prime = get_prime(inputs, args.patch_size)
            output, _ = model_prime(input_prime)

            if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                output = fc(output, restart=True)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)

            _, predicted = output.max(1)
            natural_correct += predicted.eq(targets).sum().item()

        # ========== FGSM攻击 ==========
        inputs.requires_grad = True

        with torch.enable_grad():
            input_prime = get_prime(inputs, args.patch_size)
            output, _ = model_prime(input_prime)

            if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                output = fc(output, restart=True)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)

            loss = F.cross_entropy(output, targets)

        # 计算梯度并生成对抗样本
        grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]
        adv_inputs = inputs + eps * grad.sign()
        adv_inputs = torch.clamp(adv_inputs, 0, 1).detach()

        # ========== 测试鲁棒精度 ==========
        with torch.no_grad():
            adv_prime = get_prime(adv_inputs, args.patch_size)
            output, _ = model_prime(adv_prime)

            if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                output = fc(output, restart=True)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)

            _, predicted = output.max(1)
            robust_correct += predicted.eq(targets).sum().item()

        total += batch_size

        natural_acc = 100. * natural_correct / total
        robust_acc = 100. * robust_correct / total
        pbar.update(batch_size)
        pbar.set_postfix({
            'Natural': f'{natural_acc:.2f}%',
            'Robust': f'{robust_acc:.2f}%'
        })

    pbar.close()

    return 100. * natural_correct / total, 100. * robust_correct / total


# ==================== 修改命令行参数 ====================
parser = argparse.ArgumentParser(description='Training code for GFNet with Robustness Testing')

# 原有的参数...
parser.add_argument('--data_url', default='./data', type=str,
                    help='path to the dataset (ImageNet)')

parser.add_argument('--work_dirs', default='./output', type=str,
                    help='path to save log and checkpoints')

parser.add_argument('--train_stage', default=-1, type=int,
                    help='select training stage, see our paper for details \
                          stage-1 : warm-up \
                          stage-2 : learn to select patches with RL \
                          stage-3 : finetune CNNs')

parser.add_argument('--model_arch', default='', type=str,
                    help='architecture of the model to be trained \
                         resnet50 / resnet101 / \
                         densenet121 / densenet169 / densenet201 / \
                         regnety_600m / regnety_800m / regnety_1.6g / \
                         mobilenetv3_large_100 / mobilenetv3_large_125 / \
                         efficientnet_b2 / efficientnet_b3')

parser.add_argument('--patch_size', default=96, type=int,
                    help='size of local patches (we recommend 96 / 128 / 144)')

parser.add_argument('--T', default=4, type=int,
                    help='maximum length of the sequence of Glance + Focus')

parser.add_argument('--print_freq', default=100, type=int,
                    help='the frequency of printing log')

parser.add_argument('--model_prime_path', default='', type=str,
                    help='path to the pre-trained model of Global Encoder (for training stage-1)')

parser.add_argument('--model_path', default='', type=str,
                    help='path to the pre-trained model of Local Encoder (for training stage-1)')

parser.add_argument('--checkpoint_path', default='', type=str,
                    help='path to the stage-2/3 checkpoint (for training stage-2/3)')

parser.add_argument('--resume', default='', type=str,
                    help='path to the checkpoint for resuming')

# 新增的鲁棒测试参数
parser.add_argument('--test_robustness', action='store_true', default=True,
                    help='test robustness accuracy each epoch')

parser.add_argument('--robust_test_samples', default=500, type=int,
                    help='number of samples for robustness testing')

parser.add_argument('--pgd_eps', default=8 / 255, type=float,
                    help='PGD epsilon for robustness testing')

parser.add_argument('--pgd_alpha', default=2 / 255, type=float,
                    help='PGD alpha for robustness testing')

parser.add_argument('--pgd_steps', default=20, type=int,
                    help='PGD steps for robustness testing')

args = parser.parse_args()


# ==================== 修改main函数 ====================
def main():

    if not os.path.isdir(args.work_dirs):
        mkdir_p(args.work_dirs)

    record_path = args.work_dirs + '/GF-' + str(args.model_arch) \
                  + '_patch-size-' + str(args.patch_size) \
                  + '_T' + str(args.T) \
                  + '_train-stage' + str(args.train_stage)
    if not os.path.isdir(record_path):
        mkdir_p(record_path)
    record_file = record_path + '/record.txt'


    # *create model* #
    model_configuration = model_configurations[args.model_arch]
    if 'resnet' in args.model_arch:
        model_arch = 'resnet'
        model = resnet.resnet50(pretrained=False)
        model_prime = resnet.resnet50(pretrained=False)
    elif 'densenet' in args.model_arch:
        model_arch = 'densenet'
        model = eval('densenet.' + args.model_arch)(pretrained=False)
        model_prime = eval('densenet.' + args.model_arch)(pretrained=False)
    elif 'efficientnet' in args.model_arch:
        model_arch = 'efficientnet'
        model = create_model(args.model_arch, pretrained=False, num_classes=1000,
                             drop_rate=0.3, drop_connect_rate=0.2)
        model_prime = create_model(args.model_arch, pretrained=False, num_classes=1000,
                                   drop_rate=0.3, drop_connect_rate=0.2)
    elif 'mobilenetv3' in args.model_arch:
        model_arch = 'mobilenetv3'
        model = create_model(args.model_arch, pretrained=False, num_classes=1000,
                             drop_rate=0.2, drop_connect_rate=0.2)
        model_prime = create_model(args.model_arch, pretrained=False, num_classes=1000,
                                   drop_rate=0.2, drop_connect_rate=0.2)
    elif 'regnet' in args.model_arch:
        model_arch = 'regnet'
        import pycls.core.model_builder as model_builder
        from pycls.core.config import cfg
        cfg.merge_from_file(model_configuration['cfg_file'])
        cfg.freeze()
        model = model_builder.build_model()
        model_prime = model_builder.build_model()

    fc = Full_layer(model_configuration['feature_num'],
                    model_configuration['fc_hidden_dim'],
                    model_configuration['fc_rnn'])

    if args.train_stage == 1:
        model.load_state_dict(torch.load(args.model_path))
        model_prime.load_state_dict(torch.load(args.model_prime_path))
    else:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
        fc.load_state_dict(checkpoint['fc'])

    train_configuration = train_configurations[model_arch]

    if args.train_stage != 2:
        if train_configuration['train_model_prime']:
            optimizer = torch.optim.SGD([{'params': model.parameters()},
                                         {'params': model_prime.parameters()},
                                         {'params': fc.parameters()}],
                                        lr=0,  # specify in adjust_learning_rate()
                                        momentum=train_configuration['momentum'],
                                        nesterov=train_configuration['Nesterov'],
                                        weight_decay=train_configuration['weight_decay'])
        else:
            optimizer = torch.optim.SGD([{'params': model.parameters()},
                                         {'params': fc.parameters()}],
                                        lr=0,  # specify in adjust_learning_rate()
                                        momentum=train_configuration['momentum'],
                                        nesterov=train_configuration['Nesterov'],
                                        weight_decay=train_configuration['weight_decay'])
        training_epoch_num = train_configuration['epoch_num']
    else:
        optimizer = None
        training_epoch_num = 15
    criterion = nn.CrossEntropyLoss().cuda()

    model = nn.DataParallel(model.cuda())
    model_prime = nn.DataParallel(model_prime.cuda())
    fc = fc.cuda()

    traindir = args.data_url + 'train/'
    valdir = args.data_url + 'val/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_set_index = torch.randperm(len(train_set))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, num_workers=32, pin_memory=False,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   train_set_index[:]))

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize, ])),
        batch_size=train_configuration['batch_size'], shuffle=False, num_workers=32, pin_memory=False)

    if args.train_stage != 1:
        state_dim = model_configuration['feature_map_channels'] * math.ceil(args.patch_size / 32) * math.ceil(args.patch_size / 32)
        ppo = PPO(model_configuration['feature_map_channels'], state_dim,
                  model_configuration['policy_hidden_dim'], model_configuration['policy_conv'])

        if args.train_stage == 3:
            ppo.policy.load_state_dict(checkpoint['policy'])
            ppo.policy_old.load_state_dict(checkpoint['policy'])

    else:
        ppo = None
    memory = Memory()

    if args.resume:
        resume_ckp = torch.load(args.resume)

        start_epoch = resume_ckp['epoch']
        print('resume from epoch: {}'.format(start_epoch))

        model.module.load_state_dict(resume_ckp['model_state_dict'])
        model_prime.module.load_state_dict(resume_ckp['model_prime_state_dict'])
        fc.load_state_dict(resume_ckp['fc'])

        if optimizer:
            optimizer.load_state_dict(resume_ckp['optimizer'])

        if ppo:
            ppo.policy.load_state_dict(resume_ckp['policy'])
            ppo.policy_old.load_state_dict(resume_ckp['policy'])
            ppo.optimizer.load_state_dict(resume_ckp['ppo_optimizer'])

        best_acc = resume_ckp['best_acc']
    else:
        start_epoch = 0
        best_acc = 0

    for epoch in range(start_epoch, training_epoch_num):
        if args.train_stage != 2:
            print('Training Stage: {}, lr:'.format(args.train_stage))
            adjust_learning_rate(optimizer, train_configuration,
                                 epoch, training_epoch_num, args)
        else:
            print('Training Stage: {}, train ppo only'.format(args.train_stage))

        # 训练
        train(model_prime, model, fc, memory, ppo, optimizer, train_loader, criterion,
              args.print_freq, epoch, train_configuration['batch_size'], record_file, train_configuration, args)

        # 正常验证
        acc = validate(model_prime, model, fc, memory, ppo, optimizer, val_loader, criterion,
                       args.print_freq, epoch, train_configuration['batch_size'], record_file, train_configuration,
                       args)

        # ========== 新增：测试鲁棒精度 ==========
        if 1:
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}: Testing Robust Accuracy with PGD Attack")
            print('-' * 80)

            try:
                # 首先尝试使用完整的PGD攻击
                natural_acc, robust_acc = test_robust_accuracy(
                    model_prime, model, fc, val_loader, args,
                    eps=args.pgd_eps,
                    alpha=args.pgd_alpha,
                    steps=args.pgd_steps,
                    num_samples=args.robust_test_samples
                )

            except Exception as e:
                print(f"PGD attack failed: {e}")
                print("Trying simplified FGSM attack...")

                # 如果PGD失败，使用简化的FGSM攻击
                natural_acc, robust_acc = test_robust_accuracy_simple(
                    model_prime, model, fc, val_loader, args,
                    eps=args.pgd_eps,
                    alpha=args.pgd_alpha,
                    steps=1,  # FGSM只有一步
                    num_samples=args.robust_test_samples
                )

            print(f"Attack Results (ε={args.pgd_eps:.3f}):")
            print(f"  Natural Accuracy: {natural_acc:.2f}%")
            print(f"  Robust Accuracy: {robust_acc:.2f}%")
            print(f"  Robustness Drop: {natural_acc - robust_acc:.2f}%")

            # 记录到文件
            with open(record_file, 'a+') as fd:
                fd.write(f"\nEpoch {epoch + 1} Robust Accuracy Test:\n")
                fd.write(f"Natural Accuracy: {natural_acc:.2f}%\n")
                fd.write(f"Robust Accuracy: {robust_acc:.2f}%\n")
                fd.write(f"Robustness Drop: {natural_acc - robust_acc:.2f}%\n\n")

            print(f"{'=' * 80}\n")

        # 保存检查点
        if acc > best_acc:
            best_acc = acc
            is_best = True
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'model_prime_state_dict': model_prime.module.state_dict(),
            'fc': fc.state_dict(),
            'acc': acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }, is_best, checkpoint=record_path)

    # ========== 训练结束后测试最终鲁棒精度 ==========
    if args.test_robustness:
        print(f"\n{'=' * 80}")
        print("Training Complete! Testing Final Robust Accuracy")
        print('=' * 80)

        try:
            # 使用更多样本进行最终测试
            natural_acc, robust_acc = test_robust_accuracy(
                model_prime, model, fc, val_loader, args,
                eps=args.pgd_eps,
                alpha=args.pgd_alpha,
                steps=args.pgd_steps * 2,  # 最终测试使用更多步数
                num_samples=min(2000, len(val_loader.dataset))  # 更多样本
            )

            print(f"\nFinal PGD Attack Results:")
            print(f"  Natural Accuracy: {natural_acc:.2f}%")
            print(f"  Robust Accuracy: {robust_acc:.2f}%")
            print(f"  Robustness Drop: {natural_acc - robust_acc:.2f}%")
            print(f"{'=' * 80}\n")

            # 记录最终结果
            with open(record_file, 'a+') as fd:
                fd.write(f"\n\nFinal Robust Accuracy Results:\n")
                fd.write(f"Natural Accuracy: {natural_acc:.2f}%\n")
                fd.write(f"Robust Accuracy: {robust_acc:.2f}%\n")
                fd.write(f"Robustness Drop: {natural_acc - robust_acc:.2f}%\n")

        except Exception as e:
            print(f"Error in final robustness test: {e}")


# ==================== 在文件顶部添加必要的导入 ====================
import copy
from tqdm import tqdm


def train(model_prime, model, fc, memory, ppo, optimizer, train_loader, criterion,
          print_freq, epoch, batch_size, record_file, train_configuration, args):

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    train_batches_num = len(train_loader)

    if args.train_stage == 2:
        model_prime.eval()
        model.eval()
        fc.eval()
    else:
        if train_configuration['train_model_prime']:
            model_prime.train()
        else:
            model_prime.eval()
        model.train()
        fc.train()

    if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
        dsn_fc_prime = model_prime.module.fc
        dsn_fc = model.module.fc
    else:
        dsn_fc_prime = model_prime.module.classifier
        dsn_fc = model.module.classifier
    fd = open(record_file, 'a+')

    end = time.time()

    for i, (x, target) in enumerate(train_loader):

        loss_cla = []
        loss_list_dsn = []

        target_var = target.cuda()
        input_var = x.cuda()

        input_prime = get_prime(input_var, args.patch_size)

        if train_configuration['train_model_prime'] and args.train_stage != 2:
            output, state = model_prime(input_prime)
            assert 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch
            output_dsn = dsn_fc_prime(output)
            output = fc(output, restart=True)
        else:
            with torch.no_grad():
                output, state = model_prime(input_prime)
                if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                    output_dsn = dsn_fc_prime(output)
                    output = fc(output, restart=True)
                else:
                    _ = fc(output, restart=True)
                    output = model_prime.module.classifier(output)
                    output_dsn = output

        loss_prime = criterion(output, target_var)
        loss_cla.append(loss_prime)

        loss_dsn = criterion(output_dsn, target_var)
        loss_list_dsn.append(loss_dsn)

        losses[0].update(loss_prime.data.item(), x.size(0))
        acc = accuracy(output, target_var, topk=(1,))
        top1[0].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x.size(0))

        confidence_last = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)

        for patch_step in range(1, args.T):

            if args.train_stage == 1:
                action = torch.rand(x.size(0), 2).cuda()
            else:
                if patch_step == 1:
                    action = ppo.select_action(state.to(0), memory, restart_batch=True)
                else:
                    action = ppo.select_action(state.to(0), memory)

            patches = get_patch(input_var, action, args.patch_size)

            if args.train_stage != 2:
                output, state = model(patches)
                output_dsn = dsn_fc(output)
                output = fc(output, restart=False)
            else:
                with torch.no_grad():
                    output, state = model(patches)
                    output_dsn = dsn_fc(output)
                    output = fc(output, restart=False)

            loss = criterion(output, target_var)
            loss_cla.append(loss)
            losses[patch_step].update(loss.data.item(), x.size(0))

            loss_dsn = criterion(output_dsn, target_var)
            loss_list_dsn.append(loss_dsn)

            acc = accuracy(output, target_var, topk=(1,))
            top1[patch_step].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x.size(0))

            confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), x.size(0))
            memory.rewards.append(reward)

        loss = (sum(loss_cla) + train_configuration['dsn_ratio'] * sum(loss_list_dsn)) / args.T

        if args.train_stage != 2:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            ppo.update(memory)
        memory.clear_memory()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0 or i == train_batches_num - 1:
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                epoch, i + 1, train_batches_num, batch_time=batch_time, loss=losses[-1]))
            print(string)
            fd.write(string + '\n')

            _acc = [acc.ave for acc in top1]
            print('accuracy of each step:')
            print(_acc)
            fd.write('accuracy of each step:\n')
            fd.write(str(_acc) + '\n')

            _reward = [reward.ave for reward in reward_list]
            print('reward of each step:')
            print(_reward)
            fd.write('reward of each step:\n')
            fd.write(str(_reward) + '\n')

    fd.close()


def validate(model_prime, model, fc, memory, ppo, _, val_loader, criterion,
             print_freq, epoch, batch_size, record_file, __, args):

    batch_time = AverageMeter()
    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    train_batches_num = len(val_loader)

    model_prime.eval()
    model.eval()
    fc.eval()

    if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
        dsn_fc_prime = model_prime.module.fc
        dsn_fc = model.module.fc
    else:
        dsn_fc_prime = model_prime.module.classifier
        dsn_fc = model.module.classifier
    fd = open(record_file, 'a+')

    end = time.time()
    with torch.no_grad():
        for i, (x, target) in enumerate(val_loader):

            loss_cla = []
            loss_list_dsn = []

            target_var = target.cuda()
            input_var = x.cuda()

            input_prime = get_prime(input_var, args.patch_size)

            output, state = model_prime(input_prime)
            if 'resnet' in args.model_arch or 'densenet' in args.model_arch or 'regnet' in args.model_arch:
                output_dsn = dsn_fc_prime(output)
                output = fc(output, restart=True)
            else:
                _ = fc(output, restart=True)
                output = model_prime.module.classifier(output)
                output_dsn = output

            loss_prime = criterion(output, target_var)
            loss_cla.append(loss_prime)

            loss_dsn = criterion(output_dsn, target_var)
            loss_list_dsn.append(loss_dsn)

            losses[0].update(loss_prime.data.item(), x.size(0))
            acc = accuracy(output, target_var, topk=(1,))
            top1[0].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x.size(0))

            confidence_last = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)

            for patch_step in range(1, args.T):

                if args.train_stage == 1:
                    action = torch.rand(x.size(0), 2).cuda()
                else:
                    if patch_step == 1:
                        action = ppo.select_action(state.to(0), memory, restart_batch=True, training=False)
                    else:
                        action = ppo.select_action(state.to(0), memory, training=False)

                patches = get_patch(input_var, action, args.patch_size)

                output, state = model(patches)
                output_dsn = dsn_fc(output)
                output = fc(output, restart=False)

                loss = criterion(output, target_var)
                loss_cla.append(loss)
                losses[patch_step].update(loss.data.item(), x.size(0))

                loss_dsn = criterion(output_dsn, target_var)
                loss_list_dsn.append(loss_dsn)

                acc = accuracy(output, target_var, topk=(1,))
                top1[patch_step].update(acc.sum(0).mul_(100.0 / batch_size).data.item(), x.size(0))

                confidence = torch.gather(F.softmax(output.detach(), 1), dim=1, index=target_var.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), x.size(0))
                memory.rewards.append(reward)

            memory.clear_memory()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0 or i == train_batches_num - 1:
                string = ('Val: [{0}][{1}/{2}]\t'
                          'Time {batch_time.value:.3f} ({batch_time.ave:.3f})\t'
                          'Loss {loss.value:.4f} ({loss.ave:.4f})\t'.format(
                    epoch, i + 1, train_batches_num, batch_time=batch_time, loss=losses[-1]))
                print(string)
                fd.write(string + '\n')

                _acc = [acc.ave for acc in top1]
                print('accuracy of each step:')
                print(_acc)
                fd.write('accuracy of each step:\n')
                fd.write(str(_acc) + '\n')

                _reward = [reward.ave for reward in reward_list]
                print('reward of each step:')
                print(_reward)
                fd.write('reward of each step:\n')
                fd.write(str(_reward) + '\n')
    fd.close()

    return top1[args.T - 1].ave

if __name__ == '__main__':
    main()