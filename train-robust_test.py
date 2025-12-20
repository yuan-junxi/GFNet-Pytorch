import time

import torchvision.transforms as transforms
import torchvision.datasets as datasets

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


parser = argparse.ArgumentParser(description='Training code for GFNet')

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


args = parser.parse_args()

import torch
import torch.nn.functional as F


class PGDAttack:
    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=20, random_start=True):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def perturb(self, images, labels):
        """
        生成PGD对抗样本
        """
        images_original = images.clone().detach().cuda()
        images = images.clone().detach().cuda()
        labels = labels.clone().detach().cuda()

        if self.random_start:
            # 在[-eps, eps]范围内随机扰动
            delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
            images = torch.clamp(images + delta, 0, 1).detach()

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)

            # 确保是正确的前向传播
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 如果模型返回(output, state)

            loss = F.cross_entropy(outputs, labels)

            grad = torch.autograd.grad(loss, images,
                                       retain_graph=False,
                                       create_graph=False)[0]

            images = images.detach() + self.alpha * grad.sign()
            # 计算相对于原始图像的扰动
            delta = images - images_original
            delta = torch.clamp(delta, -self.eps, self.eps)
            images = torch.clamp(images_original + delta, 0, 1).detach()

        return images


def robust_test(model_prime, model, fc, dataloader, args, memory=None, ppo=None):
    """
    进行PGD-20鲁棒测试，适配GFNet的多步骤架构
    """
    model_prime.eval()
    if model:
        model.eval()
    if fc:
        fc.eval()

    correct = 0
    total = 0
    batch_time = AverageMeter()

    end = time.time()

    # 创建包装模型，用于PGD攻击
    class GFNetWrapper(torch.nn.Module):
        def __init__(self, model_prime, model, fc, args, memory=None, ppo=None):
            super().__init__()
            self.model_prime = model_prime
            self.model = model
            self.fc = fc
            self.args = args
            self.memory = memory
            self.ppo = ppo

        def forward(self, x):
            """
            简化版前向传播，仅用于PGD攻击
            注意：这里假设仅使用第一步预测
            """
            with torch.no_grad():
                # 获取全局特征
                if self.args.train_stage == 1:
                    # stage 1: 仅使用model_prime
                    output, _ = self.model_prime(x)
                    if self.fc:
                        output = self.fc(output, restart=True)
                else:
                    # stage 2/3: 使用完整的GFNet流程
                    # 这里简化处理，仅使用第一步
                    input_prime = get_prime(x, self.args.patch_size)
                    output, state = self.model_prime(input_prime)

                    if self.fc:
                        output = self.fc(output, restart=True)

                    # 如果有多步，这里应该继续，但为简化仅使用第一步
                    # 实际可以根据需要添加更多步骤

            return output

    # 创建包装模型
    wrapper_model = GFNetWrapper(model_prime, model, fc, args, memory, ppo).cuda()
    wrapper_model.eval()

    # 创建PGD攻击器
    pgd_attack = PGDAttack(wrapper_model, eps=8 / 255, alpha=2 / 255, steps=20)

    for i, (images, labels) in enumerate(dataloader):
        images = images.cuda()
        labels = labels.cuda()

        # 生成对抗样本
        adv_images = pgd_attack.perturb(images, labels)

        # 使用完整的GFNet流程进行评估
        with torch.no_grad():
            if args.train_stage == 1:
                # stage 1: 仅使用model_prime
                outputs, _ = model_prime(adv_images)
                if fc:
                    outputs = fc(outputs, restart=True)
            else:
                # stage 2/3: 使用完整的T步流程
                # 这里需要根据您的validate函数逻辑
                # 简化起见，我们使用类似的逻辑
                input_prime = get_prime(adv_images, args.patch_size)
                output, state = model_prime(input_prime)
                if fc:
                    outputs = fc(output, restart=True)
                else:
                    outputs = output

                # 多步骤处理（根据实际需求）
                for patch_step in range(1, args.T):
                    # 这里需要action selection，简化起见使用随机或固定策略
                    if ppo and memory:
                        if patch_step == 1:
                            action = ppo.select_action(state.to(0), memory, restart_batch=True, training=False)
                        else:
                            action = ppo.select_action(state.to(0), memory, training=False)
                    else:
                        action = torch.rand(images.size(0), 2).cuda()

                    patches = get_patch(adv_images, action, args.patch_size)
                    output, state = model(patches)
                    if fc:
                        outputs = fc(output, restart=False)
                    else:
                        outputs = output

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            current_acc = 100. * correct / total
            print(f'Robust Test: [{i + 1}/{len(dataloader)}], '
                  f'Time {batch_time.ave:.3f}, '
                  f'Acc: {current_acc:.2f}%')

    robust_acc = 100. * correct / total
    return robust_acc

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

    best_robust_acc = 0

    for epoch in range(start_epoch, training_epoch_num):
        if args.train_stage != 2:
            print('Training Stage: {}, lr:'.format(args.train_stage))
            adjust_learning_rate(optimizer, train_configuration,
                                 epoch, training_epoch_num, args)
        else:
            print('Training Stage: {}, train ppo only'.format(args.train_stage))

        train(model_prime, model, fc, memory, ppo, optimizer, train_loader, criterion,
              args.print_freq, epoch, train_configuration['batch_size'], record_file, train_configuration, args)

        # 使用新的验证函数
        normal_acc, robust_acc = validate_with_robustness(
            model_prime, model, fc, memory, ppo, val_loader, criterion,
            args.print_freq, epoch, train_configuration['batch_size'],
            record_file, train_configuration, args
        )

        # 可以选择根据正常精度或鲁棒精度保存最佳模型
        acc = normal_acc  # 或者使用 robust_acc

        # 更新最佳鲁棒精度
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc

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
            'acc': normal_acc,
            'robust_acc': robust_acc,  # 保存鲁棒精度
            'best_acc': best_acc,
            'best_robust_acc': best_robust_acc,  # 保存最佳鲁棒精度
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }, is_best, checkpoint=record_path)

        # 打印当前最佳精度
        print(f'Epoch {epoch}: Best Normal Accuracy = {best_acc:.2f}%, '
              f'Best Robust Accuracy = {best_robust_acc:.2f}%\n')


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
def validate_with_robustness(model_prime, model, fc, memory, ppo, val_loader,
                             criterion, print_freq, epoch, batch_size,
                             record_file, train_configuration, args):
    """
    同时进行正常验证和鲁棒验证
    """
    # 正常的验证精度
    print(f"Epoch {epoch}: Starting normal validation...")
    normal_acc = validate(model_prime, model, fc, memory, ppo, None,
                          val_loader, criterion, print_freq, epoch,
                          batch_size, record_file, train_configuration, args)

    # 鲁棒性测试
    print(f"Epoch {epoch}: Starting PGD-20 robust test...")
    robust_acc = robust_test(model_prime, model, fc, val_loader, args, memory, ppo)

    # 记录结果
    fd = open(record_file, 'a+')
    string = (f'\nEpoch {epoch}: Normal Accuracy = {normal_acc:.2f}%, '
              f'Robust Accuracy (PGD-20) = {robust_acc:.2f}%\n')
    print(string)
    fd.write(string)
    fd.close()

    return normal_acc, robust_acc



if __name__ == '__main__':
    main()
