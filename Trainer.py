import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
import datetime
import math
from sklearn.metrics import confusion_matrix
import warnings
from sam import SAM


class Trainer(object):
    def __init__(self, args, model=None, train_loader=None, val_loader=None, weighted_train_loader=None,
                 per_class_num=[], log=None):
        # 初始化函数
        self.args = args  # 存储命令行参数
        self.device = args.gpu  # 使用的 GPU ID
        self.print_freq = args.print_freq  # 打印频率
        self.lr = args.lr  # 初始学习率
        self.label_weighting = args.label_weighting  # 标签加权
        self.epochs = args.epochs  # 总训练轮数
        self.start_epoch = args.start_epoch  # 起始训练轮数
        self.use_cuda = True  # 是否使用 CUDA
        self.num_classes = args.num_classes  # 类别数量
        self.train_loader = train_loader  # 训练数据加载器
        self.val_loader = val_loader  # 验证数据加载器
        self.weighted_train_loader = weighted_train_loader  # 加权训练数据加载器
        self.per_cls_weights = None  # 每个类别的权重
        self.cls_num_list = per_class_num  # 每个类别的样本数量
        self.contrast_weight = args.contrast_weight  # 对比损失权重
        self.model = model  # 模型

        # 优化器
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(base_optimizer=base_optimizer, rho=args.rho, params=model.parameters(), lr=args.lr,
                                 momentum=args.momentum, weight_decay=args.weight_decay)
        self.rho = args.rho
        # 学习率调度器
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.log = log  # 日志记录器
        self.beta = args.beta  # 混合数据增强的参数
        self.update_weight()  # 更新每个类别的权重

    def update_weight(self):
        # 更新每个类别的权重
        per_cls_weights = 1.0 / (np.array(self.cls_num_list) ** self.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.cls_num_list)
        self.per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()

    def train(self):
        # 训练函数
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            alpha = 1 - (epoch / self.epochs) ** 2  # alpha 是一个与当前训练轮数相关的参数
            batch_time = AverageMeter('Time', ':6.3f')  # 记录每个批次的时间
            data_time = AverageMeter('Data', ':6.3f')  # 记录数据加载时间
            losses = AverageMeter('Loss', ':.4e')  # 记录损失
            top1 = AverageMeter('Acc@1', ':6.2f')  # 记录 top-1 准确率
            top5 = AverageMeter('Acc@5', ':6.2f')  # 记录 top-5 准确率

            # 切换到训练模式
            self.model.train()
            end = time.time()
            weighted_train_loader = iter(self.weighted_train_loader)

            for i, (inputs, targets) in enumerate(self.train_loader):
                input_org_1 = inputs[0]
                input_org_2 = inputs[1]
                target_org = targets

                try:
                    input_invs, target_invs = next(weighted_train_loader)
                except:
                    weighted_train_loader = iter(self.weighted_train_loader)
                    input_invs, target_invs = next(weighted_train_loader)

                input_invs_1 = input_invs[0][:input_org_1.size()[0]]
                input_invs_2 = input_invs[1][:input_org_2.size()[0]]

                one_hot_org = torch.zeros(target_org.size(0), self.num_classes).scatter_(1, target_org.view(-1, 1), 1)
                one_hot_org_w = self.per_cls_weights.cpu() * one_hot_org
                one_hot_invs = torch.zeros(target_invs.size(0), self.num_classes).scatter_(1, target_invs.view(-1, 1),
                                                                                           1)
                one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
                one_hot_invs_w = self.per_cls_weights.cpu() * one_hot_invs

                input_org_1 = input_org_1.cuda()
                input_org_2 = input_org_2.cuda()
                input_invs_1 = input_invs_1.cuda()
                input_invs_2 = input_invs_2.cuda()

                one_hot_org = one_hot_org.cuda()
                one_hot_org_w = one_hot_org_w.cuda()
                one_hot_invs = one_hot_invs.cuda()
                one_hot_invs_w = one_hot_invs_w.cuda()

                # 记录数据加载时间
                data_time.update(time.time() - end)

                # 数据增强
                lam = np.random.beta(self.beta, self.beta)
                mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1,
                                                                                         org2=input_org_2,
                                                                                         invs1=input_invs_1,
                                                                                         invs2=input_invs_2,
                                                                                         label_org=one_hot_org,
                                                                                         label_invs=one_hot_invs,
                                                                                         label_org_w=one_hot_org_w,
                                                                                         label_invs_w=one_hot_invs_w)

                output_1, output_cb_1, z1, p1 = self.model(mix_x, train=True)
                output_2, output_cb_2, z2, p2 = self.model(cut_x, train=True)
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (1 - alpha) * rebalance_loss

                losses.update(loss.item(), inputs[0].size(0))



                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.first_step(zero_grad=True)


                output_1, output_cb_1, z1, p1 = self.model(mix_x, train=True)
                output_2, output_cb_2, z2, p2 = self.model(cut_x, train=True)
                contrastive_loss = self.SimSiamLoss(p1, z2) + self.SimSiamLoss(p2, z1)

                loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1))
                loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1))
                loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1))
                loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1))

                balance_loss = loss_mix + loss_cut
                rebalance_loss = loss_mix_w + loss_cut_w

                loss = alpha * balance_loss + (
                            1 - alpha) * rebalance_loss + self.contrast_weight * contrastive_loss  # 为分析硬编码对比权重

                loss.backward()
                self.optimizer.second_step(zero_grad=True)

                # 记录经过的时间
                batch_time.update(time.time() - end)
                end = time.time()
                if i % self.print_freq == 0:
                    output = ('Epoch: [{0}/{1}][{2}/{3}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch + 1, self.epochs, i, len(self.train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))  # TODO
                    print(output)
            # 在验证集上进行评估
            acc1 = self.validate(epoch=epoch)
            if self.args.dataset == 'ImageNet-LT' or self.args.dataset == 'iNaturelist2018':
                self.adjust_learning_rate(self.optimizer, epoch, self.args)
            else:
                self.train_scheduler.step()
            # 记住最佳的 acc@1 并保存检查点
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1': best_acc1,
            }, is_best, epoch + 1)

    def validate(self, epoch=None):
        # 验证函数
        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps  # 防止除零错误

        # 切换到评估模式
        self.model.eval()
        all_preds = []
        all_targets = []

        confidence = np.array([])
        pred_class = np.array([])
        true_class = np.array([])

        with torch.no_grad():
            end = time.time()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.cuda()

                # 计算输出
                output = self.model(input, train=False)

                # 计算准确率
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1.item(), input.size(0))
                top5.update(acc5.item(), input.size(0))

                # 记录经过的时间
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                if i % self.print_freq == 0:
                    output = ('Test: [{0}/{1}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(self.val_loader), batch_time=batch_time, top1=top1, top5=top5))
                    print(output)
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)  # 每个类别的样本数量
            cls_hit = np.diag(cf)  # 每个类别的正确预测数量
            cls_acc = cls_hit / cls_cnt  # 每个类别的准确率
            output = (
                'EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(epoch=epoch + 1,
                                                                                                    flag='val',
                                                                                                    top1=top1,
                                                                                                    top5=top5))

            self.log.info(output)
            out_cls_acc = '%s Class Accuracy: %s' % (
                'val', (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))

            many_shot = self.cls_num_list > 100
            medium_shot = (self.cls_num_list <= 100) & (self.cls_num_list > 20)
            few_shot = self.cls_num_list <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
        return top1.avg

    def SimSiamLoss(self, p, z, version='simplified'):  # SimSiam 对比损失
        z = z.detach()  # 停止梯度更新

        if version == 'original':
            p = F.normalize(p, dim=1)  # l2 归一化
            z = F.normalize(z, dim=1)  # l2 归一化
            return -(p * z).sum(dim=1).mean()

        elif version == 'simplified':  # 更简单的计算方法
            return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception

    def adjust_learning_rate(self, optimizer, epoch, args):
        # 调整学习率
        warmup_epochs = 15
        lr = self.lr
        if epoch <= warmup_epochs:
            lr = self.lr / warmup_epochs * (epoch + 1)
        else:  # 余弦学习率调度
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (self.epochs - warmup_epochs + 1)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr




