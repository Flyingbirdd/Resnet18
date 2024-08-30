import os
import argparse
import torch
from utils import util
from utils.util import *
from model import ResNet_cifar
from model import Resnet_LT
from imbalance_data import cifar10Imbanlance, cifar100Imbanlance, dataset_lt_data




# 主函数，负责解析参数、加载模型和数据集、并调用验证函数进行评估
def main():
    args = parser.parse_args()

    # 如果指定了GPU，则输出使用的GPU信息
    if args.gpu is not None:
        print("Use GPU: {} for testing".format(args.gpu))

    # 创建模型
    model = get_model(args)

    # 将模型转移到指定的GPU上
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # 如果没有指定GPU，使用DataParallel将模型分配到所有可用的GPU上
        model = torch.nn.DataParallel(model).cuda()

    # 从指定的检查点加载模型
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # 加载验证数据集
    val_dataset = get_dataset(args)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)

    print("Testing started!")
    # 切换到评估模式
    model.eval()
    # 调用验证函数进行评估
    validate(model, val_loader, args)


# 如果该脚本作为主程序运行，解析命令行参数并调用主函数
if __name__ == '__main__':
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Global and Local Mixture Consistency Cumulative Learning")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help="cifar10, cifar100, ImageNet-LT, iNaturelist2018")
    parser.add_argument('--root', type=str, default='/data/', help="dataset setting")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                        choices=('resnet18', 'resnet32', 'resnet50', 'resnext50_32x4d'))
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ',
                        choices=('10', '100', '1000', '8142'))
    parser.add_argument('-b', '--batch_size', default=100, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='model path', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--root_model', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--store_name', type=str, default='GLMC-CVPR2023/output/')
    parser.add_argument('--dir_train_txt', type=str, default="GLMC-CVPR2023/data/data_txt/iNaturalist18_train.txt")
    parser.add_argument('--dir_test_txt', type=str, default="GLMC-CVPR2023/data/data_txt/iNaturalist18_val.txt")

    # 调用主函数
    main()
# 定义一个验证函数，用于计算模型在验证集上的精度
def validate(model, val_loader, args):
    # 定义两个计数器，分别用于记录top-1和top-5的精度
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # 切换到评估模式
    model.eval()
    # 在评估时不需要计算梯度，加快计算速度
    with torch.no_grad():
        # 遍历验证集数据
        for i, (input, target) in enumerate(val_loader):
            # 将输入数据和目标标签转移到GPU上
            input = input.cuda()
            target = target.cuda()
            # 计算模型输出
            output = model(input, train=False)
            # 计算top-1和top-5的精度
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # 更新计数器
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            # 输出当前batch的精度
            output = 'Testing:  ' + str(i) + ' Prec@1:  ' + str(top1.val) + ' Prec@5:  ' + str(top5.val)
            print(output, end="\r")
        # 最终输出整个验证集的平均精度
        output = (
            '{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(flag='val', top1=top1, top5=top5))
        print(output)

# 根据输入的参数选择并返回相应的模型
def get_model(args):
    # 如果数据集是ImageNet-LT或iNaturalist2018，则选择ResNeXt50模型
    if args.dataset == "ImageNet-LT" or args.dataset == "iNaturelist2018":
        net = Resnet_LT.resnext50_32x4d(num_classes=args.num_classes)
        print("=> creating model '{}'".format('resnext50_32x4d'))
    else:
        # 否则，根据指定的架构选择不同版本的ResNet模型
        if args.arch == 'resnet18':
            net = ResNet_cifar.resnet18(num_class=args.num_classes)
        elif args.arch == 'resnet32':
            net = ResNet_cifar.resnet32(num_class=args.num_classes)
        print("=> creating model '{}'".format(args.arch))
    return net


# 根据输入参数加载并返回相应的数据集
def get_dataset(args):
    # 获取数据集的转换操作（如数据增强等）
    _, transform_val = util.get_transform(args.dataset)

    # 加载CIFAR-10数据集
    if args.dataset == 'cifar10':
        testset = cifar10Imbanlance.Cifar10Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                      transform=transform_val,
                                                      file_path=os.path.join(args.root, 'cifar-10-batches-py/'))
        print("load cifar10")
        return testset

    # 加载CIFAR-100数据集
    if args.dataset == 'cifar100':
        testset = cifar100Imbanlance.Cifar100Imbanlance(imbanlance_rate=args.imbanlance_rate, train=False,
                                                        transform=transform_val,
                                                        file_path=os.path.join(args.root, 'cifar-100-python/'))
        print("load cifar100")
        return testset

