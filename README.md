模型训练，
我在github上修改下这个文件

python main.py --dataset cifar10 -a resnet18 --num_classes 10 --imbanlance_rate 0.02 --beta 0.5 --lr 0.01 --epochs 400 -b 64 --momentum 0.9 --weight_decay 5e-3 --resample_weighting 0.0 --label_weighting 1.2 --contrast_weight 1 --rho 0.05


