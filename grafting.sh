mkdir -pv checkpoint/baseline_cifar10_mobilenetv2;
CUDA_VISIBLE_DEVICES=1 nohup python grafting.py --s checkpoint/baseline_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 >checkpoint/baseline_cifar10_mobilenetv2/1.log &

mkdir -pv checkpoint/grafting_cifar10_mobilenetv2;
CUDA_VISIBLE_DEVICES=2 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 2 --i 1 >checkpoint/grafting_cifar10_mobilenetv2/1.log &
CUDA_VISIBLE_DEVICES=3 nohup python grafting.py  --s checkpoint/grafting_cifar10_mobilenetv2 --cifar 10  --model MobileNetV2 --num 2 --i 2 >checkpoint/grafting_cifar10_mobilenetv2/2.log &
