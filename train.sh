# efficientdet-d0  1080 Ti 单卡 efficientdet-d0 freeze backbone max batchsize 30 not freeze backbone max batchsize 18
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --num_workers 8 --batch=18 --lr 0.006
# efficientdet-d1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --num_workers 4 --batch=8 --lr 0.03
#efficientdet-d2
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --num_workers 8 --batch=4 --lr 0.02
#CUDA_VISIBLE_DEVICES=0 python3 train.py --num_workers 4 --batch 30 --lr 0.00025