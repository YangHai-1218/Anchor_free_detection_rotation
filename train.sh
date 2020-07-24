CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train.py --num_workers 16 --batch=4
# CUDA_VISIBLE_DEVICES=0 python3 train.py --num_workers 4 --path ../EfficientDet/datasets/coco2017/  --batch 5
