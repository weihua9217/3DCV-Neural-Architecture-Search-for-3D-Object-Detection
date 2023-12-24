




# Neural Architecture Search for 3d Object Detection

# Install container and create image

```
sudo docker run -it --name uni3d2 --gpus all --shm-size 8G -p 20001:8888 -v /home/weihua9217/Uni3D:/data nvcr.io/nvidia/pytorch:22.12-py3 bash 
pip install -r requirement.txt
pip install -r requirement_point.txt
pip install opencv-python==4.5.5.64
```

## Training

- Single GPU
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=1234 --use_env train_3d.py app:configs/cls_kitti.yml

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=1235 --use_env train_3d.py app:configs/cls_kitti_original.yml

- 2 GPU
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=1234 --use_env train_3d.py app:configs/cls_kitti.yml

CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=1234 --use_env train_3d.py app:configs/cls_kitti_original.yml

- Evaluation
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=1235 --use_env test_3d.py app:configs/cls_kitti.yml


