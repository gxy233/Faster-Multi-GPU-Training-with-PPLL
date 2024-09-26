CUDA_VISIBLE_DEVICES=2,3 python training/trainingFw.py --exp k2_vit_224_p4

CUDA_VISIBLE_DEVICES=2,3 python training/trainingFw.py --exp k2_resnet32


CUDA_VISIBLE_DEVICES=4,5,3,7 python training/trainingFw.py --exp k4_vit_224_p4


# in lab
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
srun --gres=gpu:2080:2 --time 60 --export=ALL,CUDA_VISIBLE_DEVICES=2,3 python training/trainingFw.py --exp k2_resnet32
srun --gres=gpu:2 --time 60 --export=ALL,CUDA_VISIBLE_DEVICES=2,3 python training/trainingFw.py --exp k2_resnet32_t
srun --gres=gpu --time 60 python train.py --dataset cifar10 --model resnet --layers 32 --droprate 0.0 --no 0 --cos_lr --local_module_num 16  --local_loss_mode cross_entropy --aux_net_widen 1 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.5 --ixx_2 0   --ixy_2 0  --momentum 0.995


srun --gres=gpu:2:2080 --time 60 --export=ALL,CUDA_VISIBLE_DEVICES=2,3 python training/trainingFw.py --exp k2_vit_224_p4
srun --gres=gpu:4:2080 --time 30 --export=ALL,CUDA_VISIBLE_DEVICES=0,1,2,3 python training/trainingFw.py --exp k4_vit_224_p4