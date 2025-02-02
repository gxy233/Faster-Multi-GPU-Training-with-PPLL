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


##############     patch=16    ############## 
# PPLL
CUDA_VISIBLE_DEVICES=0,1,2,3 python training/trainingFw.py --exp k4_vit_224_p16
CUDA_VISIBLE_DEVICES=0,1 python training/trainingFw.py --exp k2_vit_224_p16

## ddp 数据并行
CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/chengqixu/gxy/temp_name1/train_vitddp.py
CUDA_VISIBLE_DEVICES=0,1 python /home/chengqixu/gxy/temp_name1/train_vitddp.py

## 原网络
python train.py






ssh -p 22000 chengqixu@ink-ellie.usc.edu

srun --cpus-per-task 16 --nodelist ink-ellie --gres=gpu:2080:2 --time 10 python train.py     --dataset cifar10     --model resnetinfopp     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

srun --cpus-per-task 16 --nodelist ink-ellie --gres=gpu:2080:2 --time 10 python /home/chengqixu/gxy/temp_name1/models/resnetInfoPro_gxy_ppll.py



CUDA_VISIBLE_DEVICES=0,1,2,3 python training/trainingFw.py --exp k4_vit_224_p16

# resnet  local learning k=16  
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:1 --time 10 python train.py     --dataset cifar10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:1 --time 10 python train.py     --dataset cifar10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995


# resnet  e2e
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:1 --time 10 python train.py     --dataset cifar10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 1      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:1 --time 10 python train.py     --dataset stl10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 1      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

# resnet pp stage=2 和stage=4  
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:2 --time 10 python train.py     --dataset cifar10     --model resnetpps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:1 --time 10 python train.py     --dataset cifar10     --model resnetpps4     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist dill-sage --gres=gpu:4 --time 10 python train.py     --dataset cifar10     --model resnetpps4     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

python train.py     --dataset cifar10     --model resnetpps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:2 --time 10 python train.py     --dataset stl10     --model resnetpps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

# resnet ppll   new fw
# stage=2 cifar
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:2 --time 180 python train.py     --dataset cifar10     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist allegro-adams --gres=gpu:2 --time 10 python train.py     --dataset cifar10     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist  ink-ellie --gres=gpu:2 --time 2400  python train.py     --dataset cifar10     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

# stage=2 svhn
srun --cpus-per-task 32 --nodelist dill-sage --gres=gpu:2 --time 180 python train.py     --dataset svhn     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist  ink-ellie --gres=gpu:2 --time 2400  python train.py     --dataset svhn     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995

# stage=4 cifar
srun --cpus-per-task 32 --nodelist  ink-ellie --gres=gpu:4 --time 10  python train.py     --dataset cifar10     --model resnetinfopps4     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
srun --cpus-per-task 32 --nodelist  allegro-adams --gres=gpu:4 --time 10  python train.py     --dataset cifar10     --model resnetinfopps4     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995



srun --cpus-per-task 32 --nodelist  allegro-adams --gres=gpu:2 --time 2400  python train.py     --dataset stl10     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995


# resnet ppll stage=2 和stage=4  old fw
srun --cpus-per-task 32 --nodelist  ink-ellie --gres=gpu:2 --time 240  python /home/chengqixu/gxy/temp_name1/models/resnetInfoPro_gxy_ppll.py





srun --cpus-per-task 32 --nodelist ink-ellie --gres=gpu:1 --time 2400  python train.py    --dataset svhn     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 16      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
CUDA_VISIBLE_DEVICES=7,8 python train.py     --dataset cifar10     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
python train.py     --dataset cifar10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995


CUDA_VISIBLE_DEVICES=5,6,7,8 python train.py     --dataset cifar10     --model resnetinfopps4     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995


CUDA_VISIBLE_DEVICES=7,8 python train.py     --dataset cifar10     --model resnetinfopps2     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 8      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995



python train_emaup.py     --dataset cifar10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 4      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995


python train.py     --dataset stl10     --model resnet     --layers 32     --droprate 0.0     --no 0     --cos_lr     --local_module_num 1      --local_loss_mode cross_entropy     --aux_net_widen 1     --aux_net_feature_dim 128     --ixx_1 5     --ixy_1 0.5     --ixx_2 0       --ixy_2 0      --momentum 0.995
