export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python training/trainingFw.py --exp k4_vit_224_p4
