set -euxo pipefail

# FP32 TRAINING
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --name jiif_4 --model JIIF --scale 4 --sample_q 30720 --input_size 256 --train_batch 1 --epoch 200 --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2 --exp jiif_4

## NIPQ 8bit training
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --name jiif_4 --model JIIF --scale 4 --sample_q 30720 --input_size 256 --train_batch 1  --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2 --quantize --exp jiif_4_nipq_8bit --checkpoint workspace/jiif_4/checkpoints/jiif_4.pth.tar --eval_interval 10 --last_fp --epoch 180 --ft_epoch 20 --optim adam --target 8

## KETI weight pruning 0.1
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --name jiif_4 --model JIIF --scale 4 --sample_q 30720 --input_size 256 --train_batch 1  --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2 --pruning --exp jiif_4_pruning_0.1 --checkpoint workspace/jiif_4/checkpoints/jiif_4.pth.tar --eval_interval 10 --epoch 80 --prune_ratio 0.1

## KETI weight pruning 0.1 + nipq 8bit
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main.py --name jiif_4 --model JIIF --scale 4 --sample_q 30720 --input_size 256 --train_batch 1  --eval_interval 10 --lr 0.0001 --lr_step 60 --lr_gamma 0.2 --quantize --exp jiif_4_pruning_0.1_nipq_8bit --checkpoint workspace/jiif_4_pruning_0.1/checkpoints/jiif_4.pth.tar --eval_interval 10 --last_fp --epoch 180 --ft_epoch 20 --optim adam --target 8