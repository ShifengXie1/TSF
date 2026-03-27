export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=20
export CKPT_DIR="../ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=1152
export PERIODICITY=24
export ALIGN_CONST=0.4
export NORM_CONST=0.4
export RGB_MA_KERNEL=5

for PRED_LEN in 96 192 336 720; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model VisionTS \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_rgb_decomposition_$PRED_LEN \
    --model_id VisionTSRGB_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 10 \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST \
    --rgb_mode decomposition \
    --rgb_ma_kernel $RGB_MA_KERNEL \
    --rgb_channel_scales 1.0 1.0 1.0
done;
