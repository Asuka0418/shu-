python train_kfold.py \
  --data_root_dir /........... \
  --clinical_csv  /........... \
  --label-col NIH评估 --num_classes 4 \
  --n_splits 1 --epochs 1 --patience 1 \
  --batch_size 1 --lr 1 --weight_decay 1 --max_patches 1  \
  --results_dir /........... --loss dice_ce --lambda_dice 1  --device cuda:0 --monitor F1 --save_att
