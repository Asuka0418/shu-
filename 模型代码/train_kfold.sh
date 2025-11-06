# python train_kfold.py \
#   --data_root_dir /home/pt_select \
#   --clinical_csv  /data/wl2p/GABMIL-main/dataset_csv/25new.csv \
#   --label-col NIH评估 --num_classes 4 \
#   --n_splits 5 --epochs 100 --patience 25 \
#   --batch_size 1 --lr 5e-3 --weight_decay 5e-4 \
#   --results_dir /data/wl2p/GABMIL-main/results/2025data_2/fuse1-4 --loss dice_ce --lambda_dice 0.3 --device cuda:0 --monitor F1 --save_att


python train_kfold.py \
  --data_root_dir /data/wl2p/pt_files_all \
  --clinical_csv  /data/wl2p/pjh/dataset_csv/all_1_other_sites.csv \
  --label-col NIH评估 --num_classes 4 \
  --n_splits 5 --epochs 200 --patience 50 \
  --batch_size 1 --lr 5e-4 --weight_decay 2e-3 --max_patches 7000   \
  --results_dir /data/wl2p/pjh/results/all_other_sites_3 --loss dice_ce --lambda_dice 0.75  --device cuda:1 --monitor F1 --save_att
#/home/25data_uni_new_GA/features/pt_files  /home/20+22data_uni/features/pt_files --loss dice_ce --lambda_dice 0.5