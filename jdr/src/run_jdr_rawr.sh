#! /bin/sh
#
# Run ONLY JDR (no rewiring baselines) on the 9 EPR/RAwR datasets.

for net in GCN GPRGNN
do
  for dataset in Actor Chameleon Citeseer Cora Cornell PubMed Squirrel Texas Wisconsin
  do
      python train_model.py --dataset $dataset --dataset_source epr_rawr --epr_root ../.. --data_split rawr --net $net --no-wandb_log --random_sort No --denoise_default $net
  done
done
