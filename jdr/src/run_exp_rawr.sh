#! /bin/sh
#
# Run JDR and rewiring baselines on the same 9 datasets/split used in EPR/RAwR.

for net in GCN GPRGNN
do
  for dataset in Actor Chameleon Citeseer Cora Cornell PubMed Squirrel Texas Wisconsin
  do
      python train_model.py --dataset $dataset --dataset_source epr_rawr --epr_root ../.. --data_split rawr --net $net --no-wandb_log --random_sort No
      python train_model.py --dataset $dataset --dataset_source epr_rawr --epr_root ../.. --data_split rawr --net $net --no-wandb_log --random_sort No --rewire_default ppr
      python train_model.py --dataset $dataset --dataset_source epr_rawr --epr_root ../.. --data_split rawr --net $net --no-wandb_log --random_sort No --rewire_default fosr
      if [ "$dataset" != "Squirrel" ]; then
        python train_model.py --dataset $dataset --dataset_source epr_rawr --epr_root ../.. --data_split rawr --net $net --no-wandb_log --random_sort No --rewire_default borf
      fi
      python train_model.py --dataset $dataset --dataset_source epr_rawr --epr_root ../.. --data_split rawr --net $net --no-wandb_log --random_sort No --denoise_default $net
  done
done
