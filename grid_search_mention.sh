export PYTHONPATH="$PWD"

mv experiments_mention.conf experiments.conf


export TPU_NAME=tensorflow-tpu
export GCP_PROJECT=xiaoyli-20-04-274510

#for lr in 1e-5 2e-5 3e-5 4e-5 5e-5 6e-5 7e-5 8e-5 9e-5 1e-4 2e-4 3e-4 4e-4 5e-4 6e-4;do
for lr in 1e-5;do
  config="spanbert_base_mention_lr${lr}";
  echo $config
  python3 train.py $config
done
