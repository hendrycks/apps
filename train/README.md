# Training

## How to train

We use the following command to run and train.  Note the configuration file is called deepspeed_config.json.

    USE_TF=NO deepspeed tune_apps_gpt.py  \
    --save-dir=/path/to/save_dir  \
    --load=/data/sauravkadavath/gcp/megagpu/gpt2_1500_codelm_deepspeed_megagpu__CONT/checkpoint-4350 \
    --apps-train-files /data/hendrycks/apps-beta/train \
    --apps-dataroot /data/hendrycks/apps-beta/train/ \
    --grad-acc-steps=8 \
    --epochs=10 \
    --fp16 \
    --deepspeed deepspeed_config.json \
    --batch-size-per-replica=2
