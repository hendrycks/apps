# Training

## How to train

First use `apps_create_split.py` to create the `train.json` and `test.json`. Note the paths specified in `apps_create_split.py` should point to relative paths from training directory or absolute paths.

We use the following command to run and train.  Note the configuration file is called deepspeed_config.json.

    USE_TF=NO deepspeed tune_apps_gpt.py  \
    --save-dir=/path/to/save_dir  \
    --load=/path/to/model \  # Can be used to restart from checkpoint
    --apps-train-files ~/apps/train \
    --apps-dataroot ~/apps/train/ \
    --grad-acc-steps=8 \
    --epochs=10 \
    --fp16 \
    --deepspeed deepspeed_config.json \
    --batch-size-per-replica=2
