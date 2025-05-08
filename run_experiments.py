import wandb
import os
import yaml
import random

api_key = os.getenv("WANDB_API_KEY")
wandb_entity = os.getenv("WANDB_ENTITY")
wandb.login(key=api_key)
run_setup = os.getenv("RUN_SETUP_PATH")

def read_config(path_to_yml):
    with open(path_to_yml, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

run = wandb.init(
    entity="a1226419-university-of-vienna",
    project="master thesis",
    # open config from a yml path saved in run_setup
    config= read_config(run_setup)
)

# Simulate training.
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    # Log metrics to wandb.
    run.log({"acc": acc, "loss": loss})

# Finish the run and upload any remaining data.
run.finish()