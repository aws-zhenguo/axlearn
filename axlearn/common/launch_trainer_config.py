# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

import os
from absl import app, flags

from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function
from axlearn.experiments.text.gpt.common import mesh_shape_from_axes

TP_DEGREE = os.environ.get("TP_DEGREE", 4)
TRAIN_BATCH_SIZE = os.environ.get("TRAIN_BATCH_SIZE", 16)
NUM_LAYERS = os.environ.get("NUM_LAYERS", 8)
SIMULATION = os.environ.get("SIMULATION", False)

def update_trainer_config(trainer_config):
    trainer_config.model.decoder.transformer.set(num_layers=NUM_LAYERS)
    trainer_config.set(max_step=101)
    trainer_config.input.batcher.set(global_batch_size=TRAIN_BATCH_SIZE)
    # save_every_n_steps
    trainer_config.checkpointer.save_policy.set(n=100)
    # trainer_config.mesh_axis_names = ('pipeline', 'data', 'expert', 'fsdp', 'seq', 'model')
    trainer_config.mesh_shape = mesh_shape_from_axes(data=1, fsdp=-1, model=4)

    # adjust hidden dimension if doing simulation in single ndoe
    if SIMULATION:
        trainer_config.model.decoder.set(dim=trainer_config.model.decoder.dim // TP_DEGREE)
    return trainer_config


def main(_):
    measurement.initialize(flags.FLAGS)
    launch.setup()
    trainer_config = launch_trainer.get_trainer_config()
    trainer_config = update_trainer_config(trainer_config)
    trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))
    launch_trainer.run_trainer(trainer_config)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
