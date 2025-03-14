# Copyright © 2023 Apple Inc.

"""Main function for launching the trainer."""

import os
import time
from absl import app, flags
import threading
import re
from queue import Queue
from typing import Optional, Dict, Any

from axlearn.common import launch, launch_trainer, measurement
from axlearn.common.config import config_for_function
from axlearn.common import measurement
from axlearn.common.checkpointer_orbax import OrbaxCheckpointer
from axlearn.common.checkpointer import parse_step_from_dir
from orbax.checkpoint.logging import abstract_logger
from axlearn.common.config import REQUIRED, Required, config_class, maybe_set_config
import jax
import mlflow
from datetime import datetime
import hashlib

from absl import logging

from axlearn.experiments.text.gpt.common import mesh_shape_from_axes

NUM_NODES = int(os.environ.get("NUM_NODES", 2))
TP_DEGREE = int(os.environ.get("TP_DEGREE", 4))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", NUM_NODES * 64 // TP_DEGREE))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 8))
SAVE_EVERY_N_STEPS = int(os.environ.get("SAVE_EVERY_N_STEPS", 10))
CHECKPOINTER_TYPE = os.environ.get("CHECKPOINTER_TYPE")
MAX_STEPS = int(os.environ.get("MAX_STEPS", 200))
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "scale_out_training_metrics")

print("NUM_NODES", NUM_NODES)
print("TP_DEGREE", TP_DEGREE)
print("NUM_LAYERS", NUM_LAYERS)
print("TRAIN_BATCH_SIZE", TRAIN_BATCH_SIZE)

PROCESS_INDEX = os.environ["NEURON_PJRT_PROCESS_INDEX"]


def update_trainer_config(trainer_config):
    # config checkpointer
    existing_save_policy = trainer_config.checkpointer.save_policy

    if CHECKPOINTER_TYPE == "OrbaxCheckpointer":
        trainer_config.checkpointer = MyOrbaxCheckpointer.default_config()

    trainer_config.checkpointer.keep_last_n = 100
    # orbax checkpointer does not have keep_every_n_steps option
    # trainer_config.checkpointer.set(keep_every_n_steps=100000)
    trainer_config.checkpointer.save_policy = existing_save_policy
    trainer_config.checkpointer.save_policy.set(n=SAVE_EVERY_N_STEPS)

    # config trainer
    trainer_config.model.decoder.transformer.set(num_layers=NUM_LAYERS)
    trainer_config.set(max_step=MAX_STEPS)
    trainer_config.input.input_dispatcher.set(global_logical_batch_size=TRAIN_BATCH_SIZE)

    # trainer_config.mesh_shape = mesh_shape_from_axes(data=1, fsdp=-1, model=4)

    return trainer_config


class MLFlowReporter:
    _mlflow_initialized = False
    _mlflow_run = None
    _lock = threading.Lock()  # Thread-safe initialization lock

    def __init__(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._report_metrics, daemon=True)
        self.thread.start()

    @classmethod
    def _initialize_mlflow(cls):
        """Initialize MLflow once for all instances."""
        with cls._lock:
            if not cls._mlflow_initialized:
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
                mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

                # Generate run name
                run_name = cls._generate_run_name()

                # Start run with generated name and store it at class level
                cls._mlflow_run = mlflow.start_run(run_name=run_name)

                cls._mlflow_initialized = True

    @classmethod
    def _generate_run_name(cls):
        """Generate MLflow run name based on environment."""
        if os.environ.get("POD_UID"):
            # Kubernetes environment
            pod_uid = os.environ.get("POD_UID", "")
            hostname = os.environ.get("PMIX_HOSTNAME", "")

            # Generate 4-letter hash of POD_UID
            pod_hash = hashlib.md5(pod_uid.encode()).hexdigest()[:4]
            return f"{pod_hash}-{hostname}"

        elif os.environ.get("SLURM_JOBID"):
            # SLURM environment
            job_id = os.environ.get("SLURM_JOBID", "")
            job_name = os.environ.get("SLURM_JOB_NAME", "")
            return f"{job_id}_{job_name}_{PROCESS_INDEX}"

        else:
            # Default case
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_environment_tags(self):
        """Collect relevant environment information as tags."""
        tags = {}

        if os.environ.get("POD_UID"):
            # Kubernetes environment tags
            tags.update(
                {
                    "environment": "kubernetes",
                    **os.environ,
                    # "pod_uid": os.environ.get("POD_UID", ""),
                    # "hostname": os.environ.get("HOSTNAME", ""),
                }
            )
            # remove unneeded tags
            remove_tags = ["LS_COLORS", "LESSCLOSE", "TERM", "LESSOPEN", "SHLVL", "LC_CTYPE"]
            for tag in remove_tags:
                if tag in tags:
                    tags.pop(tag)

        elif os.environ.get("SLURM_JOBID"):
            # SLURM environment tags
            tags.update(
                {
                    "environment": "slurm",
                    "slurm_job_id": os.environ.get("SLURM_JOBID", ""),
                    "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
                    "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
                    "process_index": os.environ.get("NEURON_PJRT_PROCESS_INDEX", ""),
                }
            )

        return tags

    def _report_metrics(self):
        self._initialize_mlflow()

        # Set environment tags for this instance
        env_tags = self._get_environment_tags()
        for key, value in env_tags.items():
            mlflow.set_tag(key, value)

        while True:
            item = self.queue.get()
            if item is None:  # Shutdown signal
                break

            item_type, data = item

            try:
                if item_type == "metric":
                    metric_name, value, step, timestamp = data
                    mlflow.log_metric(metric_name, value, step=step, timestamp=timestamp)
                elif item_type == "param":
                    param_name, value = data
                    mlflow.log_param(param_name, value)
                elif item_type == "tag":
                    tag_name, value = data
                    mlflow.set_tag(tag_name, value)
            except Exception as e:
                print(f"Error logging {item_type} {data}: {e}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        if step is None:
            step = None
        self.queue.put(("metric", (name, value, step, int(datetime.now().timestamp()))))

    def log_param(self, name: str, value: Any):
        if value is not None:
            self.queue.put(("param", (name, value)))

    def log_tag(self, name: str, value: str):
        if value is not None:
            self.queue.put(("tag", (name, value)))

    def log_config(self, config: Dict[str, Any], prefix: str = ""):
        """Recursively log configuration as parameters."""

        def _sanitize_key(key: str) -> str:
            # Replace invalid characters with underscores
            sanitized = re.sub(r"[^\w\-\. :/]", "_", key)
            # Replace multiple consecutive underscores with a single one
            sanitized = re.sub(r"_+", "_", sanitized)
            return sanitized

        def _flatten_config(cfg: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
            items = []
            for k, v in cfg.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(_flatten_config(v, new_key).items())
                else:
                    items.append((_sanitize_key(new_key), v))
            return dict(items)

        flattened = _flatten_config(config)
        for key, value in flattened.items():
            param_name = f"{prefix}{key}" if prefix else key
            # Convert complex objects to string representation
            if not isinstance(value, (str, int, float, bool)):
                value = str(value)
            self.log_param(param_name, value)

    def close(self):
        self.queue.put(None)  # Shutdown signal
        self.thread.join()
        mlflow.end_run()


@measurement.register_recorder("scale_out_recorder")
class ScaleOutRecorder(measurement.Recorder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reporter = MLFlowReporter()
        self.last_step_time = None
        self.last_step_number = None
        self.job_start_time = None
        self.allow_list = [
            r"/jax/checkpoint.*",
            r"/jax/orbax.*",
            r"/axlearn/common.*",
            # Add other patterns here
        ]

    def record(self, event: measurement.Event, *args, **kwargs):
        current_time = datetime.now()

        # Record timestamp for every event
        if args and event == measurement.Event.START_STEP:
            step = args[0]
            self.reporter.log_metric(f"{event.value}_time", current_time.timestamp(), step=step)
        else:
            self.reporter.log_metric(f"{event.value}_time", current_time.timestamp())

        if event == measurement.Event.START_JOB:
            self.job_start_time = current_time

        elif event == measurement.Event.END_JOB:
            if self.job_start_time:
                job_duration = (current_time - self.job_start_time).total_seconds()
                self.reporter.log_metric("job_duration", job_duration)

            # Record duration for the last step if exists
            if self.last_step_time and self.last_step_number:
                last_step_duration = (current_time - self.last_step_time).total_seconds()
                self.reporter.log_metric(
                    "step_duration", last_step_duration, step=self.last_step_number
                )

        elif event == measurement.Event.START_STEP:
            current_step = args[0] if args else None
            current_time = datetime.now()

            # Calculate duration between steps
            if self.last_step_time and self.last_step_number:
                step_duration = (current_time - self.last_step_time).total_seconds()
                self.reporter.log_metric("step_duration", step_duration, step=self.last_step_number)

            # Update last step info
            self.last_step_time = current_time
            self.last_step_number = current_step

    def start_monitoring(self, *args, **kwargs):
        def event_duration_callback(event: str, duration_secs: float, **kwargs):
            logging.info(f"[{PROCESS_INDEX}] {event}: val: {duration_secs}")

            if not any(re.match(pattern, event) for pattern in self.allow_list):
                return

            metric_name = event.lstrip("/").replace("/", "_")
            self.reporter.log_metric(f"{metric_name}", duration_secs, step=kwargs.get("step"))

        jax.monitoring.register_event_duration_secs_listener(event_duration_callback)

    def __del__(self):
        self.reporter.close()


class MyOrbaxCheckpointer(OrbaxCheckpointer):
    def __init__(self, cfg, *, parent):
        super().__init__(cfg, parent=parent)
        self._manager._logger = StandardLogger()

    def restore(self, *, step=None, state):
        # There is an existing bug in Axlearn which fails here during job_start
        # when step=None is passed to try restore.
        # Orbax does not expect step to be None and fails the job throwing an exception.

        # The right way to handle this is to try to get the latest checkpoint if step=None and
        # if no latest checkpoing is present return immediately without calling orbax layer.
        if step == None:
            try:
                ckpt_dir = self.latest_checkpoint_path(self.config.dir)
                step = parse_step_from_dir(ckpt_dir)
            except IndexError:
                logging.info("Could not find any completed checkpoints under %s", self.config.dir)
                return step, state

        return super().restore(step=step, state=state)


class StandardLogger(abstract_logger.AbstractLogger):
    def __init__(self):
        super().__init__()
        self.recorder = measurement.global_recorder

    def log_entry(self, msg, *args, **kwargs):
        logging.info(f"[{PROCESS_INDEX}] {msg}")

        if isinstance(msg, dict):
            for key, value in msg.items():
                if isinstance(value, (int, float)):
                    self.recorder.reporter.log_metric(
                        f"checkpoint_{key}", value, step=msg.get("step", 0)
                    )


def main(_):
    measurement.global_recorder = ScaleOutRecorder(
        measurement.Recorder.default_config().set(name="ScaleOutRecorder")
    )

    setup_start_time = time.time()
    launch.setup()
    setup_end_time = time.time()

    trainer_config = launch_trainer.get_trainer_config()
    trainer_config.set(recorder=config_for_function(lambda: measurement.global_recorder))

    trainer_config = update_trainer_config(trainer_config)

    # Log model configuration to MLflow
    try:
        config_dict = trainer_config.to_dict()
        measurement.global_recorder.reporter.log_config(config_dict, prefix="model_config.")
    except Exception as e:
        print(f"Error logging model configuration: {e}")

    measurement.start_monitoring()
    jax.monitoring.record_event_duration_secs(
        "/axlearn/common/launch/setup_duration_sec", setup_end_time - setup_start_time
    )
    trainer_start_time = time.time()
    launch_trainer.run_trainer(trainer_config)
    trainer_end_time = time.time()
    jax.monitoring.record_event_duration_secs(
        "/axlearn/common/launch/run_trainer_duration_sec", trainer_end_time - trainer_start_time
    )

    # wait for mlflow to finish publishing metrics
    time.sleep(180)


if __name__ == "__main__":
    measurement.define_flags()
    app.run(main)
