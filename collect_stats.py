import pandas as pd
from loguru import logger


with open("recovery_profiles/recovery_pyinstrument_2nodes.txt", "r") as f:
    lines = f.readlines()


def process_line(line):
    # return time, function name, file path
    values = line.split()
    return values[-3], values[-2], values[-1]


breakdown = {
    "init tensorflow": {"function_name": "<module>", "file_name": "axlearn/common/file_system.py"},
    "init trainer": {"function_name": "<module>", "file_name": "axlearn/common/trainer.py"},
    "set up distribtued": {"function_name": "setup", "file_name": "axlearn/common/launch.py"},
    "get configuration": {
        "function_name": "get_trainer_config",
        "file_name": "axlearn/common/launch_trainer.py",
    },
    "init learner & model": {
        "function_name": "Learner.__call__",
        "file_name": "axlearn/common/module.py",
    },
    "init input": {
        "function_name": "Input.dataset",
        "file_name": "axlearn/common/input_tf_data.py",
    },
    "load checkpoint": {
        "function_name": "SpmdTrainer._prepare_training",
        "file_name": "axlearn/common/trainer.py",
    },
}


def match_line(line, pattern):
    time, function_name, file_name = process_line(line)
    if function_name == pattern["function_name"] and pattern["file_name"] in file_name:
        return time
    else:
        return None


results = dict()
for section, pattern in breakdown.items():
    for line in lines:
        try:
            time = match_line(line, pattern)
        except IndexError:
            continue

        if time is not None:
            if section in results:
                logger.warning(
                    f"duplicate function_name or file_name for section {section}, pattern {pattern}."
                )
            results[section] = time

steps = results.keys()
df = pd.DataFrame.from_dict({"steps": steps, "time": [results[step] for step in steps]})
print(df.to_markdown(index=False))
