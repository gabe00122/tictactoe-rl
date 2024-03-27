import subprocess
import datetime
import json
import os
from typing import NamedTuple, Optional, Any


class TrainSettings(NamedTuple):
    git_hash: Optional[str]
    timestamp: Optional[str]
    seed: int
    total_steps: int
    jit_steps: int
    env_num: int
    agent_settings: Any
    opponent: str


def save_settings(path: str | os.PathLike[str], settings: TrainSettings) -> None:
    with open(path, "w") as file:
        json.dump(settings, file, indent=2)


def load_settings(path: str | os.PathLike[str], update_stamp: bool = False) -> TrainSettings:
    with open(path, "r") as file:
        settings = json.load(file)

    if update_stamp:
        settings = get_stamp() | settings

    return settings


def get_stamp():
    return {
        'git_hash': get_git_revision_hash(),
        'timestamp': get_timestamp(),
    }


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()
