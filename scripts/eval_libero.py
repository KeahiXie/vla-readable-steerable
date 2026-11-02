"""Start lerobot-eval with env.task_ids working, without changing the installed package.

Two problems at the pinned commit:
1. The field is typed Iterable[int], which the command-line parser can't read.
2. LiberoEnv.gym_kwargs hard-codes "task_ids": [1], so the option would be ignored and every
   eval would silently run task 1 of the suite.
"""
from collections.abc import Iterable

import draccus
import yaml

from lerobot.envs.configs import LiberoEnv


def decode_int_list(raw_value, path):
    values = yaml.safe_load(raw_value) if isinstance(raw_value, str) else raw_value
    return [int(value) for value in values]


def gym_kwargs(self):
    if not self.task_ids:
        raise ValueError("Pass --env.task_ids explicitly, for example --env.task_ids=[8]")
    return {"obs_type": self.obs_type, "render_mode": self.render_mode, "task_ids": list(self.task_ids)}


draccus.decode.register(Iterable[int], decode_int_list)
LiberoEnv.gym_kwargs = property(gym_kwargs)

if __name__ == "__main__":
    from lerobot.scripts.lerobot_eval import main

    main()
