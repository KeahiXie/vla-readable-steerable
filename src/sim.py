"""Drive LIBERO-Goal directly: same observation pipeline as lerobot-eval, plus state save and restore."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_libero  # noqa: E402,F401  (applies the task_ids fix)

from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env, make_env_pre_post_processors  # noqa: E402
from lerobot.envs.utils import add_envs_task, preprocess_observation  # noqa: E402

TASK_IDS = {"plate": 8, "stove": 1}
TASK_TEXT = {"plate": "put the bowl on the plate", "stove": "put the bowl on the stove"}
BOWL, PLATE, STOVE = "akita_black_bowl_1_main", "plate_1_main", "flat_stove_1_burner"
TARGETS = {"plate": ["on", "akita_black_bowl_1", "plate_1"], "stove": ["on", "akita_black_bowl_1", "flat_stove_1_cook_region"]}


class Scene:
    """One LIBERO-Goal task. The vector env has a single sub-env, so everything has batch size 1."""

    def __init__(self, name, policy_config):
        self.name = name
        self.config = LiberoEnv(task="libero_goal", task_ids=[TASK_IDS[name]])
        self.vector = make_env(self.config, n_envs=1)["libero_goal"][TASK_IDS[name]]
        self.env = self.vector.envs[0]
        # This step flips images 180 degrees and builds the 8-number state, as lerobot-eval does.
        self.env_pre, self.env_post = make_env_pre_post_processors(self.config, policy_config)

    @property
    def sim(self):
        return self.env._env.sim

    def reset(self, seed):
        observation, _ = self.vector.reset(seed=[seed])
        return observation

    def step(self, action):
        """Step the inner env. LiberoEnv.step resets the scene on success, which would erase the landing."""
        raw, _, _, _ = self.env._env.step(action[0])
        return add_batch(self.env._format_raw_obs(raw)), bool(self.env._env.check_success())

    def landed_on(self):
        """Which targets the bowl is on right now, by the same check the tasks use for success."""
        inner = self.env._env.env
        return [name for name, goal in TARGETS.items() if inner._eval_predicate(goal)]

    def save_state(self):
        return self.env._env.get_sim_state().copy()

    def restore(self, state):
        """Load a saved state and return its observation in vector-env form.

        Re-rendering is not pixel-identical to the observation seen while stepping, so always use
        this returned observation as the snapshot's input.

        The gripper's command lives outside the simulator state and starts open after a reset, which
        drops a held bowl within 4 steps. Every snapshot is taken after the lift, so set it closed.
        """
        raw = self.env._env.regenerate_obs_from_state(state)
        inner = self.env._env.env
        inner.robots[0].gripper.current_action = np.array([-1.0, 1.0])
        inner.timestep, inner.done = 0, False  # restores don't reset the episode clock
        return add_batch(self.env._format_raw_obs(raw))

    def position(self, body):
        return self.sim.data.body_xpos[self.sim.model.body_name2id(body)].copy()

    def gripper_opening(self):
        return float(np.abs(self.sim.data.qpos[self.env._env.robots[0]._ref_gripper_joint_pos_indexes]).sum())

    def batch(self, observation, instruction, preprocessor):
        """Vector-env observation -> model batch, with the instruction chosen explicitly."""
        processed = preprocess_observation(observation)
        processed = add_envs_task(self.vector, processed)
        processed["task"] = [TASK_TEXT[instruction]]
        processed = self.env_pre(processed)
        return preprocessor(processed)

    def close(self):
        self.vector.close()


def add_batch(tree):
    if isinstance(tree, dict):
        return {key: add_batch(value) for key, value in tree.items()}
    return np.asarray(tree)[None]


def env_action(chunk_step, postprocessor, scene):
    """One normalized action from a chunk -> the numpy action the simulator takes."""
    action = postprocessor(chunk_step)
    action = scene.env_post({"action": action})["action"]
    return action.to("cpu").numpy()
