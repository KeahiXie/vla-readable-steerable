"""Fetch recorded demo frames and turn them into exact model inputs. Plumbing, no research logic."""
import io

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def _dataset_file(pins, filename):
    return hf_hub_download(
        pins["dataset"], filename, repo_type="dataset", revision=pins["dataset_revision"]
    )


def _to_chw_float(cell):
    """Stored PNG bytes -> CHW float tensor in [0, 1], the form the policy expects."""
    array = np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def _index_range(fs, pins, name):
    """Read the min and max of the `index` column from a file's footer, without downloading it."""
    path = f"datasets/{pins['dataset']}@{pins['dataset_revision']}/{name}"
    with fs.open(path) as handle:
        meta = pq.ParquetFile(handle).metadata
        column = meta.schema.names.index("index")
        stats = [meta.row_group(i).column(column).statistics for i in range(meta.num_row_groups)]
    return min(s.min for s in stats), max(s.max for s in stats)


def _find_data_file(fs, pins, names, global_index):
    """Binary search by each file's real index range.

    meta/episodes data/file_index is wrong at this dataset revision (it puts episode 379 in
    file-025, which actually holds episodes 61-63), so it isn't used.
    """
    low, high = 0, len(names) - 1
    while low <= high:
        middle = (low + high) // 2
        first, last = _index_range(fs, pins, names[middle])
        if global_index < first:
            high = middle - 1
        elif global_index > last:
            low = middle + 1
        else:
            return names[middle]
    raise LookupError(f"no data file holds index {global_index}")


def load_frames(pins, snapshots):
    """One frame per episode, from the first n_episodes of the task in episode order."""
    episodes = pq.read_table(_dataset_file(pins, "meta/episodes/chunk-000/file-000.parquet")).to_pandas()
    tasks = pq.read_table(_dataset_file(pins, "meta/tasks.parquet")).to_pandas()
    task_index = int(tasks.loc[snapshots["task"], "task_index"])
    chosen = episodes[episodes["tasks"].apply(lambda tasks: snapshots["task"] in list(tasks))]
    chosen = chosen.sort_values("episode_index").head(snapshots["n_episodes"])

    fs = HfFileSystem()
    root = f"datasets/{pins['dataset']}@{pins['dataset_revision']}/"
    names = sorted(
        path.split("@", 1)[1].split("/", 1)[1] for path in fs.glob(root + "data/*/*.parquet")
    )

    frames = []
    for _, episode in chosen.iterrows():
        offset = int(snapshots["frame_fraction"] * int(episode["length"]))
        global_index = int(episode["dataset_from_index"]) + offset
        name = _find_data_file(fs, pins, names, global_index)
        table = pq.read_table(_dataset_file(pins, name), filters=[("index", "=", global_index)]).to_pylist()
        assert len(table) == 1, f"expected one row for index {global_index}, got {len(table)}"
        row = table[0]
        assert row["episode_index"] == int(episode["episode_index"]), (global_index, row["episode_index"])
        assert row["frame_index"] == offset, (global_index, row["frame_index"], offset)
        assert row["task_index"] == task_index, (global_index, row["task_index"], task_index)
        frames.append(
            {
                "episode_index": int(episode["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "observation.images.image": _to_chw_float(row[IMAGE_KEYS[0]]),
                "observation.images.image2": _to_chw_float(row[IMAGE_KEYS[1]]),
                "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
                "recorded_action": row["action"],
            }
        )
    return frames


def make_batch(frame, instruction, preprocessor):
    """Same image and state, chosen instruction, through the checkpoint's own normalizer."""
    raw = {key: frame[key].clone() for key in (*IMAGE_KEYS, "observation.state")}
    raw["task"] = instruction
    return preprocessor(raw)


def prompt_inputs(policy, batch):
    """The exact tokens and pixels generate_actions would feed the model, with no action text."""
    model = policy.model
    inputs, _ = model.create_input_tokens(
        states=batch["observation.state"],
        images=model.prepare_images(batch),
        lang_text=batch["task"],
        actions=None,
    )
    return inputs
