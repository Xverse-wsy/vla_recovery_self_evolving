"""Lossless, chunked NumPy trajectories with explicit action-space provenance.

Each shard stores post-action observations and transitions. initial.npz stores
observation zero. Thus T actions always correspond to T+1 observations, without
duplicating RGB arrays or mixing OSC and joint-position actions. No pickle,
lossy video decoding, model calls or random-number generators are involved.
"""

import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np


SCHEMA_VERSION = 1


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "biufc":
        raise TypeError(f"Expected numeric array, got {result.dtype}")
    return result.copy()


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_npz(path, values):
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temp.replace(path)


def observation_arrays(raw, policy_observation=None, sim_state=None):
    result = {}
    for key, value in raw.items():
        if isinstance(value, (np.ndarray, np.number, bool, int, float)):
            result["raw/" + key] = array(value)
    if not result:
        raise ValueError("No numeric observations to record")
    # Policy RGB is a reversible 180-degree rotation of raw RGB. Store it once.
    if policy_observation is not None and "states" in policy_observation:
        state = array(policy_observation["states"])
        result["policy/state"] = state[0] if state.ndim == 2 and state.shape[0] == 1 else state
    if sim_state is not None:
        result["sim/state"] = array(sim_state)
    return result


class EpisodeRecorder:
    def __init__(self, output, metadata, shard_steps=16):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        if shard_steps < 1:
            raise ValueError("shard_steps must be positive")
        self.shard_steps = shard_steps
        self.metadata = dict(metadata, schema_version=SCHEMA_VERSION,
                             created_at_utc=datetime.now(timezone.utc).isoformat(),
                             format="lossless_npz_shards", complete=False,
                             steps=0, observations=0, shards=[],
                             raw_image_orientation="LIBERO raw; rotate RGB 180 degrees for pi0.5/WCM",
                             action_padding="zero-padded to 8; slice using action_dim",
                             depth="raw simulator depth buffer when present; not metric depth",
                             action_alignment="observation[t], action[t], observation[t+1]",
                             compression="NumPy ZIP/DEFLATE, lossless; allow_pickle=False",
                             crash_semantics="Exceptions flush pending observations; abrupt process/host death can lose the unflushed shard. complete=false is not a finished training episode.")
        self.pending = []
        self.steps = self.queries = 0
        self.started = self.closed = False
        self.keys = self.shapes = None
        self.action_log = (self.output / "transitions.jsonl").open("x")
        self.event_log = (self.output / "events.jsonl").open("x")
        atomic_json(self.output / "manifest.json", self.metadata)

    def begin(self, raw, *, policy_observation=None, sim_state=None, wcm=None):
        if self.started:
            raise RuntimeError("Episode already started")
        values = observation_arrays(raw, policy_observation, sim_state)
        self.keys = set(values)
        self.shapes = {key: (value.shape, value.dtype) for key, value in values.items()}
        atomic_npz(self.output / "initial.npz", values)
        self.started = True
        self.metadata.update(observations=1, observation_fields={
            key: dict(shape=list(v.shape), dtype=str(v.dtype)) for key, v in values.items()},
            initial_wcm=wcm, sim_state_available="sim/state" in values)
        atomic_json(self.output / "manifest.json", self.metadata)

    def record_query(self, actions, *, policy_observation, model_actions=None):
        self.queries += 1
        values = {"actions": array(actions)}
        if model_actions is not None:
            values["model_actions"] = array(model_actions)
        for key in ("main_images", "wrist_images", "states"):
            if key in policy_observation:
                values["policy_input/" + key] = array(policy_observation[key])
        directory = self.output / "queries"
        directory.mkdir(exist_ok=True)
        atomic_npz(directory / f"query_{self.queries:06d}.npz", values)
        return self.queries

    def append(self, action, raw, *, owner, action_space, reward, success,
               terminated, truncated, policy_query=0, chunk_index=-1,
               policy_observation=None, sim_state=None, wcm=None):
        if not self.started or self.closed:
            raise RuntimeError("Recorder must be started and open")
        action = array(action)
        if action.ndim != 1 or len(action) not in (7, 8) or not np.isfinite(action).all():
            raise ValueError("Expected finite 7-D OSC or 8-D joint action")
        values = observation_arrays(raw, policy_observation, sim_state)
        if set(values) != self.keys or any((v.shape, v.dtype) != self.shapes[k] for k, v in values.items()):
            raise ValueError("Observation schema changed within the episode")
        step = self.steps + 1
        padded = np.zeros(8, dtype=np.float64)
        padded[:len(action)] = action
        record = dict(step=step, owner=owner, action_space=action_space,
                      action=action.tolist(), action_dim=len(action), action_dtype=str(action.dtype),
                      reward=float(reward), success=bool(success),
                      terminated=bool(terminated), truncated=bool(truncated),
                      policy_query=int(policy_query), chunk_index=int(chunk_index),
                      monotonic_ns=time.monotonic_ns(),
                      wcm_value=float(wcm["value"]) if wcm else None,
                      wcm_reason=wcm.get("reason") if wcm else None,
                      wcm_available=wcm is not None)
        self.action_log.write(json.dumps(record, allow_nan=False) + "\n")
        self.action_log.flush()
        arrays = dict(values)
        for key, value in record.items():
            if key != "action" and key != "wcm_value":
                arrays["transition/" + key] = np.asarray("" if value is None else value)
        arrays["transition/action"] = padded
        arrays["transition/wcm_value"] = np.asarray(record["wcm_value"] if wcm else np.nan)
        self.pending.append(arrays)
        self.steps = step
        if len(self.pending) >= self.shard_steps:
            self.flush()

    def event(self, name, **data):
        record = dict(event=name, step=self.steps)
        record.update(data)
        self.event_log.write(json.dumps(record, allow_nan=False) + "\n")
        self.event_log.flush()

    def flush(self):
        if not self.pending:
            return
        first = self.steps - len(self.pending) + 1
        name = f"steps_{first:06d}_{self.steps:06d}.npz"
        values = {key: np.stack([row[key] for row in self.pending]) for key in self.pending[0]}
        atomic_npz(self.output / name, values)
        digest = hashlib.sha256((self.output / name).read_bytes()).hexdigest()
        self.metadata["shards"].append(dict(file=name, first_step=first,
                                             last_step=self.steps, sha256=digest))
        self.pending.clear()
        self.metadata.update(steps=self.steps, observations=self.steps + 1, policy_queries=self.queries)
        atomic_json(self.output / "manifest.json", self.metadata)

    def finish(self, outcome, *, success=False, error=None):
        if self.closed:
            return self.summary()
        self.flush()
        self.metadata.update(complete=bool(self.started and error is None), outcome=outcome,
                             success=bool(success), error=error, steps=self.steps,
                             observations=self.steps + int(self.started), policy_queries=self.queries)
        atomic_json(self.output / "manifest.json", self.metadata)
        self.action_log.close()
        self.event_log.close()
        self.closed = True
        return self.summary()

    def summary(self):
        return dict(path=str(self.output), schema_version=SCHEMA_VERSION,
                    complete=self.metadata["complete"], steps=self.steps,
                    observations=self.steps + int(self.started), shards=len(self.metadata["shards"]),
                    format="lossless_npz_shards")


def iter_transitions(directory, *, require_complete=True):
    """Yield aligned numeric training transitions, keeping controller spaces separate."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text())
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported trajectory schema")
    if require_complete and not metadata["complete"]:
        raise ValueError("Incomplete/exceptional episode: inspect before using for training")
    with np.load(directory / "initial.npz", allow_pickle=False) as data:
        previous = {key: data[key].copy() for key in data.files}
    seen = 0
    for shard in metadata["shards"]:
        path = directory / shard["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise ValueError(f"Corrupt trajectory shard: {path}")
        with np.load(path, allow_pickle=False) as data:
            block = {key: data[key] for key in data.files}
            for i, step in enumerate(block["transition/step"]):
                if int(step) != seen + 1:
                    raise ValueError("Nonconsecutive trajectory steps")
                after = {key: block[key][i].copy() for key in previous}
                transition = {key.removeprefix("transition/"): block[key][i].copy()
                              for key in block if key.startswith("transition/")}
                transition["action"] = transition["action"][:int(transition["action_dim"])]
                yield dict(observation=previous, next_observation=after, **transition)
                previous, seen = after, seen + 1
    if seen != metadata["steps"]:
        raise ValueError("Trajectory count does not match manifest")
