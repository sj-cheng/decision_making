"""Explicit lifecycle adapters around the frozen HARL HASAC training chain.

The original actor/critic networks, replay calculation, sequential update, and
full run loop are retained.  Two expressions in the frozen ``train`` method
are replaced: actor-temperature loss excludes retired samples; critic's
temperature target counts active action dimensions in each replay row.
The exact source and unified diff can be exported with ``train_source_audit``.
No frozen author file is changed and no extra stochastic forward is added.
"""
from __future__ import annotations

import difflib
import hashlib
import inspect
from pathlib import Path
import sys
import textwrap

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "current/e45u_official_hasac_calibration_20261102/official_source"
sys.path.insert(0, str(FROZEN))

from harl.algorithms.actors.hasac import HASAC
from harl.common.buffers.off_policy_buffer_ep import OffPolicyBufferEP
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
import harl.runners.off_policy_ha_runner as author_module

FROZEN_TRAIN_FILE_SHA256 = "7f206620c3d21f4f3d2b83ba59ed148c07a84bfc0b47a2a9c378b87bd3c3ddd9"


def active_flags(obs, device):
    """Return the explicitly encoded current ego lifecycle flag."""
    # The frozen runner supplies NumPy arrays. Validate their flag on CPU and
    # transfer only that column, rather than re-copying the whole observation
    # or adding a CUDA synchronization to every stochastic forward.
    if isinstance(obs, np.ndarray):
        flag = obs[..., -1:]
        if not np.isin(flag, (0, 1)).all():
            raise ValueError("The final actor feature must be the binary ego active flag")
        return torch.as_tensor(flag, dtype=torch.float32, device=device)
    active = torch.as_tensor(obs, dtype=torch.float32, device=device)[..., -1:]
    if not bool(torch.all((active == 0) | (active == 1))):
        raise ValueError("The final actor feature must be the binary ego active flag")
    return active


class RetirementHASAC(HASAC):
    """Retired actuators output deterministic zero action and zero entropy."""

    def get_actions(self, obs, available_actions=None, stochastic=True):
        action = super().get_actions(obs, available_actions, stochastic)
        return action * active_flags(obs, self.device)

    def get_actions_with_logprobs(self, obs, available_actions=None, stochastic=True):
        action, logp = super().get_actions_with_logprobs(obs, available_actions, stochastic)
        active = active_flags(obs, self.device)
        return action * active, logp * active


def masked_actor_alpha_loss(log_alpha, log_prob_with_target, active):
    """Same alpha objective, normalized over active samples of this actor.

    The ordinary frozen replay sample supplies its original stochastic logp;
    retired rows have neither an entropy target nor an alpha gradient.
    """
    active = active.to(device=log_prob_with_target.device, dtype=log_prob_with_target.dtype)
    count = active.sum()
    if not bool(count > 0):
        raise RuntimeError("No active actor rows: retirement replay guard must resample this minibatch")
    return -(log_alpha * log_prob_with_target * active).sum() / count


def retirement_target_entropy(target_entropy, valid_transitions):
    """Per-row joint entropy target: sum the active agents' action dimensions."""
    target = torch.as_tensor(target_entropy, dtype=valid_transitions.dtype,
                             device=valid_transitions.device).reshape(-1, 1, 1)
    return (target * valid_transitions).sum(dim=0)


def _derive_train():
    file = Path(author_module.__file__).resolve()
    if file != (FROZEN / "harl/runners/off_policy_ha_runner.py").resolve():
        raise RuntimeError("Mixed HARL imports: load the E45-U frozen source before any older HARL adapter")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    if digest != FROZEN_TRAIN_FILE_SHA256:
        raise RuntimeError("Frozen author train source changed")
    original = textwrap.dedent(inspect.getsource(OffPolicyHARunner.train))
    replacements = (
        ("alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()",
         "alpha_loss = masked_actor_alpha_loss(\n"
         "                        self.log_alpha[agent_id], log_prob, sp_valid_transition[agent_id]\n"
         "                    )"),
        ("self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))",
         "self.critic.update_alpha(\n"
         "                    logp_actions, retirement_target_entropy(self.target_entropy, sp_valid_transition)\n"
         "                )"),
    )
    adapted = original
    for old, new in replacements:
        if adapted.count(old) != 1:
            raise RuntimeError("Expected exactly one frozen source expression: " + old)
        adapted = adapted.replace(old, new)
    namespace = dict(vars(author_module))
    namespace.update(masked_actor_alpha_loss=masked_actor_alpha_loss,
                     retirement_target_entropy=retirement_target_entropy)
    exec(compile(adapted, "<e45v_frozen_train_two_lifecycle_expressions>", "exec"), namespace)
    method = namespace["train"]
    method.__doc__ = "Frozen official train with exactly two documented retirement-temperature expressions."
    audit = {
        "author_commit": "b1af98b0dbab72a2eee9d160751cd09aedbb8ce2",
        "author_file_sha256": digest,
        "original_method_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "derived_method_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "replacement_count": len(replacements),
        "diff": "".join(difflib.unified_diff(original.splitlines(True), adapted.splitlines(True),
                                             fromfile="frozen_author/train", tofile="E45V/train")),
        "scope": "Only actor alpha active-row reduction and per-row critic entropy target; all other update code retained",
    }
    return method, audit, original, adapted


retirement_train, _TRAIN_AUDIT, ORIGINAL_TRAIN_SOURCE, RETIREMENT_TRAIN_SOURCE = _derive_train()


def train_source_audit():
    """Return a serializable provenance record, including the exact two-hunk diff."""
    return dict(_TRAIN_AUDIT)


def validate_replay_lifecycle(sample):
    obs, actions, valid = sample[1], sample[2], sample[6]
    if not np.array_equal(obs[..., -1:], valid):
        raise RuntimeError("Replay lifecycle mask differs from pre-action actor active feature")
    if not np.isin(valid, (0, 1)).all():
        raise RuntimeError("Nonbinary replay lifecycle mask")
    if np.any(np.abs(actions[valid[..., 0] == 0]) > 1e-7):
        raise RuntimeError("Retired replay actions must be zero")
    if np.any(valid.sum(axis=0) <= 0):
        raise RuntimeError("Replay contains a pre-action team state with no active actuator")


class RetirementReplayEP(OffPolicyBufferEP):
    """Author EP replay with a finite guard against an all-retired actor batch.

    On ordinary batches, uses the exact author sample unchanged.  A batch
    missing all valid rows for an actor would divide by zero in the frozen
    active-mask policy loss; that exceptional batch is explicitly redrawn.
    """
    max_retirement_resamples = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retirement_resamples = 0

    def sample(self):
        for _ in range(self.max_retirement_resamples):
            sample = super().sample()
            validate_replay_lifecycle(sample)
            if np.all(sample[6].sum(axis=(1, 2)) > 0):
                return sample
            self.retirement_resamples += 1
        raise RuntimeError("Retirement replay cannot supply active rows for every actor after 100 draws")


class RetirementRunnerMixin:
    """Use with OffPolicyHARunner: construction and full run loop stay upstream.

    Register ``RetirementHASAC`` for the HASAC actor and ``RetirementReplayEP``
    for EP storage at the scoped environment integration boundary.  Enforce
    continuous EP HASAC, unshared actors and active-mask policy loss.
    """
    train = retirement_train

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.args["algo"] != "hasac" or self.state_type != "EP" or self.num_agents != 2
                or self.share_param or not self.algo_args["algo"]["use_policy_active_masks"]
                or not self.algo_args["algo"]["auto_alpha"]):
            raise RuntimeError("E45-V requires two unshared EP HASAC actors, active policy masks, and automatic temperature")
        if not all(isinstance(actor, RetirementHASAC) for actor in self.actor):
            raise RuntimeError("Register RetirementHASAC before the full author constructor")
        if not isinstance(self.buffer, RetirementReplayEP):
            raise RuntimeError("Register RetirementReplayEP before the full author constructor")

    def insert(self, data):
        if self.args["algo"] != "hasac" or self.state_type != "EP":
            raise RuntimeError("E45-V retirement adapter supports only EP HASAC")
        obs = np.asarray(data[1])
        active = obs[..., -1:].astype(np.float32)
        if not np.isin(active, (0, 1)).all():
            raise RuntimeError("Current observations do not carry binary lifecycle flags")
        if np.any(active.sum(axis=0) <= 0):
            raise RuntimeError("Attempt to insert a pre-action state with no active pursuer")
        # Warmup draws do not know the current obs.  The environment ignores
        # retired actuators; record the actual zero retired intent in replay.
        adjusted = list(data)
        adjusted[2] = np.asarray(data[2]) * active
        old_idx = self.buffer.idx
        super().insert(tuple(adjusted))
        rows = (old_idx + np.arange(obs.shape[1])) % self.buffer.buffer_size
        for agent in range(self.num_agents):
            self.buffer.valid_transitions[agent][rows] = active[agent]
        # The upstream agent_deaths bookkeeping is not relied on: current
        # active flags above remain correct even if a different env finishes.
        if not hasattr(self, "retirement_insert_audit"):
            self.retirement_insert_audit = {"team_rows": 0, "retired_actor_rows": 0,
                                             "warmup_nonzero_retired_intents_masked": 0}
        self.retirement_insert_audit["team_rows"] += len(rows)
        self.retirement_insert_audit["retired_actor_rows"] += int((active == 0).sum())
        changed = np.any(np.abs(np.asarray(data[2])) > 1e-7, axis=-1) & (active[..., 0] == 0)
        self.retirement_insert_audit["warmup_nonzero_retired_intents_masked"] += int(changed.sum())

