"""Structured Single Actor with the unmodified Stable-Baselines3 PPO updater.

Actions in SB3's rollout buffer are the original four-dimensional Gaussian
latents z.  The task environment applies tanh(z), masks retired actuators, and
uses the existing target-free safe mapping.  PPO probabilities and analytic
entropy are explicitly measured in latent space, over active coordinates only.
No inverse tanh of a rounded/saturated executed action is used.

SB3 requires a finite Box.  Its wide +/-1e6 latent Box is only an execution
guard: this policy rejects nonfinite/out-of-Box samples before SB3 sees them,
so SB3's generic action clipping must remain a no-op throughout valid runs.
The actor mean/log-standard-deviation retain Single's existing clamps.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any

from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn import functional as F
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from learning.e45b_joint_network import KEYS, Network


LATENT_BOUND = 1e6
OBS_SHAPES = {
    "maps": (8, 4), "map_mask": (8,), "agents": (2, 6),
    "particles": (2, 64, 3), "weights": (2, 64),
    "history": (16, 14), "history_mask": (16,), "public": (8,),
}


def structured_observation_space() -> spaces.Dict:
    return spaces.Dict({name: spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
                        for name, shape in OBS_SHAPES.items()})


def latent_action_space() -> spaces.Box:
    return spaces.Box(-LATENT_BOUND, LATENT_BOUND, shape=(4,), dtype=np.float32)


class MaskedJointGaussian(Distribution):
    """A Gaussian density on the active-coordinate subspace.

    Retired coordinates are deterministic zero and contribute zero log-density,
    entropy and KL.  The mask is observed state, not a sampled action or a role.
    """

    def __init__(self):
        super().__init__()
        self.mean_actions = self.log_std = self.active = None

    def proba_distribution_net(self, *args, **kwargs):
        raise NotImplementedError("The preserved Single network provides mean and log_std heads")

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor,
                           active: torch.Tensor) -> "MaskedJointGaussian":
        if mean_actions.shape != log_std.shape or mean_actions.shape != active.shape:
            raise ValueError("Gaussian mean/log_std/active mask must have matching (batch,4) shape")
        if mean_actions.ndim != 2 or mean_actions.shape[-1] != 4:
            raise ValueError("Expected joint four-dimensional latent actions")
        self.mean_actions, self.log_std, self.active = mean_actions, log_std, active
        self.distribution = Normal(mean_actions, log_std.exp())
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return (self.distribution.log_prob(actions) * self.active).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return (self.distribution.entropy() * self.active).sum(dim=-1)

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample() * self.active

    def mode(self) -> torch.Tensor:
        return self.mean_actions * self.active

    def actions_from_params(self, mean_actions, log_std, active, deterministic=False):
        return self.proba_distribution(mean_actions, log_std, active).get_actions(deterministic)

    def log_prob_from_params(self, mean_actions, log_std, active):
        actions = self.actions_from_params(mean_actions, log_std, active)
        return actions, self.log_prob(actions)

    def tanh_log_prob_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Audit-only density, with a stable Jacobian and no inverse transform."""
        log_jacobian = 2. * (np.log(2.) - latent - F.softplus(-2. * latent))
        return self.log_prob(latent) - (log_jacobian * self.active).sum(dim=-1)

    def kl_to(self, other: "MaskedJointGaussian") -> torch.Tensor:
        if not torch.equal(self.active, other.active):
            raise ValueError("KL comparison requires the same state-dependent action subspace")
        return (kl_divergence(self.distribution, other.distribution) * self.active).sum(-1)


class StructuredJointPolicy(ActorCriticPolicy):
    """Single's actor architecture with an independent, fresh value network.

    Parameters inherited from ActorCriticPolicy are retained for SB3 save/load
    compatibility; the standard MLP/action heads are not built or used.  PPO's
    collection, GAE, minibatching, clipping, KL stopping and optimizer update
    remain the installed SB3 implementation.
    """

    def __init__(self, observation_space, action_space, lr_schedule,
                 pretrained_path=None, initialization_seed=None, actor_width=None,
                 **kwargs):
        if kwargs.get("use_sde", False):
            raise ValueError("E45-W uses Single's state-dependent Gaussian, not gSDE")
        if kwargs.get("squash_output", False):
            raise ValueError("The environment, not SB3, transforms the stored Gaussian latent")
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("StructuredJointPolicy requires the preserved Dict observations")
        for name, shape in OBS_SHAPES.items():
            if name not in observation_space.spaces or observation_space[name].shape != shape:
                raise ValueError(f"Missing/malformed Single observation block: {name}")
        if (not isinstance(action_space, spaces.Box) or action_space.shape != (4,)
                or not np.isfinite(action_space.low).all() or not np.isfinite(action_space.high).all()
                or np.any(action_space.low > -1e4) or np.any(action_space.high < 1e4)):
            raise ValueError("Use a finite, wide latent Box, preferably latent_action_space(); not a [-1,1] Box")
        self.pretrained_path = str(pretrained_path) if pretrained_path is not None else None
        self.initialization_seed = initialization_seed
        self.actor_width = actor_width
        kwargs.setdefault("normalize_images", False)
        kwargs.setdefault("ortho_init", False)
        kwargs.setdefault("optimizer_kwargs", {"eps": 1e-5})
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        # Both B/C consume exactly the same random construction before C loads
        # the frozen actor.  Their independently initialized critics therefore
        # agree when initialization_seed agrees, while Actor C alone is loaded.
        if self.initialization_seed is None:
            self.actor_network = Network("single", width=self.actor_width)
            self.critic_network = Network("single", width=self.actor_network.width)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.initialization_seed))
                self.actor_network = Network("single", width=self.actor_width)
                self.critic_network = Network("single", width=self.actor_network.width)
        if self.pretrained_path is not None:
            self.load_pretrained_actor(self.pretrained_path)
        # Old leaf values are preserved only as inert actor-state metadata.
        # They never enter PPO values, actor features, loss, or its optimizer.
        for p in self.actor_network.value.parameters(): p.requires_grad_(False)
        for p in self.critic_network.policy.parameters(): p.requires_grad_(False)
        self.optimizer = self.optimizer_class([p for p in self.parameters() if p.requires_grad],
                                              lr=lr_schedule(1), **self.optimizer_kwargs)

    def load_pretrained_actor(self, path):
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if str(data["kind"]) != "single" or int(data["width"]) != self.actor_network.width:
                    raise ValueError("Pretrained checkpoint is not the matching Single architecture")
                state = {k: torch.as_tensor(data[k], dtype=torch.float32)
                         for k in data.files if k not in ("kind", "width")}
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
        if state["trunk.0.weight"].shape[0] != self.actor_network.width:
            raise ValueError("Pretrained checkpoint width differs from the declared random-init architecture")
        self.actor_network.load_state_dict(state, strict=True)

    def actor_state_numpy(self):
        """CPU NumPy export for independent frozen-Actor evaluation workers."""
        return {key: value.detach().cpu().numpy().copy()
                for key, value in self.actor_network.state_dict().items()}

    def load_actor_state(self, state):
        """Load only the Actor; the fresh critic and optimizer are untouched."""
        converted = {key: torch.as_tensor(value, device=self.device)
                     for key, value in state.items()}
        self.actor_network.load_state_dict(converted, strict=True)

    def _observations(self, obs):
        result = {}
        for key in KEYS:
            value = torch.as_tensor(obs[key], dtype=torch.float32, device=self.device)
            if value.ndim == len(OBS_SHAPES[key]): value = value.unsqueeze(0)
            result[key] = value
        return result

    def latent_parameters(self, obs):
        x = self._observations(obs)
        parameters, _ = self.actor_network(x)
        _, mean, log_std = parameters
        active = x["agents"][:, :, 4].repeat_interleave(2, dim=-1)
        return mean[:, 0], log_std[:, 0], active

    def get_distribution(self, obs) -> MaskedJointGaussian:
        mean, log_std, active = self.latent_parameters(obs)
        return MaskedJointGaussian().proba_distribution(mean, log_std, active)

    def _check_latents(self, actions):
        low = torch.as_tensor(self.action_space.low, device=actions.device)
        high = torch.as_tensor(self.action_space.high, device=actions.device)
        if not bool(torch.isfinite(actions).all() and (actions > low).all() and (actions < high).all()):
            raise RuntimeError("Latent guard exceeded; abort before SB3 generic clipping changes sampled actions")

    def forward(self, obs, deterministic=False):
        x = self._observations(obs)
        distribution = self.get_distribution(x)
        actions = distribution.get_actions(deterministic)
        self._check_latents(actions)
        _, values = self.critic_network(x)
        return actions, values.unsqueeze(-1), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        x = self._observations(obs)
        actions = actions.reshape(-1, 4)
        self._check_latents(actions)
        distribution = self.get_distribution(x)
        _, values = self.critic_network(x)
        return values.unsqueeze(-1), distribution.log_prob(actions), distribution.entropy()

    def _predict(self, observation, deterministic=False):
        actions = self.get_distribution(observation).get_actions(deterministic)
        self._check_latents(actions)
        return actions

    def predict_values(self, obs):
        _, values = self.critic_network(self._observations(obs))
        return values.unsqueeze(-1)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        params = super()._get_constructor_parameters()
        # A saved policy includes its actual weights and must not depend on the
        # original pretrained artifact remaining accessible during reloading.
        params.update(pretrained_path=None, initialization_seed=self.initialization_seed,
                      actor_width=self.actor_network.width)
        return params


def provider_audit():
    """Capture the real installed updater and collection sources for provenance."""
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
    files = {"ppo": inspect.getsourcefile(PPO),
             "on_policy_collection": inspect.getsourcefile(OnPolicyAlgorithm)}
    return {
        "provider": "Stable-Baselines3", "version": stable_baselines3.__version__,
        "source_files": {key: {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                         for key, path in files.items()},
        "PPO_train_is_unmodified_provider": True,
        "policy_adapter_scope": "preserved structured Single actor, independent fresh critic, active-coordinate Gaussian latent density",
        "entropy_space": "analytic Gaussian latent entropy; not executed displacement entropy",
        "action_contract": "finite wide Box guard; policy checks before provider action clipping; env tanh then target-free safe map",
        "official_sources": ["https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/ppo/ppo.html",
                             "https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/common/on_policy_algorithm.html"],
    }
