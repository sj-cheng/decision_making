"""Public demonstration supervision for unchanged HARL HASAC actors.

An optimizer pre-step hook adds the auxiliary gradient to the original
actor-loss gradient before its single Adam step. Author source is untouched.
"""
import numpy as np
import torch


def public_teacher(data, device):
    return {key: torch.as_tensor(data[key], dtype=torch.float32, device=device)
            for key in ('obs', 'action', 'active')}


def latent_bc_loss(actor, observations, actions, active):
    # The caller passes only public actor features and teacher intent; no
    # centralized/privileged replay state is accepted by this function.
    net = actor.actor
    x = net.feature_extractor(observations) if net.feature_extractor is not None else observations
    mu = net.mu_layer(net.net(x))
    target = torch.atanh(actions.clamp(-1 + 1e-4, 1 - 1e-4))
    return ((mu - target).square() * active).sum() / (2 * active.sum().clamp_min(1))


def supervised_losses(model, teacher):
    return [latent_bc_loss(actor, teacher['obs'][:, i], teacher['action'][:, i],
                           teacher['active'][:, i])
            for i, actor in enumerate(model.actor)]


def initialize_bc_std(model, std=.15):
    # Both RL arms inherit the same explicit initial exploration scale.
    for actor in model.actor:
        with torch.no_grad():
            actor.actor.log_std_layer.weight.zero_()
            actor.actor.log_std_layer.bias.fill_(float(np.log(std)))


def bc_step(model, teacher):
    losses = []
    for i, actor in enumerate(model.actor):
        actor.turn_on_grad()
        # BC supervises the mean. A constant std is an explicitly registered
        # initialization, not a fitted certainty estimate from six episodes.
        for parameter in actor.actor.log_std_layer.parameters():
            parameter.requires_grad_(False)
        loss = latent_bc_loss(actor, teacher['obs'][:, i], teacher['action'][:, i], teacher['active'][:, i])
        actor.actor_optimizer.zero_grad()
        loss.backward()
        actor.actor_optimizer.step()
        actor.turn_off_grad()
        losses.append(float(loss.detach()))
    return losses


def install_anchor(model, teacher, weight):
    if weight < 0:
        raise ValueError('Negative supervision weight')
    stats = dict(weight=float(weight), calls=0, last_losses=[None, None])
    handles = []
    if weight == 0:
        return handles, stats
    for i, actor in enumerate(model.actor):
        def add_gradient(optimizer, args, kwargs, i=i, actor=actor):
            with torch.enable_grad():
                loss = latent_bc_loss(actor, teacher['obs'][:, i], teacher['action'][:, i], teacher['active'][:, i])
                (weight * loss).backward()
            stats['calls'] += 1
            stats['last_losses'][i] = float(loss.detach())
        handles.append(actor.actor_optimizer.register_step_pre_hook(add_gradient))
    return handles, stats


@torch.no_grad()
def fit_metrics(model, teacher):
    mse = []
    for i, actor in enumerate(model.actor):
        pred = actor.get_actions(teacher['obs'][:, i], stochastic=False)
        mask = teacher['active'][:, i]
        error = ((pred - teacher['action'][:, i]).square() * mask).sum() / (2 * mask.sum().clamp_min(1))
        mse.append(float(error))
    return dict(latent_mse=[float(x) for x in supervised_losses(model, teacher)], action_mse=mse)
