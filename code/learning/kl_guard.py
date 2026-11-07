"""Accept an Adam step only when cumulative, whole-batch KL stays bounded.

The KL callback must compare against the immutable distribution at the start
of the on-policy batch, never the preceding minibatch. Rejected trials restore
both parameters and optimizer state. This module does not change old trainers.
"""
import copy
import math
import torch


def accepted_actor_step(parameters, optimizer, loss_closure, whole_batch_kl,
                        maximum_kl=.02, gradient_clip=.5, maximum_backtracks=12):
    parameters = tuple(parameters)
    if not parameters or maximum_kl <= 0. or maximum_backtracks < 0:
        raise ValueError('Positive KL limit and nonempty actor parameters required')
    optimized = tuple(p for group in optimizer.param_groups for p in group['params'])
    if len(parameters) != len(optimized) or {id(p) for p in parameters} != {id(p) for p in optimized}:
        raise ValueError('The guarded parameter set must match the entire actor optimizer')
    with torch.no_grad():
        initial_kl = float(whole_batch_kl())
    if not math.isfinite(initial_kl) or initial_kl > maximum_kl:
        raise ValueError('Actor is already outside the immutable batch KL constraint')
    values = [p.detach().clone() for p in parameters]
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    base_rates = [group['lr'] for group in optimizer.param_groups]

    def restore():
        with torch.no_grad():
            for parameter, value in zip(parameters, values):
                parameter.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer))

    try:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_closure()
        if loss.ndim or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite or nonscalar actor loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        gradients = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
        attempts = []
        for trial in range(maximum_backtracks+1):
            restore()
            scale = .5**trial
            for group, rate in zip(optimizer.param_groups, base_rates):
                group['lr'] = rate*scale
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = None if gradient is None else gradient.clone()
            optimizer.step()
            with torch.no_grad():
                candidate_kl = float(whole_batch_kl())
                finite_parameters = all(bool(torch.isfinite(p).all()) for p in parameters)
            accepted = finite_parameters and math.isfinite(candidate_kl) and candidate_kl <= maximum_kl
            attempts.append(dict(scale=scale, cumulative_kl=candidate_kl, accepted=accepted))
            if accepted:
                # Preserve accepted Adam moments/step count; the next proposal
                # starts from the configured rate rather than a hidden schedule.
                for group, rate in zip(optimizer.param_groups, base_rates):
                    group['lr'] = rate
                optimizer.zero_grad(set_to_none=True)
                return dict(accepted=True, initial_kl=initial_kl, final_kl=candidate_kl,
                            loss=float(loss.detach()), gradient_norm=float(norm), attempts=attempts)
        restore()
        for parameter, gradient in zip(parameters, initial_gradients):
            parameter.grad = gradient
        return dict(accepted=False, initial_kl=initial_kl, final_kl=initial_kl,
                    loss=float(loss.detach()), gradient_norm=float(norm), attempts=attempts)
    except BaseException:
        restore()
        for parameter, gradient in zip(parameters, initial_gradients):
            parameter.grad = gradient
        raise
