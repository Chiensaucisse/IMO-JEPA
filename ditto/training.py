import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import lambda_return, max_cos, encode_expert


def imagine_rollout(
    wm_iface, actor, critic, expert_emb, expert_act, expert_goal, cfg
):
    """Imagine rollouts in the world model and collect trajectories.
    
    Args:
        wm_iface: World model interface
        actor: Policy actor network
        critic: Value critic network
        expert_emb: Expert embeddings (B, T, D)
        expert_act: Expert actions (B, T, A)
        expert_goal: Goal states (B, G)
        cfg: Config object
        
    Returns:
        Dict with rollout data: rewards, values, entropies, actions, etc.
    """
    HS, H = cfg.history_size, cfg.imagine_horizon

    ctx = wm_iface.init_context(expert_emb, expert_act)
    s_t = expert_emb[:, HS - 1]  # (B, D)
    g = expert_goal

    rewards, values, target_values = [], [], []
    entropies, log_pis, pi_actions, mus = [], [], [], []

    for h in range(H):
        v, v_tgt = critic(s_t.detach())
        a_t, log_p_t, entropy_t, mu_t = actor(s_t, g)

        ctx, s_next = wm_iface.step(ctx, a_t)
        s_t = s_next
        s_next_expert = expert_emb[:, HS + h]
        r = max_cos(s_next, s_next_expert)

        rewards.append(r)
        values.append(v)
        target_values.append(v_tgt)
        entropies.append(entropy_t)
        log_pis.append(log_p_t)
        pi_actions.append(a_t)
        mus.append(mu_t)

    return {
        'rewards': rewards,
        'values': torch.stack(values),
        'target_values': torch.stack(target_values),
        'entropy': torch.stack(entropies).mean(),
        'log_pis': torch.stack(log_pis),
        'pi_actions': torch.stack(pi_actions),
        'mus': torch.stack(mus),
    }


def train_step_bc(wm_iface, actor, batch, cfg, opt_a, step, writer=None):
    """Behavioral cloning training step for latent representations.
    
    Args:
        wm_iface: World model interface
        actor: Policy actor network
        batch: Data batch
        cfg: Config object
        opt_a: Actor optimizer
        step: Training step number
        writer: TensorBoard writer (optional)
        
    Returns:
        Dict with loss values
    """
    with torch.no_grad():
        expert_emb, expert_act = encode_expert(wm_iface, batch)

    B, T, D = expert_emb.size()
    goal = batch['goal_state']
    if goal.ndim == 2:
        goal = goal.unsqueeze(1).expand(B, T, -1)

    goal_flat = goal.reshape(B * T, -1).float()
    emb_flat = expert_emb.reshape(B * T, D)
    a_flat, _log_p, entropy_flat, _mu = actor(emb_flat, goal_flat)
    a = a_flat.reshape(B, T, -1)

    bc_loss = F.mse_loss(a, expert_act)
    entropy_loss = -cfg.eta * entropy_flat.mean()
    loss = cfg.bc_alpha * bc_loss + entropy_loss

    opt_a.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
    opt_a.step()

    if writer is not None:
        writer.add_scalar('train/bc_loss', bc_loss.item(), step)
        writer.add_scalar('train/entropy', entropy_flat.mean().item(), step)
    return {'bc': bc_loss.item(), 'entropy': entropy_flat.mean().item()}


def train_step_bc_pixels(actor, batch, cfg, opt_a, step, writer=None):
    """Behavioral cloning training step directly from pixels.
    
    Args:
        actor: Pixel-based actor network
        batch: Data batch
        cfg: Config object
        opt_a: Actor optimizer
        step: Training step number
        writer: TensorBoard writer (optional)
        
    Returns:
        Dict with loss values
    """
    HS = cfg.history_size
    pixels = batch['pixels'][:, HS - 1]  # (B, 3, H, W)
    target = batch['action'][:, HS - 1]  # (B, A)
    goal = batch['goal_state'][:, HS - 1]  # (B, G)

    a, _log_p, entropy, _mu = actor(pixels, goal)
    bc_loss = F.mse_loss(a, target)
    entropy_loss = -cfg.eta * entropy.mean()
    loss = cfg.bc_alpha * bc_loss + entropy_loss

    opt_a.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
    opt_a.step()

    if writer is not None:
        writer.add_scalar('train/bc_pixels_loss', bc_loss.item(), step)
        writer.add_scalar('train/entropy', entropy.mean().item(), step)
    return {'bc': bc_loss.item()}


def train_step(
    wm_iface, actor, critic, batch, cfg, opt_a, opt_c, step, writer=None
):
    """
    Combines:
    - Critic: Value estimation from imagined rollouts
    - Actor: Policy learning with BC auxiliary loss and entropy regularization
    
    Args:
        wm_iface: World model interface
        actor: Policy actor network
        critic: Value critic network
        batch: Data batch
        cfg: Config object
        opt_a: Actor optimizer
        opt_c: Critic optimizer
        step: Training step number
        writer: TensorBoard writer (optional)
        
    Returns:
        Dict with loss values
    """
    with torch.no_grad():
        expert_emb, expert_act = encode_expert(wm_iface, batch)
    goal = batch['goal_state']
    if goal.ndim == 3:
        goal = goal[:, cfg.history_size - 1]  # (B, G)
    expert_goal = goal.float()
    out = imagine_rollout(
        wm_iface, actor, critic, expert_emb, expert_act, expert_goal, cfg
    )

    rewards = out['rewards']  # (H, B)
    values = out['values']  # (H, B), grad to critic
    target_values = out['target_values'].detach()  # (H, B)
    log_pis = out['log_pis']  # (H, B), grad to actor
    pi_actions = out['pi_actions']  # (H, B, A), grad to actor
    mus = out['mus']

    returns = lambda_return(rewards, target_values, cfg.lam, cfg.gamma)

    HS, H = cfg.history_size, cfg.imagine_horizon
    expert_pi_target = expert_act[:, HS - 1 : HS - 1 + H].transpose(
        0, 1
    )  # (H, B, A)
    critic_loss = 0.5 * F.mse_loss(values[:-1], returns[:-1].detach())
    actor_value_loss = -returns[:-1].mean()
    bc_loss = F.mse_loss(pi_actions[:-1], expert_pi_target[:-1])
    entropy_loss = -out['entropy']

    policy_loss = (
        actor_value_loss
        + cfg.bc_alpha * bc_loss
        + cfg.eta * entropy_loss
    )

    opt_c.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.v.parameters(), cfg.grad_clip)
    opt_c.step()

    opt_a.zero_grad(set_to_none=True)
    policy_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
    opt_a.step()

    if step % cfg.target_update_interval == 0:
        critic.sync_target()

    if writer is not None:
        writer.add_scalar('train/critic_loss', critic_loss.item(), step)
        writer.add_scalar('train/actor_loss', policy_loss.item(), step)
        writer.add_scalar('train/bc_loss', bc_loss.item(), step)
        writer.add_scalar('train/entropy', out['entropy'].item(), step)
        writer.add_scalar('train/return_mean', returns.mean().item(), step)
        writer.add_scalar(
            'train/reward_mean', torch.stack(rewards).mean().item(), step
        )

    return {
        'critic': critic_loss.item(),
        'actor': policy_loss.item(),
        'bc': bc_loss.item(),
    }
