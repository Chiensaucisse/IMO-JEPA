
import random
import torch
import numpy as np


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def lambda_return(rewards, values, lam, gamma):
    """Compute lambda returns  from rewards and value estimates.
    
    Args:
        rewards: List or tensor of rewards of length H
        values: List or tensor of value estimates of length H+1
        lam: Lambda parameter for GAE (0 to 1)
        gamma: Discount factor
        
    Returns:
        Tensor of returns with same shape as values[:-1]
    """
    if torch.is_tensor(rewards):
        rewards = list(rewards.unbind(0))
    if torch.is_tensor(values):
        values = list(values.unbind(0))

    R = values[-1]
    out = [R]

    for r_t, v_tp1 in zip(rewards[-2::-1], values[:0:-1]):
        R = r_t + gamma * ((1 - lam) * v_tp1 + lam * R)
        out.insert(0, R)

    return torch.stack(out)


def max_cos(a, b):
    """Compute max cosine similarity: (a·b) / max(||a||, ||b||)^2.
    
    This is a normalized similarity metric that's robust to scale differences.
    Used as reward signal for imitating expert trajectories.
    
    Args:
        a: Tensor of shape (..., D)
        b: Tensor of shape (..., D)
        
    Returns:
        Similarity scores of shape (...)
    """
    na = a.norm(dim=-1, keepdim=True)
    nb = b.norm(dim=-1, keepdim=True)
    max_n = torch.maximum(na, nb).squeeze(-1)
    return (a * b).sum(-1) / (max_n.pow(2) + 1e-8)


def encode_expert(wm_iface, batch):

    info = wm_iface.encode_episode(batch)
    return info


class DittoConfig:
   
    
    def __init__(
        self,
        method: str = 'ditto',
        lewm_ckpt: str = '',
        log_dir: str = './experiments/ditto/tb',
        save_dir: str = './experiments/ditto/checkpoints',
        checkpoint_path: str = None,
        run_name: str = 'ditto',
        
        # Environment & data
        image_size: int = 224,
        history_size: int = 3,
        imagine_horizon: int = 12,
        frameskip: int = 1,
        seq_len: int = 15,
        embed_dim: int = 192,
        
        # Training
        batch_size: int = 64,
        num_workers: int = 0,
        train_split: float = 0.9,
        epochs: int = 50,
        seed: int = 42,
        device: str = 'cuda',
        
        # Architecture
        actor_hidden: int = 512,
        actor_layers: int = 3,
        critic_hidden: int = 512,
        critic_layers: int = 3,
        
        # Optimization
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        
        # Loss weights
        gamma: float = 0.95,
        lam: float = 0.95,
        actor_alpha: float = 1.0,
        bc_alpha: float = 1.0,
        eta: float = 1e-3,
        mu_l2: float = 1e-3,
        target_update_interval: int = 100,
        
        # Evaluation
        eval_only: bool = False,
        num_evals: int = 10,
        dataset_name: str = '',
    ):
        self.method = method
        self.lewm_ckpt = lewm_ckpt
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path
        self.run_name = run_name
        
        self.image_size = image_size
        self.history_size = history_size
        self.imagine_horizon = imagine_horizon
        self.frameskip = frameskip
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.epochs = epochs
        self.seed = seed
        self.device = device
        
        self.actor_hidden = actor_hidden
        self.actor_layers = actor_layers
        self.critic_hidden = critic_hidden
        self.critic_layers = critic_layers
        
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        
        self.gamma = gamma
        self.lam = lam
        self.actor_alpha = actor_alpha
        self.bc_alpha = bc_alpha
        self.eta = eta
        self.mu_l2 = mu_l2
        self.target_update_interval = target_update_interval
        
        self.eval_only = eval_only
        self.num_evals = num_evals
        self.dataset_name = dataset_name
