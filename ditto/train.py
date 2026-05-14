import os
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import stable_worldmodel as swm

from .models import TanhGaussianActor, Critic, PixelBCActor
from .interfaces import LeWMInterface
from .training import train_step_bc, train_step_bc_pixels, train_step
from .evaluation import eval_in_env
from .utils import set_seed, DittoConfig, encode_expert


class ImageNetTransform:
    """ImageNet normalization transform for pixel data."""
    
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __call__(self, sample):
        pixels = sample['pixels'].float() / 255.0
        sample['pixels'] = (pixels - self.MEAN) / self.STD
        return sample


class Compose:
    
    def __init__(self, *transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_dataloaders(cfg: DittoConfig):
    """Build train and validation dataloaders.
    
    Args:
        cfg: Config object
        
    Returns:
        Tuple of (train_loader, val_loader, action_dim)
    """
    transform = Compose(
        ImageNetTransform(),
    )
    
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        format='lance',
        frameskip=cfg.frameskip,
        num_steps=cfg.history_size + cfg.imagine_horizon,
        transform=transform,
        keys_to_load=['pixels', 'action', 'goal_state'],
        cache_dir='/home/noe/MVA/stable-worldmodel/datasets',
    )

    # N = 500_000_000
    action_dim = dataset.get_dim('action') * cfg.frameskip
    # indices = list(range(min(N, len(dataset))))
    # dataset = Subset(dataset, indices)

    train_len = int(len(dataset) * cfg.train_split)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    return (
        DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        ),
        action_dim,
    )


def build_world_model(cfg: DittoConfig, action_dim: int, device):
    """Load and initialize LeWM world model.
    
    Args:
        cfg: Config object with lewm_ckpt path
        action_dim: Dimension of actions
        device: Torch device
        
    Returns:
        Tuple of (wm, wm_iface)
    """
    from .wm_training import build_lewm_model, LeWMConfig, BackboneConfig
    
    lewm_cfg = LeWMConfig(
        history_size=cfg.history_size,
        num_preds=1,
        embed_dim=cfg.embed_dim,
        backbone=BackboneConfig(),
    )
    
    wm = build_lewm_model(lewm_cfg, action_dim=action_dim).to(device).eval()
    
    if cfg.lewm_ckpt:
        checkpoint = torch.load(cfg.lewm_ckpt, map_location='cpu')
        if 'model' in checkpoint:
            wm.load_state_dict(checkpoint['model'])
        else:
            wm.load_state_dict(checkpoint)
        print(f'Loaded LeWM checkpoint from {cfg.lewm_ckpt}')
    
    wm.requires_grad_(False)
    wm_iface = LeWMInterface(wm, cfg.history_size, cfg.embed_dim)
    
    return wm, wm_iface


def main(cfg: DittoConfig = None):

    if cfg is None:
        cfg = DittoConfig()
    
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, action_dim = build_dataloaders(cfg)
    
    wm = None
    wm_iface = None
    actor = None
    
    if cfg.method == 'bc_pixels':
        wm_iface = None
        actor = PixelBCActor(
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
            action_dim=action_dim,
            hidden=cfg.actor_hidden,
            layers=cfg.actor_layers,
        ).to(device)

    else:
        wm, wm_iface = build_world_model(cfg, action_dim, device)
        
        actor = TanhGaussianActor(
            embed_dim=wm_iface.embed_dim,
            action_dim=action_dim,
            hidden=cfg.actor_hidden,
            layers=cfg.actor_layers,
            goal_scale=cfg.image_size,
        ).to(device)

    critic = None
    opt_c = None
    if cfg.method == 'ditto':
        critic = Critic(
            cfg.embed_dim, cfg.critic_hidden, cfg.critic_layers
        ).to(device)
        opt_c = torch.optim.Adam(
            critic.parameters(),
            lr=cfg.lr_critic,
            weight_decay=cfg.weight_decay,
        )

    if cfg.checkpoint_path:
        checkpoint = torch.load(cfg.checkpoint_path, map_location='cpu')
        actor.load_state_dict(checkpoint['actor'])
        print(
            f'Loaded checkpoint from {cfg.checkpoint_path} at epoch {checkpoint["epoch"]}'
        )
        if critic is not None:
            critic.load_state_dict(checkpoint['critic'])

    opt_a = torch.optim.Adam(
        actor.parameters(), lr=cfg.lr_actor, weight_decay=cfg.weight_decay
    )

    if not cfg.eval_only:
        run_id = f'{cfg.run_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, run_id))
        os.makedirs(cfg.save_dir, exist_ok=True)

        step = 0
        for epoch in range(cfg.epochs):
            actor.train()
            if critic is not None:
                critic.train()
            
            for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
                batch = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                }

                batch['action'] = torch.nan_to_num(batch['action'], 0.0)

                if cfg.method == 'bc_pixels':
                    train_step_bc_pixels(
                        actor, batch, cfg, opt_a, step, writer
                    )

                elif cfg.method == 'bc_latent':
                    train_step_bc(
                        wm_iface, actor, batch, cfg, opt_a, step, writer
                    )
                elif cfg.method == 'ditto':
                    train_step(
                        wm_iface,
                        actor,
                        critic,
                        batch,
                        cfg,
                        opt_a,
                        opt_c,
                        step,
                        writer,
                    )

                step += 1

            actor.eval()
            if critic is not None:
                critic.eval()

            if epoch % 10 == 0:
                torch.save(
                    {
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict()
                        if critic is not None
                        else None,
                        'epoch': epoch,
                    },
                    os.path.join(
                        cfg.save_dir,
                        f'{cfg.method}_{epoch}.pth',
                    ),
                )
                print(
                    f'Model saved to {os.path.join(cfg.save_dir, f"{cfg.method}_{epoch}.pth")}'
                )

        writer.close()

    eval_in_env(
        cfg,
        wm_iface,
        actor,
        video_dir=os.path.join(cfg.save_dir, 'videos'),
    )


if __name__ == '__main__':
    main()
