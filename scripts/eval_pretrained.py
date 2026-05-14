import argparse
import torch
from pathlib import Path

from ditto.utils import DittoConfig
from ditto.models import GaussianActor, Critic
from ditto.interfaces import LeWMInterface
from ditto.evaluation import eval_in_env
from ditto.train import build_world_model


def evaluate_policy(checkpoint_path, cfg=None):
    """Evaluate a trained DITTO policy.
    
    Args:
        checkpoint_path: Path to checkpoint file
        cfg: DittoConfig object (if None, uses defaults)
    """
    if cfg is None:
        cfg = DittoConfig(
            lewm_ckpt='./experiments/checkpoints/lewm_custom.pth',
            eval_only=True,
            num_evals=10,
            save_dir='./results',
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
   
    wm, wm_iface = build_world_model(cfg, action_dim=2, device=device)
    
   
    actor = GaussianActor(
        embed_dim=wm_iface.embed_dim,
        action_dim=2,
        hidden=cfg.actor_hidden,
        layers=cfg.actor_layers,
        goal_scale=cfg.image_size,
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint['actor'])
    print(f'Loaded checkpoint from {checkpoint_path}')
    
    actor.eval()
    
    # Run evaluation
    eval_in_env(
        cfg,
        wm_iface,
        actor,
        video_dir=cfg.save_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DITTO policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--wm_ckpt', type=str, default='./experiments/checkpoints/lewm_custom.pth',)
    parser.add_argument('--num-evals', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save videos')
    
    args = parser.parse_args()
    
    cfg = DittoConfig(
        lewm_ckpt=args.wm_ckpt,
        eval_only=True,
        num_evals=args.num_evals,
        save_dir=args.save_dir,
    )
    
    evaluate_policy(args.checkpoint, cfg)
