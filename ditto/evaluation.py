import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio
from torchvision.transforms import v2 as T

from stable_worldmodel.policy import BasePolicy
import stable_worldmodel as swm


class DittoEnvPolicy(BasePolicy):
    """Policy wrapper for DITTO actor to interact with environments."""
    
    def __init__(
        self,
        wm_iface,
        actor,
        image_size,
        device,
        expert=None,
    ):
        super().__init__()
        self.type = 'ditto'
        self.wm_iface = wm_iface
        self.actor = actor
        self.device = device

        self.tf = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=image_size),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.expert = expert

    @torch.no_grad()
    def get_action(self, infos, **kwargs):
        """Get action from policy given observations.
        
        Args:
            infos: Dict with 'pixels' (uint8) and 'goal_state' keys
            
        Returns:
            Action array (B, A)
        """
        px = infos['pixels']
        if px.ndim == 5:
            px = px[:, -1]

        goal = torch.as_tensor(infos['goal_state'], dtype=torch.float32).to(
            self.device
        )
        if goal.ndim == 1:
            goal = goal.unsqueeze(0)

        frames = torch.stack([self.tf(p) for p in px]).to(self.device)
        if isinstance(self.actor, PixelBCActor):
            goal = torch.as_tensor(
                infos['goal_state'], dtype=torch.float32
            ).to(self.device)
            if goal.ndim == 1:
                goal = goal.unsqueeze(0)
            _, _, _, mu = self.actor(frames, goal)
        else:
            s = self.wm_iface.encode_state(frames.unsqueeze(1))
            _, _, _, mu = self.actor(s, goal)

        if mu.shape[-1] > 2:
            mu = mu[:, :2]

        return mu.cpu().numpy().astype(np.float32)


def _annotate_frame(frame, text_lines, goal=None):
    """Annotate a frame with text and goal marker.
    
    Args:
        frame: RGB frame (H, W, 3) as uint8
        text_lines: List of strings to render
        goal: Goal position (2,) or None
        
    Returns:
        Annotated frame as uint8 array
    """
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('DejaVuSansMono.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    pad = 4
    line_h = 16
    box_h = pad * 2 + line_h * len(text_lines)
    draw.rectangle([(0, 0), (img.width, box_h)], fill=(0, 0, 0))
    if goal is not None:
        g = np.asarray(goal).reshape(-1).astype(np.float32)
        gx, gy = float(g[0]), float(g[1])
        gx = max(0, min(img.width - 1, gx))
        gy = max(0, min(img.height - 1, gy))
        r = 6

        draw.line(
            [(gx - r - 4, gy), (gx + r + 4, gy)], fill=(0, 255, 0), width=1
        )
        draw.line(
            [(gx, gy - r - 4), (gx, gy + r + 4)], fill=(0, 255, 0), width=1
        )

        draw.ellipse(
            [(gx - r, gy - r), (gx + r, gy + r)],
            fill=(0, 255, 0),
            outline=(0, 0, 0),
            width=1,
        )

    for i, line in enumerate(text_lines):
        draw.text(
            (pad, pad + i * line_h), line, fill=(255, 255, 255), font=font
        )
    return np.asarray(img)


@torch.no_grad()
def eval_in_env(
    cfg,
    wm_iface,
    actor,
    video_dir,
    max_steps=500,
):
    """Evaluate policy in environment and record trajectories videos.
    
    Args:
        cfg: Config object
        wm_iface: World model interface
        actor: Policy actor network
        video_dir: Directory to save videos
        max_steps: Maximum steps per episode
    """
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    out_dir = video_dir / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)

    world = swm.World(
        'swm/TwoRoom-v1',
        num_envs=cfg.num_evals,
        image_shape=(224, 224),
        max_episode_steps=max_steps,
    )

    actor.eval()
    policy = DittoEnvPolicy(
        wm_iface=wm_iface,
        actor=actor,
        image_size=cfg.image_size,
        device=cfg.device,
    )

    world.set_policy(policy)

    frames = [[] for _ in range(cfg.num_evals)]
    success = np.zeros(cfg.num_evals, dtype=bool)
    step_counter = np.zeros(cfg.num_evals, dtype=int)

    def on_step(w):
        px = w.infos['pixels']
        if px.ndim == 5:
            px = px[:, -1]
        d2g = w.infos['distance_to_target'].squeeze(-1)
        hit = d2g < 17
        nonlocal_success = success
        nonlocal_success |= hit
        goal_all = w.infos['goal_state']

        for i in range(w.num_envs):
            goal_i = np.asarray(goal_all[i]).reshape(-1)[:2]
            ann = _annotate_frame(
                np.ascontiguousarray(px[i].astype(np.uint8)),
                [
                    f'd2g={float(d2g[i].item()):.1f}',
                ],
                goal_i,
            )
            frames[i].append(ann)
            step_counter[i] += 1

    for env_idx, _ep in world._run_iter(
        episodes=cfg.num_evals,
        on_step=on_step,
        max_steps=max_steps,
        mode='wait',
        seed=cfg.seed,
    ):
        pass

    world.close()
    sr = success.sum()
    print(
        f'success_rate={sr:.3f}  successes={int(success.sum())}/{cfg.num_evals}'
    )

    for i in range(cfg.num_evals):
        if frames[i]:
            imageio.mimsave(
                out_dir / f'rollout_{i}.gif',
                frames[i],
                fps=30,
            )
            print(f'Video {i} saved to {out_dir}')
