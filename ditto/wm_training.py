"""
LeWM (Latent Embedding World Model) training module.
Standalone training code for the world model.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel

import stable_worldmodel as swm
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor
from stable_worldmodel.wm.loss import SIGReg


@dataclass
class BackboneConfig:
    name: str = 'facebook/dinov2-small'


@dataclass
class LeWMConfig:
    
    # Data
    dataset_name: str = ''
    image_size: int = 224
    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 1
    
    # Training
    batch_size: int = 64
    num_workers: int = 4
    train_split: float = 0.9
    epochs: int = 20
    lr: float = 5e-5
    weight_decay: float = 1e-3
    seed: int = 42
    device: str = 'cuda'
    
    # Architecture
    embed_dim: int = 192
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 2048
    predictor_dim_head: int = 64
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0
    
    # Loss
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    
    # Checkpoints
    checkpoint_path: str = None
    log_dir: str = './experiments/lewm/tb'
    save_dir: str = './experiments/checkpoints'
    run_name: str = 'lewm_training'
    
    # Other
    backbone: BackboneConfig = None
    
    def __post_init__(self):
        if self.backbone is None:
            self.backbone = BackboneConfig()


class ImageNetTransform:
    
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


class ColumnNormalizer:
    """Normalize a column in the dataset."""
    
    def __init__(self, dataset, key):
        data = torch.from_numpy(dataset.get_col_data(key)[:]).float()
        data = data[~torch.isnan(data).any(dim=1)]
        self.key = key
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True).clamp(min=1e-6)

    def __call__(self, sample):
        sample[self.key] = (sample[self.key] - self.mean) / self.std
        return sample


class PixelDecoder(nn.Module):
    """Decoder for reconstructing pixels from embeddings."""
    
    def __init__(self, embed_dim, image_size, init_grid=7, init_channels=256):
        super().__init__()
        self.image_size = image_size
        self.init_grid = init_grid
        self.init_channels = init_channels

        self.proj = nn.Linear(embed_dim, init_channels * init_grid * init_grid)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(init_channels, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, emb):
        x = self.proj(emb)
        x = x.view(-1, self.init_channels, self.init_grid, self.init_grid)
        x = self.upsample(x)

        if x.shape[-1] != self.image_size:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
        return x


def get_backbone(backbone_name: str):

    backbone = AutoModel.from_pretrained(backbone_name)
    if hasattr(backbone, 'vision_model'):
        backbone = backbone.vision_model
    
    hidden_size = backbone.config.hidden_size
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    
    return backbone, hidden_size


def build_lewm_model(cfg: LeWMConfig, action_dim: int):

    backbone, hidden_size = get_backbone(cfg.backbone.name)
    embed_dim = cfg.embed_dim

    predictor = Predictor(
        num_frames=cfg.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_size,
        output_dim=hidden_size,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        dim_head=cfg.predictor_dim_head,
        dropout=cfg.predictor_dropout,
        emb_dropout=cfg.predictor_emb_dropout,
    )

    action_encoder = Embedder(
        input_dim=action_dim,
        emb_dim=embed_dim,
    )

    projector = MLP(
        input_dim=hidden_size,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    pred_proj = MLP(
        input_dim=hidden_size,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    model = LeWM(
        encoder=backbone,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )

    # Add decoder for reconstruction
    model.decoder = PixelDecoder(embed_dim=embed_dim, image_size=cfg.image_size)

    return model


def build_dataloaders(cfg: LeWMConfig):

    transform = Compose(
        ImageNetTransform(),
        ColumnNormalizer(swm.data.load_dataset(cfg.dataset_name), 'action'),
    )
    
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        frameskip=cfg.frameskip,
        num_steps=cfg.history_size + cfg.num_preds,
        cache_dir='/home/noe/MVA/stable-worldmodel/datasets',
    )
    dataset.transform = transform
    action_dim = dataset.get_dim('action') * cfg.frameskip

    train_len = int(len(dataset) * cfg.train_split)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    
    return train_loader, val_loader, action_dim


def compute_loss(model, batch, cfg: LeWMConfig, sigreg):
    """Compute LeWM loss.
    
    Args:
        model: LeWM model
        batch: Data batch
        cfg: Config object
        sigreg: SIGReg loss module
        
    Returns:
        Dict with loss values
    """
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    info = model.encode(batch)

    emb = info['emb']  # (B, T, D)
    act_emb = info['act_emb']  # (B, T, D_act)

    ctx_emb = emb[:, :cfg.history_size]
    ctx_act = act_emb[:, :cfg.history_size]

    tgt_emb = emb[:, cfg.num_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)

    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = sigreg(emb.transpose(0, 1))  # expects (T, B, D)

    loss = pred_loss + cfg.sigreg_weight * sigreg_loss
    
    return {
        'loss': loss,
        'pred_loss': pred_loss,
        'sigreg_loss': sigreg_loss,
    }


def train_epoch(model, loader, optimizer, device, cfg: LeWMConfig, sigreg, writer=None, global_step=0):
   
    model.train()
    model.encoder.eval()  # Keep encoder frozen

    loss_buffer = []

    for batch in tqdm(loader):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items() if isinstance(value, torch.Tensor)
        }

        losses = compute_loss(model, batch, cfg, sigreg)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if writer is not None:
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/pred_loss', losses['pred_loss'].item(), global_step)
            writer.add_scalar('train/sigreg_loss', losses['sigreg_loss'].item(), global_step)

        loss_buffer.append(loss.item())
        global_step += 1

    return {
        'loss': sum(loss_buffer) / len(loss_buffer),
    }, global_step


@torch.no_grad()
def eval_epoch(model, loader, device, cfg: LeWMConfig, sigreg, writer=None, epoch=0):
    
    model.eval()
    loss_buffer = []

    for batch in tqdm(loader):
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items() if isinstance(value, torch.Tensor)
        }

        losses = compute_loss(model, batch, cfg, sigreg)
        loss = losses['loss']
        loss_buffer.append(loss.item())

    metrics = {'loss': sum(loss_buffer) / len(loss_buffer)}
    
    if writer is not None:
        writer.add_scalar('val/loss', metrics['loss'], epoch)

    return metrics


def train_lewm(cfg: LeWMConfig = None):
  
    if cfg is None:
        cfg = LeWMConfig()
    
    # Setup
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    
    # Logging
    run_id = f'{cfg.run_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, run_id))
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    # Data
    print("Loading data...")
    train_loader, val_loader, action_dim = build_dataloaders(cfg)
    
    # Model
    print("Building model...")
    model = build_lewm_model(cfg, action_dim).to(device)
    
    if cfg.checkpoint_path:
        print(f"Loading checkpoint from {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
    
    # Loss and optimizer
    sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    

    print(f"Starting training for {cfg.epochs} epochs...")
    best_val = float('inf')
    global_step = 0
    
    for epoch in range(cfg.epochs):
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, device, cfg, sigreg, writer, global_step
        )
        val_metrics = eval_epoch(model, val_loader, device, cfg, sigreg, writer, epoch)
        
        print(
            f'Epoch {epoch + 1} | '
            f'train_loss={train_metrics["loss"]:.6f} | '
            f'val_loss={val_metrics["loss"]:.6f}'
        )
        
        # Save checkpoint
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            save_path = os.path.join(cfg.save_dir, f'lewm_best.pth')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_metrics['loss'],
                    'config': cfg,
                },
                save_path,
            )
            print(f"Saved best model to {save_path}")
    
    writer.close()
    print("Training complete!")


if __name__ == '__main__':
    cfg = LeWMConfig(
        dataset_name='./datasets/datasets/tworoom_10000.lance',
        batch_size=16,
        epochs=20,
        sigreg_weight=0.1,
        frameskip=1,
    )
    train_lewm(cfg)
