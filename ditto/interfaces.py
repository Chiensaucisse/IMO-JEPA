import torch


class LeWMInterface:
    """Interface for LeWM."""
    
    def __init__(self, wm, history_size: int, embed_dim: int):
        self.wm = wm
        self.history_size = history_size
        self._embed_dim = embed_dim

    @property
    def embed_dim(self):
        return self._embed_dim

    def encode_episode(self, batch):
        """Encode a full episode into embeddings and actions.
        
        Args:
            batch: Dict with 'pixels' and 'action' tensors
            
        Returns:
            Tuple of (embeddings, actions)
        """
        info = self.wm.encode(
            {'pixels': batch['pixels'], 'action': batch['action']}
        )
        return info['emb'], batch['action']

    def encode_state(self, pixels):
        """Encode a single state into embedding.
        
        Args:
            pixels: Batch of frames (B, T, C, H, W)
            
        Returns:
            Embeddings (B, D)
        """
        info = self.wm.encode({'pixels': pixels})
        return info['emb'][:, 0]

    def init_context(self, expert_emb, expert_act):
        """Initialize rollout context from expert trajectory.
        
        Args:
            expert_emb: Expert embeddings (B, T, D)
            expert_act: Expert actions (B, T, A)
            
        Returns:
            Context dict for rollout
        """
        HS = self.history_size
        return {
            'emb_ctx': expert_emb[:, :HS].clone(),
            'raw_act_ctx': expert_act[:, :HS].clone(),
        }

    def step(self, ctx, raw_action):
        """Step forward in the world model with an action.
        
        Args:
            ctx: Context dict from init_context()
            raw_action: Action to take (B, A)
            
        Returns:
            Tuple of (updated_context, next_embedding)
        """
        ctx['raw_act_ctx'] = torch.cat(
            [ctx['raw_act_ctx'][:, 1:], raw_action.unsqueeze(1)], dim=1
        )
        act_emb = self.wm.action_encoder(ctx['raw_act_ctx'])
        next_emb = self.wm.predict(ctx['emb_ctx'], act_emb)[:, -1:]
        ctx['emb_ctx'] = torch.cat([ctx['emb_ctx'][:, 1:], next_emb], dim=1)
        return ctx, next_emb.squeeze(1)
