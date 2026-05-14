# Offline imitation learning inside JEPA World Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A PyTorch implementation of **DITTO** — a policy learning method that combines world model imagination with behavioral cloning and reinforcement learning for visual control tasks.

## What This Is

- Implementation of the DITTO method ([DeMoss et al., 2023](https://arxiv.org/abs/2302.03086))
- Built upon [`stable-worldmodel`](https://github.com/jbinas/stable-worldmodel) library.

- Demonstrated on the Tworoom navigation environment


## Installation

```bash
# Clone repository
git clone https://github.com/Chiensaucisse/DITTO-JEPA
cd ditto-jepa

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Data 

If you want to collect your own dataset:

```bash
# Collect 100 trajectories 
python scripts/data/collect_custom.py --num-traj 100

# Collect with custom parameters
python scripts/data/collect_custom.py \
  --num-traj 1000 \
  --cache-dir ./datasets \
  --num-envs 20 \
  --action-noise 1.5

```

### 3. Train World Model 

If you need a world model checkpoint, train LeWM:
```bash
python examples/train_lewm_tworoom.py
```

Or use a pre-trained checkpoint.

### 4. Train DITTO Policy

```bash
python examples/train_ditto_tworoom.py
```

### 5. Evaluate Policy

```bash
python examples/eval_pretrained.py \
  --checkpoint ./checkpoints/ditto_tworoom_100_ditto_0.pth \
  --num-evals 10
```

## Training Methods

### Behavioral Cloning (BC)
- **`bc_pixels`**: Learn directly from raw pixels
- **`bc_latent`**: Learn in world model's latent space

### DITTO (Behavioral Cloning + RL)
- **`ditto`**: Actor-Critic learning with:
  - Value loss from imagined rollouts
  - Behavioral cloning auxiliary loss (optional)
  - Entropy regularization for exploration


## Results

To be add

## Code Organization

### Key Files Explained

**`ditto/models.py`**
- Neural network architectures for actor, critic, and encoders

**`ditto/interfaces.py`**
- Abstraction layer for world model interaction
- `LeWMInterface` class for encoding episodes and stepping through the model


**`ditto/training.py`**
- Training loops for different methods (BC, DITTO)
- Imagination rollout with world model
- Generalized Advantage Estimation return computation

**`ditto/evaluation.py`**
- Policy rollout in actual environment
- Video generation with goal visualization
- Success rate metrics

**`ditto/utils.py`**
- Configuration management
- Loss utilities
- Hyperparameter definitions

**`scripts/data/collect_custom.py`**
- Data collection from TwoRoom environment
- Creates expert trajectories in swm data formats
- Configurable noise, action repeats, and trajectory count
- Use: `python scripts/data/collect_custom.py --num-traj 100 --cache-dir /path/data_folder/`



## Extending the Code

### Adding a Custom World Model

Create a new interface by subclassing the base pattern:

```python
# ditto/interfaces.py
class CustomWMInterface:
    def __init__(self, wm, history_size: int):
        self.wm = wm
        self.history_size = history_size
    
    @property
    def embed_dim(self):
        return self.wm.embedding_dim
    
    def encode_episode(self, batch):
        # Your encoding logic
        return embeddings, actions
    
    def encode_state(self, pixels):
        # Encode single frame
        return embedding
    
    def init_context(self, expert_emb, expert_act):
        # Initialize rollout context
        return context
    
    def step(self, ctx, action):
        # Predict next state
        return updated_context, next_embedding
```

### Custom reward signal

Modify training loops in `ditto/training.py` to experiment with different reward signals:

```python
# In imagine_rollout(), replace:
r = max_cos(s_next, s_next_expert)

# With custom reward:
r = custom_reward_function(s_next, s_next_expert, ...)
```


## Citation

The original DITTO paper:

```bibtex
@article{demoss2023ditto,
  title={Ditto: Offline imitation learning with world models},
  author={DeMoss, Benjamin and Duckworth, Paul and Foerster, Jakob and Hawes, Nick and Posner, Ingmar},
  journal={arXiv preprint arXiv:2302.03086},
  year={2023}
}
```



## Acknowledgments

- Built on top of [stable-worldmodel](https://github.com/jbinas/stable-worldmodel)
- Vision backbones from [Hugging Face transformers](https://huggingface.co/models)
