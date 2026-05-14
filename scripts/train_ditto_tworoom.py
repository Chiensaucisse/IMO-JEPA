from ditto.utils import DittoConfig
from ditto.train import main


def main_example():
        
    cfg = DittoConfig(

        method='ditto',  # Options: 'bc_pixels', 'bc_latent', 'ditto'
        wm_type='lewm',
        
        # Checkpoints
        lewm_ckpt='./experiments/checkpoints/lewm_custom.pth',
        dataset_name='./datasets/datasets/tworoom_100.lance',
        
        # Training
        batch_size=8,
        epochs=100,
        num_workers=4,
        
        # Loss weights
        bc_alpha=1.0,
        eta=5e-2,
        mu_l2=0.0,
        
        # Evaluation
        eval_only=False,
        num_evals=10,
        
        # Directories
        save_dir='./checkpoints',
        log_dir='./logs',
        run_name='ditto_tworoom',
    )
    
    main(cfg)


if __name__ == '__main__':
    main_example()
