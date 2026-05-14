from ditto.wm_training import LeWMConfig, train_lewm


def main():

    
    cfg = LeWMConfig(

        dataset_name='./datasets/datasets/tworoom_10000.lance',
        

        batch_size=16,
        num_workers=4,
        epochs=20,
        lr=5e-5,
        
 
        sigreg_weight=0.1,
        
        save_dir='./experiments/checkpoints',
        log_dir='./experiments/lewm/tb',
        run_name='lewm_tworoom',
    )
    
    
    train_lewm(cfg)


if __name__ == '__main__':
    main()
