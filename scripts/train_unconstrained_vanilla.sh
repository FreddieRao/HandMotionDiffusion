rm -r save/unconstrained_vanilla_v9_SP_ANNO_l1000
python -m train.train_mdm --save_dir save/unconstrained_vanilla_v9_SP_ANNO_l1000 --dataset brics-hands-SP-ANNO \
    --batch_size 32  --save_interval 1000 --diffusion_steps 50
