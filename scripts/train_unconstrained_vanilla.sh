rm -r save/unconstrained_vanilla_v5_SP_sall_step50_bs32_split_test
python -m train.train_mdm --save_dir save/unconstrained_vanilla_v5_SP_sall_step50_bs32_split_test --dataset brics-hands-SP \
    --batch_size 32  --save_interval 1000 --diffusion_steps 50
