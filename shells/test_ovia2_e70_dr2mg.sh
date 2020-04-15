/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0 \
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \
--dataroot /research/dept7/glchen/datasets/dlsopc_datasets/viasep/via2/dmo \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--results_dir /research/dept7/glchen/github/dls-dmo/results \
--name ovia2_e70_dr2mg \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--epoch 50 \
--num_test 1000 \
--eval \
--norm batch
