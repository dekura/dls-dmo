/research/dept7/glchen/miniconda3/envs/guojin/bin/python test_mask_green.py \
--gpu_ids 0,1,2,3 \
--checkpoints_dir /research/dept7/glchen/github/dls-dmo/checkpoints \
--dataroot /research/dept7/glchen/datasets/dlsopc_datasets/layouts05frac48via12/dmo \
--netG new_dcupp \
--netD naive6_nl \
--display_winsize 1024 \
--preprocess resize_and_crop \
--dataset_mode aligned \
--load_size 1024 \
--crop_size 1024 \
--results_dir /research/dept7/glchen/github/dls-dmo/results \
--name layouts05frac48via12_e100_dr2mg_1024_1024 \
--model pix2pix \
--input_nc 3 \
--output_nc 3 \
--direction AtoB \
--num_test 512 \
--eval \
--norm batch
