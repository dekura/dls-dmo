# remote server
python=/research/dept7/glchen/miniconda3/envs/guojin/bin/python

$python flow.py \
--name fcvia1 \
--load_size 2048 \
--crop_size 1024 \
--for_dmo \
--test_ratio 1 \
--max_pervia 10000 \
--set_byvia \
--gen_via_lists 1 \
--gen_only_test \
--gds_path /research/dept7/glchen/github/pix2pixHD/datasets/fc_pre/results/ispd19test \
--out_folder /research/dept7/glchen/datasets/dlsopc_datasets/fcviasep


# one more thing to compress the testbg

# local test


# python=/usr/local/miniconda3/envs/pytorch/bin/python
# $python flow.py \
# --name testforpredataset \
# --load_size 2048 \
# --crop_size 1024 \
# --for_dmo \
# --for_dls \
# --set_byvia \
# --gds_path /Users/dekura/Downloads/testforpredataset/ \
# --out_folder /Users/dekura/Downloads/testforpreout/