# remote server
python=/research/d4/gds/gjchen21/miniconda3/envs/torch1.2/bin/python

$python flow.py \
--name ispdhaoyusep \
--load_size 2048 \
--crop_size 2048 \
--test_ratio 1 \
--for_dls \
--gds_path /research/d4/gds/gjchen21/datasets/datasets/gds/haoyu_ispd/gds \
--out_folder /research/d4/gds/gjchen21/datasets/datasets/dlsopc_datasets/ispdhaoyusep \
--gen_only_test \


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