# remote server
python=/research/dept7/glchen/miniconda3/envs/guojin/bin/python

$python flow.py \
--name layouts05frac48via12 \
--load_size 2048 \
--crop_size 1024 \
--for_dmo \
--for_dls \
--set_byvia \
--gen_via_lists 1,2,3,4 \
--gds_path /research/dept7/glchen/datasets/gds/layouts05frac48via12/gds \
--out_folder /research/dept7/glchen/datasets/dlsopc_datasets/viasep


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