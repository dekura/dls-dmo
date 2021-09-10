# phase='train'
phase='test'

name='ovia1'
dataroot='/research/dept7/glchen/datasets/dlsopc_datasets/viasep/via2/dmo'

# model='pix2pix'
model='pix2pixw'
uppscale=1

epoch=50
half_iter=`expr $epoch / 2`
p_freq=500

##### test phases ####
# test_epoch=80
test_num=1000

if [ $phase = "test" ]; then
    ext='.sh'
    gpu_num=1
else
    ext='.cmd'
    gpu_num=4
fi

# for training test
# ext='.sh'
# gpu_num=1


file_name=$phase'_'$name'_e'$epoch'_dr2mg'$ext
echo $file_name

if [ $phase = "test" ]; then
    python gen_dmo.py \
--name $name \
--file_name $file_name \
--gpu_num $gpu_num \
--model $model \
--test_dmo \
--half_iter $half_iter \
--test_num $test_num \
--dataroot $dataroot \
--uppscale $uppscale

else
    python gen_dmo.py \
--name $name \
--file_name $file_name \
--gpu_num $gpu_num \
--model $model \
--half_iter $half_iter \
--dataroot $dataroot \
--uppscale $uppscale
fi

code $file_name

# --epoch $test_epoch \
