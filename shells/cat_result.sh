root_path='/research/dept7/glchen/github/pixel2pixel/results/'
dc='dcupp_naive'
end='_50epoch/'
info='test_latest/'
path=$root_path$dc$end$info
txt=`cd $path; find *.txt`
path=$path$txt
echo $path
echo $txt
cat $path

cat_txt(){
    root_path='/research/dept7/glchen/github/pixel2pixel/results/'
    dc='dcupp_naive'
    index=$1
    end='_50epoch/'
    info='test_latest/'
    path=$root_path$dc$index$end$info
    txt=`cd $path; find *.txt`
    path=$path$txt
    echo $path
    cat $path
}
for ((i=2; i<9; i++)); do
    cat_txt $i
done
# for data in ${array[@]}
# do
    # echo ${data}
# done