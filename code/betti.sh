mkdir ${2}
python betti.py --source ${1} --img_size 640 --dist_thres 10  --target_upper_size 80 --target_lower_size 20 --betti_thres 0 --num_workers 64 --save_dir ${2}

