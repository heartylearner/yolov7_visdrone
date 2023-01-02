# yolov7_visdrone
DIP Final Project: Fine-tune [yolov7](https://github.com/WongKinYiu/yolov7) pretrained model with Visdrone-Dataset.
## Reference:
- https://github.com/WongKinYiu/yolov7
- https://github.com/ultralytics/yolov5
- https://github.com/VisDrone/VisDrone-Dataset

## settings
```shell
pip install -r requirements.txt
```
## devices
You should have at least 10GB gpu ram available

## Fine-tuned model 
[download link](https://drive.google.com/file/d/17hIqMa_aFY-41jkw9XbtJOd5bXCvjyod/view?usp=sharing)

## Testing(fine-tuning method)
```shell
python detect.py --weights best_vis_50.pt --conf 0.25 --img-size 640 --source output_dir/
```

## Testing(betti_number filtering method)
```shell
CUDA_VISIBLE_DEVICES=0 bash betti.sh testing_data/ output_dir/
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source output_dir/
```

## betti.sh explanation
python betti.py --source ${1} --img_size 640 --dist_thres 10  --target_upper_size 80 --target_lower_size 20 --betti_thres 0 --num_workers 64 --save_dir ${2}
# img_size: reshape image to img_size during processing
# dist_thres: RGB L2 distance , threshold of deciding wether to add an edge
# target_upper_size: specified upper bound of the object size you want to detect after reshaping
# target_lower_size: specified lower bound of the object size you want to detect after reshaping
# betti_thres: betti number threshold for filtering
# num_worker: specify how many worker available in your cuda
# save_dir: specify where to store the results

