# yolov7_visdrone
DIP Final Project: Fine-tune [yolov7](https://github.com/WongKinYiu/yolov7) pretrained model with Visdrone-Dataset.
## Reference:
- https://github.com/WongKinYiu/yolov7
- https://github.com/ultralytics/yolov5
- https://github.com/VisDrone/VisDrone-Dataset

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
  
