# CNN base 3D vehicle detection using LiDAR by Chainer
Reference:
- VoxelNet [link](https://arxiv.org/pdf/1711.06396.pdf)

# Prepare datasets
## KITTI dataset
- 3D Detection Dataset [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)  
- (in data folder) train/val split [link](http://www.cs.toronto.edu/objprop3d/downloads.php)

# Execution
```
######## Training by kitti dataset ########
python train.py experiments/orig_voxelnet/orig_voxelnet_single.yml

######## Evaluation by kitti dataset ########
python evaluation.py experiments/orig_voxelnet/orig_voxelnet_eval.yml --gpu 0 --nms_thresh 0.5 --thresh 0.5

・Provided by KITTI competition
cd devkit_object/cpp
g++ -O3 -DNDEBUG -o evaluate_object evaluate_object.cpp
./evaluate_object ./results/result_dir

######## Inference by kitti dataset ########
python demo.py experiments/orig_voxelnet/orig_voxelnet_demo.yml --gpu 0

######## Visualize by kitti dataset ########
・Compate inputs for network by threshold
python visualize.py --type input --config experiments/orig_voxelnet/orig_voxelnet_viz.yml
・Calculate statistic of raw data
python visualize.py --type stats --config experiments/orig_voxelnet/orig_voxelnet_stats.yml
```

# Compare inputs for network by threshold
<img src="images/compare_thres1.png" />
<img src="images/compare_thres2.png" />
<img src="images/compare_thres3.png" />

# Statistic of raw data
<img src="images/statistic_x.png" />
<img src="images/statistic_y.png" />

# Requirement
- Python3
- Chainer
- Cupy
- ChainerCV
- OpenCV
