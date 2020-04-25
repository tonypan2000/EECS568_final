# SURF-SLAM

![Our Final Results](https://raw.githubusercontent.com/audrow/navarch-568-proj/master/docs/seq00_result.png)

Authors: Snehal Chavan, Nadha Gafoor, Audrow Nash, Ming-Yuan Yu, and Xinzhe Zhang

A MATLAB implementation of ORB-SLAM [1] using SURF features. This is our final project for *EECS 568: Mobile Robotics* during the Winter 2018 semester at the university of Michigan.

## Setup
1. Install MATLAB and the computer vision toolbox.
2. Clone this repository.
```
git clone git@github.com:audrow/navarch-568-proj.git WORKSPACE_DIR
```
3. Run `src/run.m` from the repositories home directory. This run will show you several figures including the path from visual odometry, proposed loop closures, the map after the loop closures, and the error between the final map and ground truth.

To reproduce our full results, download the full [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and run sequence zero.

## Resources
* [Report](https://github.com/audrow/navarch-568-proj/blob/master/docs/report.pdf)
* [Presentation](https://github.com/audrow/navarch-568-proj/blob/master/docs/presentation.pdf)

## Reference

[1] Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. “Orb-slam: a versatile and accurate monocular slam system.” IEEE Transactions on Robotics, 31(5): 1147–1163, 2015
