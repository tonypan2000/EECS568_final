# InEKF Localization and Semantic Mapping on the KITTI Dataset

This is our final project git repository for EECS 568: Mobile Robotics: Methods and Algorithms. Our project is InEKF Localization and Semantic Mapping on the KITTI Dataset.

You can see our final presentation video of our program localizing and building a map with KITTI [here](https://www.youtube.com/watch?v=A9tSE8NMWzA).

[![](http://img.youtube.com/vi/A9tSE8NMWzA/0.jpg)](https://www.youtube.com/watch?v=A9tSE8NMWzA "Mobile Robotics Final Presentation")

You can find our final report [here](https://github.com/tonypan2000/EECS568_final/blob/master/EECS_568_Final_Report.pdf).

These instructions will get you a copy of the project up and running on your local machine.

## Left-InEKF Localization
Click for full video

[![](http://img.youtube.com/vi/7E5PInxk9EU/0.jpg)](http://www.youtube.com/watch?v=7E5PInxk9EU "Left-InEKF Localization Trajectory on KITTI") 

### Prerequisites

* MATLAB

Since our code is written in MATLAB, you can run our program on any OS platform.

### Running Localization Program

First, you need to generate a trajectory with our Left-invariant EKF. To do so, edit line 5 of `Main.m` to feed it the input dataset folder name. For example, for the dataset `0009`, the line of code should look like:

```
filename = '2011_09_26_drive_0079_sync';
```

After that, simply run `InEKF_Main.m` and it will save the poses in SE3 as 12 by 1 vectors in a `.txt` file. The name of the file will be that of the dataset appended by "poses.txt". For instance, the poses generated by the dataset in folder `2011_09_26_drive_0079_sync` will be named `2011_09_26_drive_0079_sync_poses.txt`.

## Semantic Mapping with LiDAR Points

For test, we are using `gtFine_trainvaltest.zip (241MB) [md5]` as labels, and using `leftImg8bit_trainvaltest.zip (11GB) [md5]` as raw input images for training. To further train or test the unet these files should be downloaded from the CityScapes dataset and placed in `unet/training/`

When validating our training outcome, we are using the images of `val` subfolder of the corresponding datasets.

To test our code, we are now using deducted train, label and val sets.

### Dependencies

* Python 3
* Anaconda
* Cuda, CUDNN
* OpenCV
* CMake
* Eigen
* [Octomap](https://github.com/OctoMap/octomap): version 1.9.5, devel branch

### OctoMap Usage

```
mkdir build
cd build
cmake ..
make
./mapping
```

#### Files in data

* `*.bin`: KITTI LiDAR data.
* `*_pose.txt`: ground truth poses of KITTI odometry sequence.
* `calib_cam_to_cam.txt`, `calib_imu_to_velo.txt`, and `calib_velo_to_cam.txt`: calibration transformation matrix
* `*mapping.ot`: test LiDAR mapping.

`*.ot` files are for visualization with `Octovis`.


## Datasets Used

* [KITTI](http://www.cvlibs.net/datasets/kitti/) Vision Benchmark Suite
* [CityScapes](https://www.cityscapes-dataset.com/downloads/) Semantic Understanding of Urban Street Scenes

## License: 
We used the Octomap library, and their licenses are:
  * octomap: [New BSD License](Semantic Mapping/LICENSE_OctoMap.txt)
  * octovis and related libraries: [GPL](Semantic Mapping/LICENSE_Octovis.txt)

## Authors

* [Mu-Ti Chung](https://github.com/mutichung) - <mtchung@umich.edu>
* [James Cooney](https://github.com/jpc4kp) - <cooneyj@umich.edu>
* [Tony Pan](https://github.com/tonypan2000) - <tonypan@umich.edu>
* [Shoutian Wang](https://github.com/BoomSky0416) - <shoutian@umich.edu>
* [Haoxiang Wu] - <hxwu@umich.edu>

## Acknowledgments

Thank you to our instructor [Maani Ghaffari](https://www.maanighaffari.com/) for making this possible.
