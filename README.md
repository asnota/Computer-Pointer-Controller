# Computer Pointer Controller

The project allows controlling the cursor of the mouse by a human gaze. Four pretrained models are used to detect human face, head pose, facial landmarks and, finally, human eye gaze.

## Project Set Up and Installation
This project uses OpenVino distribution to perform inference, please find the instruction on this dependency installation following the link:
<a href="https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html">
https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html</a>

The models used for the inference must be downloaded locally, please follow the steps described below:
1. Clone the repository
2. Go to the directory with the OpenVino installation to reach the downloader tool (your path may differ depending on the OpenVino distribution location on your computer as well as the version on OpenVino you have installed, please note that the installation instruction are given for Window):
```cd C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\open_model_zoo\tools\downloader```

3. Install dependencies for downloader.py:
```pip install requests pyyaml```

4. Download each of 4 models using the downloader, precising the output directory:
```
python downloader.py --name head_pose_estimation_adas_0001 -o C:\Users\frup75275\Documents\OpenVinoProject3
python downloader.py --name face-detection-adas-binary-0001 -o C:\Users\frup75275\Documents\OpenVinoProject3
python downloader.py --name landmarks-regression-retail-0009 -o C:\Users\frup75275\Documents\OpenVinoProject3
python downloader.py --name gaze-estimation-adas-0002 -o C:\Users\frup75275\Documents\OpenVinoProject3
```
5. Create virtual environment in pour project directory:
```
python -m venv env
```
6. Install dependencies:
```
pip install requirements.txt
```


## Demo
1. Initiatize OpenVino environment
```
cd C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\bin
setupvars.bat
```
2. Activate previously created virtual environment from your project directory:
```
.\env\Scripts\activate
```
3. Go to the /src subfolder inside your project directory:
```
cd \src
```
4. Run the main.py file with arguments as below:
```
python main.py
```
You may specify the optional arguments if you intend to use another input, device or models, for example here is a request with additional arguments:
```
python main.py 
-fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr ../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml \ 
-hp ../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml \ 
-ge ../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml \ 
-i ../bin/demo.mp4
```
5. You may also use -flags to visualize the output of models used in the project (by default it's set to visualize the output from GazeEstimationModel):
```
-flags fd
```
fd: for FaceDetectionModel
lr: for LandmarkRegressionModel
hp: for HeadPoseEstimationModel
ge: for GazeEstimationModel

## Documentation
The project contains 4 subfolders (src, intel, bin, benchmarks), where /src folder holds the code, /intel folder contains models, /bin folder contains the example video file and /benchmarks folder contains the graphs taken out of benchmarks tests.
The main script is maintained in main.py file, whereas the classes to inference the models in model_ prefixed files. Two additional classes - input_feeder.py and mouse_controller.py provide additional handling on the batch feed and mouse manipulation respectively.
The root directory contains README.md and requirement.txt files which should help with the required installations and project run.

## Benchmarks
The benchmark tests were performed in DL Workbench metrics tool developped by Intel using CelebA Dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
The available in DL Workbench pretrained model from OpenVino model zoo (face-detection-adas-0001) was tested with parallel stream from 1 to 2 and batch size range from 1 to 30 (with a batch step of 10) on a CPU device.  

## Results
The graph below shows that the lowest latency were achieved with a batch size of 1 (42.22 ms), whereas batch sizes of 10 (884.9 ms), 20 (1,811.86 ms) and 30 (2,633.22). 
The throughput for the last 3 cases had minor fluctuations (11.36 fps - 12.12 fps).
The significant increase in throuput was achieved by parallel streams augmentation (4), where the batch size influenced mostly the latency: 1,824.31 ms for 30 batches vs 1,189.09 ms for 10 batches, at minor fluctuations in a throughput (31,96 fps vs 32,59 fps respectively).
![Group inference results](/benchmarking/Group_inference_results.png)

The execution time by layer also shows that convolution took the most time for this model, therefore any optimization might first address the possibilities of the convolution layers optimization. 
![face-detection-adas-0001](/benchmarking/face-detection-adas-0001.png)