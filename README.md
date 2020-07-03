# Computer Pointer Controller

The project allows controlling the cursor of the mouse by a human gaze. Four pretrained models are used to detect human face, head pose, facial landmarks and, finally, human eye gaze.

## Project Set Up and Installation
This project uses OpenVino distribution to perform inference, please find the instruction on this dependency installation followith the link:
<a href="https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html">
https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_windows.html</a>

The models used for the inference must be downloaded locally, please follow the following steps:
1. Go to the directory with the OpenVino installation to reach the downloader tool (your path may differ depending on the OpenVino distribution location on your computer as well as the version on OpenVino you have installed, please note that the installation instruction are given for Window):
```cd C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\deployment_tools\open_model_zoo\tools\downloader```

2. Install dependencies for downloader.py:
```pip install requests pyyaml```

3. Download each of 4 models using the downloader, precising the output directory:
```
python downloader.py --name human-pose-estimation-0001 -o C:\Users\frup75275\Documents\OpenVinoProject3
python downloader.py --name face-detection-adas-binary-0001 -o C:\Users\frup75275\Documents\OpenVinoProject3
python downloader.py --name landmarks-regression-retail-0009 -o C:\Users\frup75275\Documents\OpenVinoProject3
python downloader.py --name gaze-estimation-adas-0002 -o C:\Users\frup75275\Documents\OpenVinoProject3
```

## Demo
1. Initiatize OpenVino environment
```cd C:\Program Files (x86)\IntelSWTools\openvino_2020.3.194\bin
setupvars.bat
```
2. Go to the src subfolder inside your cloned project folder:
```
cd C:\Users\frup75275\Documents\OpenVinoProject3\src
```
3. Run the main.py file with arguments as below:
```
python main.py -fd ../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml \ 
-lr ../intel/landmarks-regression-retail-0009/FP32-INT8/landmarks-regression-retail-0009.xml \ 
-hp ../intel/head-pose-estimation-adas-0001/FP32-INT8/head-pose-estimation-adas-0001.xml \ 
-ge ../intel/gaze-estimation-adas-0002/FP32-INT8/gaze-estimation-adas-0002.xml \ 
-i ../bin/demo.mp4
```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
