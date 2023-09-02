# neuCam

Python project programmed for Raspberry Pi for human and car tracking with IDing, tracking capability, and timer for overstayer.

## Pre-Configuration Requirements

Due to limitations on the file sizes that can be hosted on GitHub, I are unable to provide the YOLO weight files out of the box. However, these can easily be downloaded and configured manually. I highly recommend getting the YOLO weights and configs from [Darknet's YOLO page](https://pjreddie.com/darknet/yolo/).

### Recommended Models

    High-end Hardware: yolov3-spp
    Medium-end Hardware: yolov3-416
    Budget Hardware: yolov3-tiny

After downloading your desired weight file, place it into the public/yolo directory to complete the setup.

## Features and Capabilities

For detection, both OpenCV Haars-Cascade and YOLO is used. Haars-Cascade acts as a prefilter for YOLO
in order to save resources. JSON files are saved for overstaying cars and humans, and loaded and
save each time you start and quit.
