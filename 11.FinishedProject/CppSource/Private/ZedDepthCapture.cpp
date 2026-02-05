#include "ZedDepthCapture.h"
#include <filesystem>
//#include <iostream>


DEFINE_LOG_CATEGORY(ZedCamera);



ZedDepthCapture::ZedDepthCapture() {
    // Constructor – nothing to initialize here yet.
}

ZedDepthCapture::~ZedDepthCapture() {
    zed.close();
}

bool ZedDepthCapture::openCamera() {
    sl::InitParameters init_params;
    init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_params.camera_resolution = sl::RESOLUTION::HD1080; // Adjust resolution as needed.
    init_params.coordinate_units = sl::UNIT::METER;        // Depth in meters.

    // Optionally, set additional parameters (e.g. depth mode).
    auto err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        UE_LOG(ZedCamera, Error, TEXT("Error opening ZED cam: %s"), *FString(sl::toString(err).c_str()));
        //std::cerr << "Error opening ZED camera: " << sl::toString(err) << std::endl;
        return false;
    }
    UE_LOG(ZedCamera, Log, TEXT("Opened camera"));
    GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Yellow, TEXT("Opened!"));
    return true;
}

bool ZedDepthCapture::captureDepthImage(const std::string& outputPath) {
    // Capture a single frame.
    if (zed.grab() != sl::ERROR_CODE::SUCCESS) {
        //std::cout << "Failed to grab frame from ZED camera.\n";
        UE_LOG(ZedCamera, Error, TEXT("Failed to grab frame from ZED camera."));
        return false;
    }

    // Retrieve the depth measure (depth mask).
    sl::Mat depth;
    auto err = zed.retrieveImage(depth, sl::VIEW::DEPTH); // Retrieve image
    
    if (err == sl::ERROR_CODE::SUCCESS) {
        cv::Mat undistorted = UndistortImage(depth);
        cv::imwrite((outputPath + "test1.png").c_str(), undistorted);
        UE_LOG(ZedCamera, Log, TEXT("Undistort Save successful"));
    }

    auto state = depth.write((outputPath + "test.png").c_str());

    if (state == sl::ERROR_CODE::SUCCESS) {
        UE_LOG(ZedCamera, Log, TEXT("Save successful"));
        //std::cout << "Save successful\n";
        return true;
    }
    else {
        UE_LOG(ZedCamera, Error, TEXT("Save unsuccessful"));
        //std::cout << "Save unsuccessful\n";
        return false;
    }

    
}

cv::Mat ZedDepthCapture::UndistortImage(sl::Mat img) {
    sl::CalibrationParameters calib_params = zed.getCameraInformation().camera_configuration.calibration_parameters;
    float fx = calib_params.left_cam.fx;
    float fy = calib_params.left_cam.fy;
    float cx = calib_params.left_cam.cx;
    float cy = calib_params.left_cam.cy;
    float k1 = calib_params.left_cam.disto[0];
    float k2 = calib_params.left_cam.disto[1];
    float p1 = calib_params.left_cam.disto[2];
    float p2 = calib_params.left_cam.disto[3];
    float k3 = calib_params.left_cam.disto[4];

    //Create OpenCV camera matrix and distortion coeffs
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);

    //Convert sl::Mat to cv::Mat
    cv::Mat depth_image = cv::Mat(img.getHeight(), img.getWidth(), CV_32FC1, img.getPtr<sl::float1>());
    //cv::cvtColor(depth_image, depth_image, cv::COLOR_RGBA2GRAY);
    //cv::normalize(depth_image, depth_image, 0, 255, cv::NORM_MINMAX);

    //Get optimal camera matrix
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, depth_image.size(), 1);

    // Compute undistortion map
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), newCameraMatrix, depth_image.size(), CV_16SC2, map1, map2);

    // Undistort depth mask using remap
    cv::Mat undistorted_depth;
    cv::remap(depth_image, undistorted_depth, map1, map2, cv::INTER_NEAREST);

    return undistorted_depth;
}