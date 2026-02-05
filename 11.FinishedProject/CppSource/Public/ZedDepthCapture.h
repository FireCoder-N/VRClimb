// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

//#ifdef check
//#undef check
//#endif
//#include <opencv2/opencv.hpp>

#include "PreOpenCVHeaders.h"
#include "OpenCVHelper.h"
#include <ThirdParty/OpenCV/include/opencv2/imgproc.hpp>
#include <ThirdParty/OpenCV/include/opencv2/highgui/highgui.hpp>
#include <ThirdParty/OpenCV/include/opencv2/core.hpp>
#include <ThirdParty/OpenCV/include/opencv2/calib3d.hpp>
#include "PostOpenCVHeaders.h"

#include "Camera.hpp"
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
#include <string>
#include "Logging/LogMacros.h"

DECLARE_LOG_CATEGORY_EXTERN(ZedCamera, Log, All);

/**
 * 
 */
class ZedDepthCapture
{
public:
	ZedDepthCapture();
	~ZedDepthCapture();

	bool openCamera();

	// Captures one depth image (depth mask) and saves it as a PNG file.
	// 'outputPath' should be a valid path (e.g., "Saved/Depth/depth_mask.png")

	bool captureDepthImage(const std::string& outputPath);
	cv::Mat UndistortImage(sl::Mat img);

private:
	sl::Camera zed;
};
