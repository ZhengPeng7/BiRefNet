#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include "utils.h"

class BiRefNet
{
public:
	BiRefNet(std::string model_path, nvinfer1::ILogger& logger);
	cv::Mat predict(cv::Mat& image);
	~BiRefNet();
	
private:
	int input_w = 1024;
	int input_h = 1024;
	float mean[3] = { 123.675, 116.28, 103.53 };
	float std[3] = { 58.395, 57.12, 57.375 };

	std::vector<int> offset;

	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
	nvinfer1::IExecutionContext* context;
	nvinfer1::INetworkDefinition* network;

	void* buffer[2];
	float* depth_data;
	cudaStream_t stream;

	std::vector<float> preprocess(cv::Mat& image);
	std::vector<DichotomousImageSegmentation> postprocess(std::vector<int> mask, int img_w, int img_h);
	bool saveEngine(const std::string& filename);
};
