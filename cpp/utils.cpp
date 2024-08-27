#include "utils.h"

std::tuple<cv::Mat, int, int> resize_depth(cv::Mat& img, int w, int h)
{
	cv::Mat result;
	int nw, nh;
	int ih = img.rows;
	int iw = img.cols;
	float aspectRatio = (float)img.cols / (float)img.rows;

	if (aspectRatio >= 1)
	{
		nw = w;
		nh = int(h / aspectRatio);
	}
	else
	{
		nw = int(w * aspectRatio);
		nh = h;
	}
	cv::resize(img, img, cv::Size(nw, nh));
	result = cv::Mat::ones(cv::Size(w, h), CV_8UC1) * 128;
	cv::cvtColor(result, result, cv::COLOR_GRAY2RGB);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(h, w, CV_8UC3, 0.0);
	re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));

	std::tuple<cv::Mat, int, int> res_tuple = std::make_tuple(out, (w - nw) / 2, (h - nh) / 2);

	return res_tuple;
}