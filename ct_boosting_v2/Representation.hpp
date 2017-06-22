#pragma once

#include "opencv2/opencv.hpp"
class Representation
{
public:
	cv::InputArray _x;
	Representation(cv::InputArray& x):_x(x){}
	virtual ~Representation(){}
};