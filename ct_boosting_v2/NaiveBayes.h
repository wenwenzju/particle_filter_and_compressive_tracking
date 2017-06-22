#pragma once

#include "Representation.hpp"
#include "opencv2/opencv.hpp"
#include <vector>

class NaiveBayes
{
public:
	enum {NEGATIVE = -1, POSITIVE = 1};
	NaiveBayes();
	NaiveBayes(int featureNum);
	bool update(cv::Mat& mu, cv::Mat& sigma, int y, float ln);
	int classify(Representation& x, float* score = nullptr);
	void init(std::vector<float>& msp, std::vector<float>& msn, std::vector<float>& ssp, std::vector<float>& ssn);
	void rollBack();
private:
	std::vector<float> muPositive;
	std::vector<float> muNegative;
	std::vector<float> sigmaPositive;
	std::vector<float> sigmaNegative;
	std::vector<float> rollBackmuPos;
	std::vector<float> rollBackmuNeg;
	std::vector<float> rollBacksigmaPos;
	std::vector<float> rollBacksigmaNeg;
};