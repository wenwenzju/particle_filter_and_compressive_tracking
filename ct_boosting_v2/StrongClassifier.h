#pragma once
#include "Representation.hpp"
#include "NaiveBayes.h"
#include "opencv2/opencv.hpp"
#include <vector>

#define SMALL_CONSTANT 1e-16

class StrongClassifier
{
public:
	StrongClassifier(int weakClassifierNum, int fn);
	virtual ~StrongClassifier();
	void init(Representation& pos, Representation& neg);
	void update(Representation& pos, Representation& neg, float ln);
	float classify(Representation& x);
	void normalize(std::vector<float>& d);
protected:
	NaiveBayes** weakClassifiers;
	int weakClassifierNum;
	int featureNum;
private:
	std::vector<float> classifierWeights;
};
