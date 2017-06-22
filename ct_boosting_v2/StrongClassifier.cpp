#include "StrongClassifier.h"

StrongClassifier::StrongClassifier(int weakClassifierNum, int fn) : weakClassifierNum(weakClassifierNum), featureNum(fn), classifierWeights(weakClassifierNum, 1./weakClassifierNum)
{
	weakClassifiers = new NaiveBayes*[weakClassifierNum];
	for (int i = 0; i < weakClassifierNum; ++i) weakClassifiers[i] = new NaiveBayes(fn);
}

StrongClassifier::~StrongClassifier()
{
	for (int i = 0; i < weakClassifierNum; ++i) delete weakClassifiers[i];
	delete [] weakClassifiers;
}

void StrongClassifier::init(Representation& pos, Representation& neg)
{
	cv::Mat matPos = pos._x.getMat(), matNeg = neg._x.getMat();
	int nPos = matPos.cols, nNeg = matNeg.cols;								//number of positive and negative samples
	cv::Mat densityPos = cv::Mat::ones(1, nPos, CV_32FC1)*(1./(nPos+nNeg));
	cv::Mat densityNeg = cv::Mat::ones(1, nNeg, CV_32FC1)*(1./(nPos+nNeg));	//weights of positive and negative samples

	cv::Mat muPos(featureNum, 1, CV_32FC1),sigmaPos(featureNum, 1, CV_32FC1);
	cv::Mat muNeg(featureNum, 1, CV_32FC1),sigmaNeg(featureNum, 1, CV_32FC1);

	cv::Scalar sumPos, sumNeg, muTmp, sigmaTmp;
	bool* classifyPos = new bool[nPos];
	bool* classifyNeg = new bool[nNeg];
	int i = 0;
	float maxError = -FLT_MAX;
	int maxErrorIdx = 0;
	for (; i < weakClassifierNum; ++i)
	{
		//update weak classifier
		sumPos = cv::sum(densityPos);
		float sP = sumPos[0];
		sumNeg = cv::sum(densityNeg);
		float sN = sumNeg[0];
		for (int j = 0; j < featureNum; ++j)		//calculate mean and std deviation among positive and negative samples per feature
		{
			muTmp = cv::sum(matPos.row(j).mul(densityPos*(1./sP)));
			cv::Mat tmp = cv::Mat::ones(1, nPos, CV_32FC1)*muTmp[0];
			sigmaTmp = cv::sum((matPos.row(j)-tmp).mul(matPos.row(j)-tmp).mul(densityPos*(1./sP)));	//weighted mean and std deviation
			muPos.at<float>(j,0) = muTmp[0];
			sigmaPos.at<float>(j,0) = sqrt(sigmaTmp[0]);
			muTmp = cv::sum(matNeg.row(j).mul(densityNeg*(1./sN)));
			tmp = cv::Mat::ones(1, nNeg, CV_32FC1)*muTmp[0];
			sigmaTmp = cv::sum((matNeg.row(j)-tmp).mul(matNeg.row(j)-tmp).mul(densityNeg*(1./sN)));	//weighted mean and std deviation
			muNeg.at<float>(j,0) = muTmp[0];
			sigmaNeg.at<float>(j,0) = sqrt(sigmaTmp[0]);
		}

		weakClassifiers[i]->update(muPos, sigmaPos, NaiveBayes::POSITIVE, 0.f);
		weakClassifiers[i]->update(muNeg, sigmaNeg, NaiveBayes::NEGATIVE, 0.f);

		//calculate error rate
		float errorsPos = 0.f, errorsNeg = 0.f;
		for (int j = 0; j < nPos; ++j)
		{
			if (weakClassifiers[i]->classify(Representation(matPos.col(j))) == NaiveBayes::NEGATIVE)	//misclassified
			{errorsPos += densityPos.at<float>(0,j); classifyPos[j] = false;}
			else classifyPos[j] = true;
		}
		for (int j = 0; j < nNeg; ++j)
		{
			if (weakClassifiers[i]->classify(Representation(matNeg.col(j))) == NaiveBayes::POSITIVE)	//misclassified
			{errorsNeg += densityNeg.at<float>(0, j);classifyNeg[j] = false;}
			else classifyNeg[j] = true;
		}

		float errors = errorsPos+errorsNeg;

		classifierWeights[i] = errors > 0.5 ? 0: 0.5*log((1.f-errors+SMALL_CONSTANT)/(errors+SMALL_CONSTANT));

		//update density
		//if (errorsPos <= 0.5)
		for (int j = 0; j < nPos; ++j) 
		{
			if (classifyPos[j])			//correctly classify
				densityPos.at<float>(0,j) /= (2*(1.f-errors));
			else						//misclassified
				densityPos.at<float>(0,j) /= (2*errors);
		}
		//if (errorsNeg <= 0.5)
		for (int j = 0; j < nNeg; ++j)
		{
			if (classifyNeg[j])
				densityNeg.at<float>(0,j) /= (2*(1.f-errors));
			else						//misclassified
				densityNeg.at<float>(0,j) /= (2*errors);
		}
		sumPos = cv::sum(densityPos);
		sumNeg = cv::sum(densityNeg);
		densityPos /= (sumPos[0] + sumNeg[0]);		//normalize
		densityNeg /= (sumPos[0] + sumNeg[0]);		//normalize
	}

		delete [] classifyPos;
		delete [] classifyNeg;
}

void StrongClassifier::update(Representation& pos, Representation& neg, float ln)
{
	cv::Mat matPos = pos._x.getMat(), matNeg = neg._x.getMat();
	int nPos = matPos.cols, nNeg = matNeg.cols;								//number of positive and negative samples
	cv::Mat densityPos = cv::Mat::ones(1, nPos, CV_32FC1)*(1./(nPos+nNeg));
	cv::Mat densityNeg = cv::Mat::ones(1, nNeg, CV_32FC1)*(1./(nPos+nNeg));	//weights of positive and negative samples

	cv::Mat muPos(featureNum, 1, CV_32FC1),sigmaPos(featureNum, 1, CV_32FC1);
	cv::Mat muNeg(featureNum, 1, CV_32FC1),sigmaNeg(featureNum, 1, CV_32FC1);

	//cv::Mat muFuckPos(featureNum, 1, CV_32FC1), sigmaFuckPos(featureNum, 1, CV_32FC1);
	//cv::Mat muFuckNeg(featureNum, 1, CV_32FC1), sigmaFuckNeg(featureNum, 1, CV_32FC1);

	cv::Scalar sumPos, sumNeg, muTmp, sigmaTmp;
	bool* classifyPos = new bool[nPos];
	bool* classifyNeg = new bool[nNeg];
	int i = 0;
	float maxError = -FLT_MAX;
	int maxErrorIdx = 0;
	for (; i < weakClassifierNum; ++i)
	{
		//update weak classifier
		sumPos = cv::sum(densityPos);
		float sP = sumPos[0];
		sumNeg = cv::sum(densityNeg);
		float sN = sumNeg[0];
		for (int j = 0; j < featureNum; ++j)		//calculate mean and std deviation among positive and negative samples per feature
		{
			//cv::meanStdDev(matPos.row(j).mul(densityPos*(1./sP)), muTmp, sigmaTmp);	//weighted mean and std deviation
			muTmp = cv::sum(matPos.row(j).mul(densityPos*(1./sP)));
			cv::Mat tmp = cv::Mat::ones(1, nPos, CV_32FC1)*muTmp[0];
			sigmaTmp = cv::sum((matPos.row(j)-tmp).mul(matPos.row(j)-tmp).mul(densityPos*(1./sP)));	//weighted mean and std deviation
			muPos.at<float>(j,0) = muTmp[0];
			sigmaPos.at<float>(j,0) = sqrt(sigmaTmp[0]);
			//cv::meanStdDev(matNeg.row(j).mul(densityNeg*(1./sN)), muTmp, sigmaTmp);	//weighted mean and std deviation
			muTmp = cv::sum(matNeg.row(j).mul(densityNeg*(1./sN)));
			tmp = cv::Mat::ones(1, nNeg, CV_32FC1)*muTmp[0];
			sigmaTmp = cv::sum((matNeg.row(j)-tmp).mul(matNeg.row(j)-tmp).mul(densityNeg*(1./sN)));	//weighted mean and std deviation
			muNeg.at<float>(j,0) = muTmp[0];
			sigmaNeg.at<float>(j,0) = sqrt(sigmaTmp[0]);
		}

		weakClassifiers[i]->update(muPos, sigmaPos, NaiveBayes::POSITIVE, ln);
		weakClassifiers[i]->update(muNeg, sigmaNeg, NaiveBayes::NEGATIVE, ln);

		//calculate error rate
		float errorsPos = 0.f, errorsNeg = 0.f;
		for (int j = 0; j < nPos; ++j)
		{
			if (weakClassifiers[i]->classify(Representation(matPos.col(j))) == NaiveBayes::NEGATIVE)	//misclassified
			{errorsPos += densityPos.at<float>(0,j); classifyPos[j] = false;}
			else classifyPos[j] = true;
		}
		for (int j = 0; j < nNeg; ++j)
		{
			if (weakClassifiers[i]->classify(Representation(matNeg.col(j))) == NaiveBayes::POSITIVE)	//misclassified
			{errorsNeg += densityNeg.at<float>(0, j);classifyNeg[j] = false;}
			else classifyNeg[j] = true;
		}

		float errors = errorsPos+errorsNeg;
		//if (errorsPos > 0.5)
		//{
		//	std::cout << "Oooooooooops: " << std::endl << "the positive samples error rate " << errorsPos << " of weak classifier " << i << " is larger than 0.5!!" << std::endl;
		//	//break;
		//}
		//if (errorsNeg > 0.5)
		//{
		//	std::cout << "Oooooooooops: " << std::endl << "the negative samples error rate " << errorsNeg << " of weak classifier " << i << " is larger than 0.5!!" << std::endl;
		//}
		if (errors == 0 || errorsPos > 0.5 || errorsNeg > 0.5) break;

		classifierWeights[i] = errors > 0.5 ? 0: 0.5*log((1.f-errors+SMALL_CONSTANT)/(errors+SMALL_CONSTANT));
		if (errors > maxError) {maxError = errors; maxErrorIdx = i;}

		//std::cout << "error pos: " << errorsPos << "; error neg: " << errorsNeg << std::endl;

		//update density
		//if (errorsPos > 0.5) errorsPos = 1.-errorsPos;
		//if (errorsPos <= 0.5)
			for (int j = 0; j < nPos; ++j) 
			{
				if (classifyPos[j])			//correctly classify
					densityPos.at<float>(0,j) /= (2*(1.f-errors));
				else						//misclassified
					densityPos.at<float>(0,j) /= (2*errors);
			}
		//if (errorsNeg <= 0.5)
			for (int j = 0; j < nNeg; ++j)
			{
				if (classifyNeg[j])
					densityNeg.at<float>(0,j) /= (2*(1.f-errors));
				else						//misclassified
					densityNeg.at<float>(0,j) /= (2*errors);
			}
		sumPos = cv::sum(densityPos);
		sumNeg = cv::sum(densityNeg);
		densityPos /= (sumPos[0] + sumNeg[0]);		//normalize
		densityNeg /= (sumPos[0] + sumNeg[0]);		//normalize
		std::cout << densityPos << std::endl;
		std::cout << densityNeg << std::endl;
	}
	if (i == weakClassifierNum) weakClassifiers[maxErrorIdx]->rollBack();
	if (i != 0)
		for (; i < weakClassifierNum; ++i)				//break at weak classifier i
		{
			classifierWeights[i] = 0.;
		}

	delete [] classifyPos;
	delete [] classifyNeg;
}

float StrongClassifier::classify(Representation& x)
{
	normalize(classifierWeights);
	float res = 0.f;
	float score = 0.f;
	for (int i = 0; i < weakClassifierNum; ++i)
	{
		int lab = weakClassifiers[i]->classify(x, &score);
		res += classifierWeights[i]*score;
// 		if (lab == NaiveBayes::POSITIVE)
// 		{
// 			res += classifierWeights[i];
// 		}
// 		else
// 		{
// 			res -= classifierWeights[i];
// 		}
	}
	return res;
}

void StrongClassifier::normalize(std::vector<float>& d)
{
	float s = 0.f;
	std::for_each(d.begin(), d.end(),[&s](float f){s += f;});
	std::for_each(d.begin(), d.end(),[s](float& f){f /= s;});
}