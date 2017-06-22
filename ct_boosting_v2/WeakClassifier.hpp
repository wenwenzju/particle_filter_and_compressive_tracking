#pragma once

#include "Representation.hpp"

class WeakClassifier
{
public:
	enum {NEGATIVE = -1, POSITIVE = 1};
	virtual ~WeakClassifier(){}
	virtual bool update(Representation& x, int y, double lmb) {return false;}
	virtual int classify(Representation& x) {return 0;}
};