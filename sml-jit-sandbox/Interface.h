#ifndef INTERFACE_H
#define INTERFACE_H

#include <utility>

#include "ChoiceDict.h"

extern "C" std::pair<double,double> simulate();
extern "C" std::pair<double,double> importance(const ChoiceDict& constraints);

#endif
