#pragma once
#include <limits>

const double NAN_DOUBLE{ std::numeric_limits<double>::quiet_NaN() };
const double SMALL_NEGATIVE_VALUE{-0.000001};