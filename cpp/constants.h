#pragma once
#include <limits>

const double NAN_DOUBLE{ std::numeric_limits<double>::quiet_NaN() };
const double MAX_PREDICTED_PROBABILITY{0.9999999};
const double MIN_PREDICTED_PROBABILITY{0.0000001};