#pragma once
#include <limits>

const double NAN_DOUBLE{ std::numeric_limits<double>::quiet_NaN() };
const int MAX_ABS_EXPONENT_TO_APPLY_ON_LINEAR_PREDICTOR_IN_LOGIT_MODEL{std::min(16, std::numeric_limits<double>::max_exponent10)};
const std::string FAMILY_GAUSSIAN{"gaussian"};