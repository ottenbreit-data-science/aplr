#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"

namespace aplr_internal
{

// Centering the predictor avoids subtracting moments of a large common offset.
// These buffers belong to a Term, so parallel candidate searches do not share
// mutable scoring state.
struct PrefixMoments
{
    std::vector<long double> weight;
    std::vector<long double> weight_x;
    std::vector<long double> weight_x2;
    std::vector<long double> weight_y;
    std::vector<long double> weight_xy;
    std::vector<long double> absolute_weight_x;
    std::vector<long double> absolute_weight_y;
    std::vector<long double> absolute_weight_xy;
    long double weighted_y2{0.0L};
    long double origin{0.0L};
};

struct HingeScore
{
    double coefficient{0.0};
    double error{std::numeric_limits<double>::infinity()};
    double error_bound{0.0};
    bool reliable{true};
};

inline void update_prefix_moments(
    PrefixMoments &output,
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &sample_weight)
{
    const size_t size{static_cast<size_t>(x.size())};
    output.weight.resize(size + 1);
    output.weight_x.resize(size + 1);
    output.weight_x2.resize(size + 1);
    output.weight_y.resize(size + 1);
    output.weight_xy.resize(size + 1);
    output.absolute_weight_x.resize(size + 1);
    output.absolute_weight_y.resize(size + 1);
    output.absolute_weight_xy.resize(size + 1);
    output.weight[0] = output.weight_x[0] = output.weight_x2[0] = 0.0L;
    output.weight_y[0] = output.weight_xy[0] = 0.0L;
    output.absolute_weight_x[0] = output.absolute_weight_y[0] = output.absolute_weight_xy[0] = 0.0L;
    output.weighted_y2 = 0.0L;
    output.origin = size ? static_cast<long double>(x[0]) / 2.0L +
                               static_cast<long double>(x[size - 1]) / 2.0L : 0.0L;
    for (size_t i = 0; i < size; ++i)
    {
        const long double weight{static_cast<long double>(sample_weight[i])};
        const long double value{static_cast<long double>(x[i]) - output.origin};
        const long double response{static_cast<long double>(y[i])};
        output.weight[i + 1] = output.weight[i] + weight;
        output.weight_x[i + 1] = output.weight_x[i] + weight * value;
        output.weight_x2[i + 1] = output.weight_x2[i] + weight * value * value;
        output.weight_y[i + 1] = output.weight_y[i] + weight * response;
        output.weight_xy[i + 1] = output.weight_xy[i] + weight * value * response;
        output.absolute_weight_x[i + 1] = output.absolute_weight_x[i] + std::abs(weight * value);
        output.absolute_weight_y[i + 1] = output.absolute_weight_y[i] + std::abs(weight * response);
        output.absolute_weight_xy[i + 1] = output.absolute_weight_xy[i] + std::abs(weight * value * response);
        output.weighted_y2 += weight * response * response;
    }
}

inline PrefixMoments build_prefix_moments(
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &y,
    const Eigen::VectorXd &sample_weight)
{
    PrefixMoments output;
    update_prefix_moments(output, x, y, sample_weight);
    return output;
}

inline HingeScore score_hinge_from_prefix(
    const PrefixMoments &moments,
    double split_point,
    size_t begin,
    size_t end,
    double learning_rate,
    double penalty_factor,
    double ridge_penalty,
    double ridge_penalty_weight,
    int monotonic_constraint)
{
    HingeScore output;
    const long double weight{moments.weight[end] - moments.weight[begin]};
    const long double weight_x{moments.weight_x[end] - moments.weight_x[begin]};
    const long double weight_x2{moments.weight_x2[end] - moments.weight_x2[begin]};
    const long double weight_y{moments.weight_y[end] - moments.weight_y[begin]};
    const long double weight_xy{moments.weight_xy[end] - moments.weight_xy[begin]};
    const long double split{static_cast<long double>(split_point) - moments.origin};
    const long double numerator{weight_xy - split * weight_y};
    const long double basis_squared{
        weight_x2 - 2.0L * split * weight_x + split * split * weight};
    const long double denominator{basis_squared + (ridge_penalty > 0.0 ?
        static_cast<long double>(ridge_penalty * ridge_penalty_weight) : 0.0L)};

    // Bounds also cover subtraction of two prefixes. Use double precision here:
    // exhaustive scoring uses doubles, and on MSVC long double is a double.
    const long double roundoff{128.0L * std::numeric_limits<double>::epsilon() *
        static_cast<long double>(moments.weight.size())};
    const long double numerator_scale{
        moments.absolute_weight_xy[end] + moments.absolute_weight_xy[begin] +
        std::abs(split) * (moments.absolute_weight_y[end] + moments.absolute_weight_y[begin])};
    const long double basis_scale{
        std::abs(moments.weight_x2[end]) + std::abs(moments.weight_x2[begin]) +
        2.0L * std::abs(split) * (moments.absolute_weight_x[end] + moments.absolute_weight_x[begin]) +
        split * split * (std::abs(moments.weight[end]) + std::abs(moments.weight[begin]))};
    const long double numerator_bound{roundoff * numerator_scale};
    const long double basis_bound{roundoff * basis_scale};
    if (!std::isfinite(denominator) || denominator <= basis_bound ||
        !std::isfinite(numerator))
    {
        output.reliable = false;
        return output;
    }
    const long double coefficient{
        static_cast<long double>(learning_rate * penalty_factor) * numerator / denominator};
    if (!std::isfinite(static_cast<double>(coefficient)))
    {
        output.reliable = false;
        return output;
    }
    if (monotonic_constraint != 0 && std::abs(numerator) <= numerator_bound)
    {
        output.reliable = false;
        return output;
    }
    if ((monotonic_constraint > 0 && coefficient < 0.0L) ||
        (monotonic_constraint < 0 && coefficient > 0.0L))
        return output;

    const long double error{
        moments.weighted_y2 - 2.0L * coefficient * numerator +
        coefficient * coefficient * basis_squared};
    const long double coefficient_bound{
        std::abs(learning_rate * penalty_factor) * numerator_bound / denominator +
        std::abs(coefficient) * basis_bound / denominator};
    const long double error_bound{
        roundoff * (std::abs(moments.weighted_y2) +
            2.0L * std::abs(coefficient * numerator) + coefficient * coefficient * std::abs(basis_squared)) +
        2.0L * std::abs(coefficient) * numerator_bound +
        coefficient * coefficient * basis_bound +
        2.0L * coefficient_bound * (std::abs(numerator) + std::abs(coefficient * basis_squared))};
    output.coefficient = static_cast<double>(coefficient);
    output.error = static_cast<double>(error);
    output.error_bound = static_cast<double>(error_bound);
    output.reliable = std::isfinite(output.error) && std::isfinite(output.error_bound) && error >= -error_bound;
    return output;
}

} // namespace aplr_internal
