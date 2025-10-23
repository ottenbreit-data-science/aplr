#pragma once
#include <limits>
#include <numeric>   //std::iota
#include <algorithm> //std::sort, std::stable_sort
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <map>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "constants.h"

using namespace Eigen;

bool is_approximately_equal(double a, double b, double tolerance = std::numeric_limits<double>::epsilon())
{
    if (std::isinf(a) && std::isinf(b))
    {
        if (std::signbit(a) == std::signbit(b))
            return true;
        else
            return false;
    }

    double relative_tolerance;
    if (std::isinf(a) || std::isinf(b))
        relative_tolerance = (fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * tolerance;
    else
        relative_tolerance = (fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * tolerance;
    double absolute_tolerance{std::fmax(relative_tolerance, tolerance)};
    bool equal{fabs(a - b) <= absolute_tolerance};

    return equal;
}

bool is_approximately_zero(double a, double tolerance = std::numeric_limits<double>::epsilon())
{
    return is_approximately_equal(a, 0.0, tolerance);
}

bool all_are_equal(VectorXd &v1, VectorXd &v2)
{
    if (v1.rows() != v2.rows())
        return false;

    for (Eigen::Index i = 0; i < v1.size(); ++i)
    {
        if (!is_approximately_equal(v1[i], v2[i]))
        {
            return false;
        }
    }

    return true;
}

std::set<std::string> get_unique_strings(const std::vector<std::string> &string_vector)
{
    std::set<std::string> unique_strings{string_vector.begin(), string_vector.end()};
    return unique_strings;
}

std::vector<std::string> get_unique_strings_as_vector(const std::vector<std::string> &string_vector)
{
    std::set<std::string> unique_strings{get_unique_strings(string_vector)};
    std::vector<std::string> unique_strings_as_vector;
    unique_strings_as_vector.reserve(unique_strings.size());
    for (auto &unique_string : unique_strings)
    {
        unique_strings_as_vector.push_back(unique_string);
    }
    return unique_strings_as_vector;
}

std::set<int> get_unique_integers(const VectorXi &int_vector)
{
    std::set<int> unique_integers{int_vector.begin(), int_vector.end()};
    return unique_integers;
}

std::set<size_t> get_unique_integers(const std::vector<size_t> &size_t_vector)
{
    std::set<size_t> unique_integers{size_t_vector.begin(), size_t_vector.end()};
    return unique_integers;
}

double set_error_to_infinity_if_invalid(double error)
{
    bool error_is_invalid{!std::isfinite(error)};
    if (error_is_invalid)
        error = std::numeric_limits<double>::infinity();

    return error;
}

VectorXd calculate_mse_errors(const VectorXd &y, const VectorXd &predicted)
{
    VectorXd errors{y - predicted};
    errors = errors.array() * errors.array();
    return errors;
}

VectorXd calculate_binomial_errors(const VectorXd &y, const VectorXd &predicted)
{
    VectorXd errors{-y.array() * predicted.array().log() - (1.0 - y.array()).array() * (1.0 - predicted.array()).log()};
    return errors;
}

VectorXd calculate_poisson_errors(const VectorXd &y, const VectorXd &predicted)
{
    VectorXd errors{predicted.array() - y.array() * predicted.array().log()};
    return errors;
}

VectorXd calculate_gamma_errors(const VectorXd &y, const VectorXd &predicted)
{
    VectorXd errors{predicted.array().log() + y.array() / predicted.array() - 1};
    return errors;
}

VectorXd calculate_tweedie_errors(const VectorXd &y, const VectorXd &predicted, double dispersion_parameter = 1.5)
{
    VectorXd errors{-y.array() * predicted.array().pow(1 - dispersion_parameter) / (1 - dispersion_parameter) + predicted.array().pow(2 - dispersion_parameter) / (2 - dispersion_parameter)};
    return errors;
}

struct GroupData
{
    std::map<int, double> error;
    std::map<int, double> count;
};

GroupData calculate_group_errors_and_count(const VectorXd &y, const VectorXd &predicted, const VectorXi &group, const std::set<int> &unique_groups,
                                           const VectorXd &sample_weight)
{
    GroupData group_data;
    for (int unique_group_value : unique_groups)
    {
        group_data.error[unique_group_value] = 0.0;
        group_data.count[unique_group_value] = 0.0;
    }

    for (Eigen::Index i = 0; i < group.size(); ++i)
    {
        group_data.error[group[i]] += (y[i] - predicted[i]) * sample_weight[i];
        group_data.count[group[i]] += sample_weight[i];
    }

    for (int unique_group_value : unique_groups)
    {
        group_data.error[unique_group_value] = group_data.error[unique_group_value] / group_data.count[unique_group_value];
    }

    return group_data;
}

VectorXd calculate_group_mse_errors(const VectorXd &y, const VectorXd &predicted, const VectorXi &group, const std::set<int> &unique_groups,
                                    const VectorXd &sample_weight)
{
    GroupData group_residuals_and_count{calculate_group_errors_and_count(y, predicted, group, unique_groups, sample_weight)};

    for (int unique_group_value : unique_groups)
    {
        group_residuals_and_count.error[unique_group_value] *= group_residuals_and_count.error[unique_group_value];
    }

    VectorXd errors(y.rows());
    for (Eigen::Index i = 0; i < y.size(); ++i)
    {
        errors[i] = group_residuals_and_count.error[group[i]];
    }

    return errors;
}

VectorXd calculate_absolute_errors(const VectorXd &y, const VectorXd &predicted)
{
    VectorXd errors{(y - predicted).cwiseAbs()};

    return errors;
}

VectorXd calculate_quantile_errors(const VectorXd &y, const VectorXd &predicted, double quantile)
{
    VectorXd errors{calculate_absolute_errors(y, predicted)};
    for (Eigen::Index i = 0; i < y.size(); ++i)
    {
        if (y[i] < predicted[i])
            errors[i] *= 1 - quantile;
        else
            errors[i] *= quantile;
    }

    return errors;
}

VectorXd calculate_negative_binomial_errors(const VectorXd &y, const VectorXd &predicted, double dispersion_parameter)
{
    ArrayXd temp{dispersion_parameter * predicted.array()};
    VectorXd errors{(1 / dispersion_parameter) * (1 + temp).log() - y.array() * (temp / (1 + temp)).log()};

    return errors;
}

VectorXd calculate_cauchy_errors(const VectorXd &y, const VectorXd &predicted, double dispersion_parameter)
{
    VectorXd errors{(1 + ((y.array() - predicted.array()) / dispersion_parameter).pow(2)).log()};

    return errors;
}

VectorXd calculate_weibull_errors(const VectorXd &y, const VectorXd &predicted, double dispersion_parameter)
{
    VectorXd errors{dispersion_parameter * predicted.array().log() + (y.array() / predicted.array()).pow(dispersion_parameter)};

    return errors;
}

VectorXd calculate_huber_errors(const VectorXd &y, const VectorXd &predicted, double delta)
{
    ArrayXd residuals = y.array() - predicted.array();
    ArrayXd abs_residuals = residuals.abs();

    ArrayXd errors = (abs_residuals <= delta).select(0.5 * residuals.square(), delta * (abs_residuals - 0.5 * delta));

    return errors.matrix();
}

VectorXd calculate_exponential_power_errors(const VectorXd &y, const VectorXd &predicted, double dispersion_parameter)
{
    VectorXd errors{(y.array() - predicted.array()).cwiseAbs().pow(dispersion_parameter)};
    return errors;
}

VectorXd calculate_errors(const VectorXd &y, const VectorXd &predicted, const VectorXd &sample_weight, const std::string &loss_function = "mse",
                          double dispersion_parameter = 1.5, const VectorXi &group = VectorXi(0), const std::set<int> &unique_groups = {}, double quantile = 0.5)
{
    VectorXd errors;
    if (loss_function == "mse")
        errors = calculate_mse_errors(y, predicted);
    else if (loss_function == "binomial")
        errors = calculate_binomial_errors(y, predicted);
    else if (loss_function == "poisson")
        errors = calculate_poisson_errors(y, predicted);
    else if (loss_function == "gamma")
        errors = calculate_gamma_errors(y, predicted);
    else if (loss_function == "tweedie")
        errors = calculate_tweedie_errors(y, predicted, dispersion_parameter);
    else if (loss_function == "group_mse" || loss_function == "group_mse_cycle")
        errors = calculate_group_mse_errors(y, predicted, group, unique_groups, sample_weight);
    else if (loss_function == "mae")
        errors = calculate_absolute_errors(y, predicted);
    else if (loss_function == "quantile")
        errors = calculate_quantile_errors(y, predicted, quantile);
    else if (loss_function == "negative_binomial")
        errors = calculate_negative_binomial_errors(y, predicted, dispersion_parameter);
    else if (loss_function == "cauchy")
        errors = calculate_cauchy_errors(y, predicted, dispersion_parameter);
    else if (loss_function == "weibull")
        errors = calculate_weibull_errors(y, predicted, dispersion_parameter);
    else if (loss_function == "huber")
        errors = calculate_huber_errors(y, predicted, dispersion_parameter);
    else if (loss_function == "exponential_power")
        errors = calculate_exponential_power_errors(y, predicted, dispersion_parameter);

    errors = errors.array() * sample_weight.array();

    return errors;
}

double calculate_mse_error_one_observation(double y, double predicted)
{
    double error{y - predicted};
    error = error * error;
    return error;
}

double calculate_error_one_observation(double y, double predicted, double sample_weight)
{
    double error{calculate_mse_error_one_observation(y, predicted)};

    if (!std::isnan(sample_weight))
        error = error * sample_weight;

    return error;
}

double calculate_mean_error(const VectorXd &errors, const VectorXd &sample_weight)
{
    double error{std::numeric_limits<double>::infinity()};

    if (sample_weight.size() > 0)
        error = errors.sum() / sample_weight.sum();
    else
        error = errors.mean();

    error = set_error_to_infinity_if_invalid(error);

    return error;
}

double calculate_weighted_average(const VectorXd &values, const VectorXd &weights)
{
    if (values.size() != weights.size())
    {
        throw std::runtime_error("Values and weights must have the same size for weighted average calculation.");
    }
    if (values.size() == 0)
    {
        return NAN_DOUBLE;
    }

    double total_weight = weights.sum();
    if (is_approximately_zero(total_weight))
    {
        return NAN_DOUBLE;
    }

    double weighted_sum = (values.array() * weights.array()).sum();
    return weighted_sum / total_weight;
}

double calculate_sum_error(const VectorXd &errors)
{
    double error{errors.sum()};
    error = set_error_to_infinity_if_invalid(error);
    return error;
}

VectorXd calculate_exp_of_linear_predictor_adjusted_for_numerical_problems(const VectorXd &linear_predictor, double min_exponent, double max_exponent)
{
    VectorXd exp_of_linear_predictor{linear_predictor.array().exp()};
    double min_exp_of_linear_predictor{std::exp(min_exponent)};
    double max_exp_of_linear_predictor{std::exp(max_exponent)};
    for (Eigen::Index i = 0; i < linear_predictor.rows(); ++i)
    {
        bool linear_predictor_is_too_small{std::isless(linear_predictor[i], min_exponent)};
        if (linear_predictor_is_too_small)
        {
            exp_of_linear_predictor[i] = min_exp_of_linear_predictor;
            continue;
        }

        bool linear_predictor_is_too_large{std::isgreater(linear_predictor[i], max_exponent)};
        if (linear_predictor_is_too_large)
        {
            exp_of_linear_predictor[i] = max_exp_of_linear_predictor;
        }
    }

    return exp_of_linear_predictor;
}

VectorXd transform_linear_predictor_to_predictions(const VectorXd &linear_predictor, const std::string &link_function = "identity",
                                                   const std::function<VectorXd(VectorXd)> &calculate_custom_transform_linear_predictor_to_predictions_function = {})
{
    if (link_function == "identity")
        return linear_predictor;
    else if (link_function == "logit")
    {
        double min_exponent{-MAX_ABS_EXPONENT_TO_APPLY_ON_LINEAR_PREDICTOR_IN_LOGIT_MODEL};
        double max_exponent{MAX_ABS_EXPONENT_TO_APPLY_ON_LINEAR_PREDICTOR_IN_LOGIT_MODEL};
        VectorXd exp_of_linear_predictor{calculate_exp_of_linear_predictor_adjusted_for_numerical_problems(linear_predictor, min_exponent, max_exponent)};
        VectorXd predictions{exp_of_linear_predictor.array() / (1.0 + exp_of_linear_predictor.array())};
        return predictions;
    }
    else if (link_function == "log")
    {
        double min_exponent{std::numeric_limits<double>::min_exponent10};
        double max_exponent{std::numeric_limits<double>::max_exponent10};
        return calculate_exp_of_linear_predictor_adjusted_for_numerical_problems(linear_predictor, min_exponent, max_exponent);
    }
    else if (link_function == "custom_function")
    {
        try
        {
            return calculate_custom_transform_linear_predictor_to_predictions_function(linear_predictor);
        }
        catch (const std::exception &e)
        {
            std::string error_msg{"Error when executing calculate_custom_transform_linear_predictor_to_predictions_function: " + static_cast<std::string>(e.what())};
            throw std::runtime_error(error_msg);
        }
    }
    return VectorXd(0);
}

VectorXi sort_indexes_ascending(const VectorXd &sort_based_on_me)
{
    VectorXi idx(sort_based_on_me.size());
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&sort_based_on_me](size_t i1, size_t i2)
              { return sort_based_on_me(i1) < sort_based_on_me(i2); });

    return idx;
}

template <typename M>
M load_csv_into_eigen_matrix(const std::string &path)
{
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ','))
        {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size() / rows);
}

void save_as_csv_file(std::string fileName, MatrixXd matrix)
{
    // https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

    std::ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}

template <typename T> // Type must implement a size() method
size_t calculate_max_index_in_vector(T &vector)
{
    return vector.size() - static_cast<size_t>(1);
}

template <typename T> // Type must be an Eigen Matrix or Vector
bool matrix_has_nan_or_infinite_elements(const T &x)
{
    bool has_nan_or_infinite_elements{!x.allFinite()};
    if (has_nan_or_infinite_elements)
        return true;
    else
        return false;
}

template <typename T> // Type must be an Eigen Matrix or Vector
void throw_error_if_matrix_has_nan_or_infinite_elements(const T &x, const std::string &matrix_name)
{
    bool matrix_is_empty{x.size() == 0};
    if (matrix_is_empty)
        return;

    bool has_nan_or_infinite_elements{matrix_has_nan_or_infinite_elements(x)};
    if (has_nan_or_infinite_elements)
    {
        throw std::runtime_error(matrix_name + " has nan or infinite elements.");
    }
}

VectorXi calculate_indicator(const VectorXd &v)
{
    VectorXi indicator{VectorXi::Constant(v.rows(), 1)};
    for (Eigen::Index i = 0; i < v.size(); ++i)
    {
        if (is_approximately_zero(v[i]))
            indicator[i] = 0;
    }
    return indicator;
}

VectorXi calculate_indicator(const VectorXi &v)
{
    VectorXi indicator{VectorXi::Constant(v.rows(), 1)};
    for (Eigen::Index i = 0; i < v.size(); ++i)
    {
        if (v[i] == 0)
            indicator[i] = 0;
    }
    return indicator;
}

std::vector<size_t> sample_indexes_of_vector(int rows_in_underyling_vector, std::mt19937 &mersenne, size_t rows_to_sample)
{
    std::uniform_int_distribution<> distribution{0, rows_in_underyling_vector - 1};
    std::vector<size_t> output(rows_to_sample);
    for (size_t row = 0; row < rows_to_sample; ++row)
    {
        output[row] = distribution(mersenne);
    }
    return output;
}

double trapezoidal_integration(const VectorXd &y, const VectorXd &x)
{
    bool y_is_large_enough{y.rows() > 1};
    bool x_and_y_have_the_same_size{x.rows() == y.rows()};

    double output{NAN_DOUBLE};
    if (y_is_large_enough && x_and_y_have_the_same_size)
    {
        output = 0;
        for (Eigen::Index i = 1; i < y.size(); ++i)
        {
            double delta_y{(y[i] + y[i - 1]) / 2};
            double delta_x{x[i] - x[i - 1]};
            output += delta_y * delta_x;
        }
    }

    return output;
}

VectorXd calculate_weights_if_they_are_not_provided(const VectorXd &y_true, const VectorXd &weights = VectorXd(0))
{
    bool weights_are_not_provided{weights.size() == 0};
    if (weights_are_not_provided)
    {
        return VectorXd::Constant(y_true.size(), 1.0);
    }
    else
        return weights;
}

double calculate_gini(const VectorXd &y_true, const VectorXd &y_pred, const VectorXd &weights = VectorXd(0))
{
    VectorXd weights_used{calculate_weights_if_they_are_not_provided(y_true, weights)};

    VectorXi y_pred_sorted_index{sort_indexes_ascending(y_pred)};

    Eigen::Index normalized_cumsum_vector_rows{y_true.size() + 1};
    VectorXd normalized_cumsum_y_true{VectorXd::Constant(normalized_cumsum_vector_rows, 0.0)};
    VectorXd normalized_cumsum_weights{VectorXd::Constant(normalized_cumsum_vector_rows, 0.0)};
    for (Eigen::Index i = 1; i < normalized_cumsum_vector_rows; ++i)
    {
        normalized_cumsum_y_true[i] += normalized_cumsum_y_true[i - 1] + y_true[y_pred_sorted_index[i - 1]];
        normalized_cumsum_weights[i] += normalized_cumsum_weights[i - 1] + weights_used[y_pred_sorted_index[i - 1]];
    }
    normalized_cumsum_y_true /= y_true.sum();
    normalized_cumsum_weights /= weights_used.sum();

    double gini{1.0 - 2 * trapezoidal_integration(normalized_cumsum_y_true, normalized_cumsum_weights)};

    return gini;
}

std::vector<size_t> remove_duplicate_elements_from_vector(const std::vector<size_t> &vector)
{
    std::vector<size_t> output{vector};
    std::sort(output.begin(), output.end());
    std::vector<size_t>::iterator it;
    it = std::unique(output.begin(), output.end());
    output.resize(distance(output.begin(), it));
    return output;
}

std::vector<double> remove_duplicate_elements_from_vector(const std::vector<double> &vector)
{
    std::vector<double> output{vector};
    std::sort(output.begin(), output.end());
    std::vector<double>::iterator it;
    it = std::unique(output.begin(), output.end());
    output.resize(distance(output.begin(), it));
    return output;
}

double calculate_standard_deviation(const VectorXd &vector, const VectorXd &sample_weight = VectorXd(0))
{
    VectorXd sample_weight_used;
    bool sample_weight_is_provided{sample_weight.size() > 0};
    if (sample_weight_is_provided)
        sample_weight_used = sample_weight / sample_weight.mean();
    else
        sample_weight_used = VectorXd::Constant(vector.rows(), 1.0);
    double sum_weight{sample_weight_used.sum()};
    double weighted_average_of_vector{(vector.array() * sample_weight_used.array()).sum() / sum_weight};
    double variance{(sample_weight_used.array() * (vector.array() - weighted_average_of_vector).pow(2)).sum() / sum_weight};
    double standard_deviation{std::pow(variance, 0.5)};
    return standard_deviation;
}

MatrixXd generate_combinations_and_one_additional_column(const std::vector<std::vector<double>> &vectors)
{
    size_t num_vectors = vectors.size();
    std::vector<size_t> sizes(num_vectors);
    size_t num_rows = 1;

    for (size_t i = 0; i < num_vectors; ++i)
    {
        sizes[i] = vectors[i].size();
        num_rows *= sizes[i];
    }

    MatrixXd result(num_rows, num_vectors + 1);

    for (size_t row = 0; row < num_rows; ++row)
    {
        size_t index = row;
        for (size_t col = 0; col < num_vectors; ++col)
        {
            size_t vec_size = sizes[col];
            result(row, col) = vectors[col][index % vec_size];
            index /= vec_size;
        }
    }
    return result;
}

double calculate_quantile(const VectorXd &vector, double quantile, const VectorXd &sample_weight = VectorXd(0))
{
    if (quantile < 0.0 || quantile > 1.0)
    {
        throw std::runtime_error("Quantile must be between 0.0 and 1.0.");
    }

    const Eigen::Index n = vector.size();
    if (n == 0)
    {
        return NAN_DOUBLE;
    }

    VectorXd sample_weight_used;
    if (sample_weight.size() > 0)
    {
        if (sample_weight.size() != n)
        {
            throw std::runtime_error("Vector and sample_weight must have the same size.");
        }
        sample_weight_used = sample_weight;
    }
    else
    {
        sample_weight_used = VectorXd::Constant(n, 1.0);
    }

    if ((sample_weight_used.array() < 0.0).any())
    {
        throw std::runtime_error("Sample weights must be non-negative.");
    }

    double total_weight = sample_weight_used.sum();
    if (is_approximately_zero(total_weight))
    {
        return NAN_DOUBLE;
    }

    if (n == 1)
    {
        return vector[0];
    }

    std::vector<std::pair<double, double>> weighted_values(n);
    for (Eigen::Index i = 0; i < n; ++i)
    {
        weighted_values[i] = {vector[i], sample_weight_used[i]};
    }

    std::sort(weighted_values.begin(), weighted_values.end(),
              [](const auto &a, const auto &b)
              {
                  return a.first < b.first;
              });

    VectorXd quantile_positions(n);
    double cum_weight = 0.0;
    for (Eigen::Index i = 0; i < n; ++i)
    {
        double current_weight = weighted_values[i].second;
        cum_weight += current_weight;
        quantile_positions[i] = (cum_weight - 0.5 * current_weight) / total_weight;
    }

    auto it = std::upper_bound(quantile_positions.data(), quantile_positions.data() + n, quantile);
    Eigen::Index upper_index = std::distance(quantile_positions.data(), it);

    if (upper_index == 0)
    {
        return weighted_values[0].first;
    }
    if (upper_index >= n)
    {
        return weighted_values[n - 1].first;
    }

    Eigen::Index lower_index = upper_index - 1;

    double q_lower = quantile_positions[lower_index];
    double q_upper = quantile_positions[upper_index];
    double val_lower = weighted_values[lower_index].first;
    double val_upper = weighted_values[upper_index].first;

    if (is_approximately_equal(q_lower, q_upper))
    {
        return val_lower;
    }

    double fraction = (quantile - q_lower) / (q_upper - q_lower);
    return val_lower + fraction * (val_upper - val_lower);
}