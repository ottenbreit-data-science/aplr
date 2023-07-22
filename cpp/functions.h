#pragma once
#include <limits>
#include <numeric>   //std::iota
#include <algorithm> //std::sort, std::stable_sort
#include <vector>
#include <fstream>
#include <iostream>
#include <thread>
#include <future>
#include <random>
#include <set>
#include <map>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "constants.h"

using namespace Eigen;

template <typename TReal>
static bool is_approximately_equal(TReal a, TReal b, TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))
        return true;

    TReal diff = std::fabs(a - b);
    if (diff <= tolerance)
        return true;

    if (diff < std::fmax(std::fabs(a), std::fabs(b)) * tolerance)
        return true;

    return false;
}

template <typename TReal>
static bool is_approximately_zero(TReal a, TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    if (std::fabs(a) <= tolerance)
        return true;
    return false;
}

std::set<std::string> get_unique_strings(const std::vector<std::string> &string_vector)
{
    std::set<std::string> unique_strings{string_vector.begin(), string_vector.end()};
    return unique_strings;
}

std::set<int> get_unique_integers(const VectorXi &int_vector)
{
    std::set<int> unique_integers{int_vector.begin(), int_vector.end()};
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
    std::map<int, size_t> count;
};

GroupData calculate_group_errors_and_count(const VectorXd &y, const VectorXd &predicted, const VectorXi &group, const std::set<int> &unique_groups)
{
    GroupData group_data;
    for (int unique_group_value : unique_groups)
    {
        group_data.error[unique_group_value] = 0.0;
        group_data.count[unique_group_value] = 0;
    }
    for (Eigen::Index i = 0; i < group.size(); ++i)
    {
        group_data.error[group[i]] += y[i] - predicted[i];
        group_data.count[group[i]] += 1;
    }
    return group_data;
}

VectorXd calculate_group_mse_errors(const VectorXd &y, const VectorXd &predicted, const VectorXi &group, const std::set<int> &unique_groups)
{
    GroupData group_residuals_and_count{calculate_group_errors_and_count(y, predicted, group, unique_groups)};

    for (int unique_group_value : unique_groups)
    {
        group_residuals_and_count.error[unique_group_value] *= group_residuals_and_count.error[unique_group_value];
        group_residuals_and_count.error[unique_group_value] /= group_residuals_and_count.count[unique_group_value];
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
    VectorXd errors{dispersion_parameter * predicted.array().log() + (1 - dispersion_parameter) * y.array().log() +
                    (y.array() / predicted.array()).pow(dispersion_parameter)};

    return errors;
}

VectorXd calculate_errors(const VectorXd &y, const VectorXd &predicted, const VectorXd &sample_weight = VectorXd(0), const std::string &loss_function = "mse",
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
    else if (loss_function == "group_mse")
        errors = calculate_group_mse_errors(y, predicted, group, unique_groups);
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

    if (sample_weight.size() > 0)
        errors = errors.array() * sample_weight.array();

    return errors;
}

double calculate_mse_error_one_observation(double y, double predicted)
{
    double error{y - predicted};
    error = error * error;
    return error;
}

double calculate_error_one_observation(double y, double predicted, double sample_weight = NAN_DOUBLE)
{
    double error{calculate_mse_error_one_observation(y, predicted)};

    if (!std::isnan(sample_weight))
        error = error * sample_weight;

    return error;
}

double calculate_mean_error(const VectorXd &errors, const VectorXd &sample_weight = VectorXd(0))
{
    double error{std::numeric_limits<double>::infinity()};

    if (sample_weight.size() > 0)
        error = errors.sum() / sample_weight.sum();
    else
        error = errors.mean();

    error = set_error_to_infinity_if_invalid(error);

    return error;
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

struct DistributedIndices
{
    std::vector<size_t> index_lowest;
    std::vector<size_t> index_highest;
};

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

double calculate_rankability(const VectorXd &y_true, const VectorXd &y_pred, const VectorXd &weights = VectorXd(0),
                             uint_fast32_t random_state = std::numeric_limits<uint_fast32_t>::lowest(), size_t bootstraps = 10000)
{
    bool weights_are_provided{weights.size() == y_true.size()};
    std::mt19937 mersenne{random_state};
    std::vector<size_t> first_index_in_pair{sample_indexes_of_vector(y_true.size(), mersenne, bootstraps)};
    std::vector<size_t> second_index_in_pair{sample_indexes_of_vector(y_true.size(), mersenne, bootstraps)};
    double num_pairs{0.0};
    double num_ranked_correctly{0.0};
    if (weights_are_provided)
    {
        for (size_t i = 0; i < first_index_in_pair.size(); ++i)
        {
            bool first_item_in_pair_has_higher_response_than_second_item{std::isgreater(y_true[first_index_in_pair[i]], y_true[second_index_in_pair[i]])};
            if (first_item_in_pair_has_higher_response_than_second_item)
            {
                double weight = (weights[first_index_in_pair[i]] + weights[second_index_in_pair[i]]) / 2;
                num_pairs += weight;
                bool prediction_is_also_higher{std::isgreater(y_pred[first_index_in_pair[i]], y_pred[second_index_in_pair[i]])};
                bool predictions_are_equal(is_approximately_equal(y_pred[first_index_in_pair[i]], y_pred[second_index_in_pair[i]]));
                if (prediction_is_also_higher)
                {
                    num_ranked_correctly += weight;
                }
                else if (predictions_are_equal)
                {
                    num_ranked_correctly += weight * 0.5;
                }
            }
        }
    }
    else
    {
        for (size_t i = 0; i < first_index_in_pair.size(); ++i)
        {
            bool first_item_in_pair_has_higher_response_than_second_item{std::isgreater(y_true[first_index_in_pair[i]], y_true[second_index_in_pair[i]])};
            if (first_item_in_pair_has_higher_response_than_second_item)
            {
                num_pairs += 1;
                bool prediction_is_also_higher{std::isgreater(y_pred[first_index_in_pair[i]], y_pred[second_index_in_pair[i]])};
                bool predictions_are_equal(is_approximately_equal(y_pred[first_index_in_pair[i]], y_pred[second_index_in_pair[i]]));
                if (prediction_is_also_higher)
                {
                    num_ranked_correctly += 1;
                }
                else if (predictions_are_equal)
                {
                    num_ranked_correctly += 0.5;
                }
            }
        }
    }
    double rankability{num_ranked_correctly / num_pairs};
    bool rankability_is_invalid{!std::isfinite(rankability)};
    if (rankability_is_invalid)
        rankability = 0.5;

    return rankability;
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