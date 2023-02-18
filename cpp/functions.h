#pragma once
#include <limits>
#include "../dependencies/eigen-master/Eigen/Dense"
#include <numeric> //std::iota
#include <algorithm> //std::sort, std::stable_sort
#include <vector>
#include <fstream>
#include <iostream>
#include <thread>
#include <future>
#include "constants.h"

using namespace Eigen;

template<typename TReal>
static bool is_approximately_equal(TReal a, TReal b, TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    if(std::isinf(a) && std::isinf(b) && std::signbit(a)==std::signbit(b))
        return true;

    TReal diff = std::fabs(a - b);
    if (diff <= tolerance)
        return true;

    if (diff < std::fmax(std::fabs(a), std::fabs(b)) * tolerance)
        return true;

    return false;
}

template<typename TReal>
static bool is_approximately_zero(TReal a, TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    if (std::fabs(a) <= tolerance)
        return true;
    return false;
}

double set_error_to_infinity_if_invalid(double error)
{
    bool error_is_invalid{!std::isfinite(error)};
    if(error_is_invalid)
        error=std::numeric_limits<double>::infinity();
    
    return error;    
}

VectorXd calculate_gaussian_errors(const VectorXd &y,const VectorXd &predicted)
{
    VectorXd errors{y-predicted};
    errors=errors.array()*errors.array();
    return errors;
}

VectorXd calculate_binomial_errors(const VectorXd &y,const VectorXd &predicted)
{
    VectorXd errors{-y.array() * predicted.array().log()  -  (1.0-y.array()).array() * (1.0-predicted.array()).log()};
    return errors;
}

VectorXd calculate_poisson_errors(const VectorXd &y,const VectorXd &predicted)
{
    VectorXd errors{predicted.array() - y.array()*predicted.array().log()};
    return errors;
}

VectorXd calculate_gamma_errors(const VectorXd &y,const VectorXd &predicted)
{
    VectorXd errors{predicted.array().log() + y.array()/predicted.array()-1};
    return errors;
}

VectorXd calculate_tweedie_errors(const VectorXd &y,const VectorXd &predicted,double tweedie_power=1.5)
{
    VectorXd errors{-y.array()*predicted.array().pow(1-tweedie_power) / (1-tweedie_power) + predicted.array().pow(2-tweedie_power) / (2-tweedie_power)};
    return errors;
}

VectorXd calculate_errors(const VectorXd &y,const VectorXd &predicted,const VectorXd &sample_weight=VectorXd(0),const std::string &family="gaussian",double tweedie_power=1.5)
{   
    VectorXd errors;
    if(family=="gaussian")
        errors=calculate_gaussian_errors(y,predicted);
    else if(family=="binomial")
        errors=calculate_binomial_errors(y,predicted);
    else if(family=="poisson")
        errors=calculate_poisson_errors(y,predicted);
    else if(family=="gamma")
        errors=calculate_gamma_errors(y,predicted);
    else if(family=="tweedie")
        errors=calculate_tweedie_errors(y,predicted,tweedie_power);
    
    if(sample_weight.size()>0)
        errors=errors.array()*sample_weight.array();
    
    return errors;
}

double calculate_gaussian_error_one_observation(double y,double predicted)
{
    double error{y-predicted};
    error=error*error;
    return error;
}

double calculate_error_one_observation(double y,double predicted,double sample_weight=NAN_DOUBLE)
{   
    double error{calculate_gaussian_error_one_observation(y,predicted)};    
    
    if(!std::isnan(sample_weight))
        error=error*sample_weight;

    return error;
}

double calculate_mean_error(const VectorXd &errors,const VectorXd &sample_weight=VectorXd(0))
{   
    double error{std::numeric_limits<double>::infinity()};

    if(sample_weight.size()>0)
        error=errors.sum()/sample_weight.sum();
    else
        error=errors.mean();

    error=set_error_to_infinity_if_invalid(error);
 
    return error;
}

double calculate_sum_error(const VectorXd &errors)
{   
    double error{errors.sum()};
    error=set_error_to_infinity_if_invalid(error);  
    return error;
}

VectorXd calculate_exp_of_linear_predictor_adjusted_for_numerical_problems(const VectorXd &linear_predictor, double min_exponent, double max_exponent)
{
    VectorXd exp_of_linear_predictor{linear_predictor.array().exp()};
    double min_exp_of_linear_predictor{std::exp(min_exponent)};
    double max_exp_of_linear_predictor{std::exp(max_exponent)};
    for (size_t i = 0; i < static_cast<size_t>(linear_predictor.rows()); ++i)
    {            
        bool linear_predictor_is_too_small{std::isless(linear_predictor[i], min_exponent)};
        if(linear_predictor_is_too_small)
        {
            exp_of_linear_predictor[i] = min_exp_of_linear_predictor;
            continue;
        }

        bool linear_predictor_is_too_large{std::isgreater(linear_predictor[i], max_exponent)};
        if(linear_predictor_is_too_large)
        {
            exp_of_linear_predictor[i] = max_exp_of_linear_predictor;
        }

    }

    return exp_of_linear_predictor;
}

VectorXd transform_linear_predictor_to_predictions(const VectorXd &linear_predictor, const std::string &link_function="identity", double tweedie_power=1.5)
{
    if(link_function=="identity")
        return linear_predictor;
    else if(link_function=="logit")
    {
        double min_exponent{-MAX_ABS_EXPONENT_TO_APPLY_ON_LINEAR_PREDICTOR_IN_LOGIT_MODEL};
        double max_exponent{MAX_ABS_EXPONENT_TO_APPLY_ON_LINEAR_PREDICTOR_IN_LOGIT_MODEL};
        VectorXd exp_of_linear_predictor{calculate_exp_of_linear_predictor_adjusted_for_numerical_problems(linear_predictor, min_exponent, max_exponent)};
        VectorXd predictions{exp_of_linear_predictor.array() / (1.0 + exp_of_linear_predictor.array())};
        return predictions;
    }
    else if(link_function=="log")
    {
        double min_exponent{std::numeric_limits<double>::min_exponent10};
        double max_exponent{std::numeric_limits<double>::max_exponent10};
        return calculate_exp_of_linear_predictor_adjusted_for_numerical_problems(linear_predictor, min_exponent, max_exponent);
    }
    return VectorXd(0);
}

VectorXi sort_indexes_ascending(const VectorXd &sort_based_on_me)
{
    VectorXi idx(sort_based_on_me.size());
    std::iota(idx.begin(),idx.end(),0);

    std::sort(idx.begin(), idx.end(),[&sort_based_on_me](size_t i1, size_t i2) {return sort_based_on_me(i1) < sort_based_on_me(i2);});

    return idx;
}

template<typename M>
M load_csv_into_eigen_matrix (const std::string &path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

void save_as_csv_file(std::string fileName, MatrixXd matrix)
{
    //https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
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

template <typename T> //type must implement a size() method
size_t calculate_max_index_in_vector(T &vector)
{
    return vector.size()-static_cast<size_t>(1);
}

template <typename T> //type must be an Eigen Matrix or Vector
bool matrix_has_nan_or_infinite_elements(const T &x)
{
    bool has_nan_or_infinite_elements{!x.allFinite()};
    if(has_nan_or_infinite_elements)
        return true;
    else
        return false;
}

template <typename T> //type must be an Eigen Matrix or Vector
void throw_error_if_matrix_has_nan_or_infinite_elements(const T &x, const std::string &matrix_name)
{
    bool matrix_is_empty{x.size()==0};
    if(matrix_is_empty) return;

    bool has_nan_or_infinite_elements{matrix_has_nan_or_infinite_elements(x)};
    if(has_nan_or_infinite_elements)
    {
        throw std::runtime_error(matrix_name + " has nan or infinite elements.");
    }
}

VectorXd calculate_rolling_centered_mean(const VectorXd &vector, const VectorXi &sorted_index, size_t rolling_window, const VectorXd &sample_weight=VectorXd(0))
{
    bool sample_weight_is_provided{sample_weight.rows()==vector.rows()};
    bool rolling_window_contains_one_observation{rolling_window<=1};
    bool rolling_window_encompasses_all_observations_in_validation_set{rolling_window >= static_cast<size_t>(vector.rows())};
    size_t half_rolling_window{(rolling_window-1)/2};
    
    VectorXd rolling_centered_mean;
    if(rolling_window_contains_one_observation)
        rolling_centered_mean = vector;
    else if(rolling_window_encompasses_all_observations_in_validation_set)
    {
        if(sample_weight_is_provided)
        {
            double weighted_centered_mean{(vector.array() * sample_weight.array()).sum() / sample_weight.sum()};
            rolling_centered_mean = VectorXd::Constant(vector.rows(),weighted_centered_mean);
        }
        else
            rolling_centered_mean = VectorXd::Constant(vector.rows(),vector.mean());
    }
    else
    {
        rolling_centered_mean = VectorXd::Constant(vector.rows(),0);

        size_t vector_size{static_cast<size_t>(sorted_index.rows())};
        for (size_t i = 0; i < vector_size; ++i)
        {
            size_t min_index;
            if(i<half_rolling_window)
                min_index=0;
            else
                min_index=i-half_rolling_window;
            
            size_t max_index{std::min(vector_size-1, i+half_rolling_window)};

            double rolling_centered_weighted_sum{0};
            if(sample_weight_is_provided)
            {
                double rolling_centered_sample_weight_sum{0};
                for (size_t j = min_index; j <= max_index; ++j)
                {
                    rolling_centered_weighted_sum += vector[sorted_index[j]] * sample_weight[sorted_index[j]];
                    rolling_centered_sample_weight_sum += sample_weight[sorted_index[j]];
                }
                rolling_centered_mean[sorted_index[i]] = rolling_centered_weighted_sum / rolling_centered_sample_weight_sum;
            }
            else
            {
                size_t observations{max_index-min_index+1};
                for (size_t j = min_index; j <= max_index; ++j)
                {
                    rolling_centered_mean[sorted_index[i]] += vector[sorted_index[j]];
                }
                rolling_centered_mean[sorted_index[i]] /= observations;
            }
        }
    }
    
    return rolling_centered_mean;
}

VectorXi calculate_indicator(const VectorXd &v)
{
    VectorXi indicator{VectorXi::Constant(v.rows(),1)};
    for (size_t i = 0; i < static_cast<size_t>(v.size()); ++i)
    {
        if(is_approximately_zero(v[i]))
            indicator[i]=0;
    }
    return indicator;
}

VectorXi calculate_indicator(const VectorXi &v)
{
    VectorXi indicator{VectorXi::Constant(v.rows(),1)};
    for (size_t i = 0; i < static_cast<size_t>(v.size()); ++i)
    {
        if(v[i]==0)
            indicator[i]=0;
    }
    return indicator;
}