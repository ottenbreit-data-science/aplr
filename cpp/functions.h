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

//implements relative method - do not use for comparing with zero
//use this most of the time, tolerance needs to be meaningful in your context
template<typename TReal>
static bool check_if_approximately_equal(TReal a, TReal b, TReal tolerance = std::numeric_limits<TReal>::epsilon())
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

//supply tolerance that is meaningful in your context
//for example, default tolerance may not work if you are comparing double with float
template<typename TReal>
static bool check_if_approximately_zero(TReal a, TReal tolerance = std::numeric_limits<TReal>::epsilon())
{
    if (std::fabs(a) <= tolerance)
        return true;
    return false;
}

//Computes errors (for each observation) based on error metric for a vector
VectorXd calculate_errors(const VectorXd &y,const VectorXd &predicted,const VectorXd &sample_weight=VectorXd(0),bool loss_function_mse=true)
{   
    //Error per observation before adjustment for sample weights
    VectorXd residuals{y-predicted};
    if(loss_function_mse)
        residuals=residuals.array()*residuals.array();
    else
        residuals=residuals.cwiseAbs();

    //Adjusting for sample weights if specified
    if(sample_weight.size()>0)
        residuals=residuals.array()*sample_weight.array();
    
    return residuals;
}

//Computes error for one observation based on error metric
double calculate_error_one_observation(double y,double predicted,double sample_weight=NAN_DOUBLE,bool loss_function_mse=true)
{   
    //Error per observation before adjustment for sample weights
    double residual{y-predicted};
    if(loss_function_mse)
        residual=residual*residual;
    else
        residual=abs(residual);

    //Adjusting for sample weights if specified
    if(!std::isnan(sample_weight))
        residual=residual*sample_weight;
    
    return residual;
}

//Computes overall error based on errors from calculate_errors(), returning one value
double calculate_error(const VectorXd &errors,const VectorXd &sample_weight=VectorXd(0))
{   
    double error{std::numeric_limits<double>::infinity()};

    //Adjusting for sample weights if specified
    if(sample_weight.size()>0)
        error=errors.sum()/sample_weight.sum();
    else
        error=errors.mean();
    
    return error;
}

//sorts index based on v
VectorXi sort_indexes_ascending(const VectorXd &v)
{
    // initialize original index locations
    VectorXi idx(v.size());
    std::iota(idx.begin(),idx.end(),0);

    // sort indexes based on comparing values in v
    std::sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v(i1) < v(i2);});

    return idx;
}

//Loads a csv file into an Eigen matrix
template<typename M>
M load_csv (const std::string &path) {
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

//Saves an Eigen matrix as a csv file
void save_data(std::string fileName, MatrixXd matrix)
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

//For multicore distribution of elements
struct DistributedIndices
{
    std::vector<size_t> index_lowest;
    std::vector<size_t> index_highest; 
};

//Distribution of elements to multiple cores
template <typename T> //type must implement a size() method
DistributedIndices distribute_to_indices(T &collection,size_t n_jobs)
{
    size_t collection_size=static_cast<size_t>(collection.size());

    //Initializing output
    DistributedIndices output;
    output.index_lowest.reserve(collection_size);
    output.index_highest.reserve(collection_size);

    //Determining how many items to evaluate per core
    size_t available_cores{static_cast<size_t>(std::thread::hardware_concurrency())};
    if(n_jobs>1)
        available_cores=std::min(n_jobs,available_cores);
    size_t units_per_core{std::max(collection_size/available_cores,static_cast<size_t>(1))};

    //For each set of items going into one core
    for (size_t i = 0; i < collection_size; i=i+units_per_core) 
    {                
        output.index_lowest.push_back(i); 
    }
    for (size_t i = 0; i < output.index_lowest.size()-1; ++i)
    {
        output.index_highest.push_back(output.index_lowest[i+1]-1);
    }
    output.index_highest.push_back(collection_size-1);
    //Removing last bunch and adjusting the second last if necessary
    if(output.index_lowest.size()>available_cores) 
    {
        output.index_lowest.pop_back();
        output.index_highest.pop_back();
        output.index_highest[output.index_highest.size()-1]=collection_size-1;
    }

    return output;
}

template <typename T> //type must implement a size() method
size_t calculate_max_index_in_vector(T &vector)
{
    return vector.size()-static_cast<size_t>(1);
}