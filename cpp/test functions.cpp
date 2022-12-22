#include <iostream>
#include "../dependencies/eigen-master/Eigen/Dense"
#include <vector>
#include <numeric>
#include <cmath>
#include "functions.h"

using namespace Eigen;

int main()
{
    std::vector<bool> tests;
    tests.reserve(1000);

    //isapproximatelyequal
    double inf_left{-std::numeric_limits<double>::infinity()};
    double inf_right{std::numeric_limits<double>::infinity()};
    bool equal_inf_left{is_approximately_equal(inf_left,inf_left)};
    bool equal_inf_right{is_approximately_equal(inf_right,inf_right)};
    bool equal_inf_diff{!is_approximately_equal(inf_left,inf_right)};
    bool equal_inf_diff2{!is_approximately_equal(inf_right,inf_left)};
    tests.push_back(equal_inf_left);
    tests.push_back(equal_inf_right);
    tests.push_back(equal_inf_diff);
    tests.push_back(equal_inf_diff2);

    //compute_errors
    VectorXd y(5),pred(5),sample_weight(5);
    y<<1,3.3,2,4,0;
    pred<<1.4,3.3,1.5,0,1; 
    sample_weight<<0.5,0.5,1,0,1;
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"pred\n"<<pred<<"\n\n";
    std::cout<<"sample_weight\n"<<sample_weight<<"\n\n";
    VectorXd errors_mse{calculate_errors(y,pred)};
    VectorXd errors_mse_sw{calculate_errors(y,pred,sample_weight)};

    //compute_error
    //calculating errors
    double error_mse{calculate_mean_error(errors_mse)};
    std::cout<<"error_mse: "<<error_mse<<"\n\n";   
    tests.push_back((is_approximately_equal(error_mse,3.482)?true:false));
    double error_mse_sw{calculate_mean_error(errors_mse_sw,sample_weight)};
    std::cout<<"error_mse_sw: "<<error_mse_sw<<"\n\n";   
    tests.push_back((is_approximately_equal(error_mse_sw,0.4433,0.0001)?true:false));

    //calculate_rolling_centered_mean
    VectorXi sorted_index{sort_indexes_ascending(y)};
    VectorXd y_sorted(5);
    VectorXd sample_weight_sorted(5);
    for (size_t i = 0; i < y_sorted.size(); ++i)
    {
        y_sorted[i]=y[sorted_index[i]];
        sample_weight_sorted[i]=sample_weight[sorted_index[i]];
    }

    VectorXd c1{calculate_rolling_centered_mean(y,sorted_index,1)};
    std::cout<<"rolling_centered_mean_1: "<<c1.mean()<<"\n\n";
    bool test_passed{c1.isApprox(y)};
    tests.push_back(test_passed);

    VectorXd c2{calculate_rolling_centered_mean(y,sorted_index,1,sample_weight)};
    std::cout<<"rolling_centered_mean_2: "<<c2.mean()<<"\n\n";
    test_passed=c2.isApprox(y);
    tests.push_back(test_passed);

    VectorXd c3{calculate_rolling_centered_mean(y,sorted_index,100)};
    std::cout<<"rolling_centered_mean_3: "<<c3.mean()<<"\n\n";
    test_passed=is_approximately_equal(c3.mean(),y.mean());
    tests.push_back(test_passed);

    VectorXd c4{calculate_rolling_centered_mean(y,sorted_index,100,sample_weight)};
    std::cout<<"rolling_centered_mean_4: "<<c4.mean()<<"\n\n";
    double correct_weighted_average{(y.array()*sample_weight.array()).sum() / sample_weight.sum()};
    test_passed=is_approximately_equal(c4.mean(),correct_weighted_average);
    tests.push_back(test_passed);

    VectorXd c5{calculate_rolling_centered_mean(y,sorted_index,4)};
    std::cout<<"rolling_centered_mean_5: "<<c5.mean()<<"\n\n";
    VectorXd correct_weighted_average_vector(5);
    correct_weighted_average_vector[0]=(y_sorted[0]+y_sorted[1]+y_sorted[2])/3;
    correct_weighted_average_vector[1]=(y_sorted[2]+y_sorted[3]+y_sorted[4])/3;
    correct_weighted_average_vector[2]=(y_sorted[1]+y_sorted[2]+y_sorted[3])/3;
    correct_weighted_average_vector[3]=(y_sorted[3]+y_sorted[4])/2;
    correct_weighted_average_vector[4]=(y_sorted[0]+y_sorted[1])/2;
    test_passed=c5.isApprox(correct_weighted_average_vector);
    tests.push_back(test_passed);

    VectorXd c6{calculate_rolling_centered_mean(y,sorted_index,4,sample_weight)};
    std::cout<<"rolling_centered_mean_6: "<<c6.mean()<<"\n\n";
    correct_weighted_average_vector[0]=(y_sorted[0]*sample_weight_sorted[0] + y_sorted[1]*sample_weight_sorted[1] + y_sorted[2]*sample_weight_sorted[2]) / (sample_weight_sorted[0]+sample_weight_sorted[1]+sample_weight_sorted[2]);
    correct_weighted_average_vector[1]=(y_sorted[2]*sample_weight_sorted[2] + y_sorted[3]*sample_weight_sorted[3] + y_sorted[4]*sample_weight_sorted[4]) / (sample_weight_sorted[2]+sample_weight_sorted[3]+sample_weight_sorted[4]);
    correct_weighted_average_vector[2]=(y_sorted[1]*sample_weight_sorted[1] + y_sorted[2]*sample_weight_sorted[2] + y_sorted[3]*sample_weight_sorted[3]) / (sample_weight_sorted[1]+sample_weight_sorted[2]+sample_weight_sorted[3]);
    correct_weighted_average_vector[3]=(y_sorted[3]*sample_weight_sorted[3] + y_sorted[4]*sample_weight_sorted[4] )/ (sample_weight_sorted[3]+sample_weight_sorted[4]);
    correct_weighted_average_vector[4]=(y_sorted[0]*sample_weight_sorted[0] + y_sorted[1]*sample_weight_sorted[1]) / (sample_weight_sorted[0]+sample_weight_sorted[1]);
    test_passed=c6.isApprox(correct_weighted_average_vector);
    tests.push_back(test_passed);

    //testing for nan and infinity
    //matrix without nan or inf
    bool matrix_has_nan_or_inf_elements{matrix_has_nan_or_infinite_elements(y)};    
    tests.push_back(!matrix_has_nan_or_inf_elements?true:false);

    VectorXd inf(5);
    inf<<1.0, 0.2, std::numeric_limits<double>::infinity(), 0.0, 0.5;
    matrix_has_nan_or_inf_elements = matrix_has_nan_or_infinite_elements(inf);
    tests.push_back(matrix_has_nan_or_inf_elements?true:false);

    VectorXd nan(5);
    nan<<1.0, 0.2, NAN_DOUBLE, 0.0, 0.5;
    matrix_has_nan_or_inf_elements = matrix_has_nan_or_infinite_elements(nan);
    tests.push_back(matrix_has_nan_or_inf_elements?true:false);
 
    //Test summary
    std::cout<<"Test summary\n\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}