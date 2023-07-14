#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
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

    VectorXd y_true(3);
    VectorXd weights_equal(3);
    VectorXd weights_different(3);
    VectorXd y_pred_good(3);
    VectorXd y_pred_bad(3);
    VectorXd y_pred_equal(3);
    y_true<<1.0, 2.0, 3.0;
    weights_equal<<1, 1, 1;
    weights_different<<0, 0.5, 0.75;
    y_pred_good<<-1.0, 2.0, 4.0;
    y_pred_bad<<4.0, 3.0, -1.0;
    double rankability_good_ew{calculate_rankability(y_true,y_pred_good,weights_equal)};
    double rankability_bad_ew{calculate_rankability(y_true,y_pred_bad,weights_equal)};
    double rankability_equal_ew{calculate_rankability(y_true,y_pred_equal,weights_equal)};
    double rankability_good_dw{calculate_rankability(y_true,y_pred_good,weights_different)};
    double rankability_bad_dw{calculate_rankability(y_true,y_pred_bad,weights_different)};
    double rankability_equal_dw{calculate_rankability(y_true,y_pred_equal,weights_different)};
    tests.push_back(is_approximately_equal(rankability_good_ew,1.0));
    tests.push_back(is_approximately_equal(rankability_bad_ew,0.0));
    tests.push_back(is_approximately_equal(rankability_equal_ew,0.5));
    tests.push_back(is_approximately_equal(rankability_good_dw,1.0));
    tests.push_back(is_approximately_equal(rankability_bad_dw,0.0));
    tests.push_back(is_approximately_equal(rankability_equal_dw,0.5));

    VectorXd y_integration(3);
    VectorXd x_integration(3);
    y_integration<<1,2,3;
    x_integration<<4,6,8;
    double integration{trapezoidal_integration(y_integration,x_integration)};
    tests.push_back(is_approximately_equal(integration,8.0));

    VectorXd weights_none{VectorXd(0)};
    VectorXd calculated_weights_if_not_provided{calculate_weights_if_they_are_not_provided(y_true)};
    VectorXd calculated_weights_if_provided{calculate_weights_if_they_are_not_provided(y_true,weights_different)};
    tests.push_back(calculated_weights_if_not_provided==weights_equal);
    tests.push_back(calculated_weights_if_provided==weights_different);

    VectorXd y_pred(3);
    VectorXd weights_gini(3);
    y_pred<<1.0,3.0,2.0;
    weights_gini<<0.2,0.5,0.3;
    double gini{calculate_gini(y_true,y_pred,weights_gini)};
    tests.push_back(is_approximately_equal(gini,-0.1166667,0.0000001));

    VectorXi int_vector(3);
    int_vector<<1,1,2;
    std::set<int> unique_integers{get_unique_integers(int_vector)};
    bool size_is_correct{unique_integers.size()==2};
    tests.push_back(size_is_correct);

    //Test summary
    std::cout<<"Test summary\n\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}