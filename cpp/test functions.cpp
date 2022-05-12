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
    bool equal_inf_left{check_if_approximately_equal(inf_left,inf_left)};
    bool equal_inf_right{check_if_approximately_equal(inf_right,inf_right)};
    bool equal_inf_diff{!check_if_approximately_equal(inf_left,inf_right)};
    bool equal_inf_diff2{!check_if_approximately_equal(inf_right,inf_left)};
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
    VectorXd errors_mae{calculate_errors(y,pred,VectorXd(0),false)};
    VectorXd errors_mse_sw{calculate_errors(y,pred,sample_weight)};
    VectorXd errors_mae_sw{calculate_errors(y,pred,sample_weight,false)};

    //compute_error
    //calculating errors
    double error_mse{calculate_error(errors_mse)};
    std::cout<<"error_mse: "<<error_mse<<"\n\n";   
    tests.push_back((check_if_approximately_equal(error_mse,3.482)?true:false));
    double error_mae{calculate_error(errors_mae)};
    std::cout<<"error_mae: "<<error_mae<<"\n\n";   
    tests.push_back((check_if_approximately_equal(error_mae,1.18)?true:false)); 
    double error_mse_sw{calculate_error(errors_mse_sw,sample_weight)};
    std::cout<<"error_mse_sw: "<<error_mse_sw<<"\n\n";   
    tests.push_back((check_if_approximately_equal(error_mse_sw,0.4433,0.0001)?true:false));
    double error_mae_sw{calculate_error(errors_mae_sw,sample_weight)};
    std::cout<<"error_mae_sw: "<<error_mae_sw<<"\n\n";   
    tests.push_back((check_if_approximately_equal(error_mae_sw,0.5666,0.0001)?true:false));
 
    //Test summary
    std::cout<<"Test summary\n\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}