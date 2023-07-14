#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "APLRRegressor.h"
#include "term.h"


using namespace Eigen;

double calculate_custom_loss(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group)
{
    VectorXd error{(y.array()-predictions.array()).pow(2)};
    return error.mean();
}

VectorXd calculate_custom_negative_gradient(const VectorXd &y, const VectorXd &predictions, const VectorXi &group)
{
    VectorXd negative_gradient{y-predictions};
    return negative_gradient;
}

double calculate_custom_validation_error(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group)
{
    VectorXd error{(y.array()-predictions.array()).pow(3)};
    return error.mean();
}

int main()
{
    std::vector<bool> tests;
    tests.reserve(1000);

    //Model
    APLRRegressor model{APLRRegressor()};
    model.m=100;
    model.v=1.0;
    model.bins=10;
    model.n_jobs=1;
    model.loss_function="custom_function";
    model.calculate_custom_loss_function=calculate_custom_loss;
    model.calculate_custom_negative_gradient_function=calculate_custom_negative_gradient;
    model.calculate_custom_validation_error_function=calculate_custom_validation_error;
    model.verbosity=3;
    model.max_interaction_level=100;
    model.max_interactions=30;
    model.min_observations_in_split=50;
    model.ineligible_boosting_steps_added=10;
    model.max_eligible_terms=5;
    model.validation_tuning_metric="custom_function";

    //Data    
    MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
    MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")}; 
    VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};    
    VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")}; 

    VectorXd sample_weight{VectorXd::Constant(y_train.size(),1.0)};

    std::cout<<X_train;

    //Fitting
    //model.fit(X_train,y_train);
    //model.fit(X_train,y_train,sample_weight);
    //model.fit(X_train,y_train,sample_weight,{},{0,1,2,3,4,5,10,static_cast<size_t>(y_train.size()-1)});
    std::vector<size_t> validation_indexes{0,1,2,3,4,5,10,static_cast<size_t>(y_train.size()-1)};
    std::vector<size_t> prioritized_predictor_indexes{1,8};
    model.fit(X_train,y_train,sample_weight,{},validation_indexes,prioritized_predictor_indexes);
    model.fit(X_train,y_train,sample_weight,{},validation_indexes,prioritized_predictor_indexes);
    std::cout<<"feature importance\n"<<model.feature_importance<<"\n\n";

    VectorXd predictions{model.predict(X_test)};
    MatrixXd li{model.calculate_local_feature_importance(X_test)};

    //Saving results
    save_as_csv_file("data/output.csv",predictions);

    std::cout<<predictions.mean()<<"\n\n";
    tests.push_back(is_approximately_equal(predictions.mean(),23.6854,0.00001));

    std::vector<size_t> validation_indexes_from_model{model.get_validation_indexes()};
    bool validation_indexes_from_model_are_correct{validation_indexes_from_model == validation_indexes};
    tests.push_back(validation_indexes_from_model_are_correct);

    //Test summary
    std::cout<<"\n\nTest summary\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}