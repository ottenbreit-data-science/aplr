#include <iostream>
#include "term.h"
#include "../dependencies/eigen-master/Eigen/Dense"
#include <vector>
#include <numeric>
#include "APLRRegressor.h"
#include <cmath>


using namespace Eigen;

int main()
{
    //Model
    APLRRegressor model{APLRRegressor()};
    model.m=100;
    model.v=1.0;
    model.bins=10;
    model.n_jobs=1;
    model.loss_function_mse=true;
    model.verbosity=3;
    model.max_interaction_level=100;
    model.max_interactions=30;
    model.min_observations_in_split=30;
    model.ineligible_boosting_steps_added=10;
    model.max_eligible_terms=5;

    //Data    
    MatrixXd X_train{load_csv<MatrixXd>("data/X_train.csv")};
    MatrixXd X_test{load_csv<MatrixXd>("data/X_test.csv")}; 
    VectorXd y_train{load_csv<MatrixXd>("data/y_train.csv")};    
    VectorXd y_test{load_csv<MatrixXd>("data/y_test.csv")}; 

    VectorXd sample_weight{VectorXd::Constant(y_train.size(),1.0)};

    std::cout<<X_train;

    //Fitting
    //model.fit(X_train,y_train);
    //model.fit(X_train,y_train,sample_weight);
    model.fit(X_train,y_train,sample_weight,{},{0,1,2,3,4,5,10,static_cast<size_t>(y_train.size()-1)});
    std::cout<<"feature importance\n"<<model.feature_importance<<"\n\n";

    VectorXd predictions{model.predict(X_test)};
    MatrixXd li{model.calculate_local_feature_importance(X_test)};

    //Saving results
    save_data("cpp/data/output.csv",predictions);

    std::cout<<predictions.mean()<<"\n\n";

    //std::cout<<model.validation_error_steps<<"\n\n";
}