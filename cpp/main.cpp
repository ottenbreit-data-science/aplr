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
    model.v=0.5;
    model.bins=300;
    model.n_jobs=0;
    model.loss_function_mse=true;
    model.verbosity=3;
    model.min_observations_in_split=30;
    //model.max_interaction_level=0;
    model.max_interaction_level=100;
    model.max_interactions=30;
    model.ineligible_boosting_steps_added=10;
    model.max_eligible_terms=5;

    //Data    
    MatrixXd X_train{load_csv<MatrixXd>("X_train.csv")};
    MatrixXd X_test{load_csv<MatrixXd>("X_test.csv")}; 
    VectorXd y_train{load_csv<MatrixXd>("y_train.csv")};    
    VectorXd y_test{load_csv<MatrixXd>("y_test.csv")}; 

    VectorXd sample_weight{VectorXd::Constant(y_train.size(),1.0)};
    //VectorXd sample_weight{VectorXd::Random(y_train.size()).cwiseAbs()};

    //Fitting
    clock_t time_req{clock()};
    //model.fit(X_train,y_train);
    model.fit(X_train,y_train,sample_weight);
    time_req=clock()-time_req;
    std::cout<<"time elapsed: "<<std::to_string(time_req)<<"\n\n";
    
    VectorXd predictions{model.predict(X_test)};

    //Saving results
    save_data("output.csv",predictions);
    std::cout<<"min validation_error "<<model.validation_error_steps.minCoeff()<<"\n\n";
    std::cout<<check_if_approximately_equal(model.validation_error_steps.minCoeff(),6.39607,0.00001)<<"\n";

    std::cout<<"mean prediction "<<predictions.mean()<<"\n\n";
    std::cout<<check_if_approximately_equal(predictions.mean(),23.7461,0.0001)<<"\n";

    std::cout<<"best_m: "<<model.m<<"\n";

    std::cout<<"test";
}