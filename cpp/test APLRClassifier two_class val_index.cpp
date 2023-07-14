#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "APLRClassifier.h"


using namespace Eigen;

int main()
{
    std::vector<bool> tests;
    tests.reserve(1000);

    //Model
    APLRClassifier model{APLRClassifier()};
    model.m=100;
    model.v=0.5;
    model.bins=300;
    model.n_jobs=0;
    model.verbosity=3;
    model.max_interaction_level=0;
    model.max_interactions=1000;
    model.min_observations_in_split=20;
    model.ineligible_boosting_steps_added=10;
    model.max_eligible_terms=5;

    //Data    
    MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
    MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")}; 
    VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_logit.csv")};    
    VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_logit.csv")};
    std::vector<std::string> y_train_str(y_train.rows());
    std::vector<std::string> y_test_str(y_test.rows());
    VectorXd sample_weight{VectorXd::Constant(y_train.size(),1.0)};

    for (Eigen::Index i = 0; i < y_train.size(); ++i)
    {
        y_train_str[i] = std::to_string(y_train[i]);
    }
    for (Eigen::Index i = 0; i < y_test.size(); ++i)
    {
        y_test_str[i] = std::to_string(y_test[i]);
    }

    std::cout<<X_train;


    //Fitting
    //model.fit(X_train,y_train_str);
    //model.fit(X_train,y_train_str,sample_weight);
    model.fit(X_train,y_train_str,sample_weight,{},{0,1,2,3,4,5,10,static_cast<size_t>(y_train.size()-1)});
    model.fit(X_train,y_train_str,sample_weight,{},{0,1,2,3,4,5,10,static_cast<size_t>(y_train.size()-1)});
    MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test,false)};
    std::vector<std::string> predictions{model.predict(X_test,false)};
    MatrixXd local_feature_importance{model.calculate_local_feature_importance(X_test)};
    //MatrixXd lfi_model1{model.get_logit_model("0.000000").calculate_local_feature_importance(X_test)};
    //MatrixXd lfi_model2{model.get_logit_model("1.000000").calculate_local_feature_importance(X_test)};

    std::cout<<"validation_error\n"<<model.get_validation_error()<<"\n\n";
    tests.push_back(is_approximately_equal(model.get_validation_error(),0.0228939,0.000001));

    std::cout<<"predicted_class_prob_mean\n"<<predicted_class_probabilities.mean()<<"\n\n";
    tests.push_back(is_approximately_equal(predicted_class_probabilities.mean(),0.5,0.00001));

    std::cout<<"local_feature_importance_mean\n"<<local_feature_importance.mean()<<"\n\n";
    tests.push_back(is_approximately_equal(local_feature_importance.mean(),0.135719,0.00001));

    //Test summary
    std::cout<<"\n\nTest summary\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}