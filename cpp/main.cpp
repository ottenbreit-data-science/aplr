#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "APLRRegressor.h"
#include "term.h"

using namespace Eigen;

int main()
{
    // Model
    APLRRegressor model{APLRRegressor()};
    model.m = 100;
    model.v = 0.5;
    model.bins = 300;
    model.n_jobs = 0;
    model.loss_function = "mse";
    model.verbosity = 3;
    model.min_observations_in_split = 10;
    // model.max_interaction_level=0;
    model.max_interaction_level = 100;
    model.max_interactions = 30;
    model.ineligible_boosting_steps_added = 10;
    model.max_eligible_terms = 5;

    // Data
    MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("X_train.csv")};
    MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("X_test.csv")};
    VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("y_train.csv")};
    VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("y_test.csv")};

    VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};
    // VectorXd sample_weight{VectorXd::Random(y_train.size()).cwiseAbs()};

    // Fitting
    clock_t time_req{clock()};
    // model.fit(X_train,y_train);
    model.fit(X_train, y_train, sample_weight);
    time_req = clock() - time_req;
    std::cout << "time elapsed: " << std::to_string(time_req) << "\n\n";

    VectorXd predictions{model.predict(X_test)};

    // Saving results
    save_as_csv_file("output.csv", predictions);
    std::cout << "min validation_error " << model.validation_error_steps.minCoeff() << "\n\n";
    std::cout << is_approximately_equal(model.validation_error_steps.minCoeff(), 6.17133, 0.00001) << "\n";

    std::cout << "mean prediction " << predictions.mean() << "\n\n";
    std::cout << is_approximately_equal(predictions.mean(), 23.591, 0.0001) << "\n";

    std::cout << "best_m: " << model.m << "\n";

    std::cout << "test";
}