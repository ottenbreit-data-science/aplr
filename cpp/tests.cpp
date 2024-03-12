#include <cmath>
#include <iostream>
#include <vector>
#include <numeric>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "APLRRegressor.h"
#include "APLRClassifier.h"
#include "term.h"

using namespace Eigen;

double calculate_custom_loss(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)
{
    VectorXd error{(y.array() - predictions.array()).pow(2)};
    return error.mean();
}

VectorXd calculate_custom_negative_gradient(const VectorXd &y, const VectorXd &predictions, const VectorXi &group, const MatrixXd &other_data)
{
    VectorXd negative_gradient{y - predictions};
    return negative_gradient;
}

double calculate_custom_validation_error(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)
{
    VectorXd error{(y.array() - predictions.array()).pow(3)};
    return error.mean();
}

double calculate_custom_validation_error_2(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group, const MatrixXd &other_data)
{
    VectorXd error{(y.array() - predictions.array()).pow(2)};
    return error.mean();
}

VectorXd calculate_custom_transform_linear_predictor_to_predictions(const VectorXd &linear_predictor)
{
    VectorXd predictions{linear_predictor.array().exp()};
    return predictions;
}

VectorXd calculate_custom_differentiate_predictions_wrt_linear_predictor(const VectorXd &linear_predictor)
{
    VectorXd differentiated_predictions{linear_predictor.array().exp()};
    return differentiated_predictions;
}

class Tests
{
public:
    std::vector<bool> tests;
    Tests()
    {
        tests.reserve(10000);
    }

    void summarize_results()
    {
        std::cout << "\n\nTest summary\n"
                  << "Passed " << std::accumulate(tests.begin(), tests.end(), 0) << " out of " << tests.size() << " tests.";
    }

    void test_aplrregressor_cauchy_group_mse_validation()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "group_mse";

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};
        // VectorXd sample_weight{VectorXd(0)};

        VectorXi group{X_train.col(0).cast<int>()};
        // VectorXi group{VectorXi::Constant(20,1)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        model.fit(X_train, y_train, sample_weight, {}, {}, {}, {}, group);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 20.170939369337834));

        VectorXd feature_importance_on_test_set{model.calculate_feature_importance(X_test)};
        double feature_importance_on_test_set_mean{feature_importance_on_test_set.mean()};
        double feature_importance_mean{model.get_feature_importance().mean()};
        double term_importance_mean{model.get_term_importance().mean()};
        double feature_importance_first{model.get_feature_importance()[0]};
        double term_importance_first{model.get_term_importance()[0]};
        int term_base_predictor_index_max{model.get_term_main_predictor_indexes().maxCoeff()};
        int term_interaction_level_max{model.get_term_interaction_levels().maxCoeff()};
        std::cout << feature_importance_mean << "\n\n";
        std::cout << term_importance_mean << "\n\n";
        std::cout << feature_importance_first << "\n\n";
        std::cout << term_importance_first << "\n\n";
        std::cout << term_base_predictor_index_max << "\n\n";
        std::cout << term_interaction_level_max << "\n\n";
        tests.push_back(is_approximately_equal(feature_importance_on_test_set_mean, 0.3745208543129413));
        tests.push_back(is_approximately_equal(feature_importance_mean, 0.37558643075803277));
        tests.push_back(is_approximately_equal(term_importance_mean, 0.12150899891033366));
        tests.push_back(is_approximately_equal(feature_importance_first, 0.74048121167747938));
        tests.push_back(is_approximately_equal(term_importance_first, 0.85496610382351568));
        tests.push_back(term_base_predictor_index_max == 5);
        tests.push_back(term_interaction_level_max == 1);
    }

    void test_aplrregressor_cauchy_group_mse_by_prediction_validation()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "group_mse_by_prediction";
        model.group_mse_by_prediction_bins = 7;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        VectorXi group{X_train.col(0).cast<int>()};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        model.fit(X_train, y_train, sample_weight, {}, {}, {}, {}, group);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 20.104856610039558));
    }

    void test_aplrregressor_cauchy()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;
        model.boosting_steps_before_interactions_are_allowed = 60;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train, y_train, sample_weight, {}, cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 20.850035037781723));
    }

    void test_aplrregressor_custom_loss_and_validation()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 0;
        model.loss_function = "custom_function";
        model.calculate_custom_loss_function = calculate_custom_loss;
        model.calculate_custom_negative_gradient_function = calculate_custom_negative_gradient;
        model.calculate_custom_validation_error_function = calculate_custom_validation_error;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "custom_function";

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);

        std::vector<size_t> prioritized_predictor_indexes{1, 8};
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, prioritized_predictor_indexes, {}, VectorXi(0), {}, X_train);
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, prioritized_predictor_indexes, {}, VectorXi(0), {}, X_train);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.944797684016745));
    }

    void test_aplrregressor_custom_loss()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "custom_function";
        model.calculate_custom_loss_function = calculate_custom_loss;
        model.calculate_custom_negative_gradient_function = calculate_custom_negative_gradient;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.early_stopping_rounds = 10;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);

        std::vector<size_t> prioritized_predictor_indexes{1, 8};
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, prioritized_predictor_indexes, {}, VectorXi(0), {}, X_train);
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, prioritized_predictor_indexes, {}, VectorXi(0), {}, X_train);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.7035, 0.00001));
    }

    void test_aplrregressor_gamma_custom_link()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "gamma";
        model.link_function = "custom_function";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.calculate_custom_transform_linear_predictor_to_predictions_function = calculate_custom_transform_linear_predictor_to_predictions;
        model.calculate_custom_differentiate_predictions_wrt_linear_predictor_function = calculate_custom_differentiate_predictions_wrt_linear_predictor;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.5266, 0.00001));
    }

    void test_aplrregressor_gamma_custom_validation()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "gamma";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "custom_function";
        model.calculate_custom_validation_error_function = calculate_custom_validation_error_2;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.5512, 0.00001));
    }

    void test_aplrregressor_gamma_gini_weighted()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "gamma";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "negative_gini";

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.3198, 0.00001));
    }

    void test_aplrregressor_gamma_gini()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "gamma";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "negative_gini";

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        model.fit(X_train, y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.3198, 0.00001));
    }

    void test_aplrregressor_gamma()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "gamma";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "mse";

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.5512, 0.00001));
    }

    void test_aplrregressor_group_mse()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "group_mse";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        VectorXi group{X_train.col(0).cast<int>()};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        // model.fit(X_train, y_train, VectorXd(0), {}, {}, {}, {}, group);
        model.fit(X_train, y_train, sample_weight, {}, {}, {}, {}, group);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 20.82771158964184));
    }

    void test_aplrregressor_group_mse_cycle()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "group_mse_cycle";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.group_mse_by_prediction_bins = 8;
        model.group_mse_cycle_min_obs_in_bin = 28;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        std::cout << X_train;

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.533140818895273));
    }

    void test_aplrregressor_int_constr()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, {1, 8}, {0, 0, 1, -1, 0, 0, 0, 0, 0}, VectorXi(0),
                  {{1, 1, 8, 1, 8, 8}, {2, 3, 2}, {4}}, X_train);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.606326522845816));
    }

    void test_aplrregressor_inversegaussian()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "tweedie";
        model.link_function = "log";
        model.dispersion_parameter = 3.0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.validation_tuning_metric = "mae";

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        model.fit(X_train, y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.3198, 0.00001));
    }

    void test_aplrregressor_logit()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.5;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "binomial";
        model.link_function = "logit";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_logit.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_logit.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 0.0875969, 0.00001));
    }

    void test_aplrregressor_mae()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mae";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.557834093496929));
    }

    void test_aplrregressor_monotonic()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, {1, 8}, {1, 0, -1, 1, 1, 1, 1, 1, 1});
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.337125831228487));
    }

    void test_aplrregressor_monotonic_ignore_interactions()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.monotonic_constraints_ignore_interactions = true;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, {1, 8}, {1, 0, -1, 1, 1, 1, 1, 1, 1});
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 24.3013, 0.00001));
    }

    void test_aplrregressor_negative_binomial()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "negative_binomial";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 1.8694, 0.00001));
    }

    void test_aplrregressor_poisson()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "poisson";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 1.88727, 0.00001));
    }

    void test_aplrregressor_poissongamma()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "tweedie";
        model.link_function = "log";
        model.dispersion_parameter = 1.5;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 1.88553, 0.00001));
    }

    void test_aplrregressor_quantile()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "quantile";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.quantile = 0.5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.65630148869738));
    }

    void test_aplrregressor_weibull()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "weibull";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.6406, 0.00001));
    }

    void test_aplrregressor()
    {
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train);
        // model.fit(X_train,y_train,sample_weight);
        // model.fit(X_train,y_train,sample_weight,{},cv_observations);

        std::vector<size_t> prioritized_predictor_indexes{1, 8};
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, prioritized_predictor_indexes);
        model.fit(X_train, y_train, sample_weight, {}, cv_observations, prioritized_predictor_indexes);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};
        MatrixXd li{model.calculate_local_feature_contribution(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predictions.mean(), 23.7035, 0.00001));

        std::map<double, double> coefficient_shape_function = model.get_coefficient_shape_function(1);
        bool coefficient_shape_function_has_correct_length{coefficient_shape_function.size() == 27};
        bool coefficient_shape_function_value_test{is_approximately_equal(coefficient_shape_function.begin()->second, 0.04175, 0.00001)};
        tests.push_back(coefficient_shape_function_has_correct_length);
        tests.push_back(coefficient_shape_function_value_test);
    }

    void test_aplr_classifier_multi_class_other_params()
    {
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.5;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.monotonic_constraints_ignore_interactions = true;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};
        std::vector<std::string> y_train_str(y_train.rows());
        std::vector<std::string> y_test_str(y_test.rows());
        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            y_train_str[i] = std::to_string(y_train[i]);
        }
        for (Eigen::Index i = 0; i < y_test.size(); ++i)
        {
            y_test_str[i] = std::to_string(y_test[i]);
        }

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        model.fit(X_train, y_train_str, sample_weight, {}, cv_observations, {1, 8}, {0, 0, 1, -1, 0, 0, 0, 0, 0},
                  {{1, 1, 8, 1, 8, 8}, {2, 3}, {4}});
        model.fit(X_train, y_train_str, sample_weight, {}, cv_observations, {1, 8}, {0, 0, 1, -1, 0, 0, 0, 0, 0},
                  {{1, 1, 8, 1, 8, 8}, {2, 3}, {4}});
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};

        MatrixXd local_feature_importance{model.calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model3{model.get_logit_model("2.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model4{model.get_logit_model("3.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model5{model.get_logit_model("4.000000").calculate_local_feature_contribution(X_test)};

        VectorXd feature_importance{model.get_feature_importance()};
        VectorXd feature_importance_model1{model.get_logit_model("0.000000").get_feature_importance()};
        VectorXd feature_importance_model2{model.get_logit_model("1.000000").get_feature_importance()};
        VectorXd feature_importance_model3{model.get_logit_model("2.000000").get_feature_importance()};
        VectorXd feature_importance_model4{model.get_logit_model("3.000000").get_feature_importance()};
        VectorXd feature_importance_model5{model.get_logit_model("4.000000").get_feature_importance()};
        VectorXd feature_importance_correct{0.2 * (feature_importance_model1 + feature_importance_model2 + feature_importance_model3 + feature_importance_model4 + feature_importance_model5)};
        double feature_importance_mae = (feature_importance - feature_importance_correct).cwiseAbs().mean();
        tests.push_back(is_approximately_zero(feature_importance_mae));

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        tests.push_back(is_approximately_equal(model.get_cv_error(), 0.246477, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predicted_class_probabilities.mean(), 0.2, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_importance.mean() << "\n\n";
        tests.push_back(is_approximately_equal(local_feature_importance.mean(), 0.15805, 0.00001));
    }

    void test_aplrclassifier_multi_class()
    {
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.5;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};
        std::vector<std::string> y_train_str(y_train.rows());
        std::vector<std::string> y_test_str(y_test.rows());
        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            y_train_str[i] = std::to_string(y_train[i]);
        }
        for (Eigen::Index i = 0; i < y_test.size(); ++i)
        {
            y_test_str[i] = std::to_string(y_test[i]);
        }

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train_str);
        model.fit(X_train, y_train_str, sample_weight);
        model.fit(X_train, y_train_str, sample_weight);
        // model.fit(X_train,y_train_str,sample_weight);
        // model.fit(X_train,y_train_str,sample_weight,{},cv_observations);
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};

        MatrixXd local_feature_importance{model.calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model3{model.get_logit_model("2.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model4{model.get_logit_model("3.000000").calculate_local_feature_contribution(X_test)};
        MatrixXd lfi_model5{model.get_logit_model("4.000000").calculate_local_feature_contribution(X_test)};

        VectorXd feature_importance{model.get_feature_importance()};
        VectorXd feature_importance_model1{model.get_logit_model("0.000000").get_feature_importance()};
        VectorXd feature_importance_model2{model.get_logit_model("1.000000").get_feature_importance()};
        VectorXd feature_importance_model3{model.get_logit_model("2.000000").get_feature_importance()};
        VectorXd feature_importance_model4{model.get_logit_model("3.000000").get_feature_importance()};
        VectorXd feature_importance_model5{model.get_logit_model("4.000000").get_feature_importance()};
        VectorXd feature_importance_correct{0.2 * (feature_importance_model1 + feature_importance_model2 + feature_importance_model3 + feature_importance_model4 + feature_importance_model5)};
        double feature_importance_mae = (feature_importance - feature_importance_correct).cwiseAbs().mean();
        tests.push_back(is_approximately_zero(feature_importance_mae));

        std::cout << "validation_error\n"
                  << model.get_cv_error() << "\n\n";
        tests.push_back(is_approximately_equal(model.get_cv_error(), 0.227717, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predicted_class_probabilities.mean(), 0.2, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_importance.mean() << "\n\n";
        tests.push_back(is_approximately_equal(local_feature_importance.mean(), 0.154628, 0.00001));
    }

    void test_aplrclassifier_two_class_other_params()
    {
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.5;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.monotonic_constraints_ignore_interactions = true;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_logit.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_logit.csv")};
        std::vector<std::string> y_train_str(y_train.rows());
        std::vector<std::string> y_test_str(y_test.rows());
        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            y_train_str[i] = std::to_string(y_train[i]);
        }
        for (Eigen::Index i = 0; i < y_test.size(); ++i)
        {
            y_test_str[i] = std::to_string(y_test[i]);
        }

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train_str);
        model.fit(X_train, y_train_str, sample_weight, {}, cv_observations, {1, 8}, {0, 0, 1, -1, 0, 0, 0, 0, 0},
                  {{1, 1, 8, 1, 8, 8}, {2, 3}, {4}});
        model.fit(X_train, y_train_str, sample_weight, {}, cv_observations, {1, 8}, {0, 0, 1, -1, 0, 0, 0, 0, 0},
                  {{1, 1, 8, 1, 8, 8}, {2, 3}, {4}});
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};
        MatrixXd local_feature_importance{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfi_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfi_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        tests.push_back(is_approximately_equal(model.get_cv_error(), 0.29875, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_importance.mean() << "\n\n";
        tests.push_back(is_approximately_equal(local_feature_importance.mean(), 0.15288, 0.00001));
    }

    void test_aplrclassifier_two_class_val_index()
    {
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.5;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.boosting_steps_before_interactions_are_allowed = 50;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_logit.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_logit.csv")};
        std::vector<std::string> y_train_str(y_train.rows());
        std::vector<std::string> y_test_str(y_test.rows());
        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            y_train_str[i] = std::to_string(y_train[i]);
        }
        for (Eigen::Index i = 0; i < y_test.size(); ++i)
        {
            y_test_str[i] = std::to_string(y_test[i]);
        }

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train_str);
        // model.fit(X_train,y_train_str,sample_weight);
        model.fit(X_train, y_train_str, sample_weight, {}, cv_observations);
        model.fit(X_train, y_train_str, sample_weight, {}, cv_observations);
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};
        MatrixXd local_feature_importance{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfi_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfi_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        tests.push_back(is_approximately_equal(model.get_cv_error(), 0.27679347379315716));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_importance.mean() << "\n\n";
        tests.push_back(is_approximately_equal(local_feature_importance.mean(), 0.12248183904866403));
    }

    void test_aplrclassifier_two_class()
    {
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.5;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_logit.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_logit.csv")};
        std::vector<std::string> y_train_str(y_train.rows());
        std::vector<std::string> y_test_str(y_test.rows());
        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        for (Eigen::Index i = 0; i < y_train.size(); ++i)
        {
            y_train_str[i] = std::to_string(y_train[i]);
        }
        for (Eigen::Index i = 0; i < y_test.size(); ++i)
        {
            y_test_str[i] = std::to_string(y_test[i]);
        }

        MatrixXi cv_observations = MatrixXi::Constant(y_train.rows(), 2, 1);
        cv_observations.col(0)[273] = -1;
        cv_observations.col(0)[272] = -1;
        cv_observations.col(0)[271] = -1;
        cv_observations.col(0)[270] = -1;
        cv_observations.col(0)[269] = -1;
        cv_observations.col(0)[268] = -1;
        cv_observations.col(0)[267] = -1;
        cv_observations.col(0)[266] = -1;
        cv_observations.col(1) = -cv_observations.col(0);

        // Fitting
        // model.fit(X_train,y_train_str);
        model.fit(X_train, y_train_str, sample_weight);
        model.fit(X_train, y_train_str, sample_weight);
        // model.fit(X_train, y_train_str, sample_weight, {}, cv_observations);
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};
        MatrixXd local_feature_importance{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfi_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfi_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        tests.push_back(is_approximately_equal(model.get_cv_error(), 0.164915, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        tests.push_back(is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_importance.mean() << "\n\n";
        tests.push_back(is_approximately_equal(local_feature_importance.mean(), 0.125672, 0.00001));
    }

    void test_functions()
    {
        // floating point comparisons
        double inf_left{-std::numeric_limits<double>::infinity()};
        double inf_right{std::numeric_limits<double>::infinity()};
        tests.push_back(is_approximately_equal(inf_left, inf_left));
        tests.push_back(is_approximately_equal(inf_right, inf_right));
        tests.push_back(!is_approximately_equal(inf_left, inf_right));
        tests.push_back(!is_approximately_equal(inf_right, inf_left));
        tests.push_back(!is_approximately_equal(inf_left, 2.0));
        tests.push_back(!is_approximately_equal(inf_right, 2.0));
        tests.push_back(is_approximately_zero(0.0));
        tests.push_back(!is_approximately_zero(0.0000001));
        tests.push_back(!is_approximately_zero(-0.0000001));
        tests.push_back(!is_approximately_zero(inf_left));
        tests.push_back(!is_approximately_zero(inf_right));

        // compute_errors
        VectorXd y(5), pred(5), sample_weight(5);
        VectorXd sample_weight_equal{VectorXd::Constant(5, 1.0)};
        y << 1, 3.3, 2, 4, 0;
        pred << 1.4, 3.3, 1.5, 0, 1;
        sample_weight << 0.5, 0.5, 1, 0, 1;
        std::cout << "y\n"
                  << y << "\n\n";
        std::cout << "pred\n"
                  << pred << "\n\n";
        std::cout << "sample_weight\n"
                  << sample_weight << "\n\n";
        VectorXd errors_mse{calculate_errors(y, pred, sample_weight_equal)};
        VectorXd errors_mse_sw{calculate_errors(y, pred, sample_weight)};

        // compute_error
        // calculating errors
        double error_mse{calculate_mean_error(errors_mse, sample_weight_equal)};
        std::cout << "error_mse: " << error_mse << "\n\n";
        tests.push_back((is_approximately_equal(error_mse, 3.482) ? true : false));
        double error_mse_sw{calculate_mean_error(errors_mse_sw, sample_weight)};
        std::cout << "error_mse_sw: " << error_mse_sw << "\n\n";
        tests.push_back((is_approximately_equal(error_mse_sw, 0.4433, 0.0001) ? true : false));

        // testing for nan and infinity
        // matrix without nan or inf
        bool matrix_has_nan_or_inf_elements{matrix_has_nan_or_infinite_elements(y)};
        tests.push_back(!matrix_has_nan_or_inf_elements ? true : false);

        VectorXd inf(5);
        inf << 1.0, 0.2, std::numeric_limits<double>::infinity(), 0.0, 0.5;
        matrix_has_nan_or_inf_elements = matrix_has_nan_or_infinite_elements(inf);
        tests.push_back(matrix_has_nan_or_inf_elements ? true : false);

        VectorXd nan(5);
        nan << 1.0, 0.2, NAN_DOUBLE, 0.0, 0.5;
        matrix_has_nan_or_inf_elements = matrix_has_nan_or_infinite_elements(nan);
        tests.push_back(matrix_has_nan_or_inf_elements ? true : false);

        VectorXd y_true(3);
        VectorXd weights_equal(3);
        VectorXd weights_different(3);
        y_true << 1.0, 2.0, 3.0;
        weights_equal << 1, 1, 1;
        weights_different << 0, 0.5, 0.75;

        VectorXd y_integration(3);
        VectorXd x_integration(3);
        y_integration << 1, 2, 3;
        x_integration << 4, 6, 8;
        double integration{trapezoidal_integration(y_integration, x_integration)};
        tests.push_back(is_approximately_equal(integration, 8.0));

        VectorXd weights_none{VectorXd(0)};
        VectorXd calculated_weights_if_not_provided{calculate_weights_if_they_are_not_provided(y_true)};
        VectorXd calculated_weights_if_provided{calculate_weights_if_they_are_not_provided(y_true, weights_different)};
        tests.push_back(calculated_weights_if_not_provided == weights_equal);
        tests.push_back(calculated_weights_if_provided == weights_different);

        VectorXd y_pred(3);
        VectorXd weights_gini(3);
        y_pred << 1.0, 3.0, 2.0;
        weights_gini << 0.2, 0.5, 0.3;
        double gini{calculate_gini(y_true, y_pred, weights_gini)};
        tests.push_back(is_approximately_equal(gini, -0.1166667, 0.0000001));

        VectorXi int_vector(3);
        int_vector << 1, 1, 2;
        std::set<int> unique_integers{get_unique_integers(int_vector)};
        bool size_is_correct{unique_integers.size() == 2};
        tests.push_back(size_is_correct);
    }

    void test_term()
    {
        // Setting up term instance p with default values
        Term p{Term()};
        tests.push_back(p.base_term == 0 ? true : false);

        // Testing calulate values
        p.split_point = 0.5;
        p.direction_right = false;
        p.coefficient = 2;
        p.given_terms.push_back(Term(1, std::vector<Term>(0), -5.0, true, -3.0));
        p.given_terms.push_back(Term(2, std::vector<Term>(0), 5.0, false, 3.0));
        p.given_terms[0].given_terms.push_back(Term(0, std::vector<Term>(0), -0.21, false, 2.0));
        MatrixXd X{MatrixXd::Random(3, 4)}; // terms
        VectorXd values{p.calculate(X)};
        std::cout << "X\n";
        std::cout << X << "\n\n";
        std::cout << "values\n";
        std::cout << values << "\n\n";
        tests.push_back(is_approximately_zero(values[0]) && is_approximately_equal(values[1], -0.711234, 0.00001) &&
                                is_approximately_zero(values[2])
                            ? true
                            : false);

        // Testing calculate_prediction_contribution
        VectorXd contrib{p.calculate_contribution_to_linear_predictor(X)};
        std::cout << "Prediction contribution\n";
        std::cout << contrib << "\n\n";
        tests.push_back(is_approximately_equal(contrib[1], -1.42247, 0.0001) && is_approximately_zero(contrib[0]) && is_approximately_zero(contrib[2]) ? true : false);

        // Testing equals_base_terms
        bool t1{Term::equals_given_terms(p, p.given_terms[0])};
        bool t2{Term::equals_given_terms(p, p)};
        tests.push_back(t1 ? false : true);
        tests.push_back(t2 ? true : false);

        // Testing copy constructor
        p.ineligible_boosting_steps = 10;
        Term p2{p};
        bool test_cpy = Term::equals_given_terms(p, p2) && &p.given_terms != &p2.given_terms && p.coefficient == p2.coefficient && &p.coefficient != &p2.coefficient && is_approximately_equal(p.split_point, p2.split_point) && &p.split_point != &p2.split_point && p.direction_right == p2.direction_right && &p.direction_right != &p2.direction_right && p.name == p2.name && &p.name != &p2.name &&
                        p.coefficient_steps.size() == p2.coefficient_steps.size() && &p.coefficient_steps != &p2.coefficient_steps && ((p.coefficient_steps - p2.coefficient_steps).array().abs() == 0).all() && p.base_term == p2.base_term && p.ineligible_boosting_steps == 10 && p2.ineligible_boosting_steps == 0;
        tests.push_back(test_cpy);

        // Testing equals operator
        p2.coefficient = 35.2; // p2 should be equal - coefficient is not compared
        Term p3{p};            // to be unequal
        Term p4{p};            // to be unequal
        Term p5{p};            // to be unequal
        Term p6{p};            // to be unequal
        p3.split_point = 0.1;
        p4.direction_right = true;
        p5.given_terms.push_back(p3);
        p6.split_point = 0.2;
        p6.direction_right = false;
        p6.given_terms.push_back(p4);
        bool test_equals1 = (p == p2 && p2 == p ? true : false);
        bool test_equals2 = (p == p3 && p3 == p ? false : true);
        bool test_equals3 = (p == p4 && p4 == p ? false : true);
        bool test_equals4 = (p == p5 && p5 == p ? false : true);
        bool test_equals5 = (p == p6 && p6 == p ? false : true);
        p4.split_point = NAN_DOUBLE;
        p2.split_point = NAN_DOUBLE;
        bool test_equals6 = (p2 == p4 && p4 == p2 ? true : false);
        tests.push_back(test_equals1);
        tests.push_back(test_equals2);
        tests.push_back(test_equals3);
        tests.push_back(test_equals4);
        tests.push_back(test_equals5);
        tests.push_back(test_equals6);

        // Testing interaction_level method
        Term p7{Term(1)};
        p7.given_terms.push_back(Term(2));
        Term p8{Term(3)};
        size_t pil{p.get_interaction_level()};
        size_t p5il{p5.get_interaction_level()};
        size_t p7il{p7.get_interaction_level()};
        size_t p8il{p8.get_interaction_level()};
        tests.push_back(pil == 2 ? true : false);
        tests.push_back(p5il == 3 ? true : false);
        tests.push_back(p7il == 1 ? true : false);
        tests.push_back(p8il == 0 ? true : false);
    }
};

int main()
{
    Tests tests{Tests()};
    tests.test_aplrregressor_cauchy_group_mse_validation();
    tests.test_aplrregressor_cauchy_group_mse_by_prediction_validation();
    tests.test_aplrregressor_cauchy();
    tests.test_aplrregressor_custom_loss_and_validation();
    tests.test_aplrregressor_custom_loss();
    tests.test_aplrregressor_gamma_custom_link();
    tests.test_aplrregressor_gamma_custom_validation();
    tests.test_aplrregressor_gamma_gini_weighted();
    tests.test_aplrregressor_gamma_gini();
    tests.test_aplrregressor_gamma();
    tests.test_aplrregressor_group_mse();
    tests.test_aplrregressor_group_mse_cycle();
    tests.test_aplrregressor_int_constr();
    tests.test_aplrregressor_inversegaussian();
    tests.test_aplrregressor_logit();
    tests.test_aplrregressor_mae();
    tests.test_aplrregressor_monotonic();
    tests.test_aplrregressor_monotonic_ignore_interactions();
    tests.test_aplrregressor_negative_binomial();
    tests.test_aplrregressor_poisson();
    tests.test_aplrregressor_poissongamma();
    tests.test_aplrregressor_quantile();
    tests.test_aplrregressor_weibull();
    tests.test_aplrregressor();
    tests.test_aplr_classifier_multi_class_other_params();
    tests.test_aplrclassifier_multi_class();
    tests.test_aplrclassifier_two_class_other_params();
    tests.test_aplrclassifier_two_class_val_index();
    tests.test_aplrclassifier_two_class();
    tests.test_functions();
    tests.test_term();
    tests.summarize_results();
}