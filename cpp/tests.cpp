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
    struct TestResult
    {
        std::string name;
        bool passed;
    };
    std::vector<TestResult> tests;
    std::string current_test_suite_name;

    Tests()
    {
        tests.reserve(10000);
    }

    void add_test(const std::string &name, bool passed)
    {
        tests.push_back({current_test_suite_name + ": " + name, passed});
    }

    void summarize_results()
    {
        size_t passed_count = 0;
        for (const auto &test : tests)
        {
            if (test.passed)
                passed_count++;
        }
        std::cout << "\n\nTest summary\n"
                  << "Passed " << passed_count << " out of " << tests.size() << " tests.";
        if (passed_count < tests.size())
        {
            std::cout << "\n\nFailing tests:\n";
            for (const auto &test : tests)
            {
                if (!test.passed)
                {
                    std::cout << "- " << test.name << "\n";
                }
            }
        }
    }

    void test_aplrregressor_huber()
    {
        current_test_suite_name = "test_aplrregressor_huber";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "huber";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0; // This is delta for huber loss
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.652393228773949));

        // Also test huber as a validation metric
        model.validation_tuning_metric = "huber";
        model.fit(X_train, y_train, sample_weight);
        add_test("model.get_cv_error() with huber validation", is_approximately_equal(model.get_cv_error(), 1.6696312268620679));
    }

    void test_aplrregressor_huber_log_link()
    {
        current_test_suite_name = "test_aplrregressor_huber_log_link";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "huber";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 1;
        model.max_interactions = 30;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0; // This is delta for huber loss
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 1.9060494063077187));

        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.080814921153113131));
    }

    void test_aplrregressor_mean_bias_correction()
    {
        current_test_suite_name = "test_aplrregressor_mean_bias_correction";
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
        model.mean_bias_correction = true;
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.640483777279176));
    }

    void test_aplrregressor_ridge()
    {
        current_test_suite_name = "test_aplrregressor_ridge";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 300;
        model.n_jobs = 0;
        model.loss_function = "mse";
        model.link_function = "identity";
        model.verbosity = 3;
        model.max_interaction_level = 1;
        model.n_jobs = 1;
        model.ridge_penalty = 0.1;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        // model.fit(X_train,y_train);
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.570526180335573));
    }

    void test_aplrregressor_mse_predictor_min_observations_in_split()
    {
        current_test_suite_name = "test_aplrregressor_mse_predictor_min_observations_in_split";
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
        model.ridge_penalty = 0.0;

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
        model.fit(X_train, y_train, sample_weight, {}, MatrixXi(0, 0), {}, {}, VectorXi(0), {}, MatrixXd(0, 0), {}, {}, {},
                  {5, 6, 7, 8, 9, 10, 11, 12, 13});
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.788775508596238));
    }

    void test_aplrregressor_cauchy_term_limit()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_term_limit";
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
        model.min_observations_in_split = 10;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;
        model.max_terms = 5;
        model.ridge_penalty = 0.0;

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
        model.fit(X_train, y_train, sample_weight, {}, MatrixXi(0, 0), {0, 1, 2, 3, 8});
        // model.fit(X_train, y_train, sample_weight, {}, cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 16.654091872011836));
    }

    void test_aplrregressor_cauchy_predictor_specific_penalties_and_learning_rates()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_predictor_specific_penalties_and_learning_rates";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 200;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.min_observations_in_split = 10;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;
        model.penalty_for_non_linearity = 0.05;
        model.penalty_for_interactions = 0.1;
        model.ridge_penalty = 0.0;

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
        model.fit(X_train, y_train, sample_weight, {}, MatrixXi(0, 0), {}, {}, VectorXi(0), {}, MatrixXd(0, 0),
                  {0.2, 0.3, 0.4, 0.0, 0.5, 1.0, 0.7, 0.2, 0.9}, {0.1, 0.05, 0.0, 0.04, 0.07, 0.03, 0.2, 0.02, 0.09},
                  {0.1, 0.05, 0.02, 0.01, 0.07, 0.03, 0.2, 0.02, 0.09});
        // model.fit(X_train, y_train, sample_weight, {}, cv_observations);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 19.067710451454566));
    }

    void test_aplrregressor_cauchy_penalties()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_penalties";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 200;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.min_observations_in_split = 10;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;
        model.penalty_for_non_linearity = 0.05;
        model.penalty_for_interactions = 0.1;
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 20.809163574542939));
    }

    void test_aplrregressor_cauchy_linear_effects_only_first()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_linear_effects_only_first";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 200;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.min_observations_in_split = 10;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;
        model.boosting_steps_before_interactions_are_allowed = 90;
        model.num_first_steps_with_linear_effects_only = 80;
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 17.380763842227257));
    }

    void test_aplrregressor_cauchy_linear_effects_only_first_2()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_linear_effects_only_first_2";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 200;
        model.n_jobs = 1;
        model.loss_function = "cauchy";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.min_observations_in_split = 10;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.0;
        model.boosting_steps_before_interactions_are_allowed = 90;
        model.num_first_steps_with_linear_effects_only = 80;
        model.early_stopping_rounds = 1;
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 17.886569073729863));
    }

    void test_aplrregressor_cauchy_group_mse_validation()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_group_mse_validation";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 20.096177156192478));

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
        add_test("feature_importance_on_test_set_mean", is_approximately_equal(feature_importance_on_test_set_mean, 0.28154881700595819));
        add_test("feature_importance_mean", is_approximately_equal(feature_importance_mean, 0.28629814028801753));
        add_test("term_importance_mean", is_approximately_equal(term_importance_mean, 0.12843198080249971));
        add_test("feature_importance_first", is_approximately_equal(feature_importance_first, 0.5516725960373986));
        add_test("term_importance_first", is_approximately_equal(term_importance_first, 1.0431553101537596));
        add_test("term_base_predictor_index_max", term_base_predictor_index_max == 6);
        add_test("term_interaction_level_max", term_interaction_level_max == 1);
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 25.955566623662232));
    }

    void test_aplrregressor_cauchy_group_mse_by_prediction_validation()
    {
        current_test_suite_name = "test_aplrregressor_cauchy_group_mse_by_prediction_validation";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 20.096177156192478));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 26.68005452713048));
    }

    void test_aplrregressor_cauchy()
    {
        current_test_suite_name = "test_aplrregressor_cauchy";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 20.979930894644177));
    }

    void test_aplrregressor_custom_loss_and_validation()
    {
        current_test_suite_name = "test_aplrregressor_custom_loss_and_validation";
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
        model.ridge_penalty = 0.0;

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
        model.remove_provided_custom_functions();
        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 24.301339246925711));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), -64.887393290901031));
    }

    void test_aplrregressor_custom_loss()
    {
        current_test_suite_name = "test_aplrregressor_custom_loss";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 24.301339246925711, 0.00001));
    }

    void test_aplrregressor_gamma_custom_link()
    {
        current_test_suite_name = "test_aplrregressor_gamma_custom_link";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.526613939603266, 0.00001));
    }

    void test_aplrregressor_gamma_custom_validation()
    {
        current_test_suite_name = "test_aplrregressor_gamma_custom_validation";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.555068816303912));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 7.5399664045598414));
    }

    void test_aplrregressor_gamma_gini_weighted()
    {
        current_test_suite_name = "test_aplrregressor_gamma_gini_weighted";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.555068816303912));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), -0.94130723051226006));
    }

    void test_aplrregressor_gamma_gini()
    {
        current_test_suite_name = "test_aplrregressor_gamma_gini";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.555068816303912));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), -0.94130723051226006));
    }

    void test_aplrregressor_gamma()
    {
        current_test_suite_name = "test_aplrregressor_gamma";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.555068816303912));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 7.5399664045598414));
    }

    void test_aplrregressor_group_mse()
    {
        current_test_suite_name = "test_aplrregressor_group_mse";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.900212415566287));
    }

    void test_aplrregressor_group_mse_cycle()
    {
        current_test_suite_name = "test_aplrregressor_group_mse_cycle";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 24.014522054509584));
    }

    void test_aplrregressor_int_constr()
    {
        current_test_suite_name = "test_aplrregressor_int_constr";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.657546542794449));
    }

    void test_aplrregressor_inversegaussian()
    {
        current_test_suite_name = "test_aplrregressor_inversegaussian";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.320673705115034));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 2.0657033975879591));
    }

    void test_aplrregressor_logit()
    {
        current_test_suite_name = "test_aplrregressor_logit";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.05;
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 0.087596882912220717, 0.00001));
    }

    void test_aplrregressor_mae()
    {
        current_test_suite_name = "test_aplrregressor_mae";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.602543167509292));
    }

    void test_aplrregressor_monotonic()
    {
        current_test_suite_name = "test_aplrregressor_monotonic";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.34283475003015));
    }

    void test_aplrregressor_monotonic_ignore_interactions()
    {
        current_test_suite_name = "test_aplrregressor_monotonic_ignore_interactions";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 24.301339246925711, 0.00001));
    }

    void test_aplrregressor_negative_binomial()
    {
        current_test_suite_name = "test_aplrregressor_negative_binomial";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 1.8694002118421278, 0.00001));
    }

    void test_aplrregressor_poisson()
    {
        current_test_suite_name = "test_aplrregressor_poisson";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 1.8872692088161898, 0.00001));
    }

    void test_aplrregressor_poissongamma()
    {
        current_test_suite_name = "test_aplrregressor_poissongamma";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 1.8855344167602603, 0.00001));
    }

    void test_aplrregressor_quantile()
    {
        current_test_suite_name = "test_aplrregressor_quantile";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.646255799722155));
    }

    void test_aplrregressor_neg_top_quantile_mean_response()
    {
        current_test_suite_name = "test_aplrregressor_neg_top_quantile_mean_response";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.validation_tuning_metric = "neg_top_quantile_mean_response";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.quantile = 0.8;
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.592285396936951));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), -33.517015579716308));
    }

    void test_aplrregressor_bottom_quantile_mean_response()
    {
        current_test_suite_name = "test_aplrregressor_bottom_quantile_mean_response";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.validation_tuning_metric = "bottom_quantile_mean_response";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.quantile = 0.2;
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 0.5)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.592285396936951));
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 13.922224916203017));
    }

    void test_aplrregressor_weibull()
    {
        current_test_suite_name = "test_aplrregressor_weibull";
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
        model.ridge_penalty = 0.0;

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

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.640555263512187, 0.00001));
    }

    void test_aplrregressor()
    {
        current_test_suite_name = "test_aplrregressor";
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
        model.ridge_penalty = 0.0;

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
        VectorXd li_for_particular_terms{model.calculate_local_contribution_from_selected_terms(X_train, {1, 8})};
        std::vector<size_t> base_predictors_in_the_second_affiliation{model.get_base_predictors_in_each_unique_term_affiliation()[1]};
        std::vector<size_t> correct_base_predictors_in_the_second_affiliation{{1, 8}};
        std::string the_second_unique_term_affiliation{model.get_unique_term_affiliations()[1]};
        std::string the_correct_second_unique_term_affiliation{"X2 & X9"};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 24.301339246925711, 0.00001));

        std::map<double, double> main_effect_shape = model.get_main_effect_shape(1);
        bool main_effect_shape_has_correct_length{main_effect_shape.size() == 9};
        bool main_effect_shape_value_test{is_approximately_equal(main_effect_shape.begin()->second, 0)};
        bool li_for_particular_terms_has_correct_size{li_for_particular_terms.rows() == X_train.rows()};
        bool li_for_particular_terms_mean_is_correct{is_approximately_equal(li_for_particular_terms.mean(), -0.52786383485971788)};
        MatrixXd unique_term_affiliation_shape{model.get_unique_term_affiliation_shape("X2 & X9")};
        MatrixXd unique_term_affiliation_shape_for_X2{model.get_unique_term_affiliation_shape("X2")};
        VectorXd main_effect_shape_keys(main_effect_shape.size());
        std::transform(main_effect_shape.begin(), main_effect_shape.end(), main_effect_shape_keys.data(),
                       [](const std::pair<double, double> &pair)
                       { return pair.first; });
        VectorXd main_effect_shape_values(main_effect_shape.size());
        std::transform(main_effect_shape.begin(), main_effect_shape.end(), main_effect_shape_values.data(),
                       [](const std::pair<double, double> &pair)
                       { return pair.second; });
        add_test("main_effect_shape_has_correct_length", main_effect_shape_has_correct_length);
        add_test("main_effect_shape_value_test", main_effect_shape_value_test);
        add_test("li_for_particular_terms_has_correct_size", li_for_particular_terms_has_correct_size);
        add_test("li_for_particular_terms_mean_is_correct", li_for_particular_terms_mean_is_correct);
        add_test("base_predictors_in_the_second_affiliation", base_predictors_in_the_second_affiliation == correct_base_predictors_in_the_second_affiliation);
        add_test("the_second_unique_term_affiliation", the_second_unique_term_affiliation == the_correct_second_unique_term_affiliation);
        add_test("unique_term_affiliation_shape.mean()", is_approximately_equal(unique_term_affiliation_shape.mean(), 85.239971686680235));
        add_test("unique_term_affiliation_shape.rows()", unique_term_affiliation_shape.rows() == 65536);
        add_test("unique_term_affiliation_shape.cols()", unique_term_affiliation_shape.cols() == 3);
        add_test("main_effect_shape_keys == unique_term_affiliation_shape_for_X2.col(0)", main_effect_shape_keys == unique_term_affiliation_shape_for_X2.col(0));
        add_test("main_effect_shape_values == unique_term_affiliation_shape_for_X2.col(1)", main_effect_shape_values == unique_term_affiliation_shape_for_X2.col(1));
    }

    void test_aplrregressor_faster_convergence_identity()
    {
        current_test_suite_name = "test_aplrregressor_faster_convergence_identity";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "mse";
        model.link_function = "identity";
        model.verbosity = 3;
        model.max_interaction_level = 1;
        model.min_observations_in_split = 50;
        model.faster_convergence = true;
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.656267332497631));
    }

    void test_aplrregressor_faster_convergence_log()
    {
        current_test_suite_name = "test_aplrregressor_faster_convergence_log";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 0.1;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "poisson";
        model.link_function = "log";
        model.verbosity = 3;
        model.max_interaction_level = 1;
        model.min_observations_in_split = 20;
        model.faster_convergence = true;
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train_poisson.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test_poisson.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 1.8989834541884052));
    }

    void test_aplrregressor_exponential_power()
    {
        current_test_suite_name = "test_aplrregressor_exponential_power";
        // Model
        APLRRegressor model{APLRRegressor()};
        model.m = 100;
        model.v = 1.0;
        model.bins = 10;
        model.n_jobs = 1;
        model.loss_function = "exponential_power";
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 30;
        model.min_observations_in_split = 50;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.dispersion_parameter = 1.5;
        model.ridge_penalty = 0.0;

        // Data
        MatrixXd X_train{load_csv_into_eigen_matrix<MatrixXd>("data/X_train.csv")};
        MatrixXd X_test{load_csv_into_eigen_matrix<MatrixXd>("data/X_test.csv")};
        VectorXd y_train{load_csv_into_eigen_matrix<MatrixXd>("data/y_train.csv")};
        VectorXd y_test{load_csv_into_eigen_matrix<MatrixXd>("data/y_test.csv")};

        VectorXd sample_weight{VectorXd::Constant(y_train.size(), 1.0)};

        // Fitting
        model.fit(X_train, y_train, sample_weight);
        std::cout << "feature importance\n"
                  << model.feature_importance << "\n\n";

        VectorXd predictions{model.predict(X_test)};

        // Saving results
        save_as_csv_file("data/output.csv", predictions);

        std::cout << predictions.mean() << "\n\n";
        add_test("predictions.mean()", is_approximately_equal(predictions.mean(), 23.576861785661166));
    }

    void test_aplr_classifier_multi_class_other_params()
    {
        current_test_suite_name = "test_aplr_classifier_multi_class_other_params";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.monotonic_constraints_ignore_interactions = true;
        model.ridge_penalty = 0.0;

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

        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        VectorXd feature_importance{model.get_feature_importance()};
        add_test("feature_importance.mean()", is_approximately_equal(feature_importance.mean(), 0.25420178743878397));

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.24647671959943313, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.2, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.17780678779228751, 0.00001));
    }

    void test_aplrclassifier_multi_class()
    {
        current_test_suite_name = "test_aplrclassifier_multi_class";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.ridge_penalty = 0.0;

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

        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        VectorXd feature_importance{model.get_feature_importance()};
        add_test("feature_importance.mean()", is_approximately_equal(feature_importance.mean(), 0.1760445038452387));

        std::cout << "validation_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.227717, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.2, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.154628, 0.00001));
    }

    void test_aplrclassifier_two_class_other_params()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_other_params";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.monotonic_constraints_ignore_interactions = true;
        model.ridge_penalty = 0.0;

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
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.29875, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.27518427823404712, 0.00001));
    }

    void test_aplrclassifier_two_class_val_index()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_val_index";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.boosting_steps_before_interactions_are_allowed = 50;
        model.num_first_steps_with_linear_effects_only = 60;
        model.ridge_penalty = 0.0;

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
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.23802511407945728));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.10989690600027999));
    }

    void test_aplrclassifier_two_class()
    {
        current_test_suite_name = "test_aplrclassifier_two_class";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 0;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.ridge_penalty = 0.0;

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
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.16491496201017047, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.22620950269183793, 0.00001));
    }

    void test_aplrclassifier_two_class_penalties()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_penalties";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.penalty_for_non_linearity = 0.05;
        model.penalty_for_interactions = 0.1;
        model.ridge_penalty = 0.0;

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
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};
        std::vector<size_t> base_predictors_in_the_second_affiliation{model.get_base_predictors_in_each_unique_term_affiliation()[1]};
        std::vector<size_t> correct_base_predictors_in_the_second_affiliation{{0, 3, 5}};
        std::string the_second_unique_term_affiliation{model.get_unique_term_affiliations()[1]};
        std::string the_correct_second_unique_term_affiliation{"X1 & X4 & X6"};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.15942686880196807, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.05891072116542774, 0.00001));
        add_test("base_predictors_in_the_second_affiliation", base_predictors_in_the_second_affiliation == correct_base_predictors_in_the_second_affiliation);
        add_test("the_second_unique_term_affiliation", the_second_unique_term_affiliation == the_correct_second_unique_term_affiliation);
    }

    void test_aplrclassifier_two_class_predictor_specific_penalties_and_learning_rates()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_predictor_specific_penalties_and_learning_rates";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.penalty_for_non_linearity = 0.05;
        model.penalty_for_interactions = 0.1;
        model.ridge_penalty = 0.0;

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
        model.fit(X_train, y_train_str, sample_weight, {}, MatrixXi(0, 0), {}, {}, {}, {0.2, 0.3, 0.4, 0.1, 0.5, 1.0, 0.7, 0.2, 0.9},
                  {0.1, 0.05, 0.0, 0.04, 0.07, 0.03, 0.2, 0.02, 0.09}, {0.1, 0.05, 0.02, 0.01, 0.07, 0.03, 0.2, 0.02, 0.09});
        // model.fit(X_train, y_train_str, sample_weight, {}, cv_observations);
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.14420733842494515, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.10357828243742498, 0.00001));
    }

    void test_aplrclassifier_two_class_max_terms()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_max_terms";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.max_terms = 4;
        model.ridge_penalty = 0.0;

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
        // model.fit(X_train, y_train_str, sample_weight);
        // model.fit(X_train, y_train_str, sample_weight, {}, cv_observations);
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.1889066318262117, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.37047735615744898, 0.00001));
    }

    void test_aplrclassifier_two_class_predictor_min_observations_in_split()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_predictor_min_observations_in_split";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.05;
        model.bins = 300;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 100;
        model.max_interactions = 1000;
        model.min_observations_in_split = 20;
        model.ineligible_boosting_steps_added = 10;
        model.max_eligible_terms = 5;
        model.max_terms = 4;
        model.ridge_penalty = 0.0;

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
        model.fit(X_train, y_train_str, sample_weight, {}, MatrixXi(0, 0), {}, {}, {}, {}, {}, {}, {2, 3, 4, 5, 6, 7, 8, 9, 10});
        MatrixXd predicted_class_probabilities{model.predict_class_probabilities(X_test, false)};
        std::vector<std::string> predictions{model.predict(X_test, false)};
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.18970198357702517, 0.000001));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5, 0.00001));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.46597769437683995, 0.00001));
    }

    void test_aplrclassifier_two_class_ridge()
    {
        current_test_suite_name = "test_aplrclassifier_two_class_ridge";
        // Model
        APLRClassifier model{APLRClassifier()};
        model.m = 100;
        model.v = 0.5;
        model.n_jobs = 0;
        model.verbosity = 3;
        model.max_interaction_level = 1;
        model.ridge_penalty = 0.2;

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
        MatrixXd local_feature_contribution{model.calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model1{model.get_logit_model("0.000000").calculate_local_feature_contribution(X_test)};
        // MatrixXd lfc_model2{model.get_logit_model("1.000000").calculate_local_feature_contribution(X_test)};

        std::cout << "cv_error\n"
                  << model.get_cv_error() << "\n\n";
        add_test("model.get_cv_error()", is_approximately_equal(model.get_cv_error(), 0.18274188043364559));

        std::cout << "predicted_class_prob_mean\n"
                  << predicted_class_probabilities.mean() << "\n\n";
        add_test("predicted_class_probabilities.mean()", is_approximately_equal(predicted_class_probabilities.mean(), 0.5));

        std::cout << "local_feature_importance_mean\n"
                  << local_feature_contribution.mean() << "\n\n";
        add_test("local_feature_contribution.mean()", is_approximately_equal(local_feature_contribution.mean(), 0.027256376715801025));
    }

    void test_functions()
    {
        current_test_suite_name = "test_functions";
        // floating point comparisons
        double inf_left{-std::numeric_limits<double>::infinity()};
        double inf_right{std::numeric_limits<double>::infinity()};
        add_test("inf_left == inf_left", is_approximately_equal(inf_left, inf_left));
        add_test("inf_right == inf_right", is_approximately_equal(inf_right, inf_right));
        add_test("inf_left != inf_right", !is_approximately_equal(inf_left, inf_right));
        add_test("inf_right != inf_left", !is_approximately_equal(inf_right, inf_left));
        add_test("inf_left != 2.0", !is_approximately_equal(inf_left, 2.0));
        add_test("inf_right != 2.0", !is_approximately_equal(inf_right, 2.0));
        add_test("is_approximately_zero(0.0)", is_approximately_zero(0.0));
        add_test("!is_approximately_zero(0.0000001)", !is_approximately_zero(0.0000001));
        add_test("!is_approximately_zero(-0.0000001)", !is_approximately_zero(-0.0000001));
        add_test("!is_approximately_zero(inf_left)", !is_approximately_zero(inf_left));
        add_test("!is_approximately_zero(inf_right)", !is_approximately_zero(inf_right));

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
        add_test("error_mse", (is_approximately_equal(error_mse, 3.482) ? true : false));
        double error_mse_sw{calculate_mean_error(errors_mse_sw, sample_weight)};
        std::cout << "error_mse_sw: " << error_mse_sw << "\n\n";
        add_test("error_mse_sw", (is_approximately_equal(error_mse_sw, 0.4433, 0.0001) ? true : false));

        // testing for nan and infinity
        // matrix without nan or inf
        bool matrix_has_nan_or_inf_elements{matrix_has_nan_or_infinite_elements(y)};
        add_test("!matrix_has_nan_or_inf_elements", !matrix_has_nan_or_inf_elements ? true : false);

        VectorXd inf(5);
        inf << 1.0, 0.2, std::numeric_limits<double>::infinity(), 0.0, 0.5;
        matrix_has_nan_or_inf_elements = matrix_has_nan_or_infinite_elements(inf);
        add_test("matrix_has_nan_or_inf_elements with inf", matrix_has_nan_or_inf_elements ? true : false);

        VectorXd nan(5);
        nan << 1.0, 0.2, NAN_DOUBLE, 0.0, 0.5;
        matrix_has_nan_or_inf_elements = matrix_has_nan_or_infinite_elements(nan);
        add_test("matrix_has_nan_or_inf_elements with nan", matrix_has_nan_or_inf_elements ? true : false);

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
        add_test("trapezoidal_integration", is_approximately_equal(integration, 8.0));

        VectorXd weights_none{VectorXd(0)};
        VectorXd calculated_weights_if_not_provided{calculate_weights_if_they_are_not_provided(y_true)};
        VectorXd calculated_weights_if_provided{calculate_weights_if_they_are_not_provided(y_true, weights_different)};
        add_test("calculate_weights_if_they_are_not_provided (not provided)", calculated_weights_if_not_provided == weights_equal);
        add_test("calculate_weights_if_they_are_not_provided (provided)", calculated_weights_if_provided == weights_different);

        VectorXd y_pred(3);
        VectorXd weights_gini(3);
        y_pred << 1.0, 3.0, 2.0;
        weights_gini << 0.2, 0.5, 0.3;
        double gini{calculate_gini(y_true, y_pred, weights_gini)};
        add_test("calculate_gini", is_approximately_equal(gini, -0.1166667, 0.0000001));

        VectorXi int_vector(3);
        int_vector << 1, 1, 2;
        std::set<int> unique_integers{get_unique_integers(int_vector)};
        bool size_is_correct{unique_integers.size() == 2};
        add_test("get_unique_integers", size_is_correct);
    }

    void test_term()
    {
        current_test_suite_name = "test_term";
        // Setting up term instance p with default values
        Term p{Term()};
        add_test("p.base_term == 0", p.base_term == 0 ? true : false);

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
        add_test("p.calculate(X)", is_approximately_zero(values[0]) && is_approximately_equal(values[1], -0.711234, 0.00001) &&
                                           is_approximately_zero(values[2])
                                       ? true
                                       : false);

        // Testing calculate_prediction_contribution
        VectorXd contrib{p.calculate_contribution_to_linear_predictor(X)};
        std::cout << "Prediction contribution\n";
        std::cout << contrib << "\n\n";
        add_test("p.calculate_contribution_to_linear_predictor(X)", is_approximately_equal(contrib[1], -1.42247, 0.0001) && is_approximately_zero(contrib[0]) && is_approximately_zero(contrib[2]) ? true : false);

        // Testing equals_base_terms
        bool t1{Term::equals_given_terms(p, p.given_terms[0])};
        bool t2{Term::equals_given_terms(p, p)};
        add_test("!Term::equals_given_terms(p, p.given_terms[0])", t1 ? false : true);
        add_test("Term::equals_given_terms(p, p)", t2 ? true : false);

        // Testing copy constructor
        p.ineligible_boosting_steps = 10;
        Term p2{p};
        bool test_cpy = Term::equals_given_terms(p, p2) && &p.given_terms != &p2.given_terms && p.coefficient == p2.coefficient && &p.coefficient != &p2.coefficient && is_approximately_equal(p.split_point, p2.split_point) && &p.split_point != &p2.split_point && p.direction_right == p2.direction_right && &p.direction_right != &p2.direction_right && p.name == p2.name && &p.name != &p2.name &&
                        p.coefficient_steps.size() == p2.coefficient_steps.size() && &p.coefficient_steps != &p2.coefficient_steps && ((p.coefficient_steps - p2.coefficient_steps).array().abs() == 0).all() && p.base_term == p2.base_term && p.ineligible_boosting_steps == 10 && p2.ineligible_boosting_steps == 0;
        add_test("copy constructor", test_cpy);

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
        add_test("p == p2", test_equals1);
        add_test("p != p3", test_equals2);
        add_test("p != p4", test_equals3);
        add_test("p != p5", test_equals4);
        add_test("p != p6", test_equals5);
        add_test("p2 == p4 (NAN split_point)", test_equals6);

        // Testing interaction_level method
        Term p7{Term(1)};
        p7.given_terms.push_back(Term(2));
        Term p8{Term(3)};
        size_t pil{p.get_interaction_level()};
        size_t p5il{p5.get_interaction_level()};
        size_t p7il{p7.get_interaction_level()};
        size_t p8il{p8.get_interaction_level()};
        add_test("p.get_interaction_level() == 2", pil == 2 ? true : false);
        add_test("p5.get_interaction_level() == 2", p5il == 2 ? true : false);
        add_test("p7.get_interaction_level() == 1", p7il == 1 ? true : false);
        add_test("p8.get_interaction_level() == 0", p8il == 0 ? true : false);
    }

    void test_cv_results()
    {
        std::cout << "Testing CV results functionality..." << std::endl;
        current_test_suite_name = "test_cv_results";

        // 1. Setup data
        MatrixXd X(100, 2);
        VectorXd y(100);
        VectorXd sample_weight(100);
        std::mt19937 mersenne{0};
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        for (int i = 0; i < 100; ++i)
        {
            X(i, 0) = distribution(mersenne);
            X(i, 1) = distribution(mersenne);
            y(i) = X(i, 0) + X(i, 1) * 2 + distribution(mersenne);
            sample_weight(i) = 1.0 + (distribution(mersenne) + 1.0) / 2.0;
        }

        size_t cv_folds = 4;
        APLRRegressor model(10, 0.1, 0, "mse", "identity", 0, cv_folds);

        // 2. Test that accessing data before fitting throws
        bool threw = false;
        try
        {
            model.get_cv_y(0);
        }
        catch (const std::runtime_error &e)
        {
            threw = true;
        }
        add_test("access before fit throws", threw);

        // 3. Fit model
        model.fit(X, y, sample_weight);

        // 4. Test get_num_cv_folds
        add_test("get_num_cv_folds", model.get_num_cv_folds() == cv_folds);

        // 5. Test data retrieval and manually calculate cv_error
        VectorXd sample_weight_normalized = sample_weight / sample_weight.mean();
        size_t total_validation_obs = 0;
        double total_training_weight = 0.0;
        std::vector<double> fold_validation_errors_test1, fold_validation_errors_test2;
        std::vector<double> fold_training_weight_sums;

        for (size_t i = 0; i < cv_folds; ++i)
        {
            VectorXd cv_y = model.get_cv_y(i);
            VectorXd cv_preds = model.get_cv_validation_predictions(i);
            VectorXd cv_weights = model.get_cv_sample_weight(i);
            VectorXi cv_indexes = model.get_cv_validation_indexes(i);

            add_test("cv_y.size() > 0 for fold " + std::to_string(i), cv_y.size() > 0);
            add_test("cv_y.size() == cv_preds.size() for fold " + std::to_string(i), cv_y.size() == cv_preds.size());
            add_test("cv_y.size() == cv_weights.size() for fold " + std::to_string(i), cv_y.size() == cv_weights.size());
            add_test("cv_y.size() == cv_indexes.size() for fold " + std::to_string(i), cv_y.size() == cv_indexes.size());

            total_validation_obs += cv_y.size();

            // Test 1: Manually calculate validation error for this fold from get_cv_* methods
            VectorXd validation_errors1 = (cv_y - cv_preds).array().pow(2);
            double fold_validation_error1 = (validation_errors1.array() * cv_weights.array()).sum() / cv_weights.sum();
            fold_validation_errors_test1.push_back(fold_validation_error1);

            // Replicate the internal logic for calculating training weight sum for the fold
            // 1. Get training indexes for this fold
            std::vector<bool> is_validation(y.size(), false);
            for (int k = 0; k < cv_indexes.size(); ++k)
            {
                is_validation[cv_indexes[k]] = true;
            }
            std::vector<double> train_weights_for_fold;
            for (size_t k = 0; k < y.size(); ++k)
            {
                if (!is_validation[k]) // is a training observation
                {
                    train_weights_for_fold.push_back(sample_weight_normalized[k]);
                }
            }
            // 2. Sum it up (this now matches internal logic, as normalization is done before splitting)
            VectorXd train_weights_vec = Eigen::Map<VectorXd>(train_weights_for_fold.data(), train_weights_for_fold.size());
            double training_weight_sum = train_weights_vec.sum();

            fold_training_weight_sums.push_back(training_weight_sum);
            total_training_weight += training_weight_sum;

            // Test 2: Manually calculate validation error for this fold using original y/weights and the returned indexes
            VectorXd cv_y_from_indexes(cv_indexes.size());
            VectorXd cv_weights_from_indexes(cv_indexes.size());
            for (int j = 0; j < cv_indexes.size(); ++j)
            {
                cv_y_from_indexes(j) = y(cv_indexes(j));
                cv_weights_from_indexes(j) = sample_weight_normalized(cv_indexes(j));
            }
            VectorXd validation_errors2 = (cv_y_from_indexes - cv_preds).array().pow(2);
            double fold_validation_error2 = (validation_errors2.array() * cv_weights_from_indexes.array()).sum() / cv_weights_from_indexes.sum();
            fold_validation_errors_test2.push_back(fold_validation_error2);
        }
        add_test("total_validation_obs == y.size()", total_validation_obs == y.size());

        // Finalize and assert for the manual cv_error calculations
        double manual_cv_error1 = 0.0, manual_cv_error2 = 0.0;
        for (size_t i = 0; i < cv_folds; ++i)
        {
            manual_cv_error1 += fold_validation_errors_test1[i] * (fold_training_weight_sums[i] / total_training_weight);
            manual_cv_error2 += fold_validation_errors_test2[i] * (fold_training_weight_sums[i] / total_training_weight);
        }
        add_test("manual_cv_error1 == model.get_cv_error()", is_approximately_equal(manual_cv_error1, model.get_cv_error()));
        add_test("manual_cv_error2 == model.get_cv_error()", is_approximately_equal(manual_cv_error2, model.get_cv_error()));

        // 6. Test clear_cv_results
        model.clear_cv_results();
        add_test("get_num_cv_folds after clear is 0", model.get_num_cv_folds() == 0);

        // 7. Test that accessing data after clearing throws
        threw = false;
        try
        {
            model.get_cv_y(0);
        }
        catch (const std::runtime_error &e)
        {
            threw = true;
            add_test("access after clear throws (message check)", std::string(e.what()).find("not available") != std::string::npos);
        }
        add_test("access after clear throws", threw);

        // 8. Test APLRClassifier
        std::vector<std::string> y_class(100);
        for (int i = 0; i < 100; ++i)
        {
            y_class[i] = y(i) > y.mean() ? "A" : "B";
        }
        APLRClassifier classifier(10, 0.1, 0, 0, cv_folds);
        classifier.fit(X, y_class);

        // Check that data exists in one of the logit models
        APLRRegressor logit_model_before_clear = classifier.get_logit_model("A");
        add_test("classifier cv folds before clear", logit_model_before_clear.get_num_cv_folds() == cv_folds);
        add_test("classifier cv data before clear", logit_model_before_clear.get_cv_y(0).size() > 0);

        // Clear results and check again
        classifier.clear_cv_results();
        APLRRegressor logit_model_after_clear = classifier.get_logit_model("A");
        add_test("classifier cv folds after clear", logit_model_after_clear.get_num_cv_folds() == 0);

        std::cout << "CV results functionality tests passed." << std::endl;
    }
};

int main()
{
    Tests tests{Tests()};
    tests.test_aplrregressor_huber();
    tests.test_aplrregressor_huber_log_link();
    tests.test_aplrregressor_mean_bias_correction();
    tests.test_aplrregressor_ridge();
    tests.test_aplrregressor_mse_predictor_min_observations_in_split();
    tests.test_aplrregressor_cauchy_term_limit();
    tests.test_aplrregressor_cauchy_predictor_specific_penalties_and_learning_rates();
    tests.test_aplrregressor_cauchy_penalties();
    tests.test_aplrregressor_cauchy_linear_effects_only_first();
    tests.test_aplrregressor_cauchy_linear_effects_only_first_2();
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
    tests.test_aplrregressor_neg_top_quantile_mean_response();
    tests.test_aplrregressor_bottom_quantile_mean_response();
    tests.test_aplrregressor_weibull();
    tests.test_aplrregressor();
    tests.test_aplrregressor_faster_convergence_identity();
    tests.test_aplrregressor_faster_convergence_log();
    tests.test_aplrregressor_exponential_power();
    tests.test_aplr_classifier_multi_class_other_params();
    tests.test_aplrclassifier_multi_class();
    tests.test_aplrclassifier_two_class_other_params();
    tests.test_aplrclassifier_two_class_val_index();
    tests.test_aplrclassifier_two_class();
    tests.test_aplrclassifier_two_class_penalties();
    tests.test_aplrclassifier_two_class_predictor_specific_penalties_and_learning_rates();
    tests.test_aplrclassifier_two_class_max_terms();
    tests.test_aplrclassifier_two_class_predictor_min_observations_in_split();
    tests.test_aplrclassifier_two_class_ridge();
    tests.test_cv_results();
    tests.test_functions();
    tests.test_term();
    tests.summarize_results();
}