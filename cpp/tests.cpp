#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "APLRClassifier.h"
#include "APLRRegressor.h"
#include "CppDataFrame.h"
#include "Preprocessor.h"
#include "ThreadPool.h"
#include "functions.h"
#include "term.h"

using namespace Eigen;

double calculate_custom_loss(const VectorXd &y, const VectorXd &predictions, const VectorXd &, const VectorXi &, const MatrixXd &)
{
    return (y - predictions).array().square().mean();
}

VectorXd calculate_custom_negative_gradient(const VectorXd &y, const VectorXd &predictions, const VectorXi &, const MatrixXd &)
{
    return y - predictions;
}

VectorXd calculate_custom_hessian(const VectorXd &y, const VectorXd &, const VectorXi &, const MatrixXd &)
{
    return VectorXd::Ones(y.size());
}

bool approximately_equal_matrix(const MatrixXd &left, const MatrixXd &right, double tolerance = 1e-9)
{
    return left.rows() == right.rows() && left.cols() == right.cols() &&
           ((left - right).array().abs() <= tolerance).all();
}

struct RegressionFixture
{
    MatrixXd train;
    MatrixXd test;
    VectorXd response;
    VectorXd test_response;
    VectorXd weights;
};

RegressionFixture make_regression_fixture()
{
    RegressionFixture fixture{MatrixXd(24, 3), MatrixXd(8, 3), VectorXd(24), VectorXd(8), VectorXd::Ones(24)};
    for (Index row = 0; row < fixture.train.rows(); ++row)
    {
        fixture.train(row, 0) = -2.0 + row / 5.0;
        fixture.train(row, 1) = (row % 4) - 1.5;
        fixture.train(row, 2) = (row % 3 == 0) ? 0.0 : 1.0;
        fixture.response(row) = 1.5 + 2.0 * fixture.train(row, 0) - fixture.train(row, 1) + fixture.train(row, 2);
        fixture.weights(row) = 1.0 + (row % 3) * 0.25;
    }
    for (Index row = 0; row < fixture.test.rows(); ++row)
    {
        fixture.test(row, 0) = -1.8 + row / 4.0;
        fixture.test(row, 1) = (row % 4) - 1.5;
        fixture.test(row, 2) = row % 2;
        fixture.test_response(row) = 1.5 + 2.0 * fixture.test(row, 0) - fixture.test(row, 1) + fixture.test(row, 2);
    }
    return fixture;
}

CppDataFrame make_numeric_frame(const MatrixXd &matrix)
{
    return CppDataFrame::from_matrix(matrix, {"signal", "level", "flag"});
}

class Tests
{
    size_t passed = 0;
    size_t failed = 0;

    void check(const std::string &name, bool condition)
    {
        if (condition)
            ++passed;
        else
        {
            ++failed;
            std::cerr << "FAIL: " << name << "\n";
        }
    }

    template <typename Callable>
    void test(const std::string &name, Callable callable)
    {
        try
        {
            callable();
            check(name, true);
        }
        catch (const std::exception &error)
        {
            std::cerr << "FAIL: " << name << " threw: " << error.what() << "\n";
            check(name, false);
        }
    }

    APLRRegressor configured_regressor(const std::string &loss = "mse")
    {
        APLRRegressor model(12, 0.2, 7, loss, "identity", 1, 3, 6, 0, 1, 10, 2, 2, 3);
        model.max_interaction_level = 1;
        model.max_terms = 6;
        model.early_stopping_rounds = 4;
        model.penalty_for_interactions = 0.0;
        model.ridge_penalty = 0.01;
        return model;
    }

    void dataframe_and_preprocessor()
    {
        CppDataFrame frame;
        frame.add_column("number", std::vector<double>{1.0, NAN_DOUBLE, 3.0});
        frame.add_column("category", std::vector<std::string>{"a", "b", "a"});
        check("dataframe row count", frame.get_num_rows() == 3);
        check("dataframe preserves columns", frame.get_column_names_in_order() == std::vector<std::string>{"number", "category"});

        Preprocessor preprocessor;
        VectorXd weights = VectorXd::Ones(3);
        auto transformed = preprocessor.fit_transform(frame, weights);
        check("preprocessor fits", preprocessor.is_fitted());
        check("preprocessor expands missing and category columns", transformed.first.rows() == 3 && transformed.first.cols() == 4);
        check("preprocessor transform has stable shape", preprocessor.transform(frame).first.cols() == transformed.first.cols());
        MatrixXd numeric(3, 2);
        numeric << 1.0, 4.0, 2.0, 5.0, 3.0, 6.0;
        Preprocessor matrix_preprocessor;
        auto matrix_transformed = matrix_preprocessor.fit_transform(numeric, weights, {"left", "right"});
        check("matrix preprocessor fits", matrix_preprocessor.is_fitted());
        check("matrix preprocessor preserves numeric shape", matrix_transformed.first.rows() == 3 && matrix_transformed.first.cols() == 2);
        check("matrix preprocessor preserves names", matrix_transformed.second == std::vector<std::string>{"left", "right"});

        bool threw = false;
        try
        {
            frame.add_column("wrong", std::vector<double>{1.0});
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("dataframe rejects unequal columns", threw);
    }

    void imputer_encoder_and_utilities()
    {
        MedianImputer<double> imputer;
        imputer.fit({1.0, NAN_DOUBLE, 5.0}, {1.0, 1.0, 1.0});
        auto result = imputer.transform({1.0, NAN_DOUBLE, 5.0});
        check("median imputer uses observed median", is_approximately_equal(imputer.get_median(), 3.0));
        check("median imputer creates missing indicator", result.second == std::vector<double>{0.0, 1.0, 0.0});

        OneHotEncoder encoder;
        encoder.fit({"red", "blue", "red"});
        std::vector<std::vector<int>> encoded = encoder.transform({"blue", "green"});
        check("one hot encoder learns categories", encoder.get_categories().size() == 2);
        check("one hot encoder preserves row count", encoded.size() == 2 && encoded.front().size() == 2);

        VectorXd values(3);
        values << 1.0, 2.0, 3.0;
        check("mse utility", is_approximately_equal(calculate_mean_error(calculate_errors(values, values + VectorXd::Ones(3), VectorXd::Ones(3)), VectorXd::Ones(3)), 1.0));
        VectorXi integers(3);
        integers << 1, 1, 2;
        check("unique integer utility", get_unique_integers(integers).size() == 2);
    }

    void dataframe_edge_cases()
    {
        CppDataFrame frame;
        frame.add_column("z", std::vector<double>{3.0, 4.0});
        frame.add_column("a", std::vector<double>{1.0, 2.0});
        check("dataframe numeric type detection", frame.is_numeric_column("z") && !frame.empty());
        check("dataframe map names are complete", get_unique_strings(frame.get_column_names()).size() == 2);
        check("dataframe numeric access", frame.get_numeric_column("a")[1] == 2.0);
        auto matrix_result = frame.to_matrix();
        check("dataframe matrix conversion shape", matrix_result.first.rows() == 2 && matrix_result.first.cols() == 2);
        check("dataframe matrix conversion names", matrix_result.second == std::vector<std::string>{"z", "a"});

        CppDataFrame copied(frame);
        copied.add_column("a", std::vector<double>{8.0, 9.0});
        check("dataframe copy is deep", frame.get_numeric_column("a")[0] == 1.0 && copied.get_numeric_column("a")[0] == 8.0);
        CppDataFrame assigned;
        assigned = frame;
        check("dataframe assignment copies values", assigned.get_numeric_column("z")[0] == 3.0);

        CppDataFrame mixed = frame;
        mixed.add_column("kind", std::vector<std::string>{"x", "y"});
        bool wrong_type = false;
        try
        {
            mixed.to_matrix();
        }
        catch (const std::runtime_error &)
        {
            wrong_type = true;
        }
        check("dataframe rejects categorical matrix conversion", wrong_type);
        bool missing_column = false;
        try
        {
            frame.get_string_column("a");
        }
        catch (const std::runtime_error &)
        {
            missing_column = true;
        }
        check("dataframe rejects wrong column type", missing_column);
    }

    void utility_functions()
    {
        VectorXd y(3), predicted(3), weights(3);
        y << 1.0, 2.0, 4.0;
        predicted << 2.0, 1.0, 2.0;
        weights << 1.0, 2.0, 1.0;
        VectorXi groups(3);
        groups << 0, 0, 1;
        std::set<int> unique_groups{0, 1};
        check("all_are_equal", all_are_equal(y, y));
        check("loss helpers return aligned vectors", calculate_mse_errors(y, predicted).size() == 3 && calculate_absolute_errors(y, predicted).size() == 3 && calculate_huber_errors(y, predicted, 1.0).size() == 3);
        check("probability and positive loss helpers", calculate_binomial_errors(VectorXd::Constant(3, 1.0), VectorXd::Constant(3, 0.5)).size() == 3 && calculate_poisson_errors(y, predicted).size() == 3 && calculate_gamma_errors(y, predicted).size() == 3);
        check("specialized loss helpers", calculate_tweedie_errors(y, predicted).size() == 3 && calculate_negative_binomial_errors(y, predicted, 1.0).size() == 3 && calculate_cauchy_errors(y, predicted, 1.0).size() == 3 && calculate_weibull_errors(y, predicted, 1.0).size() == 3 && calculate_exponential_power_errors(y, predicted, 2.0).size() == 3);
        check("quantile and group loss helpers", calculate_quantile_errors(y, predicted, 0.5).size() == 3 && calculate_group_mse_errors(y, predicted, groups, unique_groups, weights).size() == 3);
        check("weighted average", is_approximately_equal(calculate_weighted_average(y, weights), 2.25));
        check("sum and one-observation errors", is_approximately_equal(calculate_sum_error(VectorXd::Ones(3)), 3.0) && is_approximately_equal(calculate_mse_error_one_observation(2.0, 1.0), 1.0) && is_approximately_equal(calculate_error_one_observation(2.0, 1.0, 2.0), 2.0));
        check("indicator helpers", calculate_indicator(y).size() == 3 && calculate_indicator(groups).size() == 3);
        check("standard deviation and quantile", std::isfinite(calculate_standard_deviation(y)) && std::isfinite(calculate_quantile(y, 0.5)));
        check("duplicate removal", remove_duplicate_elements_from_vector(std::vector<size_t>{2, 1, 2}).size() == 2 && remove_duplicate_elements_from_vector(std::vector<double>{2.0, 1.0, 2.0}).size() == 2);
        check("combination generation", generate_combinations_and_one_additional_column({{1.0, 2.0}, {3.0, 4.0}}).rows() == 4);
        VectorXd linear(3);
        linear << -100.0, 0.0, 100.0;
        VectorXd logit = transform_linear_predictor_to_predictions(linear, "logit");
        check("link transforms are bounded", logit[0] > 0.0 && logit[2] < 1.0 && transform_linear_predictor_to_predictions(linear, "log")[1] == 1.0);
        bool invalid_quantile = false;
        try
        {
            calculate_quantile(y, 1.1);
        }
        catch (const std::runtime_error &)
        {
            invalid_quantile = true;
        }
        check("quantile validates its range", invalid_quantile);
        bool invalid_matrix = false;
        try
        {
            VectorXd bad = y;
            bad[1] = NAN_DOUBLE;
            throw_error_if_matrix_has_nan_or_infinite_elements(bad, "bad");
        }
        catch (const std::runtime_error &)
        {
            invalid_matrix = true;
        }
        check("matrix validation rejects NaN", invalid_matrix);
    }

    void thread_pool_and_term_details()
    {
        ThreadPool pool(2);
        auto first = pool.enqueue([](int value)
                                  { return value * 2; }, 21);
        auto second = pool.enqueue([]
                                   { return 7; });
        check("thread pool returns task results", first.get() == 42 && second.get() == 7);

        Term term(0);
        term.split_point = 0.5;
        term.coefficient = 2.0;
        term.set_monotonic_constraint(1);
        check("term monotonic constraint accessors", term.get_monotonic_constraint() == 1);
        MatrixXd input(3, 1);
        input << 0.0, 0.5, 1.0;
        check("term univariate evaluation", term.calculate_without_interactions(input.col(0)).size() == 3 && term.calculate_contribution_to_linear_predictor(input).size() == 3);
        term.ineligible_boosting_steps = 4;
        term.estimated_term_importance = 2.0;
        check("term state accessors", !term.get_can_be_used_as_a_given_term() && is_approximately_equal(term.get_estimated_term_importance(), 2.0));
        Term copy = term;
        check("term copy resets ineligibility", copy.ineligible_boosting_steps == 0 && Term::equals_given_terms(term, copy));
        VectorXd gradient(3), sample_weight = VectorXd::Ones(3);
        gradient << 0.0, 1.0, 2.0;
        term.estimate_split_point(input, gradient, sample_weight, 3, 0.5, 1, false, 0.0, 0.0, 0.0, 1.0);
        check("term split estimation produces finite coefficient", std::isfinite(term.coefficient));
    }

    void regressor_fit_and_outputs()
    {
        auto fixture = make_regression_fixture();
        APLRRegressor model = configured_regressor();
        model.fit(fixture.train, fixture.response, fixture.weights, {"signal", "level", "flag"});
        VectorXd predictions = model.predict(fixture.test);
        check("regressor prediction shape", predictions.size() == fixture.test.rows());
        check("regressor predictions are finite", !matrix_has_nan_or_infinite_elements(predictions));
        check("regressor has fitted terms", model.terms.size() > 0);
        check("regressor reports finite cv error", std::isfinite(model.get_cv_error()));
        VectorXd feature_importance = model.calculate_feature_importance(fixture.test);
        VectorXd term_importance = model.calculate_term_importance(fixture.test);
        MatrixXd local_features = model.calculate_local_feature_contribution(fixture.test);
        MatrixXd local_terms = model.calculate_local_term_contribution(fixture.test);
        MatrixXd terms = model.calculate_terms(fixture.test);
        check("regressor feature importance shape", feature_importance.size() == fixture.test.cols());
        check("regressor term importance shape", term_importance.size() == static_cast<Index>(model.terms.size()));
        check("regressor local contribution shapes", local_features.rows() == fixture.test.rows() && local_features.cols() == fixture.test.cols());
        check("regressor local term contribution shape", local_terms.rows() == fixture.test.rows() && local_terms.cols() == static_cast<Index>(model.terms.size()));
        check("regressor term output shape", terms.rows() == fixture.test.rows() && terms.cols() == static_cast<Index>(model.terms.size()));
        check("regressor metadata lengths agree", model.get_term_names().size() == model.terms.size() + 1 && model.get_term_coefficients().size() == static_cast<Index>(model.terms.size() + 1) && model.get_term_affiliations().size() == model.terms.size());
        check("regressor reports optimal m", model.get_optimal_m() <= model.m);
        check("regressor reports validation metric", model.get_validation_tuning_metric() == "default");
        check("regressor feature getter shape", model.get_feature_importance().size() == fixture.test.cols());
        check("regressor term getter shape", model.get_term_importance().size() == static_cast<Index>(model.terms.size()));
        check("regressor predictor metadata shape", model.get_term_main_predictor_indexes().size() == static_cast<Index>(model.terms.size()) && model.get_term_interaction_levels().size() == static_cast<Index>(model.terms.size()));
        check("regressor unique affiliations are available", model.get_unique_term_affiliations().size() == model.get_base_predictors_in_each_unique_term_affiliation().size());
        check("regressor selected contribution shape", model.calculate_local_contribution_from_selected_terms(fixture.test, {0, 1}).size() == fixture.test.rows());
        check("regressor matrix shape API", model.get_main_effect_shape(0).size() > 0);
    }

    void regressor_losses_and_callbacks()
    {
        auto fixture = make_regression_fixture();
        for (const std::string loss : {"mse", "mae", "huber", "cauchy"})
        {
            APLRRegressor model = configured_regressor(loss);
            model.fit(fixture.train, fixture.response, fixture.weights);
            check("regressor loss " + loss + " predicts", model.predict(fixture.test).size() == fixture.test.rows());
        }

        APLRRegressor custom = configured_regressor();
        bool callback_called = false;
        custom.calculate_custom_loss_function = calculate_custom_loss;
        custom.calculate_custom_negative_gradient_function = calculate_custom_negative_gradient;
        custom.calculate_custom_hessian_function = calculate_custom_hessian;
        custom.set_progress_callback([&callback_called](const std::string &)
                                     { callback_called = true; });
        custom.fit(fixture.train, fixture.response, fixture.weights);
        check("custom loss function is installed", static_cast<bool>(custom.calculate_custom_loss_function));
        check("custom gradient function is installed", static_cast<bool>(custom.calculate_custom_negative_gradient_function));
        check("custom hessian function is installed", static_cast<bool>(custom.calculate_custom_hessian_function));
        custom.remove_provided_custom_functions();
        check("custom functions can be removed", !custom.calculate_custom_loss_function && !custom.calculate_custom_negative_gradient_function && !custom.calculate_custom_hessian_function);
        custom.clear_progress_callback();
    }

    void remaining_model_modes_and_copying()
    {
        auto fixture = make_regression_fixture();
        VectorXd positive_response = fixture.response.array() - fixture.response.minCoeff() + 1.0;
        const std::vector<std::string> positive_losses = {"poisson", "gamma", "tweedie", "negative_binomial", "weibull", "exponential_power", "quantile"};
        for (const std::string loss : positive_losses)
        {
            APLRRegressor model = configured_regressor(loss);
            model.quantile = 0.75;
            model.dispersion_parameter = 1.5;
            model.fit(fixture.train, positive_response, fixture.weights);
            check("remaining regressor loss " + loss + " predicts", model.predict(fixture.test).size() == fixture.test.rows());
        }

        APLRRegressor log_model = configured_regressor();
        log_model.link_function = "log";
        log_model.fit(fixture.train, positive_response, fixture.weights);
        check("log link produces positive predictions", (log_model.predict(fixture.test).array() > 0.0).all());

        APLRRegressor custom = configured_regressor();
        custom.calculate_custom_transform_linear_predictor_to_predictions_function = [](const VectorXd &values)
        { return values.array().exp().matrix(); };
        custom.calculate_custom_differentiate_predictions_wrt_linear_predictor_function = [](const VectorXd &values)
        { return values.array().exp().matrix(); };
        custom.calculate_custom_differentiate2_predictions_wrt_linear_predictor_function = [](const VectorXd &values)
        { return values.array().exp().matrix(); };
        check("custom link functions are installed", static_cast<bool>(custom.calculate_custom_transform_linear_predictor_to_predictions_function) && static_cast<bool>(custom.calculate_custom_differentiate_predictions_wrt_linear_predictor_function) && static_cast<bool>(custom.calculate_custom_differentiate2_predictions_wrt_linear_predictor_function));

        APLRRegressor original = configured_regressor();
        original.fit(fixture.train, fixture.response, fixture.weights);
        APLRRegressor copied(original);
        APLRRegressor assigned;
        assigned = original;
        check("regressor copy preserves predictions", approximately_equal_matrix(original.predict(fixture.test), copied.predict(fixture.test), 1e-7));
        check("regressor assignment preserves predictions", approximately_equal_matrix(original.predict(fixture.test), assigned.predict(fixture.test), 1e-7));
    }

    void dataframe_overloads_and_preprocessing()
    {
        auto fixture = make_regression_fixture();
        CppDataFrame train = make_numeric_frame(fixture.train);
        CppDataFrame test_frame = make_numeric_frame(fixture.test);
        APLRRegressor matrix_model = configured_regressor();
        APLRRegressor frame_model = configured_regressor();
        matrix_model.fit(fixture.train, fixture.response, fixture.weights, {"signal", "level", "flag"});
        frame_model.fit(train, fixture.response, fixture.weights);
        check("dataframe regressor predictions match matrix", approximately_equal_matrix(matrix_model.predict(fixture.test), frame_model.predict(test_frame), 1e-7));
        check("dataframe feature importance matches matrix", approximately_equal_matrix(matrix_model.calculate_feature_importance(fixture.test), frame_model.calculate_feature_importance(test_frame), 1e-7));
        check("dataframe term importance matches matrix", approximately_equal_matrix(matrix_model.calculate_term_importance(fixture.test), frame_model.calculate_term_importance(test_frame), 1e-7));
        check("dataframe local terms match matrix", approximately_equal_matrix(matrix_model.calculate_local_term_contribution(fixture.test), frame_model.calculate_local_term_contribution(test_frame), 1e-7));
        check("dataframe terms match matrix", approximately_equal_matrix(matrix_model.calculate_terms(fixture.test), frame_model.calculate_terms(test_frame), 1e-7));
        check("dataframe selected contribution matches matrix", approximately_equal_matrix(matrix_model.calculate_local_contribution_from_selected_terms(fixture.test, {0, 1}), frame_model.calculate_local_contribution_from_selected_terms(test_frame, {0, 1}), 1e-7));

        APLRRegressor preprocessed = configured_regressor();
        preprocessed.preprocess = true;
        preprocessed.fit(train, fixture.response, fixture.weights);
        check("preprocessed dataframe prediction shape", preprocessed.predict(test_frame).size() == fixture.test.rows());
        check("preprocessor is fitted through model", preprocessed.preprocessor.is_fitted());
    }

    void classifier_outputs()
    {
        auto fixture = make_regression_fixture();
        std::vector<std::string> labels;
        for (Index row = 0; row < fixture.response.size(); ++row)
            labels.push_back(fixture.response(row) > fixture.response.mean() ? "high" : "low");

        APLRClassifier classifier(12, 0.2, 7, 1, 3, 6, 0, 1, 10, 2, 2, 3);
        classifier.preprocess = false;
        classifier.max_terms = 6;
        classifier.early_stopping_rounds = 4;
        classifier.fit(fixture.train, labels, fixture.weights, {"signal", "level", "flag"});
        MatrixXd probabilities = classifier.predict_class_probabilities(fixture.test);
        auto predictions = classifier.predict(fixture.test);
        MatrixXd local_features = classifier.calculate_local_feature_contribution(fixture.test);
        CppDataFrame train_frame = make_numeric_frame(fixture.train);
        CppDataFrame test_frame = make_numeric_frame(fixture.test);
        APLRClassifier frame_classifier = classifier;
        frame_classifier.fit(train_frame, labels, fixture.weights);
        check("classifier discovers two categories", classifier.get_categories().size() == 2);
        check("classifier probability shape", probabilities.rows() == fixture.test.rows() && probabilities.cols() == 2);
        check("classifier probabilities normalize", ((probabilities.rowwise().sum().array() - 1.0).abs() < 1e-8).all());
        check("classifier predictions shape", predictions.size() == static_cast<size_t>(fixture.test.rows()));
        check("classifier local contribution shape", local_features.rows() == fixture.test.rows() && local_features.cols() == fixture.test.cols());
        check("classifier dataframe probabilities match matrix", approximately_equal_matrix(probabilities, frame_classifier.predict_class_probabilities(test_frame), 1e-7));
        check("classifier dataframe predictions match matrix", predictions == frame_classifier.predict(test_frame));
        check("classifier dataframe contributions have expected shape", frame_classifier.calculate_local_feature_contribution(test_frame).rows() == fixture.test.rows());
        check("classifier predictions use known categories", std::all_of(predictions.begin(), predictions.end(), [&classifier](const std::string &value)
                                                                         { return std::find(classifier.categories.begin(), classifier.categories.end(), value) != classifier.categories.end(); }));
        check("classifier cv error is finite", std::isfinite(classifier.get_cv_error()));
        check("classifier validation steps are available", classifier.get_validation_error_steps().rows() > 0 && classifier.get_validation_error_steps().cols() > 0);
        check("classifier feature importance shape", classifier.get_feature_importance().size() == fixture.test.cols());
        check("classifier logit models are available", classifier.get_logit_model(classifier.categories.front()).get_num_cv_folds() == classifier.cv_folds);
        classifier.clear_cv_results();
        check("classifier clears cv results", classifier.get_logit_model(classifier.categories.front()).get_num_cv_folds() == 0);
    }

    void multiclass_and_regressor_configuration()
    {
        auto fixture = make_regression_fixture();
        std::vector<std::string> labels;
        for (Index row = 0; row < fixture.response.size(); ++row)
            labels.push_back("class_" + std::to_string(row % 3));

        APLRClassifier classifier(8, 0.2, 11, 1, 2, 5, 0, 1, 6, 2, 2, 3);
        classifier.preprocess = false;
        classifier.fit(fixture.train, labels, fixture.weights);
        MatrixXd probabilities = classifier.predict_class_probabilities(fixture.test);
        check("multiclass classifier discovers all categories", classifier.get_categories().size() == 3);
        check("multiclass probability columns match categories", probabilities.cols() == 3);
        check("multiclass probabilities normalize", ((probabilities.rowwise().sum().array() - 1.0).abs() < 1e-8).all());
        check("multiclass unique affiliations are available", classifier.get_unique_term_affiliations().size() == classifier.get_base_predictors_in_each_unique_term_affiliation().size());

        APLRRegressor constrained = configured_regressor();
        constrained.fit(fixture.train, fixture.response, fixture.weights, {"signal", "level", "flag"}, MatrixXi(0, 0), {0}, {1, -1, 0}, {}, {{0, 1}}, {}, {0.5, 0.5, 0.5}, {0.01, 0.01, 0.01}, {0.02, 0.02, 0.02}, {2, 2, 2});
        check("regressor accepts predictor constraints", constrained.predict(fixture.test).size() == fixture.test.rows());
        double original_intercept = constrained.get_intercept();
        VectorXd before_shift = constrained.predict(fixture.test, false);
        constrained.set_intercept(original_intercept + 2.0);
        VectorXd after_shift = constrained.predict(fixture.test, false);
        check("regressor intercept setter shifts predictions", ((after_shift - before_shift).array() - 2.0).abs().maxCoeff() < 1e-8);
    }

    void parallel_and_hyperparameter_configuration()
    {
        auto fixture = make_regression_fixture();
        APLRRegressor serial = configured_regressor();
        APLRRegressor parallel = configured_regressor();
        serial.n_jobs = 1;
        parallel.n_jobs = 2;
        serial.fit(fixture.train, fixture.response, fixture.weights);
        parallel.fit(fixture.train, fixture.response, fixture.weights);
        check("parallel regressor matches serial predictions", approximately_equal_matrix(serial.predict(fixture.test), parallel.predict(fixture.test), 1e-7));
        check("parallel regressor matches serial feature importance", approximately_equal_matrix(serial.get_feature_importance(), parallel.get_feature_importance(), 1e-7));
        check("parallel regressor matches serial cv error", is_approximately_equal(serial.get_cv_error(), parallel.get_cv_error(), 1e-7));

        VectorXi groups(fixture.train.rows());
        for (Index row = 0; row < groups.size(); ++row)
            groups(row) = row % 3;
        APLRRegressor configured = configured_regressor("group_mse");
        configured.group_mse_by_prediction_bins = 3;
        configured.group_mse_cycle_min_obs_in_bin = 2;
        configured.boosting_steps_before_interactions_are_allowed = 1;
        configured.monotonic_constraints_ignore_interactions = true;
        configured.penalty_for_non_linearity = 0.1;
        configured.penalty_for_interactions = 0.1;
        configured.mean_bias_correction = true;
        configured.faster_convergence = true;
        configured.validation_ratio = 0.25;
        configured.validation_tuning_metric = "mse";
        configured.fit(fixture.train, fixture.response, fixture.weights, {}, MatrixXi(0, 0), {}, {}, groups, {{0, 1}}, {}, {0.2, 0.3, 0.4}, {0.01, 0.02, 0.03}, {0.04, 0.05, 0.06}, {2, 2, 2});
        check("configured regressor modes produce predictions", configured.predict(fixture.test).size() == fixture.test.rows());
        check("configured validation metric is reported", configured.get_validation_tuning_metric() == "mse");

        APLRRegressor custom_validation = configured_regressor();
        custom_validation.validation_tuning_metric = "custom_function";
        custom_validation.calculate_custom_validation_error_function = calculate_custom_loss;
        custom_validation.fit(fixture.train, fixture.response, fixture.weights);
        check("custom validation metric produces finite error", std::isfinite(custom_validation.get_cv_error()));

        std::vector<std::string> labels;
        for (Index row = 0; row < fixture.response.size(); ++row)
            labels.push_back(fixture.response(row) > fixture.response.mean() ? "high" : "low");
        APLRClassifier serial_classifier(10, 0.2, 7, 1, 3, 6, 0, 1, 10, 2, 2, 3);
        APLRClassifier parallel_classifier = serial_classifier;
        parallel_classifier.n_jobs = 2;
        serial_classifier.preprocess = false;
        parallel_classifier.preprocess = false;
        serial_classifier.fit(fixture.train, labels, fixture.weights);
        parallel_classifier.fit(fixture.train, labels, fixture.weights);
        check("parallel classifier matches serial probabilities", approximately_equal_matrix(serial_classifier.predict_class_probabilities(fixture.test), parallel_classifier.predict_class_probabilities(fixture.test), 1e-7));
        check("parallel classifier matches serial labels", serial_classifier.predict(fixture.test) == parallel_classifier.predict(fixture.test));
    }

    void cv_and_term_behavior()
    {
        auto fixture = make_regression_fixture();
        APLRRegressor model = configured_regressor();
        model.fit(fixture.train, fixture.response, fixture.weights);
        check("cv fold count", model.get_num_cv_folds() == 3);
        size_t validation_rows = 0;
        for (size_t fold = 0; fold < model.get_num_cv_folds(); ++fold)
        {
            check("cv fold has aligned data", model.get_cv_y(fold).size() == model.get_cv_validation_predictions(fold).size());
            validation_rows += model.get_cv_y(fold).size();
        }
        check("cv covers every observation", validation_rows == static_cast<size_t>(fixture.train.rows()));
        model.clear_cv_results();
        check("cv clear removes folds", model.get_num_cv_folds() == 0);

        Term term(0);
        term.split_point = 0.5;
        MatrixXd input(2, 1);
        input << 0.0, 1.0;
        VectorXd contribution = term.calculate(input);
        check("term evaluates both branches", contribution.size() == 2 && contribution(0) != contribution(1));
    }

    void ridge_bins_and_interactions()
    {
        MatrixXd x(12, 1);
        VectorXd gradient(12), weights = VectorXd::Ones(12);
        for (Index row = 0; row < x.rows(); ++row)
        {
            x(row, 0) = static_cast<double>(row);
            gradient(row) = 0.5 * row;
        }
        Term unregularized(0), regularized(0);
        unregularized.estimate_split_point(x, gradient, weights, 4, 0.5, 2, false, 0.0, 0.0, 0.0, 1.0);
        regularized.estimate_split_point(x, gradient, weights, 4, 0.5, 2, false, 0.0, 0.0, 10.0, 1.0);
        check("term creates bins", unregularized.bins_start_index.size() > 1 && unregularized.bins_end_index.size() == unregularized.bins_start_index.size());
        check("term bin split points align", unregularized.bins_split_points_left.size() == unregularized.bins_split_points_right.size() && unregularized.values_discretized.size() == static_cast<Index>(unregularized.bins_start_index.size()));
        check("ridge penalty produces finite coefficient", std::isfinite(unregularized.coefficient) && std::isfinite(regularized.coefficient));
        check("ridge penalty affects coefficient estimation", !is_approximately_equal(unregularized.coefficient, regularized.coefficient, 1e-8));

        auto fixture = make_regression_fixture();
        VectorXd interaction_response(fixture.response.size());
        for (Index row = 0; row < interaction_response.size(); ++row)
            interaction_response(row) = fixture.train(row, 0) * fixture.train(row, 1) + fixture.train(row, 2);
        APLRRegressor interaction_model(30, 0.5, 17, "mse", "identity", 1, 2, 6, 0, 2, 100, 2, 2, 4);
        interaction_model.penalty_for_interactions = 0.0;
        interaction_model.max_interaction_level = 2;
        interaction_model.fit(fixture.train, interaction_response, fixture.weights, {"signal", "level", "flag"}, MatrixXi(0, 0), {}, {}, {}, {{0, 1}}, {}, {}, {}, {});
        bool has_interaction = false;
        bool interactions_obey_constraint = true;
        for (Term &term : interaction_model.terms)
        {
            if (term.get_interaction_level() > 0)
            {
                has_interaction = true;
                std::set<size_t> used_predictors{term.base_term};
                for (const Term &given_term : term.given_terms)
                    used_predictors.insert(given_term.base_term);
                for (size_t predictor : used_predictors)
                    interactions_obey_constraint = interactions_obey_constraint && (predictor == 0 || predictor == 1);
            }
        }
        check("model builds interaction terms", has_interaction);
        check("interaction constraints restrict predictors", interactions_obey_constraint);
    }

    void validation_and_edge_cases()
    {
        auto fixture = make_regression_fixture();
        auto throws_fit = [&fixture](APLRRegressor &model, const VectorXd &response = VectorXd(0), const VectorXd &weights = VectorXd(0))
        {
            model.fit(fixture.train, response.size() == 0 ? fixture.response : response, weights);
        };
        APLRRegressor invalid_loss = configured_regressor();
        invalid_loss.loss_function = "unknown";
        bool threw = false;
        try
        {
            throws_fit(invalid_loss);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects unknown loss", threw);

        APLRRegressor invalid_link = configured_regressor();
        invalid_link.link_function = "unknown";
        threw = false;
        try
        {
            throws_fit(invalid_link);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects unknown link", threw);

        APLRRegressor invalid_m = configured_regressor();
        invalid_m.m = 0;
        threw = false;
        try
        {
            throws_fit(invalid_m);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects zero boosting steps", threw);

        APLRRegressor invalid_quantile = configured_regressor("quantile");
        invalid_quantile.quantile = 1.1;
        threw = false;
        try
        {
            throws_fit(invalid_quantile);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects invalid quantile", threw);

        APLRRegressor invalid_dispersion = configured_regressor("tweedie");
        invalid_dispersion.dispersion_parameter = 1.0;
        threw = false;
        try
        {
            throws_fit(invalid_dispersion);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects invalid dispersion", threw);

        APLRRegressor invalid_ratio = configured_regressor();
        invalid_ratio.validation_ratio = 1.0;
        threw = false;
        try
        {
            throws_fit(invalid_ratio);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects invalid validation ratio", threw);

        VectorXd short_response = fixture.response.head(fixture.response.size() - 1);
        APLRRegressor mismatched = configured_regressor();
        threw = false;
        try
        {
            throws_fit(mismatched, short_response);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects response length mismatch", threw);
        VectorXd negative_weights = VectorXd::Ones(fixture.response.size());
        negative_weights[0] = -1.0;
        APLRRegressor invalid_weights = configured_regressor();
        threw = false;
        try
        {
            throws_fit(invalid_weights, fixture.response, negative_weights);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("regressor rejects negative sample weights", threw);

        APLRClassifier untrained;
        threw = false;
        try
        {
            untrained.predict(fixture.test);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("classifier rejects prediction before fit", threw);
        std::vector<std::string> one_class(fixture.response.size(), "only");
        threw = false;
        try
        {
            untrained.fit(fixture.train, one_class);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("classifier rejects one category", threw);
        std::vector<std::string> short_labels(fixture.response.size() - 1, "a");
        short_labels.back() = "b";
        threw = false;
        try
        {
            untrained.fit(fixture.train, short_labels);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("classifier rejects label length mismatch", threw);

        Preprocessor preprocessor;
        CppDataFrame frame;
        frame.add_column("number", std::vector<double>{1.0, 2.0});
        threw = false;
        try
        {
            preprocessor.fit(frame, VectorXd::Ones(1));
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("preprocessor rejects weight length mismatch", threw);
        CppDataFrame categories;
        categories.add_column("kind", std::vector<std::string>{"a", "b"});
        preprocessor.fit(categories, VectorXd::Ones(2));
        auto unknown = preprocessor.transform(CppDataFrame(categories));
        check("preprocessor encodes unknown categories", unknown.first.rows() == 2);

        Term parent(0);
        parent.split_point = 0.5;
        parent.direction_right = true;
        Term interaction(1, {parent});
        MatrixXd interaction_input(2, 2);
        interaction_input << 0.0, 1.0, 1.0, 1.0;
        VectorXd masked = interaction.calculate(interaction_input);
        check("term interaction masks inactive rows", is_approximately_zero(masked[0]) && !is_approximately_zero(masked[1]));
        Term zero_rate(0);
        zero_rate.estimate_split_point(fixture.train, fixture.response, fixture.weights, 4, 0.0, 2, false, 0.0, 0.0, 0.0, 1.0);
        check("term zero learning rate makes term ineligible", zero_rate.ineligible_boosting_steps == std::numeric_limits<size_t>::max());

        ThreadPool pool(1);
        auto failed_task = pool.enqueue([]() -> int
                                        { throw std::runtime_error("task failure"); });
        threw = false;
        try
        {
            failed_task.get();
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        check("thread pool propagates task exceptions", threw);
    }

public:
    int run()
    {
        test("dataframe and preprocessor", [this]
             { dataframe_and_preprocessor(); });
        test("imputer encoder and utilities", [this]
             { imputer_encoder_and_utilities(); });
        test("dataframe edge cases", [this]
             { dataframe_edge_cases(); });
        test("utility functions", [this]
             { utility_functions(); });
        test("thread pool and term details", [this]
             { thread_pool_and_term_details(); });
        test("regressor fit and outputs", [this]
             { regressor_fit_and_outputs(); });
        test("regressor losses and callbacks", [this]
             { regressor_losses_and_callbacks(); });
        test("remaining model modes and copying", [this]
             { remaining_model_modes_and_copying(); });
        test("dataframe overloads and preprocessing", [this]
             { dataframe_overloads_and_preprocessing(); });
        test("classifier outputs", [this]
             { classifier_outputs(); });
        test("multiclass and regressor configuration", [this]
             { multiclass_and_regressor_configuration(); });
        test("parallel and hyperparameter configuration", [this]
             { parallel_and_hyperparameter_configuration(); });
        test("cv and term behavior", [this]
             { cv_and_term_behavior(); });
        test("ridge bins and interactions", [this]
             { ridge_bins_and_interactions(); });
        test("validation and edge cases", [this]
             { validation_and_edge_cases(); });
        std::cout << "Passed " << passed << " checks; failed " << failed << " checks.\n";
        return failed == 0 ? 0 : 1;
    }
};

int main()
{
    return Tests{}.run();
}
