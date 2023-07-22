#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include "APLRRegressor.h"
#include "APLRClassifier.h"

namespace py = pybind11;

std::function<double(VectorXd, VectorXd, VectorXd, VectorXi)> empty_calculate_custom_validation_error_function = {};
std::function<double(VectorXd, VectorXd, VectorXd, VectorXi)> empty_calculate_custom_loss_function = {};
std::function<VectorXd(VectorXd, VectorXd, VectorXi)> empty_calculate_custom_negative_gradient_function = {};
std::function<VectorXd(VectorXd)> empty_calculate_custom_transform_linear_predictor_to_predictions_function = {};
std::function<VectorXd(VectorXd)> empty_calculate_custom_differentiate_predictions_wrt_linear_predictor_function = {};

PYBIND11_MODULE(aplr_cpp, m)
{
    py::class_<APLRRegressor>(m, "APLRRegressor", py::module_local())
        .def(py::init<int &, double &, int &, std::string &, std::string &, int &, double &, int &, int &, int &, int &, int &, int &, int &, int &, double &, std::string &,
                      double &, std::function<double(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group)> &,
                      std::function<double(const VectorXd &y, const VectorXd &predictions, const VectorXd &sample_weight, const VectorXi &group)> &,
                      std::function<VectorXd(const VectorXd &y, const VectorXd &predictions, const VectorXi &group)> &,
                      std::function<VectorXd(const VectorXd &linear_predictor)> &, std::function<VectorXd(const VectorXd &linear_predictor)> &>(),
             py::arg("m") = 1000, py::arg("v") = 0.1, py::arg("random_state") = 0, py::arg("loss_function") = "mse", py::arg("link_function") = "identity",
             py::arg("n_jobs") = 0, py::arg("validation_ratio") = 0.2,
             py::arg("reserved_terms_times_num_x") = 100, py::arg("bins") = 300, py::arg("verbosity") = 0,
             py::arg("max_interaction_level") = 1, py::arg("max_interactions") = 100000, py::arg("min_observations_in_split") = 20,
             py::arg("ineligible_boosting_steps_added") = 10, py::arg("max_eligible_terms") = 5,
             py::arg("dispersion_parameter") = 1.5,
             py::arg("validation_tuning_metric") = "default",
             py::arg("quantile") = 0.5,
             py::arg("calculate_custom_validation_error_function") = empty_calculate_custom_validation_error_function,
             py::arg("calculate_custom_loss_function") = empty_calculate_custom_loss_function,
             py::arg("calculate_custom_negative_gradient_function") = empty_calculate_custom_negative_gradient_function,
             py::arg("calculate_custom_transform_linear_predictor_to_predictions_function") = empty_calculate_custom_transform_linear_predictor_to_predictions_function,
             py::arg("calculate_custom_differentiate_predictions_wrt_linear_predictor_function") = empty_calculate_custom_differentiate_predictions_wrt_linear_predictor_function)
        .def("fit", &APLRRegressor::fit, py::arg("X"), py::arg("y"), py::arg("sample_weight") = VectorXd(0), py::arg("X_names") = std::vector<std::string>(),
             py::arg("validation_set_indexes") = std::vector<size_t>(), py::arg("prioritized_predictors_indexes") = std::vector<size_t>(),
             py::arg("monotonic_constraints") = std::vector<int>(), py::arg("group") = VectorXi(0), py::arg("interaction_constraints") = std::vector<int>(),
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("predict", &APLRRegressor::predict, py::arg("X"), py::arg("bool cap_predictions_to_minmax_in_training") = true)
        .def("set_term_names", &APLRRegressor::set_term_names, py::arg("X_names"))
        .def("calculate_local_feature_importance", &APLRRegressor::calculate_local_feature_importance, py::arg("X"))
        .def("calculate_local_feature_importance_for_terms", &APLRRegressor::calculate_local_feature_importance_for_terms, py::arg("X"))
        .def("calculate_terms", &APLRRegressor::calculate_terms, py::arg("X"))
        .def("get_term_names", &APLRRegressor::get_term_names)
        .def("get_term_coefficients", &APLRRegressor::get_term_coefficients)
        .def("get_term_coefficient_steps", &APLRRegressor::get_term_coefficient_steps, py::arg("term_index"))
        .def("get_validation_error_steps", &APLRRegressor::get_validation_error_steps)
        .def("get_feature_importance", &APLRRegressor::get_feature_importance)
        .def("get_intercept", &APLRRegressor::get_intercept)
        .def("get_intercept_steps", &APLRRegressor::get_intercept_steps)
        .def("get_optimal_m", &APLRRegressor::get_optimal_m)
        .def("get_validation_tuning_metric", &APLRRegressor::get_validation_tuning_metric)
        .def("get_validation_indexes", &APLRRegressor::get_validation_indexes)
        .def_readwrite("intercept", &APLRRegressor::intercept)
        .def_readwrite("intercept_steps", &APLRRegressor::intercept_steps)
        .def_readwrite("m", &APLRRegressor::m)
        .def_readwrite("m_optimal", &APLRRegressor::m_optimal)
        .def_readwrite("v", &APLRRegressor::v)
        .def_readwrite("max_interaction_level", &APLRRegressor::max_interaction_level)
        .def_readwrite("max_interactions", &APLRRegressor::max_interactions)
        .def_readwrite("min_observations_in_split", &APLRRegressor::min_observations_in_split)
        .def_readwrite("interactions_eligible", &APLRRegressor::interactions_eligible)
        .def_readwrite("loss_function", &APLRRegressor::loss_function)
        .def_readwrite("link_function", &APLRRegressor::link_function)
        .def_readwrite("validation_ratio", &APLRRegressor::validation_ratio)
        .def_readwrite("validation_error_steps", &APLRRegressor::validation_error_steps)
        .def_readwrite("n_jobs", &APLRRegressor::n_jobs)
        .def_readwrite("random_state", &APLRRegressor::random_state)
        .def_readwrite("bins", &APLRRegressor::bins)
        .def_readwrite("verbosity", &APLRRegressor::verbosity)
        .def_readwrite("term_names", &APLRRegressor::term_names)
        .def_readwrite("term_coefficients", &APLRRegressor::term_coefficients)
        .def_readwrite("terms", &APLRRegressor::terms)
        .def_readwrite("ineligible_boosting_steps_added", &APLRRegressor::ineligible_boosting_steps_added)
        .def_readwrite("max_eligible_terms", &APLRRegressor::max_eligible_terms)
        .def_readwrite("number_of_base_terms", &APLRRegressor::number_of_base_terms)
        .def_readwrite("feature_importance", &APLRRegressor::feature_importance)
        .def_readwrite("dispersion_parameter", &APLRRegressor::dispersion_parameter)
        .def_readwrite("min_training_prediction_or_response", &APLRRegressor::min_training_prediction_or_response)
        .def_readwrite("max_training_prediction_or_response", &APLRRegressor::max_training_prediction_or_response)
        .def_readwrite("validation_tuning_metric", &APLRRegressor::validation_tuning_metric)
        .def_readwrite("validation_indexes", &APLRRegressor::validation_indexes)
        .def_readwrite("quantile", &APLRRegressor::quantile)
        .def_readwrite("calculate_custom_validation_error_function", &APLRRegressor::calculate_custom_validation_error_function)
        .def_readwrite("calculate_custom_loss_function", &APLRRegressor::calculate_custom_loss_function)
        .def_readwrite("calculate_custom_negative_gradient_function", &APLRRegressor::calculate_custom_negative_gradient_function)
        .def_readwrite("calculate_custom_transform_linear_predictor_to_predictions_function", &APLRRegressor::calculate_custom_transform_linear_predictor_to_predictions_function)
        .def_readwrite("calculate_custom_differentiate_predictions_wrt_linear_predictor_function", &APLRRegressor::calculate_custom_differentiate_predictions_wrt_linear_predictor_function)
        .def(py::pickle(
            [](const APLRRegressor &a) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(a.m, a.v, a.random_state, a.loss_function, a.link_function, a.n_jobs, a.validation_ratio, a.intercept, a.bins,
                                      a.verbosity, a.max_interaction_level, a.max_interactions, a.validation_error_steps, a.term_names, a.term_coefficients, a.terms,
                                      a.interactions_eligible, a.min_observations_in_split, a.ineligible_boosting_steps_added, a.max_eligible_terms,
                                      a.number_of_base_terms, a.feature_importance, a.dispersion_parameter, a.min_training_prediction_or_response,
                                      a.max_training_prediction_or_response, a.validation_tuning_metric, a.validation_indexes, a.quantile, a.m_optimal,
                                      a.intercept_steps);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 30)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                size_t m = t[0].cast<size_t>();
                double v = t[1].cast<double>();
                uint_fast32_t random_state = t[2].cast<uint_fast32_t>();
                std::string loss_function = t[3].cast<std::string>();
                std::string link_function = t[4].cast<std::string>();
                size_t n_jobs = t[5].cast<size_t>();
                double validation_ratio = t[6].cast<double>();
                double intercept = t[7].cast<double>();
                size_t bins = t[8].cast<size_t>();
                size_t verbosity = t[9].cast<size_t>();
                size_t max_interaction_level = t[10].cast<size_t>();
                size_t max_interactions = t[11].cast<size_t>();
                VectorXd validation_error_steps = t[12].cast<VectorXd>();
                std::vector<std::string> term_names = t[13].cast<std::vector<std::string>>();
                VectorXd term_coefficients = t[14].cast<VectorXd>();
                std::vector<Term> terms = t[15].cast<std::vector<Term>>();
                size_t interactions_eligible = t[16].cast<size_t>();
                size_t min_observations_in_split = t[17].cast<size_t>();
                size_t ineligible_boosting_steps_added = t[18].cast<size_t>();
                size_t max_eligible_terms = t[19].cast<size_t>();
                size_t number_of_base_terms = t[20].cast<size_t>();
                VectorXd feature_importance = t[21].cast<VectorXd>();
                double dispersion_parameter = t[22].cast<double>();
                double min_training_prediction_or_response = t[23].cast<double>();
                double max_training_prediction_or_response = t[24].cast<double>();
                std::string validation_tuning_metric = t[25].cast<std::string>();
                std::vector<size_t> validation_indexes = t[26].cast<std::vector<size_t>>();
                double quantile = t[27].cast<double>();
                size_t m_optimal = t[28].cast<size_t>();
                VectorXd intercept_steps = t[29].cast<VectorXd>();

                APLRRegressor a(m, v, random_state, loss_function, link_function, n_jobs, validation_ratio, 100, bins, verbosity, max_interaction_level,
                                max_interactions, min_observations_in_split, ineligible_boosting_steps_added, max_eligible_terms, dispersion_parameter,
                                validation_tuning_metric, quantile);
                a.intercept = intercept;
                a.validation_error_steps = validation_error_steps;
                a.term_names = term_names;
                a.term_coefficients = term_coefficients;
                a.terms = terms;
                a.interactions_eligible = interactions_eligible;
                a.number_of_base_terms = number_of_base_terms;
                a.feature_importance = feature_importance;
                a.min_training_prediction_or_response = min_training_prediction_or_response;
                a.max_training_prediction_or_response = max_training_prediction_or_response;
                a.validation_indexes = validation_indexes;
                a.m_optimal = m_optimal;
                a.intercept_steps = intercept_steps;

                return a;
            }));

    py::class_<Term>(m, "Term", py::module_local())
        .def_readwrite("name", &Term::name)
        .def_readwrite("base_term", &Term::base_term)
        .def_readwrite("given_terms", &Term::given_terms)
        .def_readwrite("split_point", &Term::split_point)
        .def_readwrite("direction_right", &Term::direction_right)
        .def_readwrite("coefficient", &Term::coefficient)
        .def_readwrite("coefficient_steps", &Term::coefficient_steps)
        .def(py::pickle(
            [](const Term &a) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(a.name, a.base_term, a.given_terms, a.split_point, a.direction_right, a.coefficient, a.coefficient_steps,
                                      a.split_point_search_errors_sum);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 8)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                std::string name = t[0].cast<std::string>();
                size_t base_term = t[1].cast<size_t>();
                std::vector<Term> given_terms = t[2].cast<std::vector<Term>>();
                double split_point = t[3].cast<double>();
                bool direction_right = t[4].cast<bool>();
                double coefficient = t[5].cast<double>();
                VectorXd coefficient_steps = t[6].cast<VectorXd>();
                double split_point_search_errors_sum = t[7].cast<double>();

                Term a(base_term, given_terms, split_point, direction_right, coefficient);
                a.name = name;
                a.coefficient_steps = coefficient_steps;
                a.split_point_search_errors_sum = split_point_search_errors_sum;

                return a;
            }));

    py::class_<APLRClassifier>(m, "APLRClassifier", py::module_local())
        .def(py::init<int &, double &, int &, int &, double &, int &, int &, int &, int &, int &, int &, int &, int &>(),
             py::arg("m") = 9000, py::arg("v") = 0.1, py::arg("random_state") = 0, py::arg("n_jobs") = 0, py::arg("validation_ratio") = 0.2,
             py::arg("reserved_terms_times_num_x") = 100, py::arg("bins") = 300, py::arg("verbosity") = 0,
             py::arg("max_interaction_level") = 1, py::arg("max_interactions") = 100000, py::arg("min_observations_in_split") = 20,
             py::arg("ineligible_boosting_steps_added") = 10, py::arg("max_eligible_terms") = 5)
        .def("fit", &APLRClassifier::fit, py::arg("X"), py::arg("y"), py::arg("sample_weight") = VectorXd(0), py::arg("X_names") = std::vector<std::string>(),
             py::arg("validation_set_indexes") = std::vector<size_t>(), py::arg("prioritized_predictors_indexes") = std::vector<size_t>(),
             py::arg("monotonic_constraints") = std::vector<int>(), py::arg("interaction_constraints") = std::vector<int>(),
             py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("predict_class_probabilities", &APLRClassifier::predict_class_probabilities, py::arg("X"), py::arg("bool cap_predictions_to_minmax_in_training") = false)
        .def("predict", &APLRClassifier::predict, py::arg("X"), py::arg("bool cap_predictions_to_minmax_in_training") = false)
        .def("calculate_local_feature_importance", &APLRClassifier::calculate_local_feature_importance, py::arg("X"))
        .def("get_categories", &APLRClassifier::get_categories)
        .def("get_logit_model", &APLRClassifier::get_logit_model, py::arg("category"))
        .def("get_validation_indexes", &APLRClassifier::get_validation_indexes)
        .def("get_validation_error_steps", &APLRClassifier::get_validation_error_steps)
        .def("get_validation_error", &APLRClassifier::get_validation_error)
        .def("get_feature_importance", &APLRClassifier::get_feature_importance)
        .def_readwrite("m", &APLRClassifier::m)
        .def_readwrite("v", &APLRClassifier::v)
        .def_readwrite("validation_ratio", &APLRClassifier::validation_ratio)
        .def_readwrite("n_jobs", &APLRClassifier::n_jobs)
        .def_readwrite("random_state", &APLRClassifier::random_state)
        .def_readwrite("bins", &APLRClassifier::bins)
        .def_readwrite("verbosity", &APLRClassifier::verbosity)
        .def_readwrite("max_interaction_level", &APLRClassifier::max_interaction_level)
        .def_readwrite("max_interactions", &APLRClassifier::max_interactions)
        .def_readwrite("min_observations_in_split", &APLRClassifier::min_observations_in_split)
        .def_readwrite("ineligible_boosting_steps_added", &APLRClassifier::ineligible_boosting_steps_added)
        .def_readwrite("max_eligible_terms", &APLRClassifier::max_eligible_terms)
        .def_readwrite("validation_indexes", &APLRClassifier::validation_indexes)
        .def_readwrite("validation_error_steps", &APLRClassifier::validation_error_steps)
        .def_readwrite("validation_error", &APLRClassifier::validation_error)
        .def_readwrite("feature_importance", &APLRClassifier::feature_importance)
        .def_readwrite("categories", &APLRClassifier::categories)
        .def_readwrite("logit_models", &APLRClassifier::logit_models)
        .def(py::pickle(
            [](const APLRClassifier &a) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(a.m, a.v, a.random_state, a.n_jobs, a.validation_ratio, a.bins, a.verbosity,
                                      a.max_interaction_level, a.max_interactions, a.min_observations_in_split, a.ineligible_boosting_steps_added,
                                      a.max_eligible_terms, a.logit_models, a.categories, a.validation_indexes, a.validation_error_steps, a.validation_error,
                                      a.feature_importance);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 18)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                size_t m = t[0].cast<size_t>();
                double v = t[1].cast<double>();
                size_t random_state = t[2].cast<size_t>();
                size_t n_jobs = t[3].cast<size_t>();
                double validation_ratio = t[4].cast<double>();
                size_t bins = t[5].cast<size_t>();
                size_t verbosity = t[6].cast<size_t>();
                size_t max_interaction_level = t[7].cast<size_t>();
                size_t max_interactions = t[8].cast<size_t>();
                size_t min_observations_in_split = t[9].cast<size_t>();
                size_t ineligible_boosting_steps_added = t[10].cast<size_t>();
                size_t max_eligible_terms = t[11].cast<size_t>();
                std::map<std::string, APLRRegressor> logit_models = t[12].cast<std::map<std::string, APLRRegressor>>();
                std::vector<std::string> categories = t[13].cast<std::vector<std::string>>();
                std::vector<size_t> validation_indexes = t[14].cast<std::vector<size_t>>();
                VectorXd validation_error_steps = t[15].cast<VectorXd>();
                double validation_error = t[16].cast<double>();
                VectorXd feature_importance = t[17].cast<VectorXd>();

                APLRClassifier a(m, v, random_state, n_jobs, validation_ratio, 100, bins, verbosity, max_interaction_level, max_interactions,
                                 min_observations_in_split, ineligible_boosting_steps_added, max_eligible_terms);
                a.logit_models = logit_models;
                a.categories = categories;
                a.validation_indexes = validation_indexes;
                a.validation_error_steps = validation_error_steps;
                a.validation_error = validation_error;
                a.feature_importance = feature_importance;

                return a;
            }));
}
