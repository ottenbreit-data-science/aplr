#include <iostream>
#include "APLRRegressor.h"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

PYBIND11_MODULE(aplr_cpp, m) {
    py::class_<APLRRegressor>(m, "APLRRegressor",py::module_local())
        .def(py::init<int&, double&, int&, bool&,int&,double&,double&,int&,int&,int&,int&,int&,int&,int&,int&>(),
            py::arg("m")=1000,py::arg("v")=0.1,py::arg("random_state")=0,py::arg("loss_function_mse")=true,
            py::arg("n_jobs")=0,py::arg("validation_ratio")=0.2,py::arg("intercept")=NAN_DOUBLE,
            py::arg("reserved_terms_times_num_x")=100,py::arg("bins")=300,py::arg("verbosity")=0,
            py::arg("max_interaction_level")=100,py::arg("max_interactions")=0,py::arg("min_observations_in_split")=20,
            py::arg("ineligible_boosting_steps_added")=10,py::arg("max_eligible_terms")=5)
        .def("fit", &APLRRegressor::fit,py::arg("X"),py::arg("y"),py::arg("sample_weight")=VectorXd(0),py::arg("X_names")=std::vector<std::string>(),
            py::arg("validation_set_indexes")=std::vector<size_t>(),py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
        .def("predict", &APLRRegressor::predict,py::arg("X"))
        .def("set_term_names", &APLRRegressor::set_term_names,py::arg("X_names"))
        .def("calculate_local_feature_importance",&APLRRegressor::calculate_local_feature_importance,py::arg("X"))
        .def("calculate_local_feature_importance_for_terms",&APLRRegressor::calculate_local_feature_importance_for_terms,py::arg("X"))
        .def("calculate_terms",&APLRRegressor::calculate_terms,py::arg("X"))
        .def("get_term_names", &APLRRegressor::get_term_names)
        .def("get_term_coefficients", &APLRRegressor::get_term_coefficients)
        .def("get_term_coefficient_steps", &APLRRegressor::get_term_coefficient_steps,py::arg("term_index"))
        .def("get_validation_error_steps", &APLRRegressor::get_validation_error_steps)
        .def("get_feature_importance", &APLRRegressor::get_feature_importance)
        .def("get_intercept", &APLRRegressor::get_intercept)
        .def("get_intercept_steps", &APLRRegressor::get_intercept_steps)
        .def("get_m", &APLRRegressor::get_m)
        .def_readwrite("intercept", &APLRRegressor::intercept)
        .def_readwrite("intercept_steps", &APLRRegressor::intercept_steps)
        .def_readwrite("m", &APLRRegressor::m)
        .def_readwrite("v", &APLRRegressor::v)
        .def_readwrite("max_interaction_level", &APLRRegressor::max_interaction_level)
        .def_readwrite("max_interactions", &APLRRegressor::max_interactions)
        .def_readwrite("min_observations_in_split", &APLRRegressor::min_observations_in_split)
        .def_readwrite("interactions_eligible", &APLRRegressor::interactions_eligible)
        .def_readwrite("loss_function_mse", &APLRRegressor::loss_function_mse)
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
        .def_readwrite("number_of_base_terms",&APLRRegressor::number_of_base_terms)
        .def_readwrite("feature_importance",&APLRRegressor::feature_importance)
        .def(py::pickle(
            [](const APLRRegressor &a) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(a.m,a.v,a.random_state,a.loss_function_mse,a.n_jobs,a.validation_ratio,a.intercept,a.bins,a.verbosity,
                    a.max_interaction_level,a.max_interactions,a.validation_error_steps,a.term_names,a.term_coefficients,a.terms,a.intercept_steps,
                    a.interactions_eligible,a.min_observations_in_split,a.ineligible_boosting_steps_added,a.max_eligible_terms,
                    a.number_of_base_terms,a.feature_importance);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 22)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                APLRRegressor a(t[0].cast<size_t>(),t[1].cast<double>(),t[2].cast<uint_fast32_t>(),t[3].cast<bool>(),t[4].cast<size_t>(),t[5].cast<double>(),
                    t[6].cast<double>(),100,t[7].cast<size_t>(),t[8].cast<size_t>(),t[9].cast<size_t>(),t[10].cast<double>(),t[17].cast<size_t>());

                a.validation_error_steps=t[11].cast<VectorXd>();
                a.term_names=t[12].cast<std::vector<std::string>>();
                a.term_coefficients=t[13].cast<VectorXd>();
                a.terms=t[14].cast<std::vector<Term>>();
                a.intercept_steps=t[15].cast<VectorXd>();
                a.interactions_eligible=t[16].cast<size_t>();
                a.ineligible_boosting_steps_added=t[18].cast<size_t>();
                a.max_eligible_terms=t[19].cast<size_t>();
                a.number_of_base_terms=t[20].cast<size_t>();
                a.feature_importance=t[21].cast<VectorXd>();

                return a;
            }
        ));

    py::class_<Term>(m, "Term",py::module_local())
        .def_readwrite("name", &Term::name)
        .def_readwrite("base_term",&Term::base_term)
        .def_readwrite("given_terms", &Term::given_terms)
        .def_readwrite("split_point", &Term::split_point)        
        .def_readwrite("direction_right", &Term::direction_right)        
        .def_readwrite("coefficient", &Term::coefficient)        
        .def_readwrite("coefficient_steps", &Term::coefficient_steps)                
        .def(py::pickle(
            [](const Term &a) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(a.name,a.base_term,a.given_terms,a.split_point,a.direction_right,a.coefficient,a.coefficient_steps,a.split_point_search_errors_sum);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 8)
                    throw std::runtime_error("Invalid state!");

                /* Create a new C++ instance */
                Term a(t[1].cast<size_t>(),t[2].cast<std::vector<Term>>(),t[3].cast<double>(),t[4].cast<bool>(),t[5].cast<double>());

                a.name=t[0].cast<std::string>();
                a.coefficient_steps=t[6].cast<VectorXd>();
                a.split_point_search_errors_sum=t[7].cast<double>();

                return a;
            }
        ));
}
