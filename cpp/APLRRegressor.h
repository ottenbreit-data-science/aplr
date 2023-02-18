#pragma once
#include <string>
#include <limits>
#include "../dependencies/eigen-master/Eigen/Dense"
#include "functions.h"
#include "term.h"
#include <vector>
#include "constants.h"
#include "functions.h"
#include <thread>
#include <future>
#include <random>

using namespace Eigen;



class APLRRegressor
{
private:
    //Fields
    size_t reserved_terms_times_num_x; //How many times number of variables in X to reserve memory for term (terms in model)
    MatrixXd X_train;
    VectorXd y_train;
    VectorXd sample_weight_train;
    MatrixXd X_validation;
    VectorXd y_validation;
    VectorXd sample_weight_validation;
    VectorXd linear_predictor_null_model;
    std::vector<Term> terms_eligible_current;
    VectorXd predictions_current;
    VectorXd predictions_current_validation;
    VectorXd neg_gradient_current;
    VectorXd neg_gradient_nullmodel_errors;
    double neg_gradient_nullmodel_errors_sum;
    size_t best_term_index;
    VectorXd linear_predictor_update;
    VectorXd linear_predictor_update_validation;
    size_t number_of_eligible_terms;
    std::vector<std::vector<size_t>> distributed_terms;
    std::vector<Term> interactions_to_consider;
    VectorXi sorted_indexes_of_errors_for_interactions_to_consider;
    bool abort_boosting;
    VectorXd linear_predictor_current;
    VectorXd linear_predictor_current_validation;
    double scaling_factor_for_log_link_function;
    std::vector<size_t> predictor_indexes;
    std::vector<size_t> prioritized_predictors_indexes;

    //Methods
    void validate_input_to_fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,const std::vector<std::string> &X_names, const std::vector<size_t> &validation_set_indexes, const std::vector<size_t> &prioritized_predictors_indexes);
    void throw_error_if_validation_set_indexes_has_invalid_indexes(const VectorXd &y, const std::vector<size_t> &validation_set_indexes);
    void throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(const MatrixXd &X, const std::vector<size_t> &prioritized_predictors_indexes);
    void define_training_and_validation_sets(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight, const std::vector<size_t> &validation_set_indexes);
    void initialize(const std::vector<size_t> &prioritized_predictors_indexes);
    bool check_if_base_term_has_only_one_unique_value(size_t base_term);
    void add_term_to_terms_eligible_current(Term &term);
    VectorXd calculate_neg_gradient_current();
    void execute_boosting_steps();
    void execute_boosting_step(size_t boosting_step);
    std::vector<size_t> find_terms_eligible_current_indexes_for_a_base_term(size_t base_term);
    void estimate_split_point_for_each_term(std::vector<Term> &terms, std::vector<size_t> &terms_indexes);
    size_t find_best_term_index(std::vector<Term> &terms, std::vector<size_t> &terms_indexes);
    void consider_interactions(const std::vector<size_t> &available_predictor_indexes);
    void determine_interactions_to_consider(const std::vector<size_t> &available_predictor_indexes);
    VectorXi find_indexes_for_terms_to_consider_as_interaction_partners();
    size_t find_out_how_many_terms_to_consider_as_interaction_partners();
    void add_necessary_given_terms_to_interaction(Term &interaction, Term &existing_model_term);
    void find_sorted_indexes_for_errors_for_interactions_to_consider();
    void add_promising_interactions_and_select_the_best_one();
    void update_intercept(size_t boosting_step);
    void select_the_best_term_and_update_errors(size_t boosting_step, bool not_evaluating_prioritized_predictors=true);
    void update_terms(size_t boosting_step);
    void update_gradient_and_errors();
    void add_new_term(size_t boosting_step);
    void calculate_and_validate_validation_error(size_t boosting_step);
    void update_term_eligibility();
    void print_summary_after_boosting_step(size_t boosting_step);
    void update_coefficients_for_all_steps();
    void print_final_summary();
    void find_optimal_m_and_update_model_accordingly();
    void name_terms(const MatrixXd &X, const std::vector<std::string> &X_names);
    void calculate_feature_importance_on_validation_set();
    void find_min_and_max_training_predictions_or_responses();
    void calculate_validation_group_mse();
    void cleanup_after_fit();
    void validate_that_model_can_be_used(const MatrixXd &X);
    void throw_error_if_family_does_not_exist();
    void throw_error_if_link_function_does_not_exist();
    VectorXd calculate_linear_predictor(const MatrixXd &X);
    void update_linear_predictor_and_predictions();
    void throw_error_if_response_contains_invalid_values(const VectorXd &y);
    void throw_error_if_response_is_not_between_0_and_1(const VectorXd &y,const std::string &error_message);
    void throw_error_if_response_is_negative(const VectorXd &y, const std::string &error_message);
    void throw_error_if_response_is_not_greater_than_zero(const VectorXd &y, const std::string &error_message);
    void throw_error_if_tweedie_power_is_invalid();
    VectorXd differentiate_predictions();
    void scale_training_observations_if_using_log_link_function();
    void revert_scaling_if_using_log_link_function();
    void cap_predictions_to_minmax_in_training(VectorXd &predictions);
    std::string compute_raw_base_term_name(const Term &term, const std::string &X_name);
    
public:
    //Fields
    double intercept;
    std::vector<Term> terms;
    size_t m; //Boosting steps to run. Can shrink to auto tuned value after running fit().
    double v; //Learning rate.
    std::string family;
    std::string link_function;
    double validation_ratio;
    size_t n_jobs; //0:using all available cores. 1:no multithreading. >1: Using a specified number of cores but not more than is available.
    uint_fast32_t random_state; //For train/validation split. If std::numeric_limits<uint_fast32_t>::lowest() then will randomly set a seed
    size_t bins; //Used if nobs>bins
    size_t verbosity; //0 none, 1 summary after running fit(), 2 each boosting step when running fit().
    std::vector<std::string> term_names;
    VectorXd term_coefficients;
    size_t max_interaction_level;
    VectorXd intercept_steps;
    size_t max_interactions; //Max interactions allowed to add (counted in interactions_eligible)
    size_t interactions_eligible; //Interactions that were eligible when training the model
    VectorXd validation_error_steps; //Validation error for each boosting step
    size_t min_observations_in_split; //Must be at least 1
    size_t ineligible_boosting_steps_added; //Determines the magnitude of ineligible_boosting_steps when set to >0 during fit(). Not used if 0.
    size_t max_eligible_terms; //Determines how many terms with ineligible_boosting_steps=0 are supposed to be left eligible
                                    //at the end of each boosting step (before decreasing ineligible_boosting_steps by one for all 
                                    //terms with ineligible_boosting_steps>0). Not used if 0.
    size_t number_of_base_terms; 
    VectorXd feature_importance; //Populated in fit() using validation set. Rows are in the same order as in X.
    double tweedie_power;
    double min_training_prediction_or_response;
    double max_training_prediction_or_response;
    double validation_group_mse;
    size_t group_size_for_validation_group_mse;

    //Methods
    APLRRegressor(size_t m=1000,double v=0.1,uint_fast32_t random_state=std::numeric_limits<uint_fast32_t>::lowest(),std::string family="gaussian",
        std::string link_function="identity", size_t n_jobs=0, double validation_ratio=0.2,double intercept=NAN_DOUBLE,
        size_t reserved_terms_times_num_x=100, size_t bins=300,size_t verbosity=0,size_t max_interaction_level=1,size_t max_interactions=100000,
        size_t min_observations_in_split=20, size_t ineligible_boosting_steps_added=10, size_t max_eligible_terms=5,double tweedie_power=1.5,
        size_t group_size_for_validation_group_mse=100);
    APLRRegressor(const APLRRegressor &other);
    ~APLRRegressor();
    void fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight=VectorXd(0),const std::vector<std::string> &X_names={},const std::vector<size_t> &validation_set_indexes={},const std::vector<size_t> &prioritized_predictors_indexes={});
    VectorXd predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training=true);
    void set_term_names(const std::vector<std::string> &X_names);
    MatrixXd calculate_local_feature_importance(const MatrixXd &X);
    MatrixXd calculate_local_feature_importance_for_terms(const MatrixXd &X);
    MatrixXd calculate_terms(const MatrixXd &X);
    std::vector<std::string> get_term_names();
    VectorXd get_term_coefficients();
    VectorXd get_term_coefficient_steps(size_t term_index);
    VectorXd get_validation_error_steps();
    VectorXd get_feature_importance();
    double get_intercept();
    VectorXd get_intercept_steps();
    size_t get_m();
    double get_validation_group_mse();
};

//Regular constructor
APLRRegressor::APLRRegressor(size_t m,double v,uint_fast32_t random_state,std::string family,std::string link_function,size_t n_jobs,
    double validation_ratio,double intercept,size_t reserved_terms_times_num_x,size_t bins,size_t verbosity,size_t max_interaction_level,
    size_t max_interactions,size_t min_observations_in_split,size_t ineligible_boosting_steps_added,size_t max_eligible_terms,double tweedie_power,
    size_t group_size_for_validation_group_mse):
        reserved_terms_times_num_x{reserved_terms_times_num_x},intercept{intercept},m{m},v{v},
        family{family},link_function{link_function},validation_ratio{validation_ratio},n_jobs{n_jobs},random_state{random_state},
        bins{bins},verbosity{verbosity},max_interaction_level{max_interaction_level},
        intercept_steps{VectorXd(0)},max_interactions{max_interactions},interactions_eligible{0},validation_error_steps{VectorXd(0)},
        min_observations_in_split{min_observations_in_split},ineligible_boosting_steps_added{ineligible_boosting_steps_added},
        max_eligible_terms{max_eligible_terms},number_of_base_terms{0},tweedie_power{tweedie_power},min_training_prediction_or_response{NAN_DOUBLE},
        max_training_prediction_or_response{NAN_DOUBLE},validation_group_mse{NAN_DOUBLE},group_size_for_validation_group_mse{group_size_for_validation_group_mse}
{
}

//Copy constructor
APLRRegressor::APLRRegressor(const APLRRegressor &other):
    reserved_terms_times_num_x{other.reserved_terms_times_num_x},intercept{other.intercept},terms{other.terms},m{other.m},v{other.v},
    family{other.family},link_function{other.link_function},validation_ratio{other.validation_ratio},
    n_jobs{other.n_jobs},random_state{other.random_state},bins{other.bins},
    verbosity{other.verbosity},term_names{other.term_names},term_coefficients{other.term_coefficients},
    max_interaction_level{other.max_interaction_level},intercept_steps{other.intercept_steps},
    max_interactions{other.max_interactions},interactions_eligible{other.interactions_eligible},validation_error_steps{other.validation_error_steps},
    min_observations_in_split{other.min_observations_in_split},ineligible_boosting_steps_added{other.ineligible_boosting_steps_added},
    max_eligible_terms{other.max_eligible_terms},number_of_base_terms{other.number_of_base_terms},
    feature_importance{other.feature_importance},tweedie_power{other.tweedie_power},min_training_prediction_or_response{other.min_training_prediction_or_response},
    max_training_prediction_or_response{other.max_training_prediction_or_response},validation_group_mse{other.validation_group_mse},
    group_size_for_validation_group_mse{other.group_size_for_validation_group_mse}
{
}

//Destructor
APLRRegressor::~APLRRegressor()
{
}

//Fits the model
//X_names specifies names for each column in X. If not specified then X1, X2, X3, ... will be used as names for each column in X.
//If validation_set_indexes.size()>0 then validation_set_indexes defines which of the indexes in X, y and sample_weight are used to validate, 
//invalidating validation_ratio. The rest of indexes are used to train. 
void APLRRegressor::fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,const std::vector<std::string> &X_names,const std::vector<size_t> &validation_set_indexes,const std::vector<size_t> &prioritized_predictors_indexes)
{
    throw_error_if_family_does_not_exist();
    throw_error_if_link_function_does_not_exist();
    throw_error_if_tweedie_power_is_invalid();
    validate_input_to_fit(X,y,sample_weight,X_names,validation_set_indexes,prioritized_predictors_indexes);
    define_training_and_validation_sets(X,y,sample_weight,validation_set_indexes);
    scale_training_observations_if_using_log_link_function();
    initialize(prioritized_predictors_indexes);
    execute_boosting_steps();
    update_coefficients_for_all_steps();
    print_final_summary();
    find_optimal_m_and_update_model_accordingly();
    revert_scaling_if_using_log_link_function();
    name_terms(X, X_names);
    calculate_feature_importance_on_validation_set();
    find_min_and_max_training_predictions_or_responses();
    calculate_validation_group_mse();
    cleanup_after_fit();
}

void APLRRegressor::throw_error_if_family_does_not_exist()
{
    bool family_exists{false};
    if(family=="gaussian")
        family_exists=true;
    else if(family=="binomial")
        family_exists=true;
    else if(family=="poisson")
        family_exists=true;
    else if(family=="gamma")
        family_exists=true;
    else if(family=="tweedie")
        family_exists=true;        
    if(!family_exists)
        throw std::runtime_error("Family "+family+" is not available in APLR.");   
}

void APLRRegressor::throw_error_if_link_function_does_not_exist()
{
    bool link_function_exists{false};
    if(link_function=="identity")
        link_function_exists=true;
    else if(link_function=="logit")
        link_function_exists=true;
    else if(link_function=="log")
        link_function_exists=true;
    if(!link_function_exists)
        throw std::runtime_error("Link function "+link_function+" is not available in APLR.");
}

void APLRRegressor::throw_error_if_tweedie_power_is_invalid()
{
    bool tweedie_power_equals_invalid_poits{is_approximately_equal(tweedie_power,1.0) || is_approximately_equal(tweedie_power,2.0)};
    bool tweedie_power_is_in_invalid_range{std::isless(tweedie_power,1.0)};
    bool tweedie_power_is_invalid{tweedie_power_equals_invalid_poits || tweedie_power_is_in_invalid_range};
    if(tweedie_power_is_invalid)
        throw std::runtime_error("Tweedie power is invalid. It must not equal 1.0 or 2.0 and cannot be below 1.0.");
}

void APLRRegressor::validate_input_to_fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,const std::vector<std::string> &X_names, const std::vector<size_t> &validation_set_indexes, const std::vector<size_t> &prioritized_predictors_indexes)
{
    if(X.rows()!=y.size()) throw std::runtime_error("X and y must have the same number of rows.");
    if(X.rows()<2) throw std::runtime_error("X and y cannot have less than two rows.");
    if(sample_weight.size()>0 && sample_weight.size()!=y.size()) throw std::runtime_error("sample_weight must have 0 or as many rows as X and y.");
    if(X_names.size()>0 && X_names.size()!=static_cast<size_t>(X.cols())) throw std::runtime_error("X_names must have as many columns as X.");
    throw_error_if_matrix_has_nan_or_infinite_elements(X, "X");
    throw_error_if_matrix_has_nan_or_infinite_elements(y, "y");
    throw_error_if_matrix_has_nan_or_infinite_elements(sample_weight, "sample_weight");
    throw_error_if_validation_set_indexes_has_invalid_indexes(y, validation_set_indexes);
    throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(X, prioritized_predictors_indexes);
    throw_error_if_response_contains_invalid_values(y);
}

void APLRRegressor::throw_error_if_validation_set_indexes_has_invalid_indexes(const VectorXd &y, const std::vector<size_t> &validation_set_indexes)
{
    bool validation_set_indexes_is_provided{validation_set_indexes.size()>0};
    if(validation_set_indexes_is_provided)
    {
        size_t max_index{*std::max_element(validation_set_indexes.begin(), validation_set_indexes.end())};
        bool validation_set_indexes_has_elements_out_of_bounds{max_index > static_cast<size_t>(y.size()-1)};
        if(validation_set_indexes_has_elements_out_of_bounds)
            throw std::runtime_error("validation_set_indexes has elements that are out of bounds.");
    }
}

void APLRRegressor::throw_error_if_prioritized_predictors_indexes_has_invalid_indexes(const MatrixXd &X, const std::vector<size_t> &prioritized_predictors_indexes)
{
    bool prioritized_predictors_indexes_is_provided{prioritized_predictors_indexes.size()>0};
    if(prioritized_predictors_indexes_is_provided)
    {
        size_t max_index{*std::max_element(prioritized_predictors_indexes.begin(), prioritized_predictors_indexes.end())};
        bool prioritized_predictors_indexes_has_elements_out_of_bounds{max_index > static_cast<size_t>(X.cols()-1)};
        if(prioritized_predictors_indexes_has_elements_out_of_bounds)
            throw std::runtime_error("prioritized_predictors_indexes has elements that are out of bounds.");
    }
}

void APLRRegressor::throw_error_if_response_contains_invalid_values(const VectorXd &y)
{
    if(link_function=="logit" || family=="binomial")
    {
        std::string error_message{"Response values for the logit link function or binomial family cannot be less than zero or greater than one."};
        throw_error_if_response_is_not_between_0_and_1(y,error_message);
    }
    else if(family=="gamma" || (family=="tweedie" && std::isgreater(tweedie_power,2)) )
    {
        std::string error_message;
        if(family=="tweedie")
            error_message="Response values for the "+family+" family when tweedie_power>2 must be greater than zero.";
        else
            error_message="Response values for the "+family+" family must be greater than zero.";
        throw_error_if_response_is_not_greater_than_zero(y,error_message);
    }
    else if(link_function=="log" || family=="poisson" || (family=="tweedie" && std::isless(tweedie_power,2) && std::isgreater(tweedie_power,1)))
    {
        std::string error_message{"Response values for the log link function or poisson family or tweedie family when tweedie_power<2 cannot be less than zero."};
        throw_error_if_response_is_negative(y,error_message);
    }
}

void APLRRegressor::throw_error_if_response_is_not_between_0_and_1(const VectorXd &y, const std::string &error_message)
{
    bool response_is_less_than_zero{(y.array()<0.0).any()};
    bool response_is_greater_than_one{(y.array()>1.0).any()};
    if(response_is_less_than_zero || response_is_greater_than_one)
        throw std::runtime_error(error_message);   
}

void APLRRegressor::throw_error_if_response_is_negative(const VectorXd &y, const std::string &error_message)
{
    bool response_is_less_than_zero{(y.array()<0.0).any()};
    if(response_is_less_than_zero)
        throw std::runtime_error(error_message);   
}

void APLRRegressor::throw_error_if_response_is_not_greater_than_zero(const VectorXd &y, const std::string &error_message)
{
    bool response_is_not_greater_than_zero{(y.array()<=0.0).any()};
    if(response_is_not_greater_than_zero)
        throw std::runtime_error(error_message);   

}

void APLRRegressor::define_training_and_validation_sets(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight, const std::vector<size_t> &validation_set_indexes)
{
    size_t y_size{static_cast<size_t>(y.size())};
    std::vector<size_t> train_indexes;
    std::vector<size_t> validation_indexes;
    bool use_validation_set_indexes{validation_set_indexes.size()>0};
    if(use_validation_set_indexes)
    {
        std::vector<size_t> all_indexes(y_size);
        std::iota(std::begin(all_indexes),std::end(all_indexes),0);
        validation_indexes=validation_set_indexes;
        train_indexes.reserve(y_size-validation_indexes.size()); 
        std::remove_copy_if(all_indexes.begin(),all_indexes.end(),std::back_inserter(train_indexes),[&validation_indexes](const size_t &arg)
            { return (std::find(validation_indexes.begin(),validation_indexes.end(),arg) != validation_indexes.end());});
    }
    else
    {
        train_indexes.reserve(y_size);
        validation_indexes.reserve(y_size);
        std::mt19937 mersenne{random_state};
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        double roll;
        for (size_t i = 0; i < y_size; ++i)
        {
            roll=distribution(mersenne);
            bool place_in_validation_set{std::isless(roll,validation_ratio)};
            if(place_in_validation_set)
            {
                validation_indexes.push_back(i);
            }
            else
            {
                train_indexes.push_back(i);
            }
        }
        train_indexes.shrink_to_fit();
        validation_indexes.shrink_to_fit();
    }
    //Defining train and test matrices
    X_train.resize(train_indexes.size(),X.cols());
    y_train.resize(train_indexes.size());
    sample_weight_train.resize(std::min(train_indexes.size(),static_cast<size_t>(sample_weight.size())));
    X_validation.resize(validation_indexes.size(),X.cols());
    y_validation.resize(validation_indexes.size());
    sample_weight_validation.resize(std::min(validation_indexes.size(),static_cast<size_t>(sample_weight.size())));
    //Populating train matrices
    for (size_t i = 0; i < train_indexes.size(); ++i)
    {
        X_train.row(i)=X.row(train_indexes[i]);
        y_train[i]=y[train_indexes[i]];
    }
    bool sample_weight_exist{sample_weight_train.size()==y_train.size()};
    if(sample_weight_exist)
    {
        for (size_t i = 0; i < train_indexes.size(); ++i)
        {
            sample_weight_train[i]=sample_weight[train_indexes[i]];
        }
    }
    //Populating test matrices
    for (size_t i = 0; i < validation_indexes.size(); ++i)
    {
        X_validation.row(i)=X.row(validation_indexes[i]);
        y_validation[i]=y[validation_indexes[i]];
    }
    sample_weight_exist = sample_weight_validation.size()==y_validation.size();
    if(sample_weight_exist)
    {
        for (size_t i = 0; i < validation_indexes.size(); ++i)
        {
            sample_weight_validation[i]=sample_weight[validation_indexes[i]];
        }
    }
}

void APLRRegressor::scale_training_observations_if_using_log_link_function()
{
    if(link_function=="log")
    {
        double inverse_scaling_factor{y_train.maxCoeff()/std::exp(1)};
        bool inverse_scaling_factor_is_not_zero{!is_approximately_zero(inverse_scaling_factor)};
        if(inverse_scaling_factor_is_not_zero)
        {
            scaling_factor_for_log_link_function=1/inverse_scaling_factor;
            y_train*=scaling_factor_for_log_link_function;
            y_validation*=scaling_factor_for_log_link_function;
        }
        else
            scaling_factor_for_log_link_function=1.0;
    }
}

void APLRRegressor::initialize(const std::vector<size_t> &prioritized_predictors_indexes)
{
    number_of_base_terms=static_cast<size_t>(X_train.cols());

    terms.reserve(X_train.cols()*reserved_terms_times_num_x);
    terms.clear();

    intercept=0;
    intercept_steps=VectorXd::Constant(m,0);

    terms_eligible_current.reserve(X_train.cols()*reserved_terms_times_num_x);
    for (size_t i = 0; i < static_cast<size_t>(X_train.cols()); ++i)
    {
        bool term_has_one_unique_value{check_if_base_term_has_only_one_unique_value(i)};
        Term copy_of_base_term{Term(i)};
        add_term_to_terms_eligible_current(copy_of_base_term);
        if(term_has_one_unique_value)
        {
            terms_eligible_current[terms_eligible_current.size()-1].ineligible_boosting_steps=std::numeric_limits<size_t>::max();
        }
    }

    predictor_indexes.resize(X_train.cols());
    for (size_t i = 0; i < static_cast<size_t>(X_train.cols()); ++i)
    {
        predictor_indexes[i]=i;
    }
    this->prioritized_predictors_indexes=prioritized_predictors_indexes;

    linear_predictor_current=VectorXd::Constant(y_train.size(),intercept);
    linear_predictor_null_model=linear_predictor_current;
    linear_predictor_current_validation=VectorXd::Constant(y_validation.size(),intercept);
    predictions_current=transform_linear_predictor_to_predictions(linear_predictor_current,link_function,tweedie_power);
    predictions_current_validation=transform_linear_predictor_to_predictions(linear_predictor_current_validation,link_function,tweedie_power);

    validation_error_steps.resize(m);
    validation_error_steps.setConstant(std::numeric_limits<double>::infinity());
                                    
    update_gradient_and_errors();
}

bool APLRRegressor::check_if_base_term_has_only_one_unique_value(size_t base_term)
{
    size_t rows{static_cast<size_t>(X_train.rows())};
    if(rows==1) return true;
    
    bool term_has_one_unique_value{true};
    for (size_t i = 1; i < rows; ++i)
    {
        bool observation_is_equal_to_previous{is_approximately_equal(X_train.col(base_term)[i], X_train.col(base_term)[i-1])};
        if(!observation_is_equal_to_previous)
        {
            term_has_one_unique_value=false;
            break;
        } 
    }

    return term_has_one_unique_value;
}

void APLRRegressor::add_term_to_terms_eligible_current(Term &term)
{
    terms_eligible_current.push_back(term);
}

VectorXd APLRRegressor::calculate_neg_gradient_current()
{
    VectorXd output;
    if(family=="gaussian")
        output=y_train-predictions_current;
    else if(family=="binomial")
        output=y_train.array() / predictions_current.array() - (y_train.array()-1.0) / (predictions_current.array()-1.0);
    else if(family=="poisson")
        output=y_train.array() / predictions_current.array() - 1;
    else if(family=="gamma")
        output=(y_train.array() - predictions_current.array()) / predictions_current.array() / predictions_current.array();
    else if(family=="tweedie")
        output=(y_train.array()-predictions_current.array()).array() * predictions_current.array().pow(-tweedie_power);
    
    if(link_function!="identity")
        output=output.array()*differentiate_predictions().array();
    
    return output;
}

VectorXd APLRRegressor::differentiate_predictions()
{
    if(link_function=="logit")
        return 1.0/4.0 * (linear_predictor_current.array()/2.0).cosh().array().pow(-2);
    else if(link_function=="log")
    {
        return linear_predictor_current.array().exp();
    }
    return VectorXd(0);
}

void APLRRegressor::execute_boosting_steps()
{
    abort_boosting = false;
    for (size_t boosting_step = 0; boosting_step < m; ++boosting_step)
    {
        execute_boosting_step(boosting_step);
        if(abort_boosting) break;
    }
}

void APLRRegressor::execute_boosting_step(size_t boosting_step)
{
    update_intercept(boosting_step);
    bool prioritize_predictors{!abort_boosting && prioritized_predictors_indexes.size()>0};
    if(prioritize_predictors)
    {
        for (auto &index:prioritized_predictors_indexes)
        {
            std::vector<size_t> terms_eligible_current_indexes_for_a_base_term{find_terms_eligible_current_indexes_for_a_base_term(index)};
            bool eligible_terms_exist{terms_eligible_current_indexes_for_a_base_term.size()>0};
            if(eligible_terms_exist)
            {
                estimate_split_point_for_each_term(terms_eligible_current, terms_eligible_current_indexes_for_a_base_term);
                best_term_index=find_best_term_index(terms_eligible_current, terms_eligible_current_indexes_for_a_base_term);
                std::vector<size_t> predictor_index{index};
                consider_interactions(predictor_index);
                select_the_best_term_and_update_errors(boosting_step,false);
            }
        }
    }
    if(!abort_boosting)
    {
        std::vector<size_t> term_indexes{create_term_indexes(terms_eligible_current)};
        estimate_split_point_for_each_term(terms_eligible_current, term_indexes);
        best_term_index=find_best_term_index(terms_eligible_current, term_indexes);
        consider_interactions(predictor_indexes);
        select_the_best_term_and_update_errors(boosting_step);
    }
    if(abort_boosting) return;
    update_term_eligibility();
    print_summary_after_boosting_step(boosting_step);
}

void APLRRegressor::update_intercept(size_t boosting_step)
{
    double intercept_update;
    if(sample_weight_train.size()==0)
        intercept_update=v*neg_gradient_current.mean();
    else
        intercept_update=v*(neg_gradient_current.array()*sample_weight_train.array()).sum()/sample_weight_train.array().sum();
    linear_predictor_update=VectorXd::Constant(neg_gradient_current.size(),intercept_update);
    linear_predictor_update_validation=VectorXd::Constant(y_validation.size(),intercept_update);
    update_linear_predictor_and_predictions();
    update_gradient_and_errors();
    calculate_and_validate_validation_error(boosting_step);
    if(!abort_boosting)
    {
        intercept+=intercept_update;
        intercept_steps[boosting_step]=intercept;
    }
}

void APLRRegressor::update_linear_predictor_and_predictions()
{
    linear_predictor_current+=linear_predictor_update;
    linear_predictor_current_validation+=linear_predictor_update_validation;
    predictions_current=transform_linear_predictor_to_predictions(linear_predictor_current,link_function,tweedie_power);
    predictions_current_validation=transform_linear_predictor_to_predictions(linear_predictor_current_validation,link_function,tweedie_power);
}

void APLRRegressor::update_gradient_and_errors()
{
    neg_gradient_current=calculate_neg_gradient_current();
    neg_gradient_nullmodel_errors=calculate_errors(neg_gradient_current,linear_predictor_null_model,sample_weight_train);
    neg_gradient_nullmodel_errors_sum=calculate_sum_error(neg_gradient_nullmodel_errors);
}

std::vector<size_t> APLRRegressor::find_terms_eligible_current_indexes_for_a_base_term(size_t base_term)
{
    std::vector<size_t> terms_eligible_current_indexes_for_a_base_term;
    terms_eligible_current_indexes_for_a_base_term.reserve(terms_eligible_current.size());
    for (size_t i = 0; i < terms_eligible_current.size(); ++i)
    {
        bool term_is_eligible{terms_eligible_current[i].base_term==base_term && terms_eligible_current[i].ineligible_boosting_steps==0};
        if(term_is_eligible)
            terms_eligible_current_indexes_for_a_base_term.push_back(i);
    }
    terms_eligible_current_indexes_for_a_base_term.shrink_to_fit();
    return terms_eligible_current_indexes_for_a_base_term;
}

void APLRRegressor::estimate_split_point_for_each_term(std::vector<Term> &terms, std::vector<size_t> &terms_indexes)
{
    bool multithreading{n_jobs!=1 && terms_indexes.size()>1};
    if(multithreading)
    {
        distributed_terms=distribute_terms_indexes_to_cores(terms_indexes,n_jobs);

        std::vector<std::thread> threads(distributed_terms.size());
        
        auto estimate_split_point_for_distributed_terms_in_one_thread=[this, &terms, &terms_indexes](size_t thread_index)
        {
            for (size_t i = 0; i < distributed_terms[thread_index].size(); ++i)
            {
                terms[terms_indexes[distributed_terms[thread_index][i]]].estimate_split_point(X_train,neg_gradient_current,sample_weight_train,bins,v,min_observations_in_split);
            }
        };
        
        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i]=std::thread(estimate_split_point_for_distributed_terms_in_one_thread,i);
        }
        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
        }

    }
    else
    {
        for (size_t i = 0; i < terms_indexes.size(); ++i)
        {
            terms[terms_indexes[i]].estimate_split_point(X_train,neg_gradient_current,sample_weight_train,bins,v,min_observations_in_split);
        }            
    }
}

size_t APLRRegressor::find_best_term_index(std::vector<Term> &terms, std::vector<size_t> &terms_indexes)
{
    size_t best_term_index{std::numeric_limits<size_t>::max()};
    double lowest_errors_sum{neg_gradient_nullmodel_errors_sum};

    for (auto &term_index:terms_indexes)
    {
        bool term_is_eligible{terms[term_index].ineligible_boosting_steps==0};
        if(term_is_eligible)
        {
            if(std::isless(terms[term_index].split_point_search_errors_sum,lowest_errors_sum))
            {
                best_term_index=term_index;
                lowest_errors_sum=terms[term_index].split_point_search_errors_sum;
            }                
        }
    }

    return best_term_index;
}

void APLRRegressor::consider_interactions(const std::vector<size_t> &available_predictor_indexes)
{
    bool consider_interactions{terms.size()>0 && max_interaction_level>0 && interactions_eligible<max_interactions};
    if(consider_interactions)
    {
        determine_interactions_to_consider(available_predictor_indexes);
        std::vector<size_t> interactions_to_consider_indexes{create_term_indexes(interactions_to_consider)};
        estimate_split_point_for_each_term(interactions_to_consider, interactions_to_consider_indexes);
        find_sorted_indexes_for_errors_for_interactions_to_consider();
        add_promising_interactions_and_select_the_best_one();
    }
}

void APLRRegressor::determine_interactions_to_consider(const std::vector<size_t> &available_predictor_indexes)
{
    interactions_to_consider=std::vector<Term>();
    interactions_to_consider.reserve(static_cast<size_t>(X_train.cols())*terms.size());

    VectorXi indexes_for_terms_to_consider_as_interaction_partners{find_indexes_for_terms_to_consider_as_interaction_partners()};
    for (auto &model_term_index:indexes_for_terms_to_consider_as_interaction_partners)
    {
        for(auto &new_term_index:available_predictor_indexes)
        {
            bool term_is_eligible{terms_eligible_current[new_term_index].ineligible_boosting_steps==0};
            if(term_is_eligible)
            {
                Term interaction{Term(new_term_index)};
                Term model_term_without_given_terms{terms[model_term_index]};
                bool model_term_has_given_terms{terms[model_term_index].given_terms.size()>0};
                if(model_term_has_given_terms)
                {
                    model_term_without_given_terms.given_terms.clear();
                    add_necessary_given_terms_to_interaction(interaction, terms[model_term_index]);
                }
                model_term_without_given_terms.cleanup_when_this_term_was_added_as_a_given_term();
                interaction.given_terms.push_back(model_term_without_given_terms);
                bool interaction_level_is_too_high{interaction.get_interaction_level()>max_interaction_level};
                if(interaction_level_is_too_high) continue;
                bool interaction_is_already_in_the_model{false};
                for (auto &term:terms)
                {
                    if(interaction==term)
                    {
                        interaction_is_already_in_the_model=true;
                        break;
                    }
                }
                if(interaction_is_already_in_the_model) continue;
                bool interaction_already_exists_in_terms_eligible_current{false};
                for(auto &term_eligible_current:terms_eligible_current)
                {
                    if(interaction.base_term==term_eligible_current.base_term && Term::equals_given_terms(interaction,term_eligible_current))
                    {
                        interaction_already_exists_in_terms_eligible_current=true;
                        break;
                    }                    
                }
                if(interaction_already_exists_in_terms_eligible_current) continue;
                interactions_to_consider.push_back(interaction); 
            }
        }                    
    }
}

VectorXi APLRRegressor::find_indexes_for_terms_to_consider_as_interaction_partners()
{
    size_t number_of_terms_to_consider_as_interaction_partners{find_out_how_many_terms_to_consider_as_interaction_partners()};
    VectorXd split_point_errors(terms.size());
    VectorXi indexes_for_terms_to_consider_as_interaction_partners(terms.size());
    size_t count{0};
    for (size_t i = 0; i < terms.size(); ++i)
    {
        if(terms[i].get_can_be_used_as_a_given_term())
        {
            split_point_errors[count] = terms[i].split_point_search_errors_sum;
            indexes_for_terms_to_consider_as_interaction_partners[count] = i;
            ++count;
        }
    }
    split_point_errors.conservativeResize(count);
    indexes_for_terms_to_consider_as_interaction_partners.conservativeResize(count);
    bool selecting_the_terms_with_lowest_previous_errors_is_necessary{number_of_terms_to_consider_as_interaction_partners < indexes_for_terms_to_consider_as_interaction_partners.size()};
    if(selecting_the_terms_with_lowest_previous_errors_is_necessary)
    {
        VectorXi sorted_indexes{sort_indexes_ascending(split_point_errors)};
        VectorXi temp_indexes(number_of_terms_to_consider_as_interaction_partners);
        for (size_t i = 0; i < number_of_terms_to_consider_as_interaction_partners; ++i)
        {
            temp_indexes[i] = indexes_for_terms_to_consider_as_interaction_partners[sorted_indexes[i]];
        }
        indexes_for_terms_to_consider_as_interaction_partners=std::move(temp_indexes);
    }
    return indexes_for_terms_to_consider_as_interaction_partners;
}

size_t APLRRegressor::find_out_how_many_terms_to_consider_as_interaction_partners()
{
    size_t terms_to_consider{terms.size()};
    if(max_eligible_terms>0)
    {
        terms_to_consider=std::min(max_eligible_terms,terms.size());
    }
    return terms_to_consider;
}

void APLRRegressor::add_necessary_given_terms_to_interaction(Term &interaction, Term &existing_model_term)
{
    bool model_term_has_multiple_given_terms{existing_model_term.given_terms.size()>1};
    if(model_term_has_multiple_given_terms)
    {
        MatrixXi value_indicator_for_each_given_term(X_train.rows(), existing_model_term.given_terms.size());
        for (size_t col = 0; col < existing_model_term.given_terms.size(); ++col)
        {
            value_indicator_for_each_given_term.col(col) = calculate_indicator(existing_model_term.given_terms[col].calculate(X_train));
        }   
        for (size_t col = 0; col < existing_model_term.given_terms.size(); ++col)
        {
            VectorXi combined_value_indicator_for_the_other_given_terms{VectorXi::Constant(X_train.rows(),1)};
            for (size_t col2 = 0; col2 < existing_model_term.given_terms.size(); ++col2)
            {
                bool is_other_given_term{col2!=col};
                if(is_other_given_term)
                {
                    combined_value_indicator_for_the_other_given_terms = combined_value_indicator_for_the_other_given_terms.array() * value_indicator_for_each_given_term.col(col2).array();
                }
            }
            
            bool given_term_provides_an_unique_zero{false};
            for (size_t row = 0; row < static_cast<size_t>(X_train.rows()); ++row)
            {
                given_term_provides_an_unique_zero = combined_value_indicator_for_the_other_given_terms[row]>0 && value_indicator_for_each_given_term.col(col)[row]==0;
                if(given_term_provides_an_unique_zero)
                    break;
            }

            bool stricter_given_term_exists{false};
            for (size_t col2 = 0; col2 < existing_model_term.given_terms.size(); ++col2)
            {
                bool is_other_given_term{col2!=col};
                if(is_other_given_term)
                {
                    bool same_base_term{existing_model_term.given_terms[col].base_term == existing_model_term.given_terms[col2].base_term};
                    bool same_direction{existing_model_term.given_terms[col].direction_right == existing_model_term.given_terms[col2].direction_right};
                    bool finite_split_point{std::isfinite(existing_model_term.given_terms[col].split_point) && std::isfinite(existing_model_term.given_terms[col2].split_point)};
                    if(same_base_term && same_direction && finite_split_point)
                    {
                        bool direction_right{existing_model_term.given_terms[col].direction_right};
                        if(direction_right)
                        {
                            stricter_given_term_exists = std::isless(existing_model_term.given_terms[col].split_point, existing_model_term.given_terms[col2].split_point);
                        }
                        else
                        {
                            stricter_given_term_exists = std::isgreater(existing_model_term.given_terms[col].split_point, existing_model_term.given_terms[col2].split_point);
                        }
                        if(stricter_given_term_exists)
                            break;
                    }
                }
            }

            if(given_term_provides_an_unique_zero && !stricter_given_term_exists)
                interaction.given_terms.push_back(existing_model_term.given_terms[col]);
        }
    }
    else
    {
        for(auto &given_term:existing_model_term.given_terms)
        {
            interaction.given_terms.push_back(given_term);
        }
    }
}

void APLRRegressor::find_sorted_indexes_for_errors_for_interactions_to_consider()
{
    VectorXd errors_for_interactions_to_consider(interactions_to_consider.size());
    for (size_t i = 0; i < interactions_to_consider.size(); ++i)
    {
        errors_for_interactions_to_consider[i]=interactions_to_consider[i].split_point_search_errors_sum;
    }
    sorted_indexes_of_errors_for_interactions_to_consider=sort_indexes_ascending(errors_for_interactions_to_consider);
}

void APLRRegressor::add_promising_interactions_and_select_the_best_one()
{
    size_t best_term_before_interactions{best_term_index};
    for (size_t j = 0; j < static_cast<size_t>(sorted_indexes_of_errors_for_interactions_to_consider.size()); ++j) //for each interaction to consider starting from lowest to highest error
    {
        bool allowed_to_add_one_interaction{interactions_eligible<max_interactions};
        if(allowed_to_add_one_interaction)
        {
            bool error_is_less_than_for_best_term_before_interactions{std::isless(interactions_to_consider[sorted_indexes_of_errors_for_interactions_to_consider[j]].split_point_search_errors_sum,terms_eligible_current[best_term_before_interactions].split_point_search_errors_sum)};
            if(error_is_less_than_for_best_term_before_interactions)
            {
                add_term_to_terms_eligible_current(interactions_to_consider[sorted_indexes_of_errors_for_interactions_to_consider[j]]);
                bool is_best_interaction{j==0};
                if(is_best_interaction)
                    best_term_index=terms_eligible_current.size()-1;        
                ++interactions_eligible;
            }
            else
                break;
        }
    }
}

void APLRRegressor::select_the_best_term_and_update_errors(size_t boosting_step, bool not_evaluating_prioritized_predictors)
{
    bool no_term_was_selected{best_term_index == std::numeric_limits<size_t>::max()};
    if(no_term_was_selected)
    {
        if(not_evaluating_prioritized_predictors) 
            abort_boosting=true;
        return;
    }

    linear_predictor_update=terms_eligible_current[best_term_index].calculate_contribution_to_linear_predictor(X_train);
    linear_predictor_update_validation=terms_eligible_current[best_term_index].calculate_contribution_to_linear_predictor(X_validation);
    double error_after_updating_term=calculate_sum_error(calculate_errors(neg_gradient_current,linear_predictor_update,sample_weight_train));
    bool no_improvement{std::isgreaterequal(error_after_updating_term,neg_gradient_nullmodel_errors_sum)};
    if(no_improvement)
    {
        if(not_evaluating_prioritized_predictors)
            abort_boosting=true;
    }
    else
    {
        update_linear_predictor_and_predictions();
        update_gradient_and_errors();
        double backup_of_validation_error{validation_error_steps[boosting_step]};
        calculate_and_validate_validation_error(boosting_step);
        if(abort_boosting)
            validation_error_steps[boosting_step]=backup_of_validation_error;
        else
            update_terms(boosting_step);
    }
}

void APLRRegressor::update_terms(size_t boosting_step)
{
    bool no_term_is_in_model{terms.size()==0};
    if(no_term_is_in_model)
        add_new_term(boosting_step);
    else
    {   
        bool found{false};
        for (size_t j = 0; j < terms.size(); ++j)
        {
            bool term_is_already_in_model{terms[j]==terms_eligible_current[best_term_index]};
            if(term_is_already_in_model)
            {
                terms[j].coefficient+=terms_eligible_current[best_term_index].coefficient;
                terms[j].coefficient_steps[boosting_step]=terms[j].coefficient;
                found=true;
                break;
            } 
        }
        if(!found) 
        {
            add_new_term(boosting_step);
        }
    }
}

void APLRRegressor::add_new_term(size_t boosting_step)
{
    terms_eligible_current[best_term_index].coefficient_steps=VectorXd::Constant(m,0);
    terms_eligible_current[best_term_index].coefficient_steps[boosting_step]=terms_eligible_current[best_term_index].coefficient;
    terms.push_back(Term(terms_eligible_current[best_term_index]));
}

void APLRRegressor::calculate_and_validate_validation_error(size_t boosting_step)
{
    validation_error_steps[boosting_step]=calculate_mean_error(calculate_errors(y_validation,predictions_current_validation,sample_weight_validation,family,tweedie_power),sample_weight_validation);
    bool validation_error_is_invalid{std::isinf(validation_error_steps[boosting_step])};
    if(validation_error_is_invalid)
    {
        abort_boosting=true;
        std::string warning_message{"Warning: Encountered numerical problems when calculating prediction errors in the previous boosting step. Not continuing with further boosting steps. One potential reason is if the combination of family and link_function is invalid."};
        std::cout<<warning_message<<"\n";
    }
}

void APLRRegressor::update_term_eligibility()
{
    number_of_eligible_terms=terms_eligible_current.size();
    bool eligibility_is_used{ineligible_boosting_steps_added>0 && max_eligible_terms>0};
    if(eligibility_is_used)
    {
        VectorXd terms_eligible_current_split_point_errors(terms_eligible_current.size());
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            terms_eligible_current_split_point_errors[i]=terms_eligible_current[i].split_point_search_errors_sum;
        }
        VectorXi sorted_split_point_errors_indexes{sort_indexes_ascending(terms_eligible_current_split_point_errors)};

        size_t count{0};
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            bool term_is_eligible_now{terms_eligible_current[sorted_split_point_errors_indexes[i]].ineligible_boosting_steps==0};
            if(term_is_eligible_now)
            {
                ++count;
                bool term_should_receive_added_boosting_steps{count>max_eligible_terms};
                if(term_should_receive_added_boosting_steps)
                    terms_eligible_current[sorted_split_point_errors_indexes[i]].ineligible_boosting_steps+=ineligible_boosting_steps_added;
            }
            else
            {
                terms_eligible_current[sorted_split_point_errors_indexes[i]].ineligible_boosting_steps-=1;
            }
        }

        number_of_eligible_terms=0;
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            bool term_is_eligible{terms_eligible_current[i].ineligible_boosting_steps==0};
            if(term_is_eligible)
                ++number_of_eligible_terms;
        }
    }
}

void APLRRegressor::print_summary_after_boosting_step(size_t boosting_step)
{
    if(verbosity>=2)
    {
        std::cout<<"Boosting step: "<<boosting_step+1<<". Unique terms: "<<terms.size()<<". Terms eligible: "<<number_of_eligible_terms<<". Validation error: "<<validation_error_steps[boosting_step]<<".\n";
    }
}

void APLRRegressor::update_coefficients_for_all_steps()
{
    for (size_t j = 0; j < m; ++j)
    {
        bool fill_down_coefficient_steps{j>0 && is_approximately_zero(intercept_steps[j]) && !is_approximately_zero(intercept_steps[j-1])};
        if(fill_down_coefficient_steps)
            intercept_steps[j]=intercept_steps[j-1];
    }

    for (size_t i = 0; i < terms.size(); ++i)
    {
        for (size_t j = 0; j < m; ++j)
        {
            bool fill_down_coefficient_steps{j>0 && is_approximately_zero(terms[i].coefficient_steps[j]) && !is_approximately_zero(terms[i].coefficient_steps[j-1])};
            if(fill_down_coefficient_steps)
                terms[i].coefficient_steps[j]=terms[i].coefficient_steps[j-1];
        }
    }
}

void APLRRegressor::print_final_summary()
{
    if(verbosity>=1)
    {
        std::cout<<"Unique terms: "<<terms.size()<<". Terms available in final boosting step: "<<terms_eligible_current.size()<<".\n";
    }
}

void APLRRegressor::find_optimal_m_and_update_model_accordingly()
{
    size_t best_boosting_step_index;
    validation_error_steps.minCoeff(&best_boosting_step_index);
    intercept=intercept_steps[best_boosting_step_index];
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].coefficient = terms[i].coefficient_steps[best_boosting_step_index];
    }
    m=best_boosting_step_index+1; 

    //Removing unused terms
    std::vector<Term> terms_new;
    terms_new.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        bool term_is_used{!is_approximately_zero(terms[i].coefficient)};
        if(term_is_used)
            terms_new.push_back(terms[i]);
    }
    terms=std::move(terms_new);
}

void APLRRegressor::revert_scaling_if_using_log_link_function()
{
    if(link_function=="log")
    {
        y_train/=scaling_factor_for_log_link_function;
        y_validation/=scaling_factor_for_log_link_function;
        intercept+=std::log(1/scaling_factor_for_log_link_function);
        for (size_t i = 0; i < static_cast<size_t>(intercept_steps.size()); ++i)
        {
            intercept_steps[i]+=std::log(1/scaling_factor_for_log_link_function);
        }
    }
}

void APLRRegressor::name_terms(const MatrixXd &X, const std::vector<std::string> &X_names)
{
    bool x_names_not_provided{X_names.size()==0};
    if(x_names_not_provided)
    {
        std::vector<std::string> temp(X.cols());
        for (size_t i = 0; i < static_cast<size_t>(X.cols()); ++i)
        {
            temp[i]="X"+std::to_string(i+1);
        }
        set_term_names(temp);
    }
    else
    {
        set_term_names(X_names);
    }
}

void APLRRegressor::set_term_names(const std::vector<std::string> &X_names)
{
    bool model_has_not_been_trained{!std::isfinite(intercept)};
    if(model_has_not_been_trained)
        throw std::runtime_error("The model must be trained with fit() before term names can be set.");

    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].name = compute_raw_base_term_name(terms[i], X_names[terms[i].base_term]);
        bool term_has_given_terms{terms[i].given_terms.size()>0};
        if(term_has_given_terms)
        {
            terms[i].name += " * I(";
            for (size_t j = 0; j < terms[i].given_terms.size(); ++j)
            {
                terms[i].name += compute_raw_base_term_name(terms[i].given_terms[j], X_names[terms[i].given_terms[j].base_term])+"*";
            }
            terms[i].name.pop_back();
            terms[i].name += "!=0)";
        }
        terms[i].name="P"+std::to_string(i)+". Interaction level: "+std::to_string(terms[i].get_interaction_level())+". "+terms[i].name;
    }

    term_names.resize(terms.size()+1);
    term_coefficients.resize(terms.size()+1);
    term_names[0]="Intercept";
    term_coefficients[0]=intercept;
    for (size_t i = 0; i < terms.size(); ++i)
    {
        term_names[i+1]=terms[i].name;
        term_coefficients[i+1]=terms[i].coefficient;
    }   
}

std::string APLRRegressor::compute_raw_base_term_name(const Term &term, const std::string &X_name)
{
    std::string name{""};
    bool is_linear_effect{std::isnan(term.split_point)};
    if(is_linear_effect)
        name=X_name;
    else
    {
        double temp_split_point{term.split_point};
        std::string sign{"-"};
        if(std::isless(temp_split_point,0))
        {
            temp_split_point=-temp_split_point;
            sign="+";
        }
        if(term.direction_right)
            name="max("+X_name+sign+std::to_string(temp_split_point)+",0)";
        else
            name="min("+X_name+sign+std::to_string(temp_split_point)+",0)";
    }
    return name;
}

void APLRRegressor::calculate_feature_importance_on_validation_set()
{
    feature_importance=VectorXd::Constant(number_of_base_terms,0);
    MatrixXd li{calculate_local_feature_importance(X_validation)};
    for (size_t i = 0; i < static_cast<size_t>(li.cols()); ++i) //for each column calculate mean abs values
    {
        feature_importance[i]=li.col(i).cwiseAbs().mean();
    }
}

MatrixXd APLRRegressor::calculate_local_feature_importance(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(),number_of_base_terms,0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        output.col(terms[i].base_term)+=contrib;
    }

    return output;
}

void APLRRegressor::find_min_and_max_training_predictions_or_responses()
{
    VectorXd training_predictions{predict(X_train,false)};
    min_training_prediction_or_response=std::max(training_predictions.minCoeff(), y_train.minCoeff());
    max_training_prediction_or_response=std::min(training_predictions.maxCoeff(), y_train.maxCoeff());
}

void APLRRegressor::calculate_validation_group_mse()
{
    VectorXd validation_predictions{predict(X_validation,false)};
    VectorXi validation_predictions_sorted_index{sort_indexes_ascending(validation_predictions)};
    VectorXd y_validation_centered{calculate_rolling_centered_mean(y_validation,validation_predictions_sorted_index,group_size_for_validation_group_mse,sample_weight_validation)};
    VectorXd validation_predictions_centered{calculate_rolling_centered_mean(validation_predictions,validation_predictions_sorted_index,group_size_for_validation_group_mse,sample_weight_validation)};

    VectorXd squared_residuals{(y_validation_centered-validation_predictions_centered).array().pow(2)};
    validation_group_mse =  squared_residuals.mean();
}

void APLRRegressor::validate_that_model_can_be_used(const MatrixXd &X)
{
    if(std::isnan(intercept) || number_of_base_terms==0) throw std::runtime_error("Model must be trained before predict() can be run.");
    if(X.rows()==0) throw std::runtime_error("X cannot have zero rows.");
    size_t cols_provided{static_cast<size_t>(X.cols())};
    if(cols_provided!=number_of_base_terms) throw std::runtime_error("X must have "+std::to_string(number_of_base_terms) +" columns but "+std::to_string(cols_provided)+" were provided.");
    throw_error_if_matrix_has_nan_or_infinite_elements(X, "X");
}

void APLRRegressor::cleanup_after_fit()
{
    terms.shrink_to_fit();
    X_train.resize(0,0);
    y_train.resize(0);
    sample_weight_train.resize(0);
    X_validation.resize(0,0);
    y_validation.resize(0);
    sample_weight_validation.resize(0);
    linear_predictor_null_model.resize(0);
    terms_eligible_current.clear();
    predictions_current.resize(0);
    predictions_current_validation.resize(0);
    neg_gradient_current.resize(0);
    neg_gradient_nullmodel_errors.resize(0);
    linear_predictor_update.resize(0);
    linear_predictor_update_validation.resize(0);
    distributed_terms.clear();
    interactions_to_consider.clear();
    sorted_indexes_of_errors_for_interactions_to_consider.resize(0);
    linear_predictor_current.resize(0);
    linear_predictor_current_validation.resize(0);
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].cleanup_after_fit();
    }
    predictor_indexes.clear();
    prioritized_predictors_indexes.clear();
}

VectorXd APLRRegressor::predict(const MatrixXd &X, bool cap_predictions_to_minmax_in_training)
{
    validate_that_model_can_be_used(X);

    VectorXd linear_predictor{calculate_linear_predictor(X)};
    VectorXd predictions{transform_linear_predictor_to_predictions(linear_predictor,link_function,tweedie_power)};

    if(cap_predictions_to_minmax_in_training)
    {
        this->cap_predictions_to_minmax_in_training(predictions);
    }

    return predictions;
}

VectorXd APLRRegressor::calculate_linear_predictor(const MatrixXd &X)
{
    VectorXd predictions{VectorXd::Constant(X.rows(),intercept)};
    for (size_t i = 0; i < terms.size(); ++i) //for each term
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        predictions+=contrib;
    }
    return predictions;    
}

void APLRRegressor::cap_predictions_to_minmax_in_training(VectorXd &predictions)
{
    for (size_t i = 0; i < static_cast<size_t>(predictions.rows()); ++i)
    {
        if(std::isgreater(predictions[i],max_training_prediction_or_response))
            predictions[i]=max_training_prediction_or_response;
        else if(std::isless(predictions[i],min_training_prediction_or_response))
            predictions[i]=min_training_prediction_or_response;
    }
}

MatrixXd APLRRegressor::calculate_local_feature_importance_for_terms(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(),terms.size(),0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd contrib{terms[i].calculate_contribution_to_linear_predictor(X)};
        output.col(i)+=contrib;
    }

    return output;
}

MatrixXd APLRRegressor::calculate_terms(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    MatrixXd output{MatrixXd::Constant(X.rows(),terms.size(),0)};

    for (size_t i = 0; i < terms.size(); ++i)
    {
        VectorXd values{terms[i].calculate(X)};
        output.col(i)+=values;
    }

    return output;
}

std::vector<std::string> APLRRegressor::get_term_names()
{
    return term_names;
}

VectorXd APLRRegressor::get_term_coefficients()
{
    return term_coefficients;
}

VectorXd APLRRegressor::get_term_coefficient_steps(size_t term_index)
{
    return terms[term_index].coefficient_steps;
}

VectorXd APLRRegressor::get_validation_error_steps()
{
    return validation_error_steps;
}

VectorXd APLRRegressor::get_feature_importance()
{
    return feature_importance;
}

double APLRRegressor::get_intercept()
{
    return intercept;
}

VectorXd APLRRegressor::get_intercept_steps()
{
    return intercept_steps;
}

size_t APLRRegressor::get_m()
{
    return m;
}

double APLRRegressor::get_validation_group_mse()
{
    return validation_group_mse;
}