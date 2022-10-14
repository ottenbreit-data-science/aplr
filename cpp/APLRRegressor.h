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
    size_t reserved_terms_times_num_x; //how many times number of variables in X to reserve memory for term (terms in model)
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
    size_t best_term;
    double lowest_error_sum;
    double error_after_updating_intercept;
    VectorXd linear_predictor_update;
    VectorXd linear_predictor_update_validation;
    double intercept_test;
    size_t number_of_eligible_terms;
    std::vector<std::vector<size_t>> distributed_terms;
    std::vector<Term> interactions_to_consider;
    VectorXi error_index_for_interactions_to_consider;
    bool abort_boosting;
    VectorXd linear_predictor_current;
    VectorXd linear_predictor_current_validation;

    //Methods
    void validate_input_to_fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,const std::vector<std::string> &X_names, const std::vector<size_t> &validation_set_indexes);
    void throw_error_if_validation_set_indexes_has_invalid_indexes(const VectorXd &y, const std::vector<size_t> &validation_set_indexes);
    void define_training_and_validation_sets(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight, const std::vector<size_t> &validation_set_indexes);
    void initialize();
    bool check_if_base_term_has_only_one_unique_value(size_t base_term);
    void add_term_to_terms_eligible_current(Term &term);
    VectorXd calculate_neg_gradient_current(const VectorXd &y,const VectorXd &predictions_current);
    void execute_boosting_steps();
    void execute_boosting_step(size_t boosting_step);
    void find_best_split_for_each_eligible_term();
    void consider_interactions();
    void determine_interactions_to_consider();
    void estimate_split_points_for_interactions_to_consider();
    void sort_errors_for_interactions_to_consider();
    void add_promising_interactions_and_select_the_best_one();
    void consider_updating_intercept();
    void select_the_best_term_and_update_errors(size_t boosting_step);
    void update_gradient_and_errors();
    void add_new_term(size_t boosting_step);
    void update_term_eligibility();
    void print_summary_after_boosting_step(size_t boosting_step);
    void update_coefficients_for_all_steps();
    void print_final_summary();
    void find_optimal_m_and_update_model_accordingly();
    void name_terms(const MatrixXd &X, const std::vector<std::string> &X_names);
    void calculate_feature_importance_on_validation_set();
    void cleanup_after_fit();
    void validate_that_model_can_be_used(const MatrixXd &X);
    void throw_error_if_family_does_not_exist();
    void throw_error_if_link_function_does_not_exist();
    VectorXd calculate_linear_predictor(const MatrixXd &X);
    void update_linear_predictor_and_predictors();
    void throw_error_if_response_contains_invalid_values(const VectorXd &y);
    void throw_error_if_response_is_not_between_0_and_1(const VectorXd &y);
    void throw_error_if_response_is_negative(const VectorXd &y);
    void throw_error_if_response_is_not_greater_than_zero(const VectorXd &y);
    void throw_error_if_tweedie_power_is_invalid();
    
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
    size_t bins; //used if nobs>bins
    size_t verbosity; //0 none, 1 summary after running fit(), 2 each boosting step when running fit().
    std::vector<std::string> term_names;
    VectorXd term_coefficients;
    size_t max_interaction_level;
    VectorXd intercept_steps;
    size_t max_interactions; //max interactions allowed to add (counted in interactions_eligible)
    size_t interactions_eligible; //interactions that were eligible when training the model
    VectorXd validation_error_steps; //validation error for each boosting step
    size_t min_observations_in_split; //Must be at least 1
    size_t ineligible_boosting_steps_added; //Determines the magnitude of ineligible_boosting_steps when set to >0 during fit(). Not used if 0.
    size_t max_eligible_terms; //Determines how many terms with ineligible_boosting_steps=0 are supposed to be left eligible
                                    //at the end of each boosting step (before decreasing ineligible_boosting_steps by one for all 
                                    //terms with ineligible_boosting_steps>0). Not used if 0.
    size_t number_of_base_terms; 
    VectorXd feature_importance; //Populated in fit() using validation set. Rows are in the same order as in X.
    double tweedie_power;

    //Methods
    APLRRegressor(size_t m=1000,double v=0.1,uint_fast32_t random_state=std::numeric_limits<uint_fast32_t>::lowest(),std::string family="gaussian",
        std::string link_function="identity", size_t n_jobs=0, double validation_ratio=0.2,double intercept=NAN_DOUBLE,
        size_t reserved_terms_times_num_x=100, size_t bins=300,size_t verbosity=0,size_t max_interaction_level=1,size_t max_interactions=100000,
        size_t min_observations_in_split=20, size_t ineligible_boosting_steps_added=10, size_t max_eligible_terms=5,double tweedie_power=1.5);
    APLRRegressor(const APLRRegressor &other);
    ~APLRRegressor();
    void fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight=VectorXd(0),const std::vector<std::string> &X_names={},const std::vector<size_t> &validation_set_indexes={});
    VectorXd predict(const MatrixXd &X);
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
};

//Regular constructor
APLRRegressor::APLRRegressor(size_t m,double v,uint_fast32_t random_state,std::string family,std::string link_function,size_t n_jobs,
    double validation_ratio,double intercept,size_t reserved_terms_times_num_x,size_t bins,size_t verbosity,size_t max_interaction_level,
    size_t max_interactions,size_t min_observations_in_split,size_t ineligible_boosting_steps_added,size_t max_eligible_terms,double tweedie_power):
        reserved_terms_times_num_x{reserved_terms_times_num_x},intercept{intercept},m{m},v{v},
        family{family},link_function{link_function},validation_ratio{validation_ratio},n_jobs{n_jobs},random_state{random_state},
        bins{bins},verbosity{verbosity},max_interaction_level{max_interaction_level},
        intercept_steps{VectorXd(0)},max_interactions{max_interactions},interactions_eligible{0},validation_error_steps{VectorXd(0)},
        min_observations_in_split{min_observations_in_split},ineligible_boosting_steps_added{ineligible_boosting_steps_added},
        max_eligible_terms{max_eligible_terms},number_of_base_terms{0},tweedie_power{tweedie_power}
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
    feature_importance{other.feature_importance},tweedie_power{other.tweedie_power}
{
}

//Destructor
APLRRegressor::~APLRRegressor()
{
}

//Fits the model
//X_names specifies names for each column in X. If not specified then X1, X2, X3, ... will be used as names for each column in X.
//If validation_set_indexes.size()>0 then validation_set_indexes defines which of the indices in X, y and sample_weight are used to validate, 
//invalidating validation_ratio. The rest of indices are used to train. 
void APLRRegressor::fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,const std::vector<std::string> &X_names,const std::vector<size_t> &validation_set_indexes)
{
    throw_error_if_family_does_not_exist();
    throw_error_if_link_function_does_not_exist();
    throw_error_if_tweedie_power_is_invalid();
    validate_input_to_fit(X,y,sample_weight,X_names,validation_set_indexes);
    define_training_and_validation_sets(X,y,sample_weight,validation_set_indexes);
    initialize();
    execute_boosting_steps();
    update_coefficients_for_all_steps();
    print_final_summary();
    find_optimal_m_and_update_model_accordingly();
    name_terms(X, X_names);
    calculate_feature_importance_on_validation_set();
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
    else if(link_function=="tweedie")
        link_function_exists=true;
    else if(link_function=="inverse")
        link_function_exists=true;        
    if(!link_function_exists)
        throw std::runtime_error("Link function "+link_function+" is not available in APLR.");
}

void APLRRegressor::throw_error_if_tweedie_power_is_invalid()
{
    bool tweedie_power_equals_invalid_poits{check_if_approximately_equal(tweedie_power,1.0) || check_if_approximately_equal(tweedie_power,2.0)};
    bool tweedie_power_is_in_invalid_range{std::isless(tweedie_power,1.0)};
    bool tweedie_power_is_invalid{tweedie_power_equals_invalid_poits || tweedie_power_is_in_invalid_range};
    if(tweedie_power_is_invalid)
        throw std::runtime_error("Tweedie power is invalid. It must not equal 1.0 or 2.0 and cannot be below 1.0.");
}

void APLRRegressor::validate_input_to_fit(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,const std::vector<std::string> &X_names, const std::vector<size_t> &validation_set_indexes)
{
    if(X.rows()!=y.size()) throw std::runtime_error("X and y must have the same number of rows.");
    if(X.rows()<2) throw std::runtime_error("X and y cannot have less than two rows.");
    if(sample_weight.size()>0 && sample_weight.size()!=y.size()) throw std::runtime_error("sample_weight must have 0 or as many rows as X and y.");
    if(X_names.size()>0 && X_names.size()!=static_cast<size_t>(X.cols())) throw std::runtime_error("X_names must have as many columns as X.");
    throw_error_if_matrix_has_nan_or_infinite_elements(X, "X");
    throw_error_if_matrix_has_nan_or_infinite_elements(y, "y");
    throw_error_if_matrix_has_nan_or_infinite_elements(sample_weight, "sample_weight");
    throw_error_if_validation_set_indexes_has_invalid_indexes(y, validation_set_indexes);
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

void APLRRegressor::throw_error_if_response_contains_invalid_values(const VectorXd &y)
{
    if(link_function=="logit")
        throw_error_if_response_is_not_between_0_and_1(y);
    else if(link_function=="log" || (link_function=="tweedie" && std::isgreater(tweedie_power,1) && std::isless(tweedie_power,2)) )
        throw_error_if_response_is_negative(y);
    else if(link_function=="inverse" || (link_function=="tweedie" && std::isgreater(tweedie_power,2)) )
        throw_error_if_response_is_not_greater_than_zero(y);
}

void APLRRegressor::throw_error_if_response_is_not_between_0_and_1(const VectorXd &y)
{
    bool response_is_less_than_zero{(y.array()<0.0).any()};
    bool response_is_greater_than_one{(y.array()>1.0).any()};
    if(response_is_less_than_zero || response_is_greater_than_one)
        throw std::runtime_error("Response values for "+link_function+" link functions cannot be less than zero or greater than one.");   
}

void APLRRegressor::throw_error_if_response_is_negative(const VectorXd &y)
{
    bool response_is_less_than_zero{(y.array()<0.0).any()};
    if(response_is_less_than_zero)
        throw std::runtime_error("Response values for "+link_function+" link functions cannot be less than zero.");   
}

void APLRRegressor::throw_error_if_response_is_not_greater_than_zero(const VectorXd &y)
{
    bool response_is_not_greater_than_zero{(y.array()<=0.0).any()};
    if(response_is_not_greater_than_zero)
        throw std::runtime_error("Response values for "+link_function+" link functions must be greater than zero.");   

}

void APLRRegressor::define_training_and_validation_sets(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight, const std::vector<size_t> &validation_set_indexes)
{
    //Defining train and validation indexes
    size_t y_size{static_cast<size_t>(y.size())};
    std::vector<size_t> train_indexes;
    std::vector<size_t> validation_indexes;
    if(validation_set_indexes.size()>0) //If validation_set_indexes is specified then the split is pre-defined and will be used
    {
        std::vector<size_t> all_indexes(y_size);
        std::iota(std::begin(all_indexes),std::end(all_indexes),0);
        validation_indexes=validation_set_indexes;
        train_indexes.reserve(y_size-validation_indexes.size()); 
        std::remove_copy_if(all_indexes.begin(),all_indexes.end(),std::back_inserter(train_indexes),[&validation_indexes](const size_t &arg)
            { return (std::find(validation_indexes.begin(),validation_indexes.end(),arg) != validation_indexes.end());});
    }
    else //Using validation_ratio to randomly sample train and validation indexes
    {
        train_indexes.reserve(y_size);
        validation_indexes.reserve(y_size);
        std::mt19937 mersenne{random_state};
        std::uniform_real_distribution<double> distribution(0.0,1.0);
        double roll;
        for (size_t i = 0; i < y_size; ++i) //for each observation place into training or test
        {
            roll=distribution(mersenne);
            if(std::isless(roll,validation_ratio)) //place in test set
            {
                validation_indexes.push_back(i);
            }
            else //place in training set
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
    if(sample_weight_train.size()==y_train.size()) //If sample weights were provided
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
    if(sample_weight_validation.size()==y_validation.size()) //If sample weights were provided
    {
        for (size_t i = 0; i < validation_indexes.size(); ++i)
        {
            sample_weight_validation[i]=sample_weight[validation_indexes[i]];
        }
    }
}

void APLRRegressor::initialize()
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
        bool observation_is_equal_to_previous{check_if_approximately_equal(X_train.col(base_term)[i], X_train.col(base_term)[i-1])};
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

//Calculates negative gradient based on y and predictions_current
VectorXd APLRRegressor::calculate_neg_gradient_current(const VectorXd &y,const VectorXd &predictions_current)
{
    VectorXd output;
    if(family=="gaussian")
        output=y-predictions_current;
    else if(family=="binomial")
        output=y.array() / predictions_current.array() - (y.array()-1.0) / (predictions_current.array()-1.0);
    else if(family=="poisson")
        output=y.array() / predictions_current.array() - 1;
    else if(family=="gamma")
        output=(y.array() - predictions_current.array()) / predictions_current.array() / predictions_current.array();
    else if(family=="tweedie")
        output=(y.array()-predictions_current.array()).array() * predictions_current.array().pow(-tweedie_power);
    return output;
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
    find_best_split_for_each_eligible_term();
    consider_interactions();
    consider_updating_intercept();
    select_the_best_term_and_update_errors(boosting_step);
    if(abort_boosting) return;
    update_term_eligibility();
    print_summary_after_boosting_step(boosting_step);
}

void APLRRegressor::find_best_split_for_each_eligible_term()
{
    best_term=std::numeric_limits<size_t>::max();
    lowest_error_sum=neg_gradient_nullmodel_errors_sum; 
    if(n_jobs!=1 && terms_eligible_current.size()>1) //if multithreading
    {
        distributed_terms=distribute_terms_to_cores(terms_eligible_current,n_jobs);

        //defining threads
        std::vector<std::thread> threads(distributed_terms.size());
        
        //lambda function to call in each thread
        auto estimate_split_point_for_distributed_terms=[this](size_t thread_index)
        {
            for (size_t i = 0; i < distributed_terms[thread_index].size(); ++i) //for each eligible term in group do evaluation
            {
                terms_eligible_current[distributed_terms[thread_index][i]].estimate_split_point(X_train,neg_gradient_current,sample_weight_train,bins,v,min_observations_in_split);
            }
        };
        
        //run and join threads
        for (size_t i = 0; i < threads.size(); ++i) //for each thread
        {
            threads[i]=std::thread(estimate_split_point_for_distributed_terms,i);
        }
        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
        }

        //Chooses best term
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            if(terms_eligible_current[i].ineligible_boosting_steps==0) //if term is actually eligible
            {
                if(std::isless(terms_eligible_current[i].split_point_search_errors_sum,lowest_error_sum))
                {
                    best_term=i;
                    lowest_error_sum=terms_eligible_current[i].split_point_search_errors_sum;
                }                
            }
        }
    }
    else //Not multithreading
    {
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            if(terms_eligible_current[i].ineligible_boosting_steps==0) //if term is actually eligible
            {
                terms_eligible_current[i].estimate_split_point(X_train,neg_gradient_current,sample_weight_train,bins,v,min_observations_in_split);
                if(std::isless(terms_eligible_current[i].split_point_search_errors_sum,lowest_error_sum))
                {
                    best_term=i;
                    lowest_error_sum=terms_eligible_current[i].split_point_search_errors_sum;
                }
            }
        }            
    }
}

void APLRRegressor::consider_interactions()
{
    bool consider_interactions{terms.size()>0 && max_interaction_level>0 && interactions_eligible<max_interactions};
    if(consider_interactions)
    {
        determine_interactions_to_consider();
        estimate_split_points_for_interactions_to_consider();
        sort_errors_for_interactions_to_consider();
        add_promising_interactions_and_select_the_best_one();
    }
}

void APLRRegressor::determine_interactions_to_consider()
{
    interactions_to_consider=std::vector<Term>();
    interactions_to_consider.reserve(static_cast<size_t>(X_train.cols())*terms.size());

    //Determining number of terms already in the model to consider
    size_t terms_to_consider{terms.size()};
    //Getting sorted split point errors indices
    VectorXd latest_split_point_errors(terms.size());
    VectorXi sorted_latest_split_point_errors_indices(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        latest_split_point_errors[i]=terms[i].split_point_search_errors_sum;
        sorted_latest_split_point_errors_indices[i]=i;
    }
    //Restricting if max_eligible_terms>0
    if(max_eligible_terms>0)
    {
        sorted_latest_split_point_errors_indices=sort_indexes_ascending(latest_split_point_errors);
        terms_to_consider=std::min(max_eligible_terms,terms.size());
    }
    for (size_t i = 0; i < terms_to_consider; ++i) //for each term already in the model or for max_eligible_terms previous best terms measured on cut point search error
    {
        for (size_t j = 0; j < static_cast<size_t>(X_train.cols()); ++j) //for each base term
        {
            if(terms_eligible_current[j].ineligible_boosting_steps==0) //if term is actually eligible
            {
                Term interaction{Term(j)};
                if(!(terms[sorted_latest_split_point_errors_indices[i]]==interaction))
                {
                    interaction.given_terms.push_back(terms[sorted_latest_split_point_errors_indices[i]]);
                    interaction.given_terms[interaction.given_terms.size()-1].cleanup_when_this_term_was_added_as_a_given_predictor();
                    bool already_exists{false};
                    for (size_t k = 0; k < terms_eligible_current.size(); ++k)
                    {
                        if(interaction.base_term==terms_eligible_current[k].base_term && Term::equals_given_terms(interaction,terms_eligible_current[k]))
                        {
                            already_exists=true;
                            break;
                        }
                    }
                    if(already_exists) continue;
                    interaction.given_terms[interaction.given_terms.size()-1].name="P"+std::to_string(sorted_latest_split_point_errors_indices[i]);
                    if(interaction.get_interaction_level()>max_interaction_level) continue;
                    interactions_to_consider.push_back(interaction);
                }
            }
        }                    
    }
}

void APLRRegressor::estimate_split_points_for_interactions_to_consider()
{
    if(n_jobs!=1 && interactions_to_consider.size()>1) //if multithreading
    {
        //Distributing eligible terms to cores
        DistributedIndices distributed_interactions{distribute_to_indices(interactions_to_consider,n_jobs)};
        
        //defining threads
        std::vector<std::thread> threads(distributed_interactions.index_lowest.size());

        //lambda function to call in each thread
        auto evaluate_group=[this](size_t lowest,size_t highest)
        {
            for (size_t i = lowest; i <= highest; ++i) //for each eligible term in group do evaluation
            {
                interactions_to_consider[i].estimate_split_point(X_train,neg_gradient_current,sample_weight_train,bins,v,min_observations_in_split);
            }
        };

        //run and join threads
        for (size_t i = 0; i < threads.size(); ++i) //for each thread
        {
            threads[i]=std::thread(evaluate_group,distributed_interactions.index_lowest[i],distributed_interactions.index_highest[i]);
        }
        for (size_t i = 0; i < threads.size(); ++i)
        {
            threads[i].join();
        }
    }
    else //not multithreading
    {
        //Testing predictiveness of interactions to add
        for (size_t j = 0; j < interactions_to_consider.size(); ++j)
        {
            interactions_to_consider[j].estimate_split_point(X_train,neg_gradient_current,sample_weight_train,bins,v,min_observations_in_split);                                
        }
    }
}

void APLRRegressor::sort_errors_for_interactions_to_consider()
{
    VectorXd errors_for_interactions_to_consider(interactions_to_consider.size()); //Sorting interactions to add after lowest error
    for (size_t i = 0; i < interactions_to_consider.size(); ++i)
    {
        errors_for_interactions_to_consider[i]=interactions_to_consider[i].split_point_search_errors_sum;
    }
    error_index_for_interactions_to_consider=sort_indexes_ascending(errors_for_interactions_to_consider);
}

void APLRRegressor::add_promising_interactions_and_select_the_best_one()
{
    size_t best_term_before_interactions{best_term};
    for (size_t j = 0; j < static_cast<size_t>(error_index_for_interactions_to_consider.size()); ++j) //for each interaction to consider starting from lowest to highest error
    {
        bool allowed_to_add_one_interaction{interactions_eligible<max_interactions};
        if(allowed_to_add_one_interaction)
        {
            bool error_is_less_than_for_best_term_before_interactions{std::isless(interactions_to_consider[error_index_for_interactions_to_consider[j]].split_point_search_errors_sum,terms_eligible_current[best_term_before_interactions].split_point_search_errors_sum)};
            if(error_is_less_than_for_best_term_before_interactions)
            {
                add_term_to_terms_eligible_current(interactions_to_consider[error_index_for_interactions_to_consider[j]]);
                bool is_best_interaction{j==0};
                if(is_best_interaction)
                {
                    best_term=terms_eligible_current.size()-1;        
                    lowest_error_sum=terms_eligible_current[best_term].split_point_search_errors_sum;
                }
                ++interactions_eligible;
            }
            else
                break; //No point in testing anymore since the index is sorted
        }
    }
}

void APLRRegressor::consider_updating_intercept()
{
    if(sample_weight_train.size()==0)
        intercept_test=neg_gradient_current.mean();
    else
        intercept_test=(neg_gradient_current.array()*sample_weight_train.array()).sum()/sample_weight_train.array().sum();
    intercept_test=intercept_test*v;
    linear_predictor_update=VectorXd::Constant(neg_gradient_current.size(),intercept_test);
    linear_predictor_update_validation=VectorXd::Constant(y_validation.size(),intercept_test);
    error_after_updating_intercept=calculate_errors(neg_gradient_current,linear_predictor_update,sample_weight_train).sum();
}

void APLRRegressor::select_the_best_term_and_update_errors(size_t boosting_step)
{
    //If intercept does best
    if(std::islessequal(error_after_updating_intercept,lowest_error_sum)) 
    {
        //Updating intercept, current predictions, gradient and errors
        lowest_error_sum=error_after_updating_intercept;
        intercept=intercept+intercept_test;
        intercept_steps[boosting_step]=intercept;
        update_linear_predictor_and_predictors();
        update_gradient_and_errors();
    }
    else //Choosing the next term and updating the model
    {
        bool no_term_was_selected{best_term == std::numeric_limits<size_t>::max()};
        if(no_term_was_selected)
        {
            abort_boosting=true;
            return;
        }

        //Updating current predictions
        VectorXd values{terms_eligible_current[best_term].calculate(X_train)};
        VectorXd values_validation{terms_eligible_current[best_term].calculate(X_validation)};
        linear_predictor_update=values*terms_eligible_current[best_term].coefficient;
        linear_predictor_update_validation=values_validation*terms_eligible_current[best_term].coefficient;
        double error_after_updating_term=calculate_errors(neg_gradient_current,linear_predictor_update,sample_weight_train).sum();
        if(std::isgreaterequal(error_after_updating_term,neg_gradient_nullmodel_errors_sum)) //if no improvement or worse then terminate search
        {
            abort_boosting=true;
            return;
        }
        else //if improvement
        {
            //Updating predictions_current, gradient and errors
            update_linear_predictor_and_predictors();
            update_gradient_and_errors();

            //Has the term been entered into the model before?
            if(terms.size()==0) //If nothing is in the model add the term
                add_new_term(boosting_step);
            else //If at least one term was added before
            {   
                //Searching in existing terms
                bool found{false};
                for (size_t j = 0; j < terms.size(); ++j)
                {
                    if(terms[j]==terms_eligible_current[best_term]) //if term was found, update coefficient and coefficient_steps
                    {
                        terms[j].coefficient+=terms_eligible_current[best_term].coefficient;
                        terms[j].coefficient_steps[boosting_step]=terms[j].coefficient;
                        found=true;
                        break;
                    } 
                }
                //term was not in the model and is added to the model
                if(!found) 
                {
                    add_new_term(boosting_step);
                }
            }
        }
    }

    validation_error_steps[boosting_step]=calculate_error(calculate_errors(y_validation,predictions_current_validation,sample_weight_validation,family,tweedie_power),sample_weight_validation);
    bool validation_error_is_invalid{!std::isfinite(validation_error_steps[boosting_step]) || std::isnan(validation_error_steps[boosting_step])};
    if(validation_error_is_invalid)
    {
        abort_boosting=true;
        std::string warning_message{"Warning: Encountered numerical problems when calculating prediction errors in the previous boosting step. Not continuing with further boosting steps."};
        bool show_additional_warning{family=="poisson" || family=="tweedie" || family=="gamma" || (link_function!="identity" && link_function!="logit")};
        if(show_additional_warning)
            warning_message+=" A reason may be too large response values.";
        std::cout<<warning_message<<"\n";
    }
}

void APLRRegressor::update_linear_predictor_and_predictors()
{
    linear_predictor_current+=linear_predictor_update;
    linear_predictor_current_validation+=linear_predictor_update_validation;
    predictions_current=transform_linear_predictor_to_predictions(linear_predictor_current,link_function,tweedie_power);
    predictions_current_validation=transform_linear_predictor_to_predictions(linear_predictor_current_validation,link_function,tweedie_power);
}

void APLRRegressor::update_gradient_and_errors()
{
    neg_gradient_current=calculate_neg_gradient_current(y_train,predictions_current);
    neg_gradient_nullmodel_errors=calculate_errors(neg_gradient_current,linear_predictor_null_model,sample_weight_train);
    neg_gradient_nullmodel_errors_sum=neg_gradient_nullmodel_errors.sum();
}

void APLRRegressor::add_new_term(size_t boosting_step)
{
    //Setting coefficient_steps
    terms_eligible_current[best_term].coefficient_steps=VectorXd::Constant(m,0);
    terms_eligible_current[best_term].coefficient_steps[boosting_step]=terms_eligible_current[best_term].coefficient;

    //Appending a copy of the chosen term to terms
    terms.push_back(Term(terms_eligible_current[best_term]));
}

void APLRRegressor::update_term_eligibility()
{
    number_of_eligible_terms=terms_eligible_current.size();
    if(ineligible_boosting_steps_added>0 && max_eligible_terms>0) //if using eligibility
    {
        //Getting sorted cut point errors indices
        VectorXd terms_eligible_current_split_point_errors(terms_eligible_current.size());
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            terms_eligible_current_split_point_errors[i]=terms_eligible_current[i].split_point_search_errors_sum;
        }
        VectorXi sorted_split_point_errors_indices{sort_indexes_ascending(terms_eligible_current_split_point_errors)};

        //Adding ineligible_boosting_steps_added to each term that will become ineligible and reducing ineligible_boosting_steps_added for
        //the terms that are already ineligible
        size_t count{0};
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            if(terms_eligible_current[sorted_split_point_errors_indices[i]].ineligible_boosting_steps==0) //If term is eligible now
            {
                ++count;
                if(count>max_eligible_terms)
                    terms_eligible_current[sorted_split_point_errors_indices[i]].ineligible_boosting_steps+=ineligible_boosting_steps_added;
            }
            else //If term is ineligible now
            {
                terms_eligible_current[sorted_split_point_errors_indices[i]].ineligible_boosting_steps-=1;
            }
        }

        //Calculating number of eligible terms
        number_of_eligible_terms=0;
        for (size_t i = 0; i < terms_eligible_current.size(); ++i)
        {
            if(terms_eligible_current[i].ineligible_boosting_steps==0)
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
    //Filling down coefficient_steps for the intercept
    for (size_t j = 0; j < m; ++j) //For each boosting step
    {
        if(j>0 && check_if_approximately_zero(intercept_steps[j]) && !check_if_approximately_zero(intercept_steps[j-1]))
            intercept_steps[j]=intercept_steps[j-1];
    }
    //Filling down coefficient_steps for each term in the model
    for (size_t i = 0; i < terms.size(); ++i) //For each term
    {
        for (size_t j = 0; j < m; ++j) //For each boosting step
        {
            if(j>0 && check_if_approximately_zero(terms[i].coefficient_steps[j]) && !check_if_approximately_zero(terms[i].coefficient_steps[j-1]))
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
    //Choosing optimal m and updating coefficients
    size_t best_boosting_step_index;
    validation_error_steps.minCoeff(&best_boosting_step_index); //boosting step with lowest error
    intercept=intercept_steps[best_boosting_step_index];
    for (size_t i = 0; i < terms.size(); ++i) //for each term set coefficient
    {
        terms[i].coefficient=terms[i].coefficient_steps[best_boosting_step_index];
    }
    m=best_boosting_step_index+1; 

    //Removing terms with zero coefficient
    std::vector<Term> terms_new;
    terms_new.reserve(terms.size());
    for (size_t i = 0; i < terms.size(); ++i)
    {
        if(!check_if_approximately_zero(terms[i].coefficient))
            terms_new.push_back(terms[i]);
    }
    terms=std::move(terms_new);
}

void APLRRegressor::name_terms(const MatrixXd &X, const std::vector<std::string> &X_names)
{
    if(X_names.size()==0) //If nothing in X_names
    {
        std::vector<std::string> temp(X.cols());
        for (size_t i = 0; i < static_cast<size_t>(X.cols()); ++i) //for each column in X, create a default name
        {
            temp[i]="X"+std::to_string(i+1);
        }
        set_term_names(temp);
    }
    else //If names were provided in X_names
    {
        set_term_names(X_names);
    }
}

//Sets meaningful names on terms. Require that the model has been trained.
//X_names is a vector containing names for terms in X in the same order as in X. 
//These names will be used to derive names for the actually used terms in the trained model.
void APLRRegressor::set_term_names(const std::vector<std::string> &X_names)
{
    if(std::isnan(intercept)) //model has not been trained
        throw std::runtime_error("The model must be trained with fit() before term names can be set.");

    for (size_t i = 0; i < terms.size(); ++i) //for each term
    {
        //Base name
        terms[i].name=X_names[terms[i].base_term];

        //Adding cut-point and direction
        if(!std::isnan(terms[i].split_point)) //If not linear effect
        {
            double temp_split_point{terms[i].split_point}; //For prettier printing (+5.0 instead 0f --5.0 as an example when split_point is negative)
            std::string sign{"-"};
            if(std::isless(temp_split_point,0))
            {
                temp_split_point=-temp_split_point;
                sign="+";
            }
            if(terms[i].direction_right)
                terms[i].name="max("+terms[i].name+sign+std::to_string(temp_split_point)+",0)";
            else
                terms[i].name="min("+terms[i].name+sign+std::to_string(temp_split_point)+",0)";
        }

        //Adding given terms
        for (size_t j = 0; j < terms[i].given_terms.size(); ++j) //for each given term
        {
            terms[i].name+=" * I("+terms[i].given_terms[j].name+"!=0)";
        }

        //Adding interaction level
        terms[i].name="P"+std::to_string(i)+". Interaction level: "+std::to_string(terms[i].get_interaction_level())+". "+terms[i].name;
    }

    //Storing model description
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

void APLRRegressor::calculate_feature_importance_on_validation_set()
{
    feature_importance=VectorXd::Constant(number_of_base_terms,0);
    MatrixXd li{calculate_local_feature_importance(X_validation)};
    for (size_t i = 0; i < static_cast<size_t>(li.cols()); ++i) //for each column calculate mean abs values
    {
        feature_importance[i]=li.col(i).cwiseAbs().mean();
    }
}

//Computes local feature importance on data X.
//Output matrix has columns for each base term in the same order as in X and observations in rows.
MatrixXd APLRRegressor::calculate_local_feature_importance(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    //Computing local feature importance
    MatrixXd output{MatrixXd::Constant(X.rows(),number_of_base_terms,0)};
    //Terms
    for (size_t i = 0; i < terms.size(); ++i) //for each term
    {
        VectorXd contrib{terms[i].calculate_prediction_contribution(X)};
        output.col(terms[i].base_term)+=contrib;
    }

    return output;
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
    error_index_for_interactions_to_consider.resize(0);
    linear_predictor_current.resize(0);
    linear_predictor_current_validation.resize(0);
    for (size_t i = 0; i < terms.size(); ++i)
    {
        terms[i].cleanup_after_fit();
    }
}

VectorXd APLRRegressor::predict(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    VectorXd linear_predictor{calculate_linear_predictor(X)};
    VectorXd predictions{transform_linear_predictor_to_predictions(linear_predictor,link_function,tweedie_power)};

    return predictions;
}

VectorXd APLRRegressor::calculate_linear_predictor(const MatrixXd &X)
{
    VectorXd predictions{VectorXd::Constant(X.rows(),intercept)};
    for (size_t i = 0; i < terms.size(); ++i) //for each term
    {
        VectorXd contrib{terms[i].calculate_prediction_contribution(X)};
        predictions+=contrib;
    }
    return predictions;    
}

MatrixXd APLRRegressor::calculate_local_feature_importance_for_terms(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    //Computing local feature importance
    MatrixXd output{MatrixXd::Constant(X.rows(),terms.size(),0)};
    //Terms
    for (size_t i = 0; i < terms.size(); ++i) //for each term
    {
        VectorXd contrib{terms[i].calculate_prediction_contribution(X)};
        output.col(i)+=contrib;
    }

    return output;
}

MatrixXd APLRRegressor::calculate_terms(const MatrixXd &X)
{
    validate_that_model_can_be_used(X);

    //Computing local feature importance
    MatrixXd output{MatrixXd::Constant(X.rows(),terms.size(),0)};
    //Terms
    for (size_t i = 0; i < terms.size(); ++i) //for each term
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