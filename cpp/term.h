#pragma once
#include <string>
#include <limits>
#include "../dependencies/eigen-master/Eigen/Dense"
#include "functions.h"
#include <functional>   // std::function
#include "constants.h"
#include <vector>
#include <algorithm>

using namespace Eigen;


//Output structure for term method calculate_given_terms_indices()
struct GivenTermsIndices
{
    VectorXi zeroed;
    VectorXi not_zeroed;
};

struct SortedData
{
    VectorXd values_sorted{VectorXd(0)};
    VectorXd y_sorted{VectorXd(0)};
    VectorXd sample_weight_sorted{VectorXd(0)};
};

class Term
{
private:
    //fields
    GivenTermsIndices given_terms_indices;
    size_t max_index;
    size_t max_index_discretized;
    size_t min_observations_in_split;
    bool loss_function_mse;
    size_t bins;
    double v;
    double error_where_given_terms_are_zero;
    SortedData sorted_vectors;
    VectorXd y_discretized;
    VectorXd errors_initial;
    double error_initial;

    //methods
    void calculate_error_where_given_terms_are_zero(const VectorXd &y, const VectorXd &sample_weight);
    void initialize_parameters_in_estimate_split_point(bool loss_function_mse,size_t bins,double v,size_t min_observations_in_split);
    void adjust_min_observations_in_split(size_t min_observations_in_split);
    void sort_vectors_ascending_by_base_term(const MatrixXd &X, const VectorXd &y,const VectorXd &sample_weight);
    SortedData sort_data(const VectorXd &values_to_sort, const VectorXd &y_to_sort, const VectorXd &sample_weight_to_sort);
    void setup_bins();
    void discretize_data_by_bin();
    void estimate_split_point_on_discretized_data();
    void calculate_coefficient_and_error_on_discretized_data(bool direction_right, double split_point);
    void estimate_coefficient_and_error_on_all_data();
    void cleanup_after_estimate_split_point();
    void cleanup_after_fit();
    void cleanup_when_this_term_was_added_as_a_given_predictor();

public:
    //fields
    std::string name;
    size_t base_term; //Index of underlying term in X to use
    std::vector<Term> given_terms; //Indexes of given terms already in the model. Will zero out term values for this term when at least one of the given terms have zero values.
    double split_point;
    bool direction_right;
    double coefficient;
    VectorXd coefficient_steps;
    double split_point_search_errors_sum; //lowest error from estimate_split_point 
    std::vector<size_t> bins_start_index; //Start index for values_sorted in bin
    std::vector<size_t> bins_end_index; //End index for values_sorted in bin
    std::vector<double> bins_split_points_left; //split_point values
    std::vector<double> bins_split_points_right; //split_point values
    size_t ineligible_boosting_steps;
    VectorXd values_discretized; //Discretized values based on split_point=nan
    VectorXd sample_weight_discretized; //Discretized sample_weight based on split_point=nan

    //methods
    Term(size_t base_term=0,const std::vector<Term> &given_terms=std::vector<Term>(0),double split_point=NAN_DOUBLE,bool direction_right=false,double coefficient=0);
    Term(const Term &other); //copy constructor
    ~Term();
    VectorXd calculate(const MatrixXd &X); //calculates term values
    VectorXd calculate_prediction_contribution(const MatrixXd &X);
    static bool equals_not_comparing_given_terms(const Term &p1,const Term &p2);
    static bool equals_given_terms(const Term &p1,const Term &p2);
    void estimate_split_point(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,bool loss_function_mse,size_t bins,double v,size_t min_observations_in_split);
    size_t get_interaction_level(size_t previous_int_level=0);
    VectorXd calculate_without_interactions(const VectorXd &x);
    void calculate_given_terms_indices(const MatrixXd &X);

    //friends
    friend bool operator== (const Term &p1, const Term &p2);
    friend class APLRRegressor;
};

//Regular constructor
Term::Term(size_t base_term,const std::vector<Term> &given_terms,double split_point,bool direction_right,double coefficient):
name{""},base_term{base_term},given_terms{given_terms},split_point{split_point},direction_right{direction_right},coefficient{coefficient},
split_point_search_errors_sum{std::numeric_limits<double>::infinity()},ineligible_boosting_steps{0}
{
}

//Copy constructor
Term::Term(const Term &other):
name{other.name},base_term{other.base_term},given_terms{other.given_terms},split_point{other.split_point},direction_right{other.direction_right},
coefficient{other.coefficient},coefficient_steps{other.coefficient_steps},split_point_search_errors_sum{other.split_point_search_errors_sum},ineligible_boosting_steps{0}
{
}

//Destructor
Term::~Term()
{
}

//Compare everything except given_terms
bool Term::equals_not_comparing_given_terms(const Term &p1,const Term &p2)
{
    bool split_point_and_direction{(check_if_approximately_equal(p1.split_point,p2.split_point) && p1.direction_right==p2.direction_right) || (std::isnan(p1.split_point) && std::isnan(p2.split_point))};
    bool base_term{p1.base_term==p2.base_term};
    return split_point_and_direction && base_term;
}

//Returns true if term other has the same unique elements in given_terms
bool Term::equals_given_terms(const Term &p1,const Term &p2)
{
    if(p1.given_terms.size()!=p2.given_terms.size()) return false;

    if(p1.given_terms.size()==0) return true;

    bool p1_isin_p2{false};

    //Checking if elements in p1.given_terms are in p2.given_terms. This also implies vica versa because p1.given_terms.size()=p2.given_terms.size().
    for (size_t i = 0; i < p1.given_terms.size(); ++i) //for all elements in p1
    {
        for (size_t j = 0; j < p2.given_terms.size(); ++j) //for all elements in p2
        {
            if(equals_not_comparing_given_terms(p1.given_terms[i],p2.given_terms[j]))
            {
                p1_isin_p2=equals_given_terms(p1.given_terms[i],p2.given_terms[j]);
            }
        }
        if(!p1_isin_p2) return false;
    }
    
    return true;
}

//Equals operator
bool operator== (const Term &p1, const Term &p2)
{
    bool cmp_ex_given{Term::equals_not_comparing_given_terms(p1,p2)};
    bool cmp{Term::equals_given_terms(p1,p2)};
    return cmp_ex_given && cmp;
}

//estimates best split point
void Term::estimate_split_point(const MatrixXd &X,const VectorXd &y,const VectorXd &sample_weight,bool loss_function_mse,size_t bins,double v,size_t min_observations_in_split)
{
    calculate_given_terms_indices(X);

    bool too_few_observations{static_cast<size_t>(given_terms_indices.not_zeroed.size())<min_observations_in_split};
    if(too_few_observations)
    {
        coefficient=0;
        split_point_search_errors_sum=std::numeric_limits<double>::infinity();
        return;
    }

    initialize_parameters_in_estimate_split_point(loss_function_mse, bins, v, min_observations_in_split);
    calculate_error_where_given_terms_are_zero(y, sample_weight);
    sort_vectors_ascending_by_base_term(X, y, sample_weight);    
    setup_bins();
    discretize_data_by_bin();
    estimate_split_point_on_discretized_data();
    estimate_coefficient_and_error_on_all_data();
    cleanup_after_estimate_split_point();
}

//Calculate indices that get zeroed out during calculate() because of given terms. Also calculates indices of those observations that do not.
void Term::calculate_given_terms_indices(const MatrixXd &X)
{
    if(given_terms.size()>0)
    {
        given_terms_indices.zeroed.resize(X.rows());
        given_terms_indices.not_zeroed.resize(X.rows());
        size_t count_zeroed{0};
        size_t count_not_zeroed{0};
        for (size_t j = 0; j < given_terms.size(); ++j)  //for each given term
        {
            VectorXd values_given_term{given_terms[j].calculate(X)};
            for (size_t i = 0; i < static_cast<size_t>(X.rows()); ++i) //for each row
            {
                if(check_if_approximately_zero(values_given_term[i])) //if zeroed out by given term
                {
                    given_terms_indices.zeroed[count_zeroed]=i;
                    ++count_zeroed;
                }
                else //if not zeroed out by given term
                {
                    given_terms_indices.not_zeroed[count_not_zeroed]=i;
                    ++count_not_zeroed;
                }
            }
        }   
        given_terms_indices.zeroed.conservativeResize(count_zeroed);
        given_terms_indices.not_zeroed.conservativeResize(count_not_zeroed);
    }
    else
    {
        given_terms_indices.zeroed.resize(0);
        given_terms_indices.not_zeroed.resize(X.rows());
        std::iota(given_terms_indices.not_zeroed.begin(),given_terms_indices.not_zeroed.end(),0);
    }
}

//Calculate term values
VectorXd Term::calculate(const MatrixXd &X)
{
    VectorXd values{calculate_without_interactions(X.col(base_term))};

    //Zeroing out where given terms have zero values
    if(given_terms.size()>0)
    {
        for (size_t j = 0; j < given_terms.size(); ++j)  //for each given term
        {
            VectorXd values_given_term{given_terms[j].calculate(X)};
            for (size_t i = 0; i < static_cast<size_t>(values.size()); ++i) //for each row
            {
                if(check_if_approximately_zero(values_given_term[i]))
                    values[i]=0;
            }
        }   
    }

    return values;
}

VectorXd Term::calculate_without_interactions(const VectorXd &x)
{
    VectorXd values;

    if(std::isnan(split_point)) //If split_point is nan then we have a linear effect and just use x values
        values=x;
    else //split_point is specified and we calculate values
    {
        if(direction_right)
            values=(x.array()-split_point).array().max(0);
        else
            values=(x.array()-split_point).array().min(0);
    }

    return values;
}

void Term::calculate_error_where_given_terms_are_zero(const VectorXd &y, const VectorXd &sample_weight)
{
    error_where_given_terms_are_zero=0;
    if(given_terms_indices.zeroed.size()>0)
    {
        if(sample_weight.size()==0)
        {
            for (size_t i = 0; i < static_cast<size_t>(given_terms_indices.zeroed.size()); ++i)
            {
                error_where_given_terms_are_zero+=calculate_error_one_observation(y[given_terms_indices.zeroed[i]],0.0,NAN_DOUBLE,loss_function_mse);
            }
        }
        else
        {
            for (size_t i = 0; i < static_cast<size_t>(given_terms_indices.zeroed.size()); ++i)
            {
                error_where_given_terms_are_zero+=calculate_error_one_observation(y[given_terms_indices.zeroed[i]],0.0,sample_weight[given_terms_indices.zeroed[i]],loss_function_mse);
            }
        }
    }
}

void Term::initialize_parameters_in_estimate_split_point(bool loss_function_mse,size_t bins,double v,size_t min_observations_in_split)
{
    this->loss_function_mse=loss_function_mse;
    this->bins=bins;
    this->v=v;
    adjust_min_observations_in_split(min_observations_in_split);
    max_index=calculate_max_index_in_vector(given_terms_indices.not_zeroed);
}

//Min observations in split, adjusting if necessary to avoid computational errors
void Term::adjust_min_observations_in_split(size_t min_observations_in_split)
{ 
    this->min_observations_in_split=std::min(min_observations_in_split,static_cast<size_t>(given_terms_indices.not_zeroed.size()));
    this->min_observations_in_split=std::max(min_observations_in_split,static_cast<size_t>(1));
}

void Term::sort_vectors_ascending_by_base_term(const MatrixXd &X, const VectorXd &y,const VectorXd &sample_weight)
{
    if(given_terms_indices.zeroed.size()>0)
    {
        //Calculating subset of values, y and sample_weight
        VectorXd values_subset(given_terms_indices.not_zeroed.size());
        VectorXd y_subset(given_terms_indices.not_zeroed.size());
        size_t count{0};
        for (size_t i = 0; i <= max_index; ++i) //for each non-zeroed observation
        {
            values_subset[count]=X.col(base_term)[given_terms_indices.not_zeroed[i]];
            y_subset[count]=y[given_terms_indices.not_zeroed[i]];
            ++count;
        }
        VectorXd sample_weight_subset(0);
        if(sample_weight.size()>0)
        {
            count=0;
            sample_weight_subset.resize(given_terms_indices.not_zeroed.size());
            for (size_t i = 0; i <= max_index; ++i) //for each non-zeroed observation
            {
                sample_weight_subset[count]=sample_weight[given_terms_indices.not_zeroed[i]];
                ++count;
            }
        }
        sorted_vectors = sort_data(values_subset, y_subset, sample_weight_subset);
    }
    else
        sorted_vectors = sort_data(X.col(base_term), y, sample_weight);
}

SortedData Term::sort_data(const VectorXd &values_to_sort, const VectorXd &y_to_sort, const VectorXd &sample_weight_to_sort)
{
    VectorXi values_sorted_index{sort_indexes_ascending(values_to_sort)};
    SortedData output;
    output.values_sorted.resize(values_sorted_index.size());
    output.y_sorted.resize(values_sorted_index.size());
    size_t max_index{values_sorted_index.size()-static_cast<size_t>(1)};
    for (size_t i = 0; i <= max_index; ++i)
    {
        output.values_sorted[i]=values_to_sort[values_sorted_index[i]];
        output.y_sorted[i]=y_to_sort[values_sorted_index[i]];
    }    
    if(sample_weight_to_sort.size()>0)
    {
        output.sample_weight_sorted.resize(values_sorted_index.size());
        for (size_t i = 0; i <= max_index; ++i)
        {
            output.sample_weight_sorted[i]=sample_weight_to_sort[values_sorted_index[i]];
        }
    }  

    return output;
}

void Term::setup_bins()
{
    if(bins_start_index.size()==0) //if not previously calculated or wrongly sized
    {
        //bins
        size_t observations_in_bin{std::max((max_index+1)/bins,static_cast<size_t>(1))};

        //Finding unique values in values_sorted - these will be used to skip i%observations_in_bin==0 criteria for general eligibility
        std::vector<double> values_sorted_unique(sorted_vectors.values_sorted.size());
        for (size_t i = 0; i <= max_index; ++i)
        {
            values_sorted_unique[i]=sorted_vectors.values_sorted[i];
        }
        auto ip{std::unique(values_sorted_unique.begin(),values_sorted_unique.end())};
        values_sorted_unique.resize(std::distance(values_sorted_unique.begin(),ip));

        bins_start_index.reserve(bins+1);
        bins_end_index.reserve(bins+1);
        //Start_index
        bins_start_index.push_back(0);
        if(bins>1)
        {
            for (size_t i = min_observations_in_split-1; i <= max_index+1-min_observations_in_split; ++i) //for each observation in an allowable range
            {
                size_t last_bin_start_index{bins_start_index[bins_start_index.size()-1]};
                bool eligible_on_spacing_between_observations{i >= last_bin_start_index + observations_in_bin || values_sorted_unique.size()<=bins};
                bool eligible_on_unique_numbers{i>0 && !check_if_approximately_equal(sorted_vectors.values_sorted[i],sorted_vectors.values_sorted[i-1])};

                bool create_bin{eligible_on_spacing_between_observations && eligible_on_unique_numbers};
                if(create_bin) 
                {
                    bins_start_index.push_back(i);
                }
            }
        }
        //End index
        if(bins_start_index.size()>0)
        {
            for (size_t i = 1; i < bins_start_index.size(); ++i)
            {
                bins_end_index.push_back(bins_start_index[i]-1);
            }
            bins_end_index.push_back(max_index);
        }   
        //Shrinking to save memory
        bins_start_index.shrink_to_fit();
        bins_end_index.shrink_to_fit();

        //split_point values
        bins_split_points_left.reserve(bins_start_index.size());
        bins_split_points_right.reserve(bins_start_index.size());
        for (size_t i = 0; i < bins_start_index.size(); ++i) //for each bin
        {
            if(bins_start_index[i]>0 && bins_start_index[i]<max_index)
            {
                bins_split_points_left.push_back(sorted_vectors.values_sorted[bins_start_index[i]]);
            } 
            if(bins_end_index[i]>0 && bins_end_index[i]<max_index)
            {
                bins_split_points_right.push_back(sorted_vectors.values_sorted[bins_end_index[i]]);
            }
        }
        bins_split_points_left.shrink_to_fit();
        bins_split_points_right.shrink_to_fit();
    }
}

void Term::discretize_data_by_bin()
{
    if(values_discretized.size()==0) //Calculate values_discretized and sample_weight_discretized if it has not been done before
    {
        values_discretized.resize(bins_start_index.size());
        for (size_t i = 0; i < bins_start_index.size(); ++i)
        {
            values_discretized[i]=sorted_vectors.values_sorted.block(bins_start_index[i],0,bins_end_index[i]-bins_start_index[i]+1,1).mean();
        }
        
        if(sorted_vectors.sample_weight_sorted.size()>0)
        {
            sample_weight_discretized.resize(bins_start_index.size());
            for (size_t i = 0; i < bins_start_index.size(); ++i)
            {
                sample_weight_discretized[i]=sorted_vectors.sample_weight_sorted.block(bins_start_index[i],0,bins_end_index[i]-bins_start_index[i]+1,1).mean();
            }
        }
    }
    y_discretized.resize(bins_start_index.size());
    for (size_t i = 0; i < bins_start_index.size(); ++i)
    {
        y_discretized[i]=sorted_vectors.y_sorted.block(bins_start_index[i],0,bins_end_index[i]-bins_start_index[i]+1,1).mean();
    }

    max_index_discretized=calculate_max_index_in_vector(values_discretized);
}

void Term::estimate_split_point_on_discretized_data()
{
    errors_initial=calculate_errors(y_discretized,VectorXd::Constant(y_discretized.size(),0.0),sample_weight_discretized,loss_function_mse);
    error_initial=errors_initial.sum();

    //FINDING BEST SPLIT ON DISCRETIZED DATA
    double split_point_temp;
    
    //Split-point nan
    calculate_coefficient_and_error_on_discretized_data(false,NAN_DOUBLE);
    double error_cp_nan{split_point_search_errors_sum};

    //Direction left: For each observation update coefficient and calculate prediction error
    double split_point_left{NAN_DOUBLE};
    double error_min_left{error_cp_nan};
    for (size_t i = 0; i < bins_split_points_left.size(); ++i) //for each bin
    {
        split_point_temp=bins_split_points_left[i];
     
        calculate_coefficient_and_error_on_discretized_data(false,split_point_temp);
        if(std::islessequal(split_point_search_errors_sum,error_min_left)) //if equal or improvement (preferring linear relationships)
        {
            error_min_left=split_point_search_errors_sum;
            split_point_left=split_point;
        }
    }

    //Direction right: For each observation update coefficient and calculate prediction error
    double split_point_right{NAN_DOUBLE};
    double error_min_right{error_cp_nan};
    for (size_t i = 0; i < bins_split_points_right.size(); ++i) //for each bin
    {
        split_point_temp=bins_split_points_right[i];
     
        calculate_coefficient_and_error_on_discretized_data(true,split_point_temp);
        if(std::islessequal(split_point_search_errors_sum,error_min_right)) //if equal or improvement (preferring linear relationships)
        {
            error_min_right=split_point_search_errors_sum;
            split_point_right=split_point;
        }
    }

    //Comparing left and right and populating direction as well as split_point
    if(std::islessequal(error_min_left,error_min_right)) //using direction left
    {
        direction_right=false;
        split_point=split_point_left;
        split_point_search_errors_sum=error_min_left;
    }
    else //using direction right
    {
        direction_right=true;
        split_point=split_point_right;
        split_point_search_errors_sum=error_min_right;
    }
}

void Term::calculate_coefficient_and_error_on_discretized_data(bool direction_right, double split_point)
{
    this->direction_right=direction_right;
    this->split_point=split_point;
    
    //Calculating values and values_sorted
    VectorXd values_sorted{calculate_without_interactions(values_discretized)};

    //Start and end indexes
    size_t index_start{0};
    size_t index_end{max_index_discretized};

    //Calculating coefficient and errors
    double xwx{0};
    double xwy{0};
    for (size_t i = index_start; i <= index_end; ++i)
    {
        if(sample_weight_discretized.size()>0)
        {
            xwx+=values_sorted[i]*values_sorted[i]*sample_weight_discretized[i];
            xwy+=values_sorted[i]*y_discretized[i]*sample_weight_discretized[i];
        }
        else
        {
            xwx+=values_sorted[i]*values_sorted[i];
            xwy+=values_sorted[i]*y_discretized[i];
        }
    }
    if(xwx!=0)
    {
        double error_reduction{0};
        coefficient=xwy/xwx*v;
        double predicted;
        double sample_weight_one_obs{NAN_DOUBLE};
        for (size_t i = index_start; i <= index_end; ++i)
        {
            predicted=values_sorted[i]*coefficient;
            if(sample_weight_discretized.size()>0)
                sample_weight_one_obs=sample_weight_discretized[i];

            error_reduction+=errors_initial[i]-calculate_error_one_observation(y_discretized[i],predicted,sample_weight_one_obs,loss_function_mse);
        }
        split_point_search_errors_sum=error_initial-error_reduction;
    }
    else
    {
        coefficient=0;
        split_point_search_errors_sum=error_initial;
    }
}

void Term::estimate_coefficient_and_error_on_all_data()
{
    sorted_vectors.values_sorted=calculate_without_interactions(sorted_vectors.values_sorted);
    double xwx{0};
    double xwy{0};
    if(sorted_vectors.sample_weight_sorted.size()>0)
    {
        xwx=(sorted_vectors.values_sorted.array()*sorted_vectors.values_sorted.array()*sorted_vectors.sample_weight_sorted.array()).sum();
        xwy=(sorted_vectors.values_sorted.array()*sorted_vectors.y_sorted.array()*sorted_vectors.sample_weight_sorted.array()).sum();
    }
    else
    {
        xwx=(sorted_vectors.values_sorted.array()*sorted_vectors.values_sorted.array()).sum();
        xwy=(sorted_vectors.values_sorted.array()*sorted_vectors.y_sorted.array()).sum();
    }
    if(xwx!=0)
    {
        coefficient=xwy/xwx*v;
        VectorXd predictions{sorted_vectors.values_sorted*coefficient};
        split_point_search_errors_sum=calculate_errors(sorted_vectors.y_sorted,predictions,sorted_vectors.sample_weight_sorted,loss_function_mse).sum()+error_where_given_terms_are_zero;
    }
    else
    {
        coefficient=0;
        split_point_search_errors_sum=std::numeric_limits<double>::infinity();
    }
}

void Term::cleanup_after_estimate_split_point()
{
    given_terms_indices.not_zeroed.resize(0);
    given_terms_indices.zeroed.resize(0);
    sorted_vectors.values_sorted.resize(0);
    sorted_vectors.y_sorted.resize(0);
    sorted_vectors.sample_weight_sorted.resize(0);
    y_discretized.resize(0);
    errors_initial.resize(0);
}

void Term::cleanup_after_fit()
{
    bins_start_index.clear();
    bins_end_index.clear();
    bins_split_points_left.clear();
    bins_split_points_right.clear();
    values_discretized.resize(0);
    sample_weight_discretized.resize(0);
}

void Term::cleanup_when_this_term_was_added_as_a_given_predictor()
{
    cleanup_after_fit();
    coefficient_steps.resize(0);
}

VectorXd Term::calculate_prediction_contribution(const MatrixXd &X)
{
    VectorXd values{calculate(X)};

    return values.array()*coefficient;
}

//Returns the interaction level of this term
size_t Term::get_interaction_level(size_t previous_int_level)
{   
    if(given_terms.size()==0) //stopping criteria
        return previous_int_level;

    //Finding maximum interaction level in given terms
    size_t max_int_level_gp{0}; 
    for (size_t i = 0; i < given_terms.size(); ++i) //for each given term
    {
        size_t int_level_given_term{given_terms[i].get_interaction_level()};
        if(int_level_given_term>max_int_level_gp)
            max_int_level_gp=int_level_given_term;
    }   

    return previous_int_level+1+max_int_level_gp;
}


//Distribution of terms to multiple cores
std::vector<std::vector<size_t>> distribute_terms_to_cores(std::vector<Term> &terms,size_t n_jobs)
{
    //Determing number of terms actually eligible
    size_t num_eligible_terms{0};
    for (size_t i = 0; i < terms.size(); ++i)
    {
        if(terms[i].ineligible_boosting_steps==0)
            ++num_eligible_terms;
    }
    
    //Determining how many items to evaluate per core
    size_t available_cores{static_cast<size_t>(std::thread::hardware_concurrency())};
    if(n_jobs>1)
        available_cores=std::min(n_jobs,available_cores);
    size_t units_per_core{std::max(num_eligible_terms/available_cores,static_cast<size_t>(1))};

    //Initializing output
    std::vector<std::vector<size_t>> output(available_cores);
    for (size_t i = 0; i < available_cores; ++i) //for each available core
    {
        output[i]=std::vector<size_t>();
        output[i].reserve(num_eligible_terms);
    }    

    //Distributing
    size_t core{0};
    size_t count{0};
    for (size_t i = 0; i < terms.size(); ++i) //for each term
    {
        if(terms[i].ineligible_boosting_steps==0) //if can be distributed to cores
        {
            output[core].push_back(i);
            ++count;
        }
        if(count>=units_per_core)
        {
            if(core<available_cores-1)
                ++core;
            else
                core=0;
            count=0;
        }
    }

    //Freeing memory
    for (size_t i = 0; i < available_cores; ++i) //for each available core
    {
        output[i].shrink_to_fit();
    }

    return output;
}
