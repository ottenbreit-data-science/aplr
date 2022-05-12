#include <iostream>
#include "term.h"
#include "../dependencies/eigen-master/Eigen/Dense"
#include <vector>
#include <numeric>
#include "functions.h"

using namespace Eigen;

int main()
{
    std::vector<bool> tests;
    tests.reserve(1000);

    //Setting up term instance p with default values
    Term p{Term()};
    tests.push_back(p.base_term==0 ? true:false);

    //Testing calulate values
    p.split_point=0.5;
    p.direction_right=false;
    p.coefficient=2;
    p.given_terms.push_back(Term(1,std::vector<Term>(0),-5.0,true,-3.0));
    p.given_terms.push_back(Term(2,std::vector<Term>(0),5.0,false,3.0));
    p.given_terms[0].given_terms.push_back(Term(0,std::vector<Term>(0),-0.21,false,2.0));
    MatrixXd X{MatrixXd::Random(3,4)}; //terms
    VectorXd values{p.calculate(X)};
    std::cout<<"X\n";
    std::cout<<X<<"\n\n";
    std::cout<<"values\n";
    std::cout<<values<<"\n\n";
    tests.push_back(check_if_approximately_zero(values[0]) && check_if_approximately_equal(values[1],-0.711234,0.00001) &&
        check_if_approximately_zero(values[2])?true:false);

    //Testing calculate_prediction_contribution
    VectorXd contrib{p.calculate_prediction_contribution(X)};
    std::cout<<"Prediction contribution\n";
    std::cout<<contrib<<"\n\n";
    tests.push_back(check_if_approximately_equal(contrib[1],-1.42247,0.0001) && check_if_approximately_zero(contrib[0]) 
        && check_if_approximately_zero(contrib[2]) ?true:false);

    //Testing equals_base_terms
    bool t1{Term::equals_given_terms(p,p.given_terms[0])};
    bool t2{Term::equals_given_terms(p,p)};
    tests.push_back(t1 ? false:true);
    tests.push_back(t2 ? true:false);

    //Testing copy constructor
    p.ineligible_boosting_steps=10;
    Term p2{p};
    bool test_cpy=Term::equals_given_terms(p,p2) && &p.given_terms != &p2.given_terms 
    && p.coefficient==p2.coefficient && &p.coefficient!=&p2.coefficient && check_if_approximately_equal(p.split_point,p2.split_point) && &p.split_point!= &p2.split_point
    && p.direction_right==p2.direction_right && &p.direction_right!=&p2.direction_right && p.name==p2.name && &p.name!=&p2.name &&
    p.coefficient_steps.size()==p2.coefficient_steps.size() && &p.coefficient_steps != &p2.coefficient_steps && ((p.coefficient_steps-p2.coefficient_steps).array().abs()==0).all()
    && p.base_term==p2.base_term && p.ineligible_boosting_steps==10 && p2.ineligible_boosting_steps==0;
    tests.push_back(test_cpy);

    //Testing equals operator
    p2.coefficient=35.2; //p2 should be equal - coefficient is not compared
    Term p3{p}; //to be unequal
    Term p4{p}; //to be unequal
    Term p5{p}; //to be unequal
    Term p6{p}; //to be unequal
    p3.split_point=0.1;
    p4.direction_right=true;
    p5.given_terms.push_back(p3);
    p6.split_point=0.2;
    p6.direction_right=false;
    p6.given_terms.push_back(p4);
    bool test_equals1=(p==p2 && p2==p ? true:false);
    bool test_equals2=(p==p3 && p3==p ? false:true);
    bool test_equals3=(p==p4 && p4==p ? false:true);
    bool test_equals4=(p==p5 && p5==p ? false:true);
    bool test_equals5=(p==p6 && p6==p ? false:true);
    p4.split_point=NAN_DOUBLE;
    p2.split_point=NAN_DOUBLE;
    bool test_equals6=(p2==p4 && p4==p2 ? true:false);
    tests.push_back(test_equals1);
    tests.push_back(test_equals2);
    tests.push_back(test_equals3);
    tests.push_back(test_equals4);
    tests.push_back(test_equals5);
    tests.push_back(test_equals6);

    //Testing interaction_level method
    Term p7{Term(1)};
    p7.given_terms.push_back(Term(2));
    Term p8{Term(3)};
    size_t pil{p.get_interaction_level()};
    size_t p5il{p5.get_interaction_level()};
    size_t p7il{p7.get_interaction_level()};
    size_t p8il{p8.get_interaction_level()};
    tests.push_back(pil==2 ? true:false);
    tests.push_back(p5il==3 ? true:false);
    tests.push_back(p7il==1 ? true:false);
    tests.push_back(p8il==0 ? true:false);
    
    //TESTING ESTIMATION OF split_point
/*    
    //Case when X=1 and Y varies, then prediction errors sum should be like for null model
    X=VectorXd::Constant(10,0);
    VectorXd y{VectorXd::Random(10)};
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    term pcp(0);
    VectorXd errors{calculate_errors(y,VectorXd::Constant(10,0))};
    //pcp.estimate_split_point(X,y,VectorXd(0),true,errors,errors.sum(),100,1,0.1);
    //std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(errors.sum(),pcp.split_point_search_errors_sum)?true:false));

    //Case when X=1 and Y varies with sample_weight=[1..], then prediction errors sum should be like for null model
    VectorXd sample_weight=VectorXd::Constant(10,1);
    std::cout<<"sample_weight\n"<<sample_weight<<"\n\n";
    pcp=term(number_of_base_terms);
    double null_prediction=0;
    VectorXd null_predictions=VectorXd::Constant(10,null_prediction);
    VectorXd errors2=calculate_errors(y,null_predictions,sample_weight);    
    pcp.estimate_split_point(X,y,sample_weight,true,errors2,errors2.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(errors.sum(),pcp.split_point_search_errors_sum)?true:false));
    
    //Case when X=1 and Y varies with sample_weight!=1, then prediction errors sum should be like for null model
    sample_weight=VectorXd::Random(10).cwiseAbs();
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"sample_weight\n"<<sample_weight<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight);    
    pcp.estimate_split_point(X,y,sample_weight,true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(errors.sum(),pcp.split_point_search_errors_sum)?true:false));

    //Case when X=1 and Y varies with sample_weight!=1 and MAE, then prediction errors sum should be like for null model
    sample_weight=VectorXd::Random(10).cwiseAbs();
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"sample_weight\n"<<sample_weight<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,false);    
    pcp.estimate_split_point(X,y,sample_weight,false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(errors.sum(),pcp.split_point_search_errors_sum)?true:false));

    //Case when X varies and Y=0, then prediction errors sum should be 0 and coefficient 0
    X=VectorXd::Random(10);
    y=VectorXd::Constant(10,0);
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    errors=calculate_errors(y,VectorXd::Constant(10,0));
    pcp.estimate_split_point(X,y,VectorXd(0),true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum) && isApproximatelyZero(pcp.coefficient)?true:false));

    //Case when X varies and Y=0 and specified sample_weight, then prediction errors sum should be 0 and coefficient 0
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,false);    
    pcp.estimate_split_point(X,y,sample_weight,false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum) && isApproximatelyZero(pcp.coefficient)?true:false));

    //Case when X varies and Y=0 and specified sample_weight and MAE, then prediction errors sum should be 0 and coefficient 0
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,true);    
    pcp.estimate_split_point(X,y,sample_weight,true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum) && isApproximatelyZero(pcp.coefficient)?true:false));

   //Case with perfect linear dependence. split_point_search_error should be 0 and coefficient 1
    X=VectorXd::Random(10);
    y=X;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(0);
    errors=calculate_errors(y,VectorXd::Constant(10,0));
    //pcp.estimate_split_point(X,y,VectorXd(0),true,errors,errors.sum(),100,1,1.0);
    //std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum,0.00001) && isApproximatelyEqual(pcp.coefficient,1.0) && std::isnan(pcp.split_point)?true:false));

   //Case with perfect linear dependence and MAE. split_point_search_error should be 0 and coefficient 1
    pcp=term(number_of_base_terms);
    errors=calculate_errors(y,VectorXd::Constant(10,0),VectorXd(0),false);
    pcp.estimate_split_point(X,y,VectorXd(0),false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum,0.00001) && isApproximatelyEqual(pcp.coefficient,1.0) && std::isnan(pcp.split_point)?true:false));

   //Case with perfect linear dependence and specified sample_weight. split_point_search_error should be 0, coefficient 1 and split_point nan
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,false);    
    pcp.estimate_split_point(X,y,sample_weight,false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum,0.00001) && isApproximatelyEqual(pcp.coefficient,1.0) && std::isnan(pcp.split_point)?true:false));

   //Case with perfect linear dependence, specified sample_weight and MAE. split_point_search_error should be 0, coefficient 1 and split_point nan
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,true);    
    pcp.estimate_split_point(X,y,sample_weight,true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyZero(pcp.split_point_search_errors_sum,0.00001) && isApproximatelyEqual(pcp.coefficient,1.0) && std::isnan(pcp.split_point)?true:false));

    //Case with linear dependence and some noise. Results from sample_weight=1 should be the same as result without specified sample_weight
    y=X.col(0)*3+VectorXd::Random(10,1);
    pcp=term(number_of_base_terms);
    //Not specified sample_weight
    errors=calculate_errors(y,VectorXd::Constant(10,0));
    pcp.estimate_split_point(X,y,VectorXd(0),true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    //Specified sample_weight
    VectorXd sample_weight_const=VectorXd::Constant(10,1);
    term pcp2=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight_const,true);    
    pcp2.estimate_split_point(X,y,sample_weight_const,true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp2.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp2.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp2.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp2.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp2.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.coefficient,pcp2.coefficient) && isApproximatelyEqual(pcp.split_point_search_errors_sum,pcp2.split_point_search_errors_sum)?true:false));

    //Case with linear dependence and some noise and MAE. Results from sample_weight=1 should be the same as result without specified sample_weight
    y=X.col(0)*3+VectorXd::Random(10,1);
    pcp=term(number_of_base_terms);
    //Not specified sample_weight
    errors=calculate_errors(y,VectorXd::Constant(10,0),VectorXd(0),false);
    pcp.estimate_split_point(X,y,VectorXd(0),false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    //Specified sample_weight
    pcp2=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight_const,false);    
    pcp2.estimate_split_point(X,y,sample_weight_const,false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp2.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp2.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp2.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp2.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp2.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.coefficient,pcp2.coefficient) && isApproximatelyEqual(pcp.split_point_search_errors_sum,pcp2.split_point_search_errors_sum)?true:false));

    //Case with perfect linear dependence except two outliers. Cut point should cut off those two. split_point=-5.2 and coefficient=1
    X=VectorXd(10);
    X<<-2.2,-0.2,-4.2,-6.2,2.8,1.8,-5.2,0.8,-3.2,-1.2;
    y=X;
    y(3)=2.8;
    y(6)=2.8;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    errors=calculate_errors(y,VectorXd::Constant(10,0));
    pcp.estimate_split_point(X,y,VectorXd(0),true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.coefficient,1.0) && isApproximatelyEqual(pcp.split_point,-5.2)?true:false));

    //Case with perfect linear dependence except two outliers. MAE. Cut point should cut off those two. split_point=2 and coefficient=1
    X=VectorXd(10);
    X<<-2.2,-0.2,-4.2,-6.2,2.8,1.8,-5.2,0.8,-3.2,-1.2;
    y=X;
    y(3)=2.8;
    y(6)=2.8;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    errors=calculate_errors(y,VectorXd::Constant(10,0),VectorXd(0));
    pcp.estimate_split_point(X,y,VectorXd(0),false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.coefficient,1.0) && isApproximatelyEqual(pcp.split_point,-5.2)?true:false));

    //Case with perfect linear dependence except two outliers and specified sample_weight that places zero weights to these outliers. 
    //split_point=nan and coefficient=1 and split_point_search_error=0 
    X=VectorXd(10);
    X<<-2.2,-0.2,-4.2,-6.2,2.8,1.8,-5.2,0.8,-3.2,-1.2;
    y=X;
    y(3)=2.8;
    y(6)=2.8;
    sample_weight=VectorXd::Constant(y.size(),1);
    sample_weight[3]=0;
    sample_weight[6]=0;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,true);    
    pcp.estimate_split_point(X,y,sample_weight,true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.coefficient,1.0) && std::isnan(pcp.split_point) && isApproximatelyZero(pcp.split_point_search_errors_sum,0.00001)?true:false));

    //Case with perfect linear dependence except two outliers and specified sample_weight and MAE that places zero weights to these outliers. 
    //coefficient=1 and split_point_search_error=0 (split_point does not really need to be nan to get the expected result)
    X=VectorXd(10);
    X<<-2.2,-0.2,-4.2,-6.2,2.8,1.8,-5.2,0.8,-3.2,-1.2;
    y=X;
    y(3)=2.8;
    y(6)=2.8;
    sample_weight[3]=0;
    sample_weight[6]=0;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    null_prediction=0;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,false);    
    pcp.estimate_split_point(X,y,sample_weight,false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.coefficient,1.0) && isApproximatelyZero(pcp.split_point_search_errors_sum,0.00001)?true:false));

    //Case with dummy variable that perfectly explains the response. Half of the error should be explained
    X=VectorXd(10);
    X<<0,1,0,0,1,1,0,1,1,0;
    y=X.array()*5+5;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    errors=calculate_errors(y,VectorXd::Constant(10,y.mean()));
    pcp.estimate_split_point(X,y,VectorXd(0),true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.split_point_search_errors_sum*2,errors.sum())?true:false));

    //Case with dummy variable that perfectly explains the response. MAE. Half of the error should be explained
    X=VectorXd(10);
    X<<0,1,0,0,1,1,0,1,1,0;
    y=X.array()*5+5;
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"y_mean\n"<<y.mean()<<"\n\n";
    pcp=term(number_of_base_terms);
    errors=calculate_errors(y,VectorXd::Constant(10,y.mean()),VectorXd(0),false);
    pcp.estimate_split_point(X,y,VectorXd(0),false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.split_point_search_errors_sum*2,errors.sum())?true:false));

   //Case with dummy variable that perfectly explains the response. Constant sample weight. Half of the error should be explained
    X=VectorXd(10);
    X<<0,1,0,0,1,1,0,1,1,0;
    y=X.array()*5+5;
    sample_weight=VectorXd::Constant(10,0.5);
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"sample_weight\n"<<sample_weight<<"\n\n";
    pcp=term(number_of_base_terms);
    null_prediction=(y.array()*sample_weight.array()).sum()/sample_weight.sum();;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,true);    
    pcp.estimate_split_point(X,y,sample_weight,true,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.split_point_search_errors_sum*2,errors.sum())?true:false));

   //Case with dummy variable that perfectly explains the response. Constant sample weight. MAE. Half of the error should be explained
    X=VectorXd(10);
    X<<0,1,0,0,1,1,0,1,1,0;
    y=X.array()*5+5;
    sample_weight=VectorXd::Constant(10,0.5);
    std::cout<<"\n\nX\n"<<X<<"\n\n";
    std::cout<<"y\n"<<y<<"\n\n";
    std::cout<<"sample_weight\n"<<sample_weight<<"\n\n";
    pcp=term(number_of_base_terms);
    null_prediction=(y.array()*sample_weight.array()).sum()/sample_weight.sum();;
    null_predictions=VectorXd::Constant(10,null_prediction);
    errors=calculate_errors(y,null_predictions,sample_weight,false);    
    pcp.estimate_split_point(X,y,sample_weight,false,errors,errors.sum(),100,1);
    std::cout<<"X_calculated\n"<<pcp.values<<"\n\n";
    std::cout<<"coefficient:"<<pcp.coefficient<<"\n";
    std::cout<<"split_point:"<<pcp.split_point<<"\n";
    std::cout<<"direction_right:"<<pcp.direction_right<<"\n";
    std::cout<<"split_point_search_error:"<<pcp.split_point_search_errors_sum<<"\n";
    std::cout<<"errors.sum():"<<errors.sum()<<"\n";
    tests.push_back((isApproximatelyEqual(pcp.split_point_search_errors_sum*2,errors.sum())?true:false));

*/
    //Test summary
    std::cout<<"\n\nTest summary\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}
