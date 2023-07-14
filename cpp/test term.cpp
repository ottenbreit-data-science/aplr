#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "term.h"
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
    tests.push_back(is_approximately_zero(values[0]) && is_approximately_equal(values[1],-0.711234,0.00001) &&
        is_approximately_zero(values[2])?true:false);

    //Testing calculate_prediction_contribution
    VectorXd contrib{p.calculate_contribution_to_linear_predictor(X)};
    std::cout<<"Prediction contribution\n";
    std::cout<<contrib<<"\n\n";
    tests.push_back(is_approximately_equal(contrib[1],-1.42247,0.0001) && is_approximately_zero(contrib[0]) 
        && is_approximately_zero(contrib[2]) ?true:false);

    //Testing equals_base_terms
    bool t1{Term::equals_given_terms(p,p.given_terms[0])};
    bool t2{Term::equals_given_terms(p,p)};
    tests.push_back(t1 ? false:true);
    tests.push_back(t2 ? true:false);

    //Testing copy constructor
    p.ineligible_boosting_steps=10;
    Term p2{p};
    bool test_cpy=Term::equals_given_terms(p,p2) && &p.given_terms != &p2.given_terms 
    && p.coefficient==p2.coefficient && &p.coefficient!=&p2.coefficient && is_approximately_equal(p.split_point,p2.split_point) && &p.split_point!= &p2.split_point
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
    
    //Test summary
    std::cout<<"\n\nTest summary\n"<<"Passed "<<std::accumulate(tests.begin(),tests.end(),0)<<" out of "<<tests.size()<<" tests.";
}
