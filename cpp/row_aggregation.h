#pragma once

#include <map>
#include <vector>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"

using namespace Eigen;

namespace aplr_internal
{

struct AggregatedRows
{
    MatrixXd X;
    VectorXd y;
    VectorXd sample_weight;
    VectorXd observation_count;
    VectorXi row_to_group;
};

inline AggregatedRows aggregate_identical_rows(
    const MatrixXd &X, const VectorXd &y, const VectorXd &sample_weight)
{
    std::map<std::vector<double>, size_t> group_for_row;
    std::vector<Eigen::Index> first_row_indexes;
    std::vector<double> weight_sums;
    std::vector<double> weighted_target_sums;
    std::vector<double> target_sums;
    std::vector<double> observation_counts;

    first_row_indexes.reserve(static_cast<size_t>(X.rows()));
    weight_sums.reserve(static_cast<size_t>(X.rows()));
    weighted_target_sums.reserve(static_cast<size_t>(X.rows()));
    target_sums.reserve(static_cast<size_t>(X.rows()));
    observation_counts.reserve(static_cast<size_t>(X.rows()));

    VectorXi row_to_group(X.rows());
    for (Eigen::Index row = 0; row < X.rows(); ++row)
    {
        std::vector<double> key(static_cast<size_t>(X.cols()));
        for (Eigen::Index column = 0; column < X.cols(); ++column)
            key[static_cast<size_t>(column)] = X(row, column);
        auto insertion = group_for_row.emplace(key, group_for_row.size());
        size_t group_index;
        if (insertion.second)
        {
            group_index = first_row_indexes.size();
            insertion.first->second = group_index;
            first_row_indexes.push_back(row);
            weight_sums.push_back(0.0);
            weighted_target_sums.push_back(0.0);
            target_sums.push_back(0.0);
            observation_counts.push_back(0.0);
        }
        else
            group_index = insertion.first->second;

        row_to_group[row] = static_cast<int>(group_index);
        weight_sums[group_index] += sample_weight[row];
        weighted_target_sums[group_index] += sample_weight[row] * y[row];
        target_sums[group_index] += y[row];
        observation_counts[group_index] += 1.0;
    }

    AggregatedRows output;
    output.row_to_group = std::move(row_to_group);
    Eigen::Index number_of_groups{
        static_cast<Eigen::Index>(first_row_indexes.size())};
    output.X.resize(number_of_groups, X.cols());
    output.y.resize(number_of_groups);
    output.sample_weight.resize(number_of_groups);
    output.observation_count.resize(number_of_groups);

    for (Eigen::Index group_index = 0; group_index < number_of_groups;
         ++group_index)
    {
        output.X.row(group_index) = X.row(first_row_indexes[group_index]);
        double weight_sum{weight_sums[group_index]};
        output.sample_weight[group_index] = weight_sum;
        output.observation_count[group_index] =
            observation_counts[group_index];
        if (weight_sum > 0.0)
            output.y[group_index] =
                weighted_target_sums[group_index] / weight_sum;
        else
            output.y[group_index] = target_sums[group_index] /
                                    observation_counts[group_index];
    }

    return output;
}

} // namespace aplr_internal
