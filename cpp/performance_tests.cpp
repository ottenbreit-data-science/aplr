#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "functions.h"
#include "prefix_split_search.h"
#include "row_aggregation.h"
#include "term.h"

using Eigen::VectorXd;

namespace
{

aplr_internal::HingeScore exhaustive_score(
    const VectorXd &x,
    const VectorXd &y,
    const VectorXd &weight,
    double split,
    size_t begin,
    size_t end,
    double learning_rate,
    double penalty_factor,
    double ridge_penalty,
    double ridge_penalty_weight,
    int monotonic_constraint)
{
    double numerator{0.0};
    double denominator{0.0};
    for (size_t i = begin; i < end; ++i)
    {
        double basis{std::isnan(split) ? x[static_cast<Eigen::Index>(i)] : x[static_cast<Eigen::Index>(i)] - split};
        numerator += basis * y[static_cast<Eigen::Index>(i)] * weight[static_cast<Eigen::Index>(i)];
        denominator += basis * basis * weight[static_cast<Eigen::Index>(i)];
    }
    if (ridge_penalty > 0.0)
        denominator += ridge_penalty * ridge_penalty_weight;
    double coefficient{learning_rate * penalty_factor * numerator / denominator};
    aplr_internal::HingeScore output;
    if (!std::isfinite(coefficient) ||
        (monotonic_constraint > 0 && coefficient < 0.0) ||
        (monotonic_constraint < 0 && coefficient > 0.0))
        return output;
    VectorXd predictions{VectorXd::Zero(x.size())};
    for (Eigen::Index i = 0; i < x.size(); ++i)
    {
        double basis{0.0};
        if (static_cast<size_t>(i) >= begin && static_cast<size_t>(i) < end)
            basis = std::isnan(split) ? x[i] : x[i] - split;
        predictions[i] = coefficient * basis;
    }
    output.coefficient = coefficient;
    output.error = calculate_sum_error(calculate_errors(y, predictions, weight));
    return output;
}

bool approximately_equal(double left, double right, double tolerance = 1e-11)
{
    if (std::isinf(left) || std::isinf(right))
        return left == right;
    return std::abs(left - right) <=
           tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

bool check_case(
    const std::string &name,
    const VectorXd &x,
    const VectorXd &y,
    const VectorXd &weight,
    double split,
    size_t begin,
    size_t end,
    int monotonic_constraint)
{
    constexpr double learning_rate{0.17};
    constexpr double penalty_factor{0.63};
    constexpr double ridge_penalty{0.001};
    constexpr double ridge_penalty_weight{7.0};
    auto prefix{aplr_internal::build_prefix_moments(x, y, weight)};
    auto actual{aplr_internal::score_hinge_from_prefix(
        prefix, split, begin, end, learning_rate, penalty_factor,
        ridge_penalty, ridge_penalty_weight, monotonic_constraint)};
    auto expected{exhaustive_score(
        x, y, weight, split, begin, end, learning_rate, penalty_factor,
        ridge_penalty, ridge_penalty_weight, monotonic_constraint)};
    bool passed{approximately_equal(actual.coefficient, expected.coefficient) &&
                approximately_equal(actual.error, expected.error)};
    if (!passed)
        std::cerr << name << " failed: coefficient " << actual.coefficient
                  << " != " << expected.coefficient << ", error "
                  << actual.error << " != " << expected.error << '\n';
    return passed;
}

bool check_reusable_prefix_buffer()
{
    VectorXd x(4);
    x << -2.0, -0.5, 1.0, 4.0;
    VectorXd y(4);
    y << 0.2, -1.0, 0.7, 2.0;
    VectorXd weight(4);
    weight << 3.0, 0.5, 1.5, 2.0;

    aplr_internal::PrefixMoments reused;
    aplr_internal::update_prefix_moments(reused, x, y, weight);

    VectorXd x_second(3);
    x_second << -3.0, 0.25, 5.0;
    VectorXd y_second(3);
    y_second << 1.1, -0.4, 0.8;
    VectorXd weight_second(3);
    weight_second << 0.2, 4.0, 1.3;
    aplr_internal::update_prefix_moments(
        reused, x_second, y_second, weight_second);
    auto fresh{aplr_internal::build_prefix_moments(
        x_second, y_second, weight_second)};

    return reused.weight == fresh.weight &&
           reused.weight_x == fresh.weight_x &&
           reused.weight_x2 == fresh.weight_x2 &&
           reused.weight_y == fresh.weight_y &&
           reused.weight_xy == fresh.weight_xy &&
           reused.absolute_weight_x == fresh.absolute_weight_x &&
           reused.absolute_weight_y == fresh.absolute_weight_y &&
           reused.absolute_weight_xy == fresh.absolute_weight_xy &&
           reused.origin == fresh.origin &&
           reused.weighted_y2 == fresh.weighted_y2;
}

bool same_term_fit(const Term &left, const Term &right)
{
    bool same_split{
        (std::isnan(left.split_point) && std::isnan(right.split_point)) ||
        left.split_point == right.split_point};
    return same_split && left.direction_right == right.direction_right &&
           approximately_equal(left.coefficient, right.coefficient, 1e-13) &&
           approximately_equal(
               left.split_point_search_errors_sum,
               right.split_point_search_errors_sum, 1e-13);
}

bool check_sort_cache_refresh()
{
    Eigen::MatrixXd X(12, 2);
    X << 3.0, -1.0,
        -2.0, 0.5,
        1.0, 2.0,
        4.0, -0.5,
        0.0, 1.5,
        -1.0, -2.0,
        2.0, 0.25,
        5.0, 3.0,
        -3.0, -1.5,
        0.5, 0.75,
        1.5, -0.25,
        -0.5, 1.0;
    VectorXd first_gradient{VectorXd::LinSpaced(12, -1.0, 1.0)};
    VectorXd second_gradient(12);
    second_gradient << 0.2, -1.5, 0.7, 2.0, -0.1, 0.9,
        -0.8, 1.2, -0.4, 0.3, 1.7, -1.1;
    VectorXd first_weight{VectorXd::Ones(12)};
    VectorXd second_weight{first_weight};
    VectorXd observation_count{VectorXd::Ones(12)};

    Term cached(0);
    cached.estimate_split_point(
        X, first_gradient, first_weight, observation_count, 6, 0.2, 2, false,
        0.1, 0.0, 0.001, 1.0);
    cached.estimate_split_point(
        X, second_gradient, second_weight, observation_count, 6, 0.2, 2, false,
        0.1, 0.0, 0.001, 1.0);
    Term fresh(0);
    fresh.estimate_split_point(
        X, second_gradient, second_weight, observation_count, 6, 0.2, 2, false,
        0.1, 0.0, 0.001, 1.0);
    if (!same_term_fit(cached, fresh))
    {
        std::cerr << "main-effect sort cache did not refresh gradients\n";
        return false;
    }

    Term gate(1, {}, 0.0, true, 1.0);
    Term reused_interaction(0, {gate});
    reused_interaction.estimate_split_point(
        X, first_gradient, first_weight, observation_count, 6, 0.2, 2, false,
        0.1, 0.1, 0.001, 1.0);
    reused_interaction.estimate_split_point(
        X, second_gradient, second_weight, observation_count, 6, 0.2, 2, false,
        0.1, 0.1, 0.001, 1.0);
    Term fresh_interaction(0, {gate});
    fresh_interaction.estimate_split_point(
        X, second_gradient, second_weight, observation_count, 6, 0.2, 2, false,
        0.1, 0.1, 0.001, 1.0);
    if (!same_term_fit(reused_interaction, fresh_interaction))
    {
        std::cerr << "interaction sorting retained stale dynamic data\n";
        return false;
    }
    return true;
}

bool check_exhaustive_selection(
    const Eigen::MatrixXd &X, const VectorXd &gradient, const VectorXd &weight,
    size_t bins, int constraint, const std::vector<Term> &gates = {}, Term *reused = nullptr)
{
    constexpr double rate{0.17};
    constexpr double nonlinearity{0.12};
    constexpr double interactions{0.09};
    constexpr double ridge{0.001};
    constexpr double ridge_weight{7.0};
    Term local(0, gates);
    Term &actual{reused ? *reused : local};
    actual.set_monotonic_constraint(constraint);
    actual.estimate_split_point(X, gradient, weight, bins, rate, 1,
        false, nonlinearity, interactions, ridge, ridge_weight);
    if (actual.values_discretized.size() == 0)
        return actual.coefficient == 0.0 && std::isinf(actual.split_point_search_errors_sum);

    std::vector<Eigen::Index> active;
    for (Eigen::Index row = 0; row < X.rows(); ++row)
    {
        bool eligible{true};
        for (auto gate : gates)
            eligible &= !is_approximately_zero(gate.calculate(X)[row]);
        if (eligible)
            active.push_back(row);
    }
    VectorXd raw_x(active.size()), raw_gradient(active.size()), raw_weight(active.size());
    for (size_t row = 0; row < active.size(); ++row)
        raw_x[row] = X(active[row], 0);
    const auto order{sort_indexes_ascending(raw_x)};
    VectorXd sorted_x(raw_x.size());
    for (Eigen::Index row = 0; row < order.size(); ++row)
    {
        sorted_x[row] = raw_x[order[row]];
        raw_gradient[row] = gradient[active[order[row]]];
        raw_weight[row] = weight[active[order[row]]];
    }
    VectorXd binned_gradient(actual.bins_start_index.size());
    for (size_t bin = 0; bin < actual.bins_start_index.size(); ++bin)
    {
        const size_t begin{actual.bins_start_index[bin]};
        const size_t length{actual.bins_end_index[bin] - begin + 1};
        // Upstream freezes bin weight sums at the first evaluation of a Term.
        const double sum_weight{actual.sample_weight_discretized[bin]};
        binned_gradient[bin] = sum_weight > 0.0 ?
            (raw_gradient.segment(begin, length).array() * raw_weight.segment(begin, length).array()).sum() / sum_weight :
            raw_gradient.segment(begin, length).mean();
    }
    bool linear_eligible{true};
    Term original(0, gates);
    const double interaction_factor{original.get_interaction_level() ? 1.0 - interactions : 1.0};
    for (const auto &gate : gates)
        linear_eligible &= gate.base_term != 0;
    const auto &x{actual.values_discretized};
    const auto &w{actual.sample_weight_discretized};
    double linear_error{linear_eligible ? exhaustive_score(x, binned_gradient, w,
        NAN_DOUBLE, 0, x.size(), rate, interaction_factor, ridge, ridge_weight, constraint).error :
        std::numeric_limits<double>::infinity()};
    double left_error{linear_error}, right_error{linear_error};
    double left_split{NAN_DOUBLE}, right_split{NAN_DOUBLE};
    for (auto split = actual.bins_split_points_left.rbegin(); split != actual.bins_split_points_left.rend(); ++split)
    {
        size_t end{0};
        while (end < static_cast<size_t>(x.size()) && x[end] < *split)
            ++end;
        auto score{exhaustive_score(x, binned_gradient, w, *split, 0, end,
            rate, interaction_factor * (1.0 - nonlinearity), ridge, ridge_weight, constraint)};
        if (score.error < left_error)
        {
            left_error = score.error;
            left_split = *split;
        }
    }
    for (double split : actual.bins_split_points_right)
    {
        size_t begin{0};
        while (begin < static_cast<size_t>(x.size()) && x[begin] <= split)
            ++begin;
        auto score{exhaustive_score(x, binned_gradient, w, split, begin, x.size(),
            rate, interaction_factor * (1.0 - nonlinearity), ridge, ridge_weight, constraint)};
        if (score.error < right_error)
        {
            right_error = score.error;
            right_split = split;
        }
    }
    const bool expected_right{!(left_error < right_error)};
    const double expected_split{expected_right ? right_split : left_split};
    const bool same_split{(std::isnan(expected_split) && std::isnan(actual.split_point)) || expected_split == actual.split_point};
    if (!same_split || expected_right != actual.direction_right)
    {
        std::cerr << "exhaustive candidate selection failed for bins=" << bins
                  << " constraint=" << constraint << " expected split=" << expected_split
                  << " actual=" << actual.split_point << '\n';
        return false;
    }
    // Rescoring uses the original sorted rows after candidate selection.
    size_t begin{0}, end{static_cast<size_t>(sorted_x.size())};
    if (!std::isnan(expected_split))
    {
        if (expected_right)
            while (begin < end && sorted_x[begin] <= expected_split)
                ++begin;
        else
        {
            end = 0;
            while (end < static_cast<size_t>(sorted_x.size()) && sorted_x[end] < expected_split)
                ++end;
        }
    }
    const auto expected{exhaustive_score(sorted_x, raw_gradient, raw_weight, expected_split,
        begin, end, rate, interaction_factor * (std::isnan(expected_split) ? 1.0 : 1.0 - nonlinearity),
        ridge, ridge_weight, constraint)};
    if (!approximately_equal(actual.coefficient, expected.coefficient, 1e-13))
    {
        std::cerr << "final coefficient differs from exhaustive score\n";
        return false;
    }
    return true;
}

bool check_candidate_selection_cases()
{
    std::mt19937 random(2026);
    std::normal_distribution<double> normal;
    for (size_t fixture = 0; fixture < 60; ++fixture)
    {
        Eigen::MatrixXd X(31, 2);
        VectorXd gradient(31), weight(31);
        for (Eigen::Index row = 0; row < X.rows(); ++row)
        {
            X(row, 0) = fixture % 3 ? normal(random) : static_cast<double>(row / 3) - 5.0;
            X(row, 1) = normal(random);
            gradient[row] = fixture % 4 ? normal(random) : (row % 2 ? 1.0 : -1.0) + 1e-13 * row;
            weight[row] = row % 7 ? std::abs(normal(random)) : 0.0;
        }
        // Large offsets, almost exact fits and cancellation must use the exact
        // scorer if the centered sufficient statistics cannot bound the winner.
        if (fixture % 5 == 0)
            X.col(0).array() += 1e12;
        if (fixture % 7 == 0)
            gradient = (X.col(0).array() * 0.3).matrix();
        for (size_t bins : {1U, 2U, 3U, 7U, 40U})
            for (int constraint : {-1, 0, 1})
            {
                Term main(0);
                if (!check_exhaustive_selection(X, gradient, weight, bins, constraint, {}, &main))
                    return false;
                VectorXd next_gradient{gradient * -0.7};
                VectorXd next_weight{weight * 1.3};
                next_weight[4] = 0.0;
                if (!check_exhaustive_selection(X, next_gradient, next_weight, bins, constraint, {}, &main))
                    return false;
                Term gate(1, {}, 0.0, true, 1.0);
                Term interaction(0, {gate});
                if (!check_exhaustive_selection(X, gradient, weight, bins, constraint, {gate}, &interaction) ||
                    !check_exhaustive_selection(X, next_gradient, next_weight, bins, constraint, {gate}, &interaction))
                    return false;
                if (fixture % 3 == 1 && !check_exhaustive_selection(X, gradient, weight,
                    bins, constraint, {Term(0, {}, 0.0, true, 1.0)}))
                    return false;
            }
    }
    Eigen::MatrixXd zero_X(8, 1);
    zero_X.col(0) = VectorXd::LinSpaced(8, -2.0, 2.0);
    if (!check_exhaustive_selection(zero_X, VectorXd::Ones(8), VectorXd::Zero(8), 3, 0))
        return false;
    Term no_active_rows(0, {Term(0, {}, 5.0, true, 1.0)});
    no_active_rows.estimate_split_point(zero_X, VectorXd::Ones(8), VectorXd::Ones(8),
        3, 0.2, 0, false, 0.0, 0.0, 0.0, 1.0);
    return no_active_rows.coefficient == 0.0 && std::isinf(no_active_rows.split_point_search_errors_sum);
}

bool check_changed_static_data()
{
    Eigen::MatrixXd X(20, 1);
    X.col(0) = VectorXd::LinSpaced(20, -3.0, 4.0);
    VectorXd gradient{VectorXd::LinSpaced(20, 0.0, 2.0)};
    VectorXd weight{VectorXd::Ones(20)};
    VectorXd count{VectorXd::Ones(20)};
    Term reused(0);
    reused.estimate_split_point(X, gradient, weight, count, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    X.col(0).reverseInPlace();
    count[3] = 7.0;
    reused.estimate_split_point(X, gradient, weight, count, 3, 0.2, 2, false, 0.1, 0.0, 0.001, 1.0);
    Term fresh(0);
    fresh.estimate_split_point(X, gradient, weight, count, 3, 0.2, 2, false, 0.1, 0.0, 0.001, 1.0);
    if (!same_term_fit(reused, fresh))
    {
        std::cerr << "cache retained stale feature values, counts or bin settings\n";
        return false;
    }
    // Adding a same-predictor gate changes eligibility and may prune the gate.
    reused.given_terms = {Term(0, {}, 0.0, true, 1.0)};
    Term gated(0, reused.given_terms);
    reused.estimate_split_point(X, gradient, weight, count, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    gated.estimate_split_point(X, gradient, weight, count, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    if (!same_term_fit(reused, gated))
        return false;
    Term copied(reused);
    Term uncached(0, reused.given_terms);
    copied.estimate_split_point(X, gradient, weight, count, 3, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    uncached.estimate_split_point(X, gradient, weight, count, 3, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    return same_term_fit(copied, uncached);
}

bool check_pruned_gate_cache_lifetime()
{
    Eigen::MatrixXd X(20, 1);
    X.col(0) = VectorXd::LinSpaced(20, -3.0, 4.0);
    const VectorXd gradient{(X.col(0).array() - 1.0).max(0.0).matrix()};
    const VectorXd weights{VectorXd::Ones(20)};
    const std::vector<Term> gates{Term(0, {}, 0.0, true, 1.0)};
    Term reused(0, gates);
    reused.estimate_split_point(X, gradient, weights, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    if (!reused.given_terms.empty())
    {
        std::cerr << "pruning fixture did not remove its same-predictor gate\n";
        return false;
    }
    const auto original_bins{reused.bins_start_index};
    const VectorXd original_values{reused.values_discretized};
    const VectorXd original_weights{reused.sample_weight_discretized};
    // Removing the gate exposes rows, but upstream retains the candidate's
    // original bin representatives, weight sums and positions through this step.
    if (!check_exhaustive_selection(X, gradient * 0.7, weights * 1.3, 8, 0, {}, &reused) ||
        reused.bins_start_index != original_bins ||
        !(reused.values_discretized.array() == original_values.array()).all() ||
        !(reused.sample_weight_discretized.array() == original_weights.array()).all())
        return false;

    Term changed(0, gates);
    changed.estimate_split_point(X, gradient, weights, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    X.col(0).reverseInPlace();
    VectorXd counts{VectorXd::Ones(20)};
    counts[3] = 4.0;
    changed.estimate_split_point(X, gradient, weights, counts, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    Term fresh(0);
    fresh.estimate_split_point(X, gradient, weights, counts, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
    if (!same_term_fit(changed, fresh))
    {
        std::cerr << "pruned gate preserved bins across an explicit input change\n";
        return false;
    }
    return true;
}

bool check_expanded_pruning_and_historical_error_scale()
{
    Eigen::MatrixXd X(18, 1);
    VectorXd gradient(18), weights(18);
    const double values[]{-3.0, -1.0, 0.5, 1.5, 3.0, 5.0};
    for (Eigen::Index row = 0; row < X.rows(); ++row)
    {
        X(row, 0) = values[row / 3];
        gradient[row] = std::max(0.0, X(row, 0) - 1.0) + 0.03 * (row % 3);
        weights[row] = 0.5 + 0.4 * (row % 3);
    }
    auto grouped{aplr_internal::aggregate_identical_rows(X, gradient, weights)};
    ExpandedScoringContext context;
    context.X = &X;
    context.negative_gradient = &gradient;
    context.sample_weight = &weights;
    auto refresh_offset = [&]() {
        grouped = aplr_internal::aggregate_identical_rows(X, gradient, weights);
        context.error_offset = (gradient.array().square() * weights.array()).sum() -
            (grouped.y.array().square() * grouped.sample_weight.array()).sum();
    };
    Term original_main(0), compressed_main(0);
    const std::vector<Term> gates{Term(0, {}, 0.0, true, 1.0)};
    Term original_pruned(0, gates), compressed_pruned(0, gates);
    double prior_offset{NAN_DOUBLE};
    for (size_t step = 0; step < 3; ++step)
    {
        refresh_offset();
        original_main.estimate_split_point(X, gradient, weights, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
        compressed_main.estimate_split_point(grouped.X, grouped.y, grouped.sample_weight,
            grouped.observation_count, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0, false, &context);
        if (!same_term_fit(original_main, compressed_main))
        {
            std::cerr << "compressed historical search error lost changing full-row variance\n";
            return false;
        }
        original_pruned.estimate_split_point(X, gradient, weights, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
        compressed_pruned.estimate_split_point(grouped.X, grouped.y, grouped.sample_weight,
            grouped.observation_count, 8, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0, false, &context);
        if (!same_term_fit(original_pruned, compressed_pruned) ||
            !compressed_pruned.given_terms.empty() ||
            (step > 0 && (original_pruned.bins_start_index != compressed_pruned.bins_start_index ||
                          original_pruned.bins_end_index != compressed_pruned.bins_end_index)))
        {
            std::cerr << "expanded fallback changed legacy bins or full-row scoring after gate pruning\n";
            return false;
        }
        if (step > 0 && context.error_offset == prior_offset)
            return false;
        prior_offset = context.error_offset;
        for (Eigen::Index row = 0; row < X.rows(); ++row)
        {
            gradient[row] = 0.8 * gradient[row] + 0.02 * (row % 3);
            weights[row] *= row % 2 ? 1.2 : 0.7;
        }
    }
    return true;
}

bool check_original_final_reduction_order()
{
    // The gate removes only rows where the indicator is already zero. Both
    // terms predict the same function, but upstream sums their errors in
    // different sorted/gated orders and compares the saved scores strictly.
    Eigen::MatrixXd X(18, 2);
    VectorXd gradient(18), weights(18);
    for (Eigen::Index row = 0; row < X.rows(); ++row)
    {
        const Eigen::Index group{row / 3};
        X(row, 0) = group >= 3 ? 1.0 : 0.0;
        X(row, 1) = group == 0 ? 3.0 : 0.25 * group;
        gradient[row] = 0.2 * (row % 7) - 0.1 * (row % 3);
        weights[row] = row == 7 ? 0.0 : 1.0 + 0.2 * (row % 5);
    }
    const auto grouped{aplr_internal::aggregate_identical_rows(X, gradient, weights)};
    ExpandedScoringContext context{&X, &gradient, &weights,
        (gradient.array().square() * weights.array()).sum() -
        (grouped.y.array().square() * grouped.sample_weight.array()).sum()};
    for (const auto &gates : {std::vector<Term>{}, std::vector<Term>{Term(1, {}, 2.0, false, 1.0)}})
    {
        Term original(0, gates), compressed(0, gates);
        original.estimate_split_point(X, gradient, weights, 8, 0.2, 1, true, 0.0, 0.0, 0.2, 3.0, true);
        compressed.estimate_split_point(grouped.X, grouped.y, grouped.sample_weight,
            grouped.observation_count, 8, 0.2, 1, true, 0.0, 0.0, 0.2, 3.0, true, &context);
        if (original.coefficient != compressed.coefficient ||
            original.split_point_search_errors_sum != compressed.split_point_search_errors_sum)
        {
            std::cerr << "original final coefficient/error reduction order was not preserved\n";
            return false;
        }
    }
    return true;
}

bool check_expanded_endpoint_eligibility()
{
    Eigen::MatrixXd compressed(3, 1);
    compressed << -2.0, 0.0, 3.0;
    VectorXd count(3); count << 3.0, 2.0, 4.0;
    VectorXd gradient(3); gradient << 0.2, -1.1, 2.0;
    VectorXd weights(3); weights << 3.0, 2.0, 4.0;
    Eigen::MatrixXd expanded(9, 1);
    VectorXd expanded_gradient(9);
    Eigen::Index row{0};
    for (Eigen::Index group = 0; group < compressed.rows(); ++group)
        for (size_t copy = 0; copy < static_cast<size_t>(count[group]); ++copy, ++row)
        {
            expanded(row, 0) = compressed(group, 0);
            expanded_gradient[row] = gradient[group];
        }
    for (size_t bins : {2U, 3U, 8U})
    {
        Term actual(0), expected(0);
        actual.estimate_split_point(compressed, gradient, weights, count, bins, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
        expected.estimate_split_point(expanded, expanded_gradient, VectorXd::Ones(9), bins, 0.2, 1, false, 0.1, 0.0, 0.001, 1.0);
        if (actual.bins_split_points_left != expected.bins_split_points_left ||
            actual.bins_split_points_right != expected.bins_split_points_right ||
            !same_term_fit(actual, expected))
        {
            std::cerr << "aggregation changed terminal-group candidate eligibility\n";
            return false;
        }
    }
    return true;
}

bool check_row_aggregation()
{
    Eigen::MatrixXd X(6, 2);
    X << 1.0, 10.0,
        1.0, 10.0,
        2.0, 20.0,
        2.0, 20.0,
        2.0, 20.0,
        3.0, 30.0;
    VectorXd y(6);
    y << 0.0, 2.0, 1.0, 3.0, 5.0, 7.0;
    VectorXd weight(6);
    weight << 1.0, 3.0, 2.0, 1.0, 4.0, 0.0;

    auto aggregated{aplr_internal::aggregate_identical_rows(X, y, weight)};
    bool shape_is_correct{aggregated.X.rows() == 3 &&
                          aggregated.X.cols() == 2};
    bool values_are_correct{
        aggregated.X.row(0).isApprox(X.row(0), 0.0) &&
        aggregated.X.row(1).isApprox(X.row(2), 0.0) &&
        aggregated.X.row(2).isApprox(X.row(5), 0.0) &&
        approximately_equal(aggregated.sample_weight[0], 4.0) &&
        approximately_equal(aggregated.sample_weight[1], 7.0) &&
        approximately_equal(aggregated.sample_weight[2], 0.0) &&
        approximately_equal(aggregated.y[0], 1.5) &&
        approximately_equal(aggregated.y[1], 25.0 / 7.0) &&
        approximately_equal(aggregated.y[2], 7.0) &&
        approximately_equal(aggregated.observation_count[0], 2.0) &&
        approximately_equal(aggregated.observation_count[1], 3.0) &&
        approximately_equal(aggregated.observation_count[2], 1.0)};

    VectorXd prediction_by_group(3);
    prediction_by_group << 1.2, 2.2, 3.4;
    double poisson_original{0.0};
    double poisson_aggregated{0.0};
    double tweedie_original{0.0};
    double tweedie_aggregated{0.0};
    const std::vector<size_t> group_for_original_row{0, 0, 1, 1, 1, 2};
    constexpr double tweedie_power{1.5};
    for (Eigen::Index row = 0; row < y.size(); ++row)
    {
        double prediction{prediction_by_group[group_for_original_row[row]]};
        poisson_original += weight[row] *
                            (prediction - y[row] * std::log(prediction));
        tweedie_original += weight[row] *
                            (-y[row] * std::pow(prediction, 1.0 - tweedie_power) /
                                 (1.0 - tweedie_power) +
                             std::pow(prediction, 2.0 - tweedie_power) /
                                 (2.0 - tweedie_power));
    }
    for (Eigen::Index row = 0; row < aggregated.y.size(); ++row)
    {
        double prediction{prediction_by_group[row]};
        poisson_aggregated += aggregated.sample_weight[row] *
                              (prediction - aggregated.y[row] * std::log(prediction));
        tweedie_aggregated += aggregated.sample_weight[row] *
                              (-aggregated.y[row] *
                                   std::pow(prediction, 1.0 - tweedie_power) /
                                   (1.0 - tweedie_power) +
                               std::pow(prediction, 2.0 - tweedie_power) /
                                   (2.0 - tweedie_power));
    }

    bool losses_are_preserved{
        approximately_equal(poisson_original, poisson_aggregated) &&
        approximately_equal(tweedie_original, tweedie_aggregated)};
    if (!(shape_is_correct && values_are_correct && losses_are_preserved))
        std::cerr << "identical-row aggregation did not preserve sufficient statistics\n";
    return shape_is_correct && values_are_correct && losses_are_preserved;
}

} // namespace

int main()
{
    VectorXd x(7);
    x << -3.0, -1.5, -0.25, 0.0, 0.75, 2.0, 4.5;
    VectorXd y(7);
    y << -1.2, -0.7, 0.1, 0.3, 0.8, 1.3, 2.1;
    VectorXd weights(7);
    weights << 0.2, 3.0, 0.7, 4.5, 1.2, 0.4, 2.8;

    bool passed{true};
    passed &= check_case("weighted left hinge", x, y, weights, 0.5, 0, 4, 0);
    passed &= check_case("weighted right hinge", x, y, weights, -0.5, 2, 7, 0);

    VectorXd monotone_y(7);
    monotone_y << -2.0, -1.0, -0.4, 0.0, 0.2, 0.6, 1.0;
    passed &= check_case(
        "increasing monotone left hinge sign", x, monotone_y, weights,
        0.5, 0, 4, 1);
    passed &= check_case(
        "decreasing monotone rejection", x, monotone_y, weights,
        0.5, 0, 4, -1);

    VectorXd near_tie_y(7);
    near_tie_y << 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0 + 1e-13;
    passed &= check_case(
        "near tie left", x, near_tie_y, weights, 0.0, 0, 3, 0);
    passed &= check_case(
        "near tie right", x, near_tie_y, weights, 0.0, 4, 7, 0);
    passed &= check_sort_cache_refresh();
    passed &= check_changed_static_data();
    passed &= check_pruned_gate_cache_lifetime();
    passed &= check_expanded_pruning_and_historical_error_scale();
    passed &= check_original_final_reduction_order();
    passed &= check_expanded_endpoint_eligibility();
    passed &= check_candidate_selection_cases();
    passed &= check_reusable_prefix_buffer();
    passed &= check_row_aggregation();

    if (!passed)
        return 1;
    std::cout << "prefix split search matches exhaustive scoring\n";
    return 0;
}
