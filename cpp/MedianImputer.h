#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include "constants.h"
#include "functions.h"

template <typename T>
class MedianImputer
{
public:
    MedianImputer() : median_(static_cast<T>(NAN_DOUBLE)), had_nans_in_fit_(false) {}

    void fit(const std::vector<T> &data, const std::vector<double> &sample_weight)
    {
        if (data.size() != sample_weight.size())
        {
            throw std::runtime_error("Data and sample_weight must have the same size.");
        }

        had_nans_in_fit_ = false;
        std::vector<std::pair<T, double>> weighted_values;
        weighted_values.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            if (std::isnan(data[i]))
                had_nans_in_fit_ = true;
            if (!std::isnan(data[i]))
            {
                if (sample_weight[i] < 0)
                {
                    throw std::runtime_error("Sample weights must be non-negative.");
                }
                weighted_values.push_back({data[i], sample_weight[i]});
            }
        }

        if (weighted_values.empty())
        {
            median_ = static_cast<T>(0.0);
            return;
        }

        std::sort(weighted_values.begin(), weighted_values.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.first < b.first;
                  });

        double total_weight = 0;
        for (const auto &pair : weighted_values)
        {
            total_weight += pair.second;
        }

        if (is_approximately_zero(total_weight))
        {
            median_ = static_cast<T>(NAN_DOUBLE);
            return;
        }

        double half_total_weight = total_weight / 2.0;
        double cum_weight = 0.0;
        for (size_t i = 0; i < weighted_values.size(); ++i)
        {
            const auto &pair = weighted_values[i];
            cum_weight += pair.second;

            if (is_approximately_equal(cum_weight, half_total_weight))
            {
                if (i + 1 < weighted_values.size())
                {
                    median_ = (pair.first + weighted_values[i + 1].first) / 2.0;
                }
                else
                {
                    median_ = pair.first;
                }
                return;
            }

            if (cum_weight > half_total_weight)
            {
                median_ = pair.first;
                return;
            }
        }
    }

    std::pair<std::vector<T>, std::vector<double>> transform(const std::vector<T> &data) const
    {
        std::vector<T> transformed_data = data;
        std::vector<double> indicator_col;

        if (had_nans_in_fit_)
        {
            indicator_col.reserve(data.size());
        }

        if (std::isnan(median_))
        {
            // If median is NaN, we can't impute, but we might still need to create an indicator column
            if (had_nans_in_fit_)
            {
                for (const auto &value : data)
                {
                    indicator_col.push_back(std::isnan(value) ? 1.0 : 0.0);
                }
            }
            return {transformed_data, indicator_col};
        }

        for (size_t i = 0; i < data.size(); ++i)
        {
            if (std::isnan(data[i]))
            {
                transformed_data[i] = median_;
            }
            if (had_nans_in_fit_)
            {
                indicator_col.push_back(std::isnan(data[i]) ? 1.0 : 0.0);
            }
        }
        return std::make_pair(transformed_data, indicator_col);
    }

    T get_median() const
    {
        return median_;
    }

    bool had_nans_in_fit() const
    {
        return had_nans_in_fit_;
    }

public:
    T median_;
    bool had_nans_in_fit_;
};
