#pragma once

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <stdexcept>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"
#include "MedianImputer.h"
#include "OneHotEncoder.h"
#include "CppDataFrame.h"

class Preprocessor
{
public:
    Preprocessor() : is_fitted_(false) {}

    void fit(const CppDataFrame &df, const Eigen::VectorXd &sample_weight)
    {
        numeric_cols_.clear();
        categorical_cols_.clear();
        
        original_column_names_ = df.get_column_names_in_order();
        for (const auto &col_name : original_column_names_)
        {
            if (df.is_numeric_column(col_name))
            {
                numeric_cols_.push_back(col_name);
            }
            else
            {
                categorical_cols_.push_back(col_name);
            }
        }

        if (df.empty())
        {
            return; // Nothing to fit
        }

        size_t num_rows = df.get_num_rows();

        if (num_rows > 0 && sample_weight.size() > 0 && static_cast<size_t>(sample_weight.size()) != num_rows)
        {
            throw std::runtime_error("sample_weight must have the same number of rows as the data.");
        }

        std::vector<double> weights;
        if (sample_weight.size() > 0)
        {
            weights.resize(sample_weight.size());
            Eigen::VectorXd::Map(&weights[0], sample_weight.size()) = sample_weight;
        }
        else if (num_rows > 0)
        {
            weights.assign(num_rows, 1.0);
        }

        // Fit MedianImputer for numeric columns
        for (const auto &col_name : numeric_cols_)
        {
            const auto &col_data = df.get_numeric_column(col_name);
            numeric_imputers_[col_name].fit(col_data, weights);
        }

        // Fit OneHotEncoder for categorical columns
        for (const auto &col_name : categorical_cols_)
        {
            const auto &col_data = df.get_string_column(col_name);
            one_hot_encoders_[col_name].fit(col_data);
        }

        // Generate and store final column names
        final_column_names_.clear();
        for (const auto &col_name : numeric_cols_)
        {
            final_column_names_.push_back(col_name);
            if (numeric_imputers_.at(col_name).had_nans_in_fit())
            {
                final_column_names_.push_back(col_name + "_is_missing");
            }
        }
        for (const auto &col_name : categorical_cols_)
        {
            const auto &encoder = one_hot_encoders_.at(col_name);
            const auto &categories = encoder.get_categories();
            for (const auto &cat : categories)
            {
                final_column_names_.push_back(col_name + "_" + cat);
            }
        }

        is_fitted_ = true;
    }

    void fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &sample_weight, const std::vector<std::string> &X_names = {})
    {
        CppDataFrame df = CppDataFrame::from_matrix(X, X_names);
        fit(df, sample_weight);
    }

    std::pair<Eigen::MatrixXd, std::vector<std::string>> transform(const CppDataFrame &df) const
    {
        if (!is_fitted_)
        {
            return df.to_matrix();
        }

        if (df.empty())
        {
            return {Eigen::MatrixXd(0, final_column_names_.size()), final_column_names_};
        }

        size_t num_rows = df.get_num_rows();
        size_t num_cols_total = final_column_names_.size();
        Eigen::MatrixXd result_matrix(num_rows, num_cols_total);
        size_t current_col_offset = 0;

        // Transform numeric columns
        for (const auto &col_name : numeric_cols_)
        {
            auto imputer_it = numeric_imputers_.find(col_name);
            if (imputer_it == numeric_imputers_.end())
                throw std::runtime_error("No fitted imputer for numeric column '" + col_name + "'.");

            const auto &col_data = df.get_numeric_column(col_name);
            auto transform_result = imputer_it->second.transform(col_data);
            const auto &transformed_col_vec = transform_result.first;
            const auto &indicator_col = transform_result.second;
            for (size_t i = 0; i < num_rows; ++i)
            {
                result_matrix(i, current_col_offset) = transformed_col_vec[i];
            }
            current_col_offset++;
            if (!indicator_col.empty())
            {
                for (size_t i = 0; i < num_rows; ++i)
                {
                    result_matrix(i, current_col_offset) = indicator_col[i];
                }
                current_col_offset++;
            }
        }

        // Transform categorical columns
        for (const auto &col_name : categorical_cols_)
        {
            auto encoder_it = one_hot_encoders_.find(col_name);
            if (encoder_it == one_hot_encoders_.end())
                throw std::runtime_error("No fitted encoder for categorical column '" + col_name + "'.");

            const auto &col_data = df.get_string_column(col_name);
            auto ohe_result = encoder_it->second.transform(col_data);
            if (num_rows > 0 && !ohe_result.empty())
            {
                for (size_t i = 0; i < num_rows; ++i)
                {
                    for (size_t j = 0; j < ohe_result[i].size(); ++j)
                    {
                        result_matrix(i, current_col_offset + j) = ohe_result[i][j];
                    }
                }
                current_col_offset += encoder_it->second.get_categories().size();
            }
        }
        return {result_matrix, final_column_names_};
    }

    std::pair<Eigen::MatrixXd, std::vector<std::string>> transform(const Eigen::MatrixXd &X, const std::vector<std::string> &X_names = {}) const
    {
        if (!is_fitted_)
        {
            return {X, X_names};
        }
        const auto& names_to_use = X_names.empty() ? original_column_names_ : X_names;
        CppDataFrame df = CppDataFrame::from_matrix(X, names_to_use);
        return transform(df);
    }

    std::pair<Eigen::MatrixXd, std::vector<std::string>> fit_transform(const CppDataFrame &df, const Eigen::VectorXd &sample_weight)
    {
        fit(df, sample_weight);
        return transform(df);
    }

    std::pair<Eigen::MatrixXd, std::vector<std::string>> fit_transform(const Eigen::MatrixXd &X, const Eigen::VectorXd &sample_weight, const std::vector<std::string> &X_names = {})
    {
        fit(X, sample_weight, X_names);
        return transform(X, X_names);
    }

    std::vector<std::string> get_original_column_names() const
    {
        return original_column_names_;
    }

    std::vector<std::string> get_transformed_column_names() const
    {
        return final_column_names_;
    }

    bool is_fitted() const
    {
        return is_fitted_;
    }

public:
    std::map<std::string, MedianImputer<double>> numeric_imputers_;
    std::map<std::string, OneHotEncoder> one_hot_encoders_;
    std::vector<std::string> numeric_cols_;
    std::vector<std::string> categorical_cols_;
    std::vector<std::string> final_column_names_;
    std::vector<std::string> original_column_names_;
    bool is_fitted_;
};
