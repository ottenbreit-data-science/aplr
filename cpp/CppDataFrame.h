#pragma once

#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include "../dependencies/eigen-3.4.0/Eigen/Dense"

class ColumnBase
{
public:
    virtual ~ColumnBase() = default;
    virtual size_t size() const = 0;
};

class NumericColumn : public ColumnBase
{
public:
    std::vector<double> data;
    size_t size() const override { return data.size(); }
};

class StringColumn : public ColumnBase
{
public:
    std::vector<std::string> data;
    size_t size() const override { return data.size(); }
};

class CppDataFrame
{
private:
    void deep_copy_from(const CppDataFrame &other)
    {
        num_rows_ = other.num_rows_;
        column_names_in_order_ = other.column_names_in_order_;
        columns_.clear();
        for (const auto &pair : other.columns_)
        {
            if (const auto *num_col = dynamic_cast<NumericColumn *>(pair.second.get()))
            {
                auto new_col = std::make_unique<NumericColumn>();
                new_col->data = num_col->data;
                columns_[pair.first] = std::move(new_col);
            }
            else if (const auto *str_col = dynamic_cast<StringColumn *>(pair.second.get()))
            {
                auto new_col = std::make_unique<StringColumn>();
                new_col->data = str_col->data;
                columns_[pair.first] = std::move(new_col);
            }
        }
    }

public:
    CppDataFrame() = default;

    // Copy constructor
    CppDataFrame(const CppDataFrame &other)
    {
        deep_copy_from(other);
    }

    // Copy assignment operator
    CppDataFrame &operator=(const CppDataFrame &other)
    {
        if (this != &other)
        {
            deep_copy_from(other);
        }
        return *this;
    }

    void add_column(const std::string &name, std::vector<double> &&data)
    {
        validate_col_length(data.size());
        auto col = std::make_unique<NumericColumn>();
        col->data = std::move(data);
        if (num_rows_ == 0 && !col->data.empty())
            num_rows_ = col->data.size();
        if (columns_.find(name) == columns_.end())
        {
            column_names_in_order_.push_back(name);
        }
        columns_[name] = std::move(col);
    }

    void add_column(const std::string &name, std::vector<std::string> &&data)
    {
        validate_col_length(data.size());
        auto col = std::make_unique<StringColumn>();
        col->data = std::move(data);
        if (num_rows_ == 0 && !col->data.empty())
            num_rows_ = col->data.size();
        if (columns_.find(name) == columns_.end())
        {
            column_names_in_order_.push_back(name);
        }
        columns_[name] = std::move(col);
    }

    const std::vector<double> &get_numeric_column(const std::string &name) const
    {
        auto it = columns_.find(name);
        if (it == columns_.end())
        {
            throw std::runtime_error("Column '" + name + "' not found.");
        }
        auto *col = dynamic_cast<NumericColumn *>(it->second.get());
        if (!col)
        {
            throw std::runtime_error("Column '" + name + "' is not numeric.");
        }
        return col->data;
    }

    const std::vector<std::string> &get_string_column(const std::string &name) const
    {
        auto it = columns_.find(name);
        if (it == columns_.end())
        {
            throw std::runtime_error("Column '" + name + "' not found.");
        }
        auto *col = dynamic_cast<StringColumn *>(it->second.get());
        if (!col)
        {
            throw std::runtime_error("Column '" + name + "' is not a string column.");
        }
        return col->data;
    }

    bool is_numeric_column(const std::string &name) const
    {
        auto it = columns_.find(name);
        if (it == columns_.end())
        {
            throw std::runtime_error("Column '" + name + "' not found.");
        }
        auto *col = dynamic_cast<NumericColumn *>(it->second.get());
        return col != nullptr;
    }

    size_t get_num_rows() const
    {
        return num_rows_;
    }

    bool empty() const
    {
        return columns_.empty();
    }

    std::vector<std::string> get_column_names() const
    {
        std::vector<std::string> names;
        names.reserve(columns_.size());
        for (const auto &pair : columns_)
        {
            names.push_back(pair.first);
        }
        return names;
    }

    std::vector<std::string> get_column_names_in_order() const
    {
        return column_names_in_order_;
    }

    std::pair<Eigen::MatrixXd, std::vector<std::string>> to_matrix() const
    {
        if (empty())
        {
            return {Eigen::MatrixXd(0, 0), {}};
        }

        const auto &col_names = get_column_names_in_order();
        Eigen::MatrixXd mat(get_num_rows(), col_names.size());

        for (size_t i = 0; i < col_names.size(); ++i)
        {
            const auto &col_name = col_names[i];
            if (is_numeric_column(col_name))
            {
                const auto &col_data = get_numeric_column(col_name);
                for (size_t j = 0; j < col_data.size(); ++j)
                {
                    mat(j, i) = col_data[j];
                }
            }
            else
            {
                throw std::runtime_error("Cannot convert DataFrame to matrix if it contains non-numeric columns.");
            }
        }
        return {mat, col_names};
    }

    static CppDataFrame from_matrix(const Eigen::MatrixXd &X, const std::vector<std::string> &X_names = {})
    {
        CppDataFrame df;
        if (X.cols() == 0)
        {
            return df;
        }

        if (!X_names.empty() && X_names.size() != static_cast<size_t>(X.cols()))
        {
            throw std::runtime_error("The number of column names must match the number of columns in the matrix.");
        }

        for (Eigen::Index i = 0; i < X.cols(); ++i)
        {
            std::string col_name = X_names.empty() ? "X" + std::to_string(i + 1) : X_names[i];
            std::vector<double> col_data(X.rows());
            for (Eigen::Index j = 0; j < X.rows(); ++j)
            {
                col_data[j] = X(j, i);
            }
            df.add_column(col_name, std::move(col_data));
        }

        return df;
    }

public:
    void validate_col_length(size_t len)
    {
        if (num_rows_ != 0 && len != num_rows_)
        {
            throw std::runtime_error("All columns in a DataFrame must have the same length.");
        }
    }

    std::map<std::string, std::unique_ptr<ColumnBase>> columns_;
    std::vector<std::string> column_names_in_order_;
    size_t num_rows_ = 0;
};
