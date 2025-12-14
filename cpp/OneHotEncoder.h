#pragma once

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>

class OneHotEncoder
{
public:
    OneHotEncoder() {}

    void fit(const std::vector<std::string> &data)
    {
        categories_.clear();
        category_map_.clear();
        for (const auto &category : data)
        {
            if (category_map_.find(category) == category_map_.end())
            {
                category_map_[category] = 0; // Temporary value
                categories_.push_back(category);
            }
        }
        // Sort categories to have a deterministic output
        std::sort(categories_.begin(), categories_.end());
        for (size_t i = 0; i < categories_.size(); ++i)
        {
            category_map_[categories_[i]] = i;
        }
    }

    std::vector<std::vector<int>> transform(const std::vector<std::string> &data) const
    {
        std::vector<std::vector<int>> encoded_data;
        encoded_data.reserve(data.size());

        for (const auto &category : data)
        {
            std::vector<int> row(categories_.size(), 0);
            auto it = category_map_.find(category);
            if (it != category_map_.end())
            {
                row[it->second] = 1;
            }
            // else, it's an unknown category, so the row is all zeros.
            encoded_data.push_back(row);
        }
        return encoded_data;
    }

    const std::vector<std::string> &get_categories() const
    {
        return categories_;
    }

public:
    std::vector<std::string> categories_;
    std::map<std::string, size_t> category_map_;
};
