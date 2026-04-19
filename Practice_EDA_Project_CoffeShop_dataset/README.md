
# Summary of Coffee Shop EDA Project Notebook

This Jupyter notebook performs **Exploratory Data Analysis (EDA)** on a Coffee Shop Sales dataset. Here's what it covers:

## Dataset Overview
- **Source:** Coffee_Shop_Sales.xlsx
- **Size:** 149,116 transactions with 11 columns
- **Date Range:** January 1 - June 30, 2023
- **Data Quality:** No missing values or duplicates

## Key Columns
- `transaction_id`, `transaction_date`, `transaction_time`
- `transaction_qty`, `unit_price`
- `store_id`, `store_location` (multiple locations)
- `product_id`, `product_category` (Coffee, Tea, Drinking Chocolate)
- `product_type`, `product_detail`

## Analysis Performed

1. **Data Exploration**
   - Loaded and inspected data structure
   - Checked for duplicates and missing values
   - Generated descriptive statistics

2. **Data Processing**
   - Combined `transaction_date` and `transaction_time` into a single `transaction_datetime` column
   - Extracted `transaction_hour` for time-based analysis

3. **Analysis Question Started**
   - *"What are the peak transaction times during the day, and how do they vary by store location?"*
   - Began extracting hour information from timestamps for hourly analysis

## Libraries Used
- pandas, numpy
- matplotlib, seaborn
- plotly (for interactive visualizations)

The notebook is structured for systematic exploration of customer behavior, sales patterns, and product performance across different store locations.
