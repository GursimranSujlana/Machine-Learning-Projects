# Salary Analysis Project

## Overview
This repository contains the analysis of a dataset that includes salary data for jobs in the fields of data science, machine learning, and data engineering. The data spans from 2020 to 2023 and includes various job categories, experience levels, and employment types.

## Dataset
The dataset includes 9,355 entries with comprehensive salary information across multiple dimensions such as job title, job category, salary currency, salary in USD, employee residence, experience level, employment type, work setting, company location, and company size.

## Analyses Performed
- Statistical analysis of salary data.
- Visual exploration of salary distributions, job categories, and employment types.
- Identification of outliers in salary data.
- Correlation analysis between years of experience and salary.
- Predictive modeling using linear regression to predict salary based on years of experience.

## Repository Structure
├── Data # Folder containing datasets used in the analysis
├── Notebooks # Jupyter notebooks with detailed analysis
├── Scripts # Python scripts for data processing and analysis
├── Images # Images and graphs generated from the analysis
├── results # Generated reports and summaries of the analysis
└── README.md # The top-level README for this project

## Usage
To run the analysis scripts, ensure that you have `pandas`, `matplotlib`, `seaborn`, and `scikit-learn` installed in your Python environment. You can install them using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn

After installation, you can run the scripts or notebooks to replicate the analysis.

Visualizations
The repository includes various visualizations generated from the dataset:

Box plots
Histograms
Bar plots
Scatter plots
Correlation heatmaps
Violin plots
Pie charts
Results
The main findings of the analysis are:

The salary distribution is right-skewed, indicating most professionals earn below the average.
Full-time employment is the predominant employment type at 99.5%.
There is no strong correlation between years of experience and salary, suggesting that experience is not the only factor influencing salary.
Predictive Model
A linear regression model was built to predict salary based on the years of experience. The model's performance on the training and testing sets is reported using Mean Squared Error (MSE) and R-squared values.

Contributing
Contributions to this project are welcome. Please consider forking the repository and submitting a pull request.

Contact
Author: GurSimran Sujlana
GitHub: https://github.com/GursimranSujlana
