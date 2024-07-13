#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# main.py

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import chi2_contingency

df = pd.read_excel("hints6_public.xlsx")

df['diabetes_num'] = df['MedConditions_Diabetes'].apply(lambda x: 0 if x == 2 else (1 if x == 1 else None))

df['hypertension_num'] = df['MedConditions_HighBP'].apply(lambda x: 0 if x == 2 else (1 if x == 1 else None))

df['hyper_dia_num'] = ((df['MedConditions_HighBP'] == 1) & (df['MedConditions_Diabetes'] == 1)).astype(int)

# Create group column based on conditions
def assign_group(row):
    if row['hypertension_num'] == 1 and row['diabetes_num'] == 1:
        return 'Hypertension+Diabetes'
    elif row['hypertension_num'] == 1 and row['diabetes_num'] == 0:
        return 'Hypertension'
    elif row['hypertension_num'] == 0 and row['diabetes_num'] == 1:
        return 'Diabetes'
    else:
        return None

df['group'] = df.apply(assign_group, axis=1)

df = df.dropna(subset=['group'])

df['group'].value_counts()


# Transform variables into new labels (table1)
df['birthgender'] = df['BirthGender'].replace({1: 'Male', 2: 'Female'})

df['marital'] = df['MaritalStatus'].replace({
    1: 'Married',
    2: 'Living as married or living with a romantic partner',
    3: 'Divorced',
    4: 'Widowed',
    5: 'Separated',
    6: 'Single, never married'
})

df['education'] = df['Education'].replace({
    1: 'Less than 8 years',
    2: '8 through 11 years',
    3: '12 years or completed high school',
    4: 'Post high school training other than college (vocational or technical)',
    5: 'Some college',
    6: 'College graduate',
    7: 'Postgraduate'
})

df['ethnicity'] = df['RaceEthn5'].replace({
    1: 'Non-Hispanic White',
    2: 'Non-Hispanic Black or African American',
    3: 'Hispanic',
    4: 'Non-Hispanic Asian',
    5: 'Non-Hispanic Other'
})

df['race'] = df['RACE_CAT2'].replace({
    11: 'White only',
    12: 'Black only',
    14: 'American Indian/Alaska Native only',
    16: 'Multiple races selected',
    31: 'Asian Indian only',
    32: 'Chinese only',
    33: 'Filipino only',
    36: 'Vietnamese only',
    37: 'Other Asian only',
    54: 'Other Pacific Islander only'
})

df['sexualorientation'] = df['SexualOrientation'].replace({
    1: 'Heterosexual',
    2: 'Homosexual',
    3: 'Bisexual',
    91: 'Other'
})

df['income'] = df['IncomeRanges'].replace({
    1: '$0 to $9,999',
    2: '$10,000 to $14,999',
    3: '$15,000 to $19,999',
    4: '$20,000 to $34,999',
    5: '$35,000 to $49,999',
    6: '$50,000 to $74,999',
    7: '$75,000 to $99,999',
    8: '$100,000 to $199,999',
    9: '$200,000 or more'
})

df['h1'] = df['GeneralHealth'].replace({
    1: 'Excellent',
    2: 'Very good',
    3: 'Good',
    4: 'Fair',
    5: 'Poor'
})

df['h6c'] = df['MedConditions_HeartCondition'].replace({
    1: 'Yes',
    2: 'No'
})

df['h6d'] = df['MedConditions_LungDisease'].replace({
    1: 'Yes',
    2: 'No'
})

df['h6e'] = df['MedConditions_Depression'].replace({
    1: 'Yes',
    2: 'No'
})

df['alcohol'] = df['DrinkDaysPerWeek'].replace({
    0: 'None',
    1: '1 day',
    2: '2 days',
    3: '3 days',
    4: '4 days',
    5: '5 days',
    6: '6 days',
    7: '7 days'
})

df['smoke'] = df['SmokeNow'].replace({
    1: 'Current',
    2: 'Former',
    3: 'Never'
})

df['occ_num'] = df['OCCUPATION_CAT'].replace({
    1: 'Employed only',
    3: 'Homemaker only',
    4: 'Student only',
    5: 'Retired only',
    6: 'Disabled only',
    7: 'Multiple Occupation statuses selected',
    8: 'Unemployed for 1 year or more only',
    9: 'Unemplyed for less than 1 year only',
    91: 'Other only'
})

continuous=['Age', 'BMI']
category=['birthgender', 'marital', 
          'education', 'ethnicity', 'race', 'occ_num','sexualorientation', 
          'income', 'h1', 'h6c', 'h6d', 'h6e', 'alcohol', 'smoke',
          'PR_RUCA_2010', 'CENSREG']

continuous_columns=['Age', 'BMI',
                    'group']
category_columns=['birthgender', 'marital', 
                  'education', 'ethnicity', 'race', 'occ_num','sexualorientation', 
                  'income', 'h1', 'h6c', 'h6d', 'h6e', 'alcohol', 'smoke',
                  'PR_RUCA_2010', 'CENSREG',
                  'group']


# ANOVA for continuous
results = []
def compute_stats_and_anova(data, variable):
    formula = f'{variable} ~ C(group)'
    model = ols(formula, data=df[continuous_columns]).fit()
    anova_table = anova_lm(model, typ=2)
    p_value = anova_table["PR(>F)"][0]

    group_stats = data.groupby('group')[variable].agg(['mean', 'std'])
    group1 = f"{group_stats.loc['Hypertension', 'mean']:.1f} ({group_stats.loc['Hypertension', 'std']:.1f})"
    group2 = f"{group_stats.loc['Diabetes', 'mean']:.1f} ({group_stats.loc['Diabetes', 'std']:.1f})"
    group3 = f"{group_stats.loc['Hypertension+Diabetes', 'mean']:.1f} ({group_stats.loc['Hypertension+Diabetes', 'std']:.1f})"
    
    return {
        'Variable': variable,
        'Group1': group1,
        'Group2': group2,
        'Group3': group3,
        'P value': f"{p_value:.3f}"
    }

for variable in continuous:
    results.append(compute_stats_and_anova(df[continuous_columns], variable))

results_df = pd.DataFrame(results)

print(results_df)




# Chi-square for categorical
category_name = ['group']

# Define special codes indicating missing or undefined values
missing_codes = [-9, -99, -7, -5, 'N/A', 'Unknown']  

# Function to determine if a value is valid (not missing)
def is_valid(value):
    return value not in missing_codes


def compute_chi_square(data, variable):

    valid_data = data[data[variable].apply(is_valid)]
    
    # Calculate contingency table
    count_table = pd.crosstab(valid_data['group'], valid_data[variable])
    
    # Calculate percentages
    percent_table = count_table.apply(lambda x: x / x.sum(), axis=1).applymap(lambda x: f"{x:.1%}")
    
    # Perform chi-square test
    chi2, p_value, _, _ = chi2_contingency(count_table)
    
    result_list = []
    for cat in count_table.columns:
        row = {
            'Variable': f"{variable}, {cat}",
            'P value': f"{p_value:.3f}"
        }
        for i, group in enumerate(count_table.index, start=1):
            row[f'Group{i}'] = f"{count_table.loc[group, cat]} ({percent_table.loc[group, cat]})"
        result_list.append(row)
    
    return result_list

results = []

for variable in category:
    results.extend(compute_chi_square(df, variable))

results_df = pd.DataFrame(results)

columns_order = ['Variable'] + [f'Group{i}' for i in range(1, len(results_df.columns) - 1)] + ['P value']
results_df = results_df[columns_order]

print(results_df)


# Define the variables based on Telehealth data (table 2 and 3)
df['d1'] = df['ReceiveTelehealthCare'].replace({1: 1, 2: 2, 3: 3, 4: 4})
df['d2'] = df['OfferedTelehealthOption'].replace({1: 1, 2: 2, 3: 3})
df['d31'] = df['THNo_PreferInPerson'].replace({1: 1, 2: 2})
df['d32'] = df['THNo_ConcernedPrivacy'].replace({1: 1, 2: 2})
df['d33'] = df['THNo_TooDifficult'].replace({1: 1, 2: 2})
df['d41'] = df['THYes_HCPRecommended'].replace({1: 1, 2: 2})
df['d42'] = df['THYes_WantedAdvice'].replace({1: 1, 2: 2})
df['d43'] = df['THYes_AvoidExposure'].replace({1: 1, 2: 2})
df['d44'] = df['THYes_Convenient'].replace({1: 1, 2: 2})
df['d45'] = df['THYes_IncludeOthers'].replace({1: 1, 2: 2})
df['d5'] = df['RecentTelehealthReason'].replace({
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6
})
df['d61'] = df['Telehealth_TechProbs'].replace({
    1: 1, 2: 2, 3: 3, 4: 4
})
df['d62'] = df['Telehealth_GoodCare'].replace({
    1: 1, 2: 2, 3: 3, 4: 4
})
df['d63'] = df['Telehealth_ConcernedPrivacy'].replace({
    1: 1, 2: 2, 3: 3, 4: 4
})


# Define special codes indicating missing or undefined values
missing_codes = [-9, -99, -7, -5, -2, -1, -6, 'N/A', 'Unknown']

# Function to determine if a value is valid (not missing)
def is_valid(value):
    return value not in missing_codes

# Function to compute chi-square and generate results, excluding negative categories
def compute_chi_square(data, variable):
    # Filter out rows where any categorical variable has negative values or special codes
    valid_data = data[data[variable].apply(is_valid)]
    
    if valid_data.shape[0] == 0:
        print(f"No valid data for variable: {variable}")
        return []
    
    # Calculate contingency table
    count_table = pd.crosstab(valid_data['group'], valid_data[variable])
    
    # Calculate percentages
    percent_table = count_table.apply(lambda x: x / x.sum(), axis=1).applymap(lambda x: f"{x:.1%}")
    
    # Perform chi-square test
    chi2, p_value, _, _ = chi2_contingency(count_table)
    
    # Prepare results for each category
    result_list = []
    for cat in count_table.columns:
        row = {
            'Variable': f"{variable}, {cat}",
            'P value': f"{p_value:.3f}"
        }
        for i, group in enumerate(count_table.index, start=1):
            row[f'Group{i}'] = f"{count_table.loc[group, cat]} ({percent_table.loc[group, cat]})"
        result_list.append(row)
    
    return result_list

# Ensure the 'group' variable exists
if 'group' not in df.columns:
    raise ValueError("The 'group' column is not present in the dataframe.")

# Results storage
results = []

# Analyze each categorical variable
for variable in ['d1', 'd2', 'd31', 'd32', 'd33', 'd41', 'd42', 'd43', 'd44', 'd45', 'd5', 'd61', 'd62', 'd63']:
    results.extend(compute_chi_square(df, variable))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Adjust column order for output format
columns_order = ['Variable'] + [f'Group{i}' for i in range(1, len(results_df.columns) - 1)] + ['P value']
results_df = results_df[columns_order]

# Output results table
print(results_df)

