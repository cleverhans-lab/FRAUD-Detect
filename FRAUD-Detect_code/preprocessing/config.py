

bin_dict_5 = {
        0.0 : "very_low",
        1.0 : "low",
        2.0 : "middle",
        3.0 : "high",
        4.0 : "very_high"
        }


bin_dict = {
        0.0 : "low",
        1.0 : "middle",
        2.0 : "high"
        }

#['age', 'hours_per_week'],
num_cols_dict ={
    'adult_income'      : ['age', 'hours_per_week', 'capital_gain', 'capital_loss'],
    'compas'            : ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count'],
    'default_credit'    : ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                            'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'],
    'marketing'         : ['duration', 'campaign', 'pdays', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed']
}


min_support_dict ={
    'adult_income'      : 0.05,
    'compas'            : 0.05,
    'default_credit'    : 0.5,
    'marketing'         : 0.5
}


dropList_dict ={
    'adult_income'      : ['gender_Female', 'gender_Male', 'race_Amer-Indian-Eskimo', 'race_Asian-Pac-Islander', 'race_Black', 'race_Other',
                            'race_White', 'marital_status_Divorced', 'marital_status_Married', 'marital_status_Separated', 'marital_status_Single', 
                            'marital_status_Widowed', 'relationship_Husband', 'relationship_Not-in-family', 'relationship_Other-relative', 
                            'relationship_Own-child', 'relationship_Unmarried', 'relationship_Wife', 'income'],
    'compas'            : ['race_African-American' , 'race_Caucasian', 'two_year_recid'],
    'default_credit'    : ['SEX_Female' , 'SEX_Male', 'DEFAULT_PAYEMENT'],
    'marketing'         : ['age_age:30-60' , 'age_age:not30-60', 'marital_divorced', 'marital_married', 'marital_single', 'marital_unknown', 'subscribed']
}

sensitive_attr_dict ={
    'adult_income'      : ['gender_Female', 'gender_Male', 'marital_status_Divorced', 'marital_status_Married', 'marital_status_Separated', 
                            'marital_status_Single', 'marital_status_Widowed'],
    'compas'            : ['race_African-American' , 'race_Caucasian'],
    'default_credit'    : ['SEX_Female' , 'SEX_Male'],
    'marketing'         : ['age_age:30-60' , 'age_age:not30-60', 'marital_divorced', 'marital_married', 'marital_single', 'marital_unknown']
}