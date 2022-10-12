import argparse
import os

from adult_income import get_adult_income
from compas import get_compas
from default_credit import get_default_credit
from marketing import get_marketing

# parser initialization
parser = argparse.ArgumentParser(description='Script for datasets preprocessing')

parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
args = parser.parse_args()


dataset = args.dataset 


if dataset == 'adult_income':
    get_adult_income(save_df=True)

elif dataset == 'compas':
    get_compas(save_df=True)
    
elif dataset == 'default_credit':
    get_default_credit(save_df=True)

elif dataset == 'marketing':
    get_marketing(save_df=True)

else:
    print('This dataset is not handled for the moment')

