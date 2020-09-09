# here we create:
    # - API calls functions (possibly several depending on needs)
    # - each function returns a df

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv

import os



def company_tolist(id_csv_file):
    """takes the csv name of deeptech id list (as 'file_name.csv') \
    and returns the flatten python list"""

    assert '.csv' in id_csv_file

    # stores the csv path inside csv_file_path
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data/', id_csv_file)

    # csv_path = os.path.join(os.path.dirname(__file__), "data/", csv)
    company = pd.read_csv(csv_file_path)
    # stores the list of list of companies id inside deeptech
    company = company.values.astype(str).tolist()

    # flattens the list of list and stores it as a simple python list
    company_list = [item for sublist in company for item in sublist]

    return company_list



def fields_tolist(fields_txt_file):
    """takes the txt file storing fields and returns a python list with fields"""

    assert '.txt' in fields_txt_file

    # stores the txt path inside txt_file_path
    txt_file_path = os.path.join(os.path.dirname(__file__), 'data/', fields_txt_file)

    file = open(txt_file_path, 'r')
    l = file.read()
    l = l.split(',')
    l[-1] = l[-1].replace('\n', '')

    return l



def getbatchdata(company_id_list, fields_list):
    """takes a company_id_list and a fields parameter, \
    which is a list of fields for the Dealroom API

    returns a pandas dataframe with each row corresponding to a company \
    and the columns corresponding to the fields list

    this function is called in getfulldata"""

    assert isinstance(company_id_list, list)

    # stores env_path for api key in .env file
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path = env_path)
    APIKEY = os.getenv('DEALROOMAPIKEY')
    URL = 'https://api.dealroom.co/api/v1'

    # converts the company_id_list and fields_list to strings \
    # of company ids and fields separated by commas
    companies_string = ','.join(company_id_list)
    fields_string = ','.join(fields_list)

    # performs get request on dealroom api
    response = requests.get(
                        url = f'{URL}/companies/batch?ids={companies_string}&fields={fields_string}',\
                        auth = (APIKEY, '')
                        )

    data = response.json()['items']

    return pd.DataFrame(data)



def getfulldata(company_csv_file, fields_txt_file):
    """the only parameter needed is the deeptech.csv name of the companies id csv \
    the function calls deeptech_tolist to the csv file in order to retrieve the entire python list of companies id \
    then the function calls getbatchdata on each 50id batches from company id list"""

    # storing the companies id in a list
    id_list = company_tolist(company_csv_file)

    # storing the fields in a list
    fields_list = fields_tolist(fields_txt_file)

    # instantiating data with the first 50 companies from id_list
    first_batch = id_list[0:50]
    data = getbatchdata(first_batch, fields_list)

    for i in np.arange(50, len(id_list), 50):
        if i <= 400:
            i_batch = id_list[i:i+50]
            data_i = getbatchdata(i_batch, fields_list)
            data = pd.concat([data, data_i], axis=0, ignore_index=True)
        elif i == 450:
            i_batch = id_list[450:len(id_list)]
            data_i = getbatchdata(i_batch, fields_list)
            data = pd.concat([data, data_i], axis=0, ignore_index=True)

    output_path = os.path.join(os.path.dirname(__file__), "rawdata")
    company_type = company_csv_file.replace('.csv', '')
    data.to_csv(f'{output_path}/{company_type}_df.csv')

    return data


def bulk_search(**kwargs):
    '''Bulk search is for searching multiple company by keywords in the name or the website'''

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path = env_path)
    APIKEY = os.getenv('DEALROOMAPIKEY')
    URL = 'https://api.dealroom.co/api/v1/companies/bulk'

    response = requests.post(
                        url = URL,\
                        auth = (APIKEY, ''),\
                        data = kwargs)


    try :
        data = response.json()['items']
    except:
        data = response.json()
        return data
    return pd.DataFrame(data)


def get_df(deep_csv, nondeep_csv, almostdeep_csv):
    """ takes the three csv files names as arguments and concat the df, \
    returns a df
    adds a 'deep_or_not' column
    adds a 0 or 1 column (1 for deep, 0 for nondeep or almost deep) """

    # stores the path of each csv file in a variable
    deep_path = os.path.join(os.path.dirname(__file__), 'rawdata/', deep_csv)
    nondeep_path = os.path.join(os.path.dirname(__file__), 'rawdata/', nondeep_csv)
    almostdeep_path = os.path.join(os.path.dirname(__file__), 'rawdata/', almostdeep_csv)

    deep = pd.read_csv(deep_path)
    nondeep = pd.read_csv(nondeep_path)
    almostdeep = pd.read_csv(almostdeep_path)

    # creating the 'deep_or_not' column
    deep['deep_or_not'] = 'deeptech'
    nondeep['deep_or_not'] = 'non_deeptech'
    almostdeep['deep_or_not'] = 'almost_deeptech'

    # creating the 'target' column
    deep['target'] = 1
    nondeep['target'] = 0
    almostdeep['target'] = 0

    if ((deep.columns != nondeep.columns).sum() == 0) & ((deep.columns != almostdeep.columns).sum() == 0):
        data = pd.concat([deep, nondeep, almostdeep], axis = 0, ignore_index = True)

    output_path = os.path.join(os.path.dirname(__file__), "rawdata")
    data.to_csv(f'{output_path}/complete_df.csv')

    return data


if __name__ == "__main__":

    import sys

    first_arg = sys.argv[1]
    second_arg = sys.argv[2]
    third_arg = sys.argv[3]

    # data = getfulldata(first_arg, second_arg)
    data_df = get_df(first_arg, second_arg, third_arg)
