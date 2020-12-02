# here we create:
    # - API calls functions (possibly several depending on needs)
    # - each function returns a df

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv, find_dotenv

import os
import json

import re


def company_tolist(id_csv_file):
    """
    takes the csv name of deeptech id list (as 'file_name.csv') \
    and returns the flatten python list
    """

    assert '.csv' in id_csv_file

    # stores the csv path inside csv_file_path
    csv_file_path = os.path.join(os.path.dirname(__file__), 'data/', id_csv_file)

    # csv_path = os.path.join(os.path.dirname(__file__), "data/", csv)
    company = pd.read_csv(csv_file_path)
    # stores the list of list of companies id inside deeptech
    company = company.values.astype(str).tolist()

    # flattens the list of list and stores it as a simple python list
    company_list = [item for sublist in company for item in sublist]

    # checks that we drop the 'id' header
    if company_list[0] == 'id':
        return company_list[1:]

    return company_list



def fields_tolist(fields_txt_file):
    """
    takes the txt file storing fields and returns a python list with fields
    """

    assert '.txt' in fields_txt_file

    # stores the txt path inside txt_file_path
    txt_file_path = os.path.join(os.path.dirname(__file__), 'data/', fields_txt_file)

    file = open(txt_file_path, 'r')
    l = file.read()
    l = l.split(',')
    l[-1] = l[-1].replace('\n', '')

    return l




def getjson(deeptech_id_csv, non_deeptech_id_csv, almost_deeptech_id_csv):
    """
    transforms the three company id csv files into a json file that is stored in data
    returns the companies dict
    """

    deeptech_list = company_tolist(deeptech_id_csv)
    non_deeptech_list = company_tolist(non_deeptech_id_csv)
    almost_deeptech_list = company_tolist(almost_deeptech_id_csv)

    companies = {
        'deeptech': deeptech_list,
        'non_deeptech': non_deeptech_list,
        'almost_deeptech': almost_deeptech_list
    }

    json_path = os.path.join(os.path.dirname(__file__), 'data/', 'companies.json')
    with open(json_path, 'w') as f:
        json.dump(companies, f)

    return companies




def getbatchdata(company_id_list, fields_list):
    """
    takes a company_id_list and a fields parameter, \
    which is a list of fields for the Dealroom API
    returns a pandas dataframe with each row corresponding to a company \
    and the columns corresponding to the fields list

    this function is called in getfulldata
    """

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

    try :
        data = response.json()['items']
    except:
        message = response.json()['message']
        problem_id = re.search(r"\d+", message).group(0)
        print(f'id {problem_id} could not be retrieved, relaunching batch without it')
        company_id_list.remove(problem_id)
        return getbatchdata(company_id_list, fields_list)

    return pd.DataFrame(data)



def getfulldata(company_dict, fields_txt_file):
    """
    takes the json object generated with getjson, and the fields_txt_file as parameters \
    makes three API calls with companies id
    """

    # storing the fields in a list
    fields_list = fields_tolist(fields_txt_file)

    # instantiating empty dataframes to store API calls results
    deep_df = pd.DataFrame(columns = fields_list)
    nondeep_df = pd.DataFrame(columns = fields_list)
    almostdeep_df = pd.DataFrame(columns = fields_list)

    df_dict = {'deeptech': deep_df, \
                'non_deeptech':nondeep_df, \
                'almost_deeptech':almostdeep_df}

    for company_type, id_list in company_dict.items():

        chunks = (len(id_list) - 1) // 50 + 1
        for i in range(chunks):
            batch = id_list[i*50:(i+1)*50]
            data_i = getbatchdata(batch, fields_list)
            df_dict[company_type] = pd.concat([df_dict[company_type], data_i], \
                                                axis=0, sort = False)

    deep_df = df_dict['deeptech']
    nondeep_df = df_dict['non_deeptech']
    almostdeep_df = df_dict['almost_deeptech']

    # creating the 'deep_or_not' column
    deep_df['deep_or_not'] = 'deeptech'
    nondeep_df['deep_or_not'] = 'non_deeptech'
    almostdeep_df['deep_or_not'] = 'almost_deeptech'

    # creating the 'target' column
    deep_df['target'] = 1
    nondeep_df['target'] = 0
    almostdeep_df['target'] = 0

    # concatenates the three dataframes
    data = pd.concat([deep_df, nondeep_df, almostdeep_df], axis = 0, ignore_index = True)

    # drop duplicates
    data.drop_duplicates(subset = 'id', inplace = True)
    data.reset_index(drop = True, inplace = True)

    X = data.drop(columns = 'target')
    y = data['target']

    output_path = os.path.join(os.path.dirname(__file__), "rawdata")
    data.to_csv(f'{output_path}/data.csv', index = False)

    return X, y

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

def company_search(name):
    # if local
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path = env_path)
    APIKEY = os.getenv('DEALROOMAPIKEY')

    URL = 'https://api.dealroom.co/api/v1/companies'
    fields_list = fields_tolist('fields_list.txt')
    fields_string = ','.join(fields_list)

    response = requests.post(
                        url = URL,\
                        auth = (APIKEY, ''),\
                        data = {'keyword':name, 'keyword_type':"name", 'keyword_match_type':"exact", 'fields': fields_string})

    try :
        data = response.json()['items']
        company = pd.DataFrame(data).head(1)
    except:
        company = response.json()

    return company

def company_search_fuzzy(name):
    # if local
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path = env_path)
    APIKEY = os.getenv('DEALROOMAPIKEY')

    URL = 'https://api.dealroom.co/api/v1/companies'
    fields_list = fields_tolist('fields_list.txt')
    fields_string = ','.join(fields_list)

    response = requests.post(
                        url = URL,\
                        auth = (APIKEY, ''),\
                        data = {'keyword':name, 'keyword_type':"name", 'keyword_match_type':"fuzzy", 'fields': fields_string})

    try :
        data = response.json()['items']
        company = pd.DataFrame(data).head(1)
    except:
        company = response.json()

    return company

# def bulk_search(**kwargs):
#     '''Bulk search is for searching multiple company by keywords in the name or the website'''
#     env_path = os.path.join(os.path.dirname(__file__), ".env")
#     load_dotenv(dotenv_path = env_path)
#     APIKEY = os.getenv('DEALROOMAPIKEY')
#     URL = 'https://api.dealroom.co/api/v1/companies/bulk'

#     response = requests.post( url = URL,auth = (APIKEY, ''),data = kwargs, headers= {"Content-Type": "application/json"} )

#     try :
#         data = response.json()['items']
#     except:
#         data = response.json()
#         return data
#     return pd.DataFrame(data)

def bulk_search(month, year):
    '''Bulk search is for searching multiple company by keywords in the name or the website'''
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path = env_path)
    APIKEY = os.getenv('DEALROOMAPIKEY')
    URL = 'https://api.dealroom.co/api/v1/companies/bulk'

    response =  requests.post(
                    url="https://api.dealroom.co/api/v1/companies/bulk",
                    data=json.dumps({
                    "form_data": {
                    "must": {"last_round_year": f'{year}',
                             "last_round_month": f'{month}',
                        "hq_locations": "France"}
                    },
                    "limit": 100,
                    "next_page_id": "",
                    "fields": "id,name,fundings,launch_year"
                }),
                    headers={
                      "Content-Type": "application/json"
                    },
                    auth=('ed9e55daea6d7c22647f70037e24ebaed82fff19', '')
                )

    try :
        data = response.json()['items']
    except:
        data = response.json()
        return data
    return pd.DataFrame(data)


if __name__ == "__main__":

    import sys


    first_arg = sys.argv[1]
    second_arg = sys.argv[2]
    third_arg = sys.argv[3]
    fourth_arg = sys.argv[4]

    """
    first_arg = 'deeptech.csv'
    second_arg = 'non_deeptech.csv'
    third_arg = 'almost_deeptech.csv'
    fourth_arg = 'fields_list.txt'
    """

    company_dict = getjson(first_arg, second_arg, third_arg)
    data = getfulldata(company_dict, fourth_arg)


