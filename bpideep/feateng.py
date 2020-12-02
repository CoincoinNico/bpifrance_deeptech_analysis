import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import numpy as np
from bpideep.list import list_industries,list_technologies,list_tags,list_background_team,list_degree_team,list_income_streams,list_investors_name,list_investor_type

data_path = os.path.join(os.path.dirname(__file__), "data")
patents_df = pd.read_csv(f"{data_path}/patents.csv")
IDZIP_DF = pd.read_csv(f"{data_path}/id_zip.csv", delimiter = ';')

KEPT_TAGS = [
            'technical_background',
             'health_industry',
             'semiconductors_industry',
             'energy_industry',
             'commission_income_streams',
             'biotechnology_tags',
             'neurology_tags',
             'saas_tags',
             'fund_investors_type',
             'Agoranov_investors_name']

TARGET_ZIP = [91, 38, 87, 35, 67]

@simple_time_tracker
def return_list(data, column):
    list_ = []
    data_ = data[column]
    for row in data_:
        for elt in row:
            elt_tagged = f"{elt}_{column}"
            if elt_tagged not in list_:
                list_.append(elt_tagged)
    return list_


@simple_time_tracker
def encoder(data, column):
    '''
    encoder function that takes a pandas dataframe (data) \
    and a column name (str) as parameters
    returns a new_df with one hot encoded columns
    '''
    list_ = return_list(data, column)
    # if the list_ is empty, return an empty dataframe
    if len(list_) == 0:
        return pd.DataFrame()
    data_encoded = pd.DataFrame(columns = list_)
    for index, row in data.iterrows():
        data_ = row[column]
        dict_ = {}
        for tag in list_:
            tag_strip = tag.strip(f"_{column}")
            if tag_strip in data_:
                encoder = 1
            else:
                encoder = 0
            dict_[tag] = encoder
        data_encoded.loc[index] = dict_
    return data_encoded



@simple_time_tracker
def return_filling(data, column):
    '''
    function that returns features 'presence' for each type \
    (deeptech, non-deeptech, all companies) in data dataframe
    '''

    def return_value(x) :
        if type(x) == str :
            if ast.literal_eval(x):
                return 1
            else :
                return 0
        if type(x) == list :
            if x:
                return 1
            else :
                return 0
    data['return_filling'] = data[column].map(lambda x : return_value(x))
    print(f'deeptech : {data[data["target"]==1]["return_filling"].value_counts()[1]/len(data[data["target"]==1])}')
    print(f'non deeptech : {data[data["target"]==0]["return_filling"].value_counts()[1]/len(data[data["target"]==0])}')
    print(f'total : {data["return_filling"].value_counts()[1] / len(data)}')


@simple_time_tracker
def background(x):
    '''
    function that extracts info from 'team' column
    enables a mapping to create a new feature:
    data['background_team'] = data['team'].map(lambda x:background(x))
    '''
    backgrounds_list = []
    team = x
    for team_member in team['items']:
        backgrounds = team_member['backgrounds']
        for tags in backgrounds:
            backgrounds_list.append(tags['name'])
    return backgrounds_list



@simple_time_tracker
def degree(x):
    '''
    function that extracts the degree name from the 'team' column in data
    enables a mapping to be applied on 'team':
    data['degree_team'] = data['team'].map(lambda x:degree(x))
    '''
    degree_list = []
    team = x
    for team_member in team['items']:
            universities = team_member['universities']['items']
            if universities and universities[0]['degree'] is not None :
                degree = universities[0]['degree']['name']
                degree_list.append(degree)
    return degree_list




@simple_time_tracker
def degree_quant(x):
    '''
    function that encodes whether or not a doctor works within the company,
    works in pair with the function degree
    enables a mapping to be applied on 'degree_team':
    data['doctor'] = data['degree_team'].map(lambda x: degree_quant(x))
    '''

    if len(x) == 0:
        return 0
    else :
        for n in range(len(x)):
            if x[n] in ['Doctor','PhD'] :
                return 1
            else :
                return 0



@simple_time_tracker
def funding_amounts_employees(data):
    '''
    function that performs the ratio between funding and employees
    '''

    funding = data['total_funding_source']
    employees = data['employees_latest']

    return funding/employees




@simple_time_tracker
def growth_stage_num(data):
    '''
    function that maps the 'growth stage' as an ordinal feature
    '''

    stage_status = data['growth_stage'].map({'mature' : 4,
                                            'late growth' : 3,
                                            'early growth' : 2,
                                            'seed' : 1})
    return stage_status



@simple_time_tracker
def industries(x):
    '''
    function that extracts info from 'industries' column through mapping
    data['industries_list'] = data['industries'].map(lambda x: industries(x))
    '''

    industries_list = []
    industries = x
    for u in range(len(industries)):
            industries_list.append(industries[u]['name'])
    return industries_list




def investors_name(x) :
    investors_list = []
    investors = x
    if investors['total'] > 0 :
        for y in range(len(investors['items'])):
                investors_list.append(investors['items'][y]['name'])
    return investors_list




def investors_type(x) :
    investors_list = []
    investors = x
    if investors['total'] > 0 :
        for y in range(len(investors['items'])):
                investors_list.append(investors['items'][y]['type'])
    return investors_list





@simple_time_tracker
def tags_reduction(encoded_dataframe, threshold = 0.02):
    '''
    function that performs a dimension reduction operation: \
    selects only tags that appear above a certain threshold (ex: 0.02)
    to be applied on the encoded tags dataframe
    '''

    tags_series_count = encoded_dataframe[encoded_dataframe == 1].sum()\
    .sort_values(ascending = False)/encoded_dataframe.shape[0]
    retained_tags = tags_series_count[tags_series_count > threshold].index.tolist()

    return encoded_dataframe[retained_tags]




def substract_date(x):
    return 2020 - x




@simple_time_tracker
def return_ratio(x):
    '''
    function that computes the growth stage over years of existance ratio
    '''

    if x['year_of_existence'] >0:
        return x['growth_stage_num']/x['year_of_existence']
    return x['growth_stage_num']




@simple_time_tracker
def feat_eng_cols(data):
    '''
    takes a pandas df as input
    global feature engineering function that performs all above mentioned \
    transformations and returns a new dataframe
    '''

    # new features in data as columns
    data['background'] = data['team'].map(lambda x:background(x))
    data['degree'] = data['team'].map(lambda x:degree(x))
    data['doctor_yesno'] = data['degree'].map(lambda x: degree_quant(x))
    data['funding_employees_ratio'] = funding_amounts_employees(data)
    data['has_strong_founder'] = data['has_strong_founder'].map({True : 1,
                                                                False : 0})
    data['has_super_founder'] = data['has_super_founder'].map({True : 1,
                                                               False : 0})
    data['growth_stage_num'] = growth_stage_num(data)
    data['industry'] = data['industries'].map(lambda x: industries(x))

    data['year'] = pd.DataFrame({'year': data['launch_year']})
    data['year_of_existence'] = data['year'].map(lambda x : substract_date(x))
    data['stage_age_ratio'] = data[['year_of_existence','growth_stage_num']]\
                                .apply(return_ratio,axis=1)
    data['investors_name'] = data['investors'].map(lambda x:investors_name(x))
    data['investors_type'] = data['investors'].map(lambda x:investors_type(x))


    # encoded features
    background_team_encoded_df = encoder(data, 'background')
    degree_team_encoded_df = encoder(data,'degree')
    industries_encoded_df = encoder(data,'industry')
    income_streams_encoded_df = encoder(data,'income_streams')
    technologies_encoded_df = encoder(data,'technologies')
    tags_encoded_df = encoder(data,'tags')
    investors_name_encoded_df = encoder(data, 'investors_name')
    investors_type_encoded_df = encoder(data, 'investors_type')

    # processed encoded features
    # tags_retained = tags_reduction(tags_encoded_df, threshold = 0)
    # background_retained = tags_reduction(background_team_encoded_df, threshold = 0)
    # industries_retained = tags_reduction(industries_encoded_df, threshold = 0)
    # investors_name_retained = tags_reduction(investors_name_encoded_df, threshold = 0)

    # to concat
    concat_df = pd.concat([
                        data[['id',
                            'doctor_yesno',
                            'funding_employees_ratio',
                            'has_strong_founder',
                            'has_super_founder',
                            'stage_age_ratio'
                            ]],
                        tags_encoded_df,
                        background_team_encoded_df,
                        industries_encoded_df,
                        degree_team_encoded_df,
                        income_streams_encoded_df,
                        technologies_encoded_df,
                        investors_name_encoded_df,
                        investors_type_encoded_df
                        ], axis = 1)

    # merge concat_df with patents to get patents info
    concat_df = concat_df.merge(patents_df[['nb_patents', 'id']], on = 'id', how = 'left')

    # we keep a trace of all features list (by group of features)
    simple_features = ['doctor_yesno',
                        'funding_employees_ratio',
                        'has_strong_founder',
                        'has_super_founder',
                        'stage_age_ratio',
                        'nb_patents']


    # selection of columns to keep
    kept_tags = KEPT_TAGS

    # necessary check that all kept_tags are in concat_df columns:
    # useful in case new entry data is given and some tags of industries differ
    no_tags = [col for col in kept_tags if col not in concat_df.columns]
    if len(no_tags) != 0:
        for col in no_tags:
            concat_df[col] = 0

    kept_cols = simple_features + kept_tags

    return concat_df[kept_cols], kept_cols



def get_kept_cols(data):
    df, kept_cols = feat_eng_cols(data)
    return kept_cols



def feat_eng(data):
    concat_df, kept_cols = feat_eng_cols(data)
    return concat_df

def get_stage_age_ratio(data):
    data['year'] = pd.DataFrame({'year': data['launch_year']})
    data['year_of_existence'] = data['year'].map(lambda x : substract_date(x))
    data['growth_stage_num'] = growth_stage_num(data)
    data['stage_age_ratio'] = data[['year_of_existence','growth_stage_num']]\
                                .apply(return_ratio,axis=1)
    return data['stage_age_ratio']


@simple_time_tracker
def zip_code(data):
    # new features in data as columns
    # import ipdb; ipdb.set_trace()
    data['degree'] = data['team'].map(lambda x:degree(x))
    data['doctor_yesno'] = data['degree'].map(lambda x: degree_quant(x))

    # merge concat_df with patents to get patents info
    if 'nb_patents' not in data.columns.tolist():
        data = data.merge(patents_df[['nb_patents', 'id']], on = 'id', how = 'left')

    simple_features = ['id',
                        'hq_locations',
                        'doctor_yesno',
                        'nb_patents']

    data = data[simple_features]

    idzip_df = IDZIP_DF.copy()
    idzip_df.set_index('id', inplace = True)
    hq_locations = data[['id', 'hq_locations']]
    def getzip(elt):
        if len(elt) == 0:
            return None
        else:
            return elt[0]
    hq_locations['hq_locations'] = hq_locations['hq_locations'].apply(lambda x: getzip(x))
    hq_locations = hq_locations.dropna(axis=0, subset=['hq_locations'])
    hq_df = pd.DataFrame(hq_locations['hq_locations'].to_list(), index=hq_locations['id'])
    hq_df = hq_df[['zip']]
    merged = hq_df.join(idzip_df).fillna(value = -1000)
    merged_clean = merged
    def convert(string):
        try:
            int(string)
            n = string[0:2]
            n = int(n)
        except:
            n = -1000
        return n
    merged_clean.zip = merged_clean.zip.apply(lambda x : convert(x))
    merged_clean.ZIP = merged_clean.ZIP.apply(lambda x : int(str(x)[0:2] if x != 0 else 0))
    merged_clean['zip_code'] = merged_clean.apply(max, axis = 1)
    final = merged_clean.drop(columns=['zip', 'ZIP'])
    df = data.set_index('id').join(final, how = 'left').fillna(value = -1)
    df['zip_code'] = df['zip_code'].astype('int')
    df.reset_index(inplace = True, drop = True)
    df['department'] = df['zip_code'].apply(lambda x: 1 if (x in TARGET_ZIP) else 0)
    df.drop(columns = ['zip_code'], inplace = True)

    return df





