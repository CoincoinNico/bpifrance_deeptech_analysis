"""This module contains function used to process data extracted on companies
and employees to generate features useful to predict the start-up category."""

import glob
import pandas as pd
import os
import re

"""A few functions used in process_company_people"""

def strip_at_chez(title):
    """Function to remove the name of the company from people job title
    when it is included via 'at or 'chez' keywords"""
    if ' at ' in title:
        return title.split(' at ', 1)[0]
    elif ' chez ' in title:
        return title.split(' chez ', 1)[0]
    else:
        return title

def lower(title):
    return title.lower()

def technical_people(title):
    "a function that assign 1 to technical people and 0 to non technical"
    technical = 0
    #Technical and excluded words are used to define whether an employee is technical.
    #These lists can be adjusted.
    technical_words = ['engineer', 'scientist', 'research', 'r&d', 'phd', 'technician', 'technical',
                    'manufacturing','process','ingénieur', 'machine learning', 'deep learning'
                   "scientifique", "chercheur", "recherche et développement", "doctorant", 'cto ',
                       ' cto', 'chief technology officer', 'cso ','supply chain'
                       'Chief Scientific Officer', "technicien", "technique",
                    "data scientist",'data science' 'clinical', 'pharmacien', 'pharmacist',
                    'clinical trial', 'drug development', 'medical', 'scientific', 'professor', 'gene therapy',
                    'preclinical', 'robotics', 'technique', 'clinique', 'ph.d', 'data analyst', 'pharmacology',
                       'chemist', 'md,', 'physiologi', 'analyst', 'maître de conférences', 'professor']
    excluded_words = ['developer', 'web', 'stack', 'développeur', 'software', 'consultant', 'c++']
    for word in technical_words:
        if word in title:
            technical = 1
    for word in excluded_words:
        if word in title:
            technical = 0
    return technical


def phd(name):
    """a function that assign 1 if PhD is in profile name or in title"""
    if type(name) == str:
        if 'phd' in name.lower():
            return 1
        elif 'ph.d'in name.lower():
            return 1
        elif 'doctorant' in name.lower():
            return 1
    return 0

def founders(title):
    """a function that assign 1 to technical people and 0 to non technical"""
    founders_words = ['ceo', 'cto ', 'cso', 'founder', 'fondateur', 'chief medical officer', 'cmo']
    #for rerun -> added cmo and excluded below
    excluded_founders = ['assistant','bras droit','right hand']
    founders = 0
    for word in founders_words:
        if word in title:
            founders = 1
    for word in excluded_founders:
        if word in title:
            founders = 0
    return founders


def strip_people_from_url(url):
    return url.strip('people').strip('/')

def build_employee_df():
    """A function that opens the csv files from scraped data and generates a dataframe"""
    #loading the data, all csvs must be with same columns/headers.
    #The routine below will load all the csvs and load data into a dataframe
    # path = os.path.join(os.path.dirname(__file__),'scraping_data/companies_people/')
    path = r'../bpideep/scraping_data/companies_people/'
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df_employee_raw = pd.concat(li, axis=0, ignore_index=True)

    # optional: generate a csv from the dataframe
    #save_path = os.path.join(os.path.dirname(__file__),'scraping_data','result_files','')
    #df_employee_raw.to_csv(save_path + 'employees_raw.csv')

    return df_employee_raw


def process_employee_data(df_employee_raw):
    """A function that processes the raw employee dataframe to generate new columns.
    New columns incudes whether the employee is technical, has a phd, or is a founder."""
    path = os.path.join(os.path.dirname(__file__),'scraping_data', 'result_files', '')
    #optional: read from csv file
    # df_employee_raw = pd.read_csv(path + 'employees_raw.csv')
    df_employee = df_employee_raw.copy()

    #Remove "/people" from LinkedIn url (necessary to match linkedin urls to the initial data from dealroom)
    df_employee['company_url'] = df_employee['web-scraper-start-url'].apply(strip_people_from_url)
    df_employee.rename(columns={'company_url':'linkedin_url', 'name':'employee_name'}, inplace=True)

    #Generates technical and phd binary columns in the dataframe.
    df_employee['title'] = df_employee['title'].apply(strip_at_chez).apply(lower)
    df_employee['technical'] = df_employee['title'].apply(technical_people)
    df_employee['phd_in_name'] = df_employee['employee_name'].apply(phd)
    df_employee['phd_in_title'] = df_employee['title'].apply(phd)
    df_employee['founder'] = df_employee['title'].apply(founders)
    df_employee['phd'] = df_employee[['phd_in_name', 'phd_in_title']].max(axis=1)
    df_employee.drop(columns=['web-scraper-order','profile', 'phd_in_name','phd_in_title', 'web-scraper-start-url'], inplace=True)
    #Generate csv
    df_employee.to_csv(path + 'employees_processed.csv')
    return df_employee

def temp_process_people_data():
    """This function is only used during the LeWagon project
    to compensate for the non_matching urls between newsest dealroom data and the old file """
    path = os.path.join(os.path.dirname(__file__),'scraping_data/result_files/')
    df_people = pd.read_csv(path +'employees.csv')
    df_people['title'] = df_people['title'].apply(strip_at_chez).apply(lower)
    #Generates technical and phd binary columns in the dataframe.
    df_people['technical'] = df_people['title'].apply(technical_people)
    df_people['phd_in_name'] = df_people['employee_name'].apply(phd)
    df_people['phd'] = df_people['title'].apply(phd)
    df_people.drop(columns=['Unnamed: 0'], inplace=True)

    #Generate csv
    df_people.to_csv(path +'employee_new.csv')

    return df_people

def companies_technical_stats(df_people):
    #Group by company to generate stats
    companies_info = df_people.groupby('linkedin_url', as_index = False ).agg(
                {'technical': 'mean',
                'phd': 'sum',
                'title':'count'})\
                        .rename(columns={'title':'employee__linkedin_count','phd':'phd_found_linkedin'})
    #Generate csv
    path = os.path.join(os.path.dirname(__file__),'scraping_data/result_files/')
    companies_info.to_csv(path + 'companies_info.csv')
    return companies_info

def open_founder_profile_files():
    """A function that opens and concatenate the csv files resulting from scrapping individual founder profiles.
    The files should be stored in '/bpifrance_deeptech_analysis/scraping_data/founder_files/'"""
    # path = os.path.join(os.path.dirname(__file__),'scraping_data/founders_files/')
    #if in a notebook use instead:
    path = r'../bpideep/scraping_data/founders_files'
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df_founders_raw = pd.concat(li, axis=0, ignore_index=True)
    df_founders_raw.rename(columns={'web-scraper-start-url':'profile-href'}, inplace=True)
    return df_founders_raw

def inline_profile(df_profile):
    """This function tranforms a dataframe containing all information for one founder
    into a single line dataframe. For instance the csv from scrapping may have 8 lines
    corresponding to a single person (for instance, 3 experiences, 3 educations, 2 types).
    This function will result in a single line with all the info in columns.
    Works in tandem with build_founder_dataframe to assemble the full founder dataframe founder by founder"""
    # First, initiate a DF with only the profile url as key (profile-href)
    line_profile = df_profile.drop(df_profile.columns.difference(['profile-href']), 1)
    line_profile = line_profile.iloc[:1]
    line_profile.reset_index(inplace = True, drop = True)
    #The function iterates through the data category returned by the scrapping (company, institution and type)
    categories = ['company', 'institution','type']
    #Each type had 3 fields
    fields = {\
              'company':['title', 'company', 'exp_description'],\
              'institution':['institution', 'degree', 'field'],\
              'type':['type', 'amount', 'text_content']\
              }
    count = 0
    #This loop creates new columns corresponding and inserts data (5 columns for experience, 5 for education and 5 for types)
    # It loops through each category one by one
    for category in categories:
        subdf = df_profile.dropna(subset=[category])
        subdf.reset_index(inplace = True, drop = True)
        #Drop all the lines which correspond to other categories
        subdf.drop(subdf.columns.difference(fields[category]), 1, inplace=True)
        items = subdf.shape[0]
        # A maximum of 5 data entry are transcribed by category
        scope = min(items, 5)
        if scope > 0:
            for i in range(1, scope):
                for field in fields[category]:
                    #Create new columns and enter corresponding data
                    subdf.loc[:,f'{field}_{i+1}'] = subdf.loc[i, field]
            #Only keep the first line of the DF (which has all the info)
            first_line = subdf.iloc[:1]
            #Join the DF to the DF initialized with the profile URL
            line_profile = line_profile.join(first_line.reset_index(drop = True))
    return line_profile

def build_founders_dataframe(df_founders_raw):
    """This function rebuilds a dataframe founder by founder, iterating through
    the list of profile_url and calling the function inline_profile on each
    Warning: This function takes time to run (30s)"""
    # Initate the dataframe with the following column names
    in_line_columns = ['profile-href', 'title', 'company', 'exp_description',
       'title_2', 'company_2', 'exp_description_2', 'title_3', 'company_3',
       'exp_description_3', 'title_4', 'company_4', 'exp_description_4',
       'title_5', 'company_5', 'exp_description_5', 'institution', 'degree',
       'field', 'institution_2', 'degree_2', 'field_2', 'institution_3', 'degree_3',
        'field_3','institution_4','degree_4', 'field_4','institution_5',
       'degree_5', 'field_5', 'type', 'amount', 'text_content', 'type_2',
       'amount_2', 'text_content_2','type_3','amount_3', 'text_content_3','type_4',
       'amount_4', 'text_content_4','type_5','amount_5', 'text_content_5']
    df_founders = pd.DataFrame(columns = in_line_columns)

    # Generate the list of profile urls
    urls = df_founders_raw['profile-href'].unique()

    # Loop through the list, select the lines corresponding to the profile url
    # and then apply the inline_profile function to assemble all founder data in one line;
    for i in range(len(urls)):
        profile = df_founders_raw[df_founders_raw['profile-href'] == urls[i]]
        df_inline_profile = inline_profile(profile)
        # Finally concatenate with the founder dataframe, line by line
        df_founders = pd.concat([df_founders, df_inline_profile], axis = 0)
    df_founders.reset_index(inplace = True, drop = True)
    return df_founders



def founder_from_institutes(row):
    """This function determines whether a founder has worked in a research institute such
    as the CNRS, CEA, etc...
    Enables a mapping to be applied on 'df_founders':
    df_founders['founder_from_institute'] = df_founders['profile-href'].apply(founder_from_institutes)
    Returns 1 or 0."""

    #Initiate search lists (the fields in which the function will look).
    #This fields are columns of the df_founders dataframe.
    companies= ['company']
    for i in range(2,6):
        companies.append(f"company_{i}")
    titles= ['title']
    for i in range(2,6):
        companies.append(f"title_{i}")
    #lists of words used to determine wheter a founder worked in a research institute.
    # A SATT is a 'société de transfert de technologie'. Their objective is to help start
    #companies from research in academic lab.
    acronyms = ['CEA', 'CNRS', 'INSERM', 'UMR', 'INRA', 'LETI', 'SATT', 'ITE', 'INRIA']
    institute_words = ['centre national de la recherche scientifique',\
                    'commissariat', 'hospitalier', 'Institut national de la recherche agronomique']
    #A founder having done a postdoc can be a strong indicator of a deeptech, so search for that.
    title_words = ['postdoc', 'post doc']
    count = 0
    #first look for presence of acronyms or institute words in founder experience (commpanies).
    for company in companies:
        text = row[company]
        for word in acronyms:
            if re.search(r'\b' + word + r'\b', str(text)):
                count = 1
        for word in institute_words:
            if word in str(text).lower():
                count = 1
    #The look for post_doc in titles
    for title in titles:
        text = row[title]
        for word in title_words:
            if word in str(text).lower():
                count = 1
    return count



def founder_has_phd(row):
    """This function determines whether a founder has a PhD degree
    Enables a mapping to be applied on 'df_founders':
    df_founders['founder_has_phd'] = df_founders['profile-href'].apply(founder_has_phd)
    Returns 1 or 0."""

    #Words used to search for phd
    degree_words = ['phd', 'ph.d', 'doctorat', 'doctor', 'philosophy']

    #Make a list of fields to search for degrees
    degrees = ['degree']
    for i in range(2,6):
        degrees.append(f"degree_{i}")

    #Make a list of of words and fields to search for job titles
    #(in case the PhD is indicated in experience and not in education)
    title_words = ['doctorant', 'phd', 'ph.d']
    titles= ['title']
    for i in range(2,6):
        titles.append(f"title_{i}")

    count = 0
    # first look in education fields (degree)
    for degree in degrees:
        text =row[degree]
        for word in degree_words:
            if word in str(text).lower():
                count = 1
    #Then look in experience fields (title)
    for title in titles:
        text = row[title]
        for word in title_words:
            if word in str(text).lower():
                count = 1
    return count


def founder_has_patents_publi(row):
    """This function determines whether has patents or public
    Enables a mapping to be applied on 'df_founders':
    df_founders['founder_has_pat_pub'] = df_founders['profile-href'].apply(founder_has_pat_pub)
    Returns 1 or 0."""
    #Make a list of fields to search for patents and publications (types)
    types= ['type']
    for i in range(2,6):
        types.append(f"type_{i}")
    #key words used in profiles.
    intel =['Patent','Publications']
    count = 0
    for item in types:
        text = row[item]
        for word in intel:
            if word == str(text):
                count = 1
    return count

def technical_founders(row):
    """A function that determines whether a founder is technical.
    This is then used to update the % of technical employees in the company
    dataframe (it can be very important for small companies where + or - 1
    technical employee can strongly impact the % of technical"""
    # Then return 1 of the founder worked in academia or has a phd.
    if row['founder_from_institute'] > 0:
        return 1
    elif row['founder_has_phd'] > 0:
        return 1
    else :
        return 0

def generate_founders_features(df_founders):
    """Function that build new features by mapping feature generating functions on the dataframe"""
    df_founders['founder_has_phd'] = df_founders.apply(founder_has_phd, axis=1)
    df_founders['founder_from_institute'] = df_founders.apply(founder_from_institutes, axis=1)
    df_founders['founder_pat_pub'] = df_founders.apply(founder_has_patents_publi, axis=1)
    df_founders['technical_founder'] = df_founders.apply(technical_founders, axis=1)
    return df_founders


def update_technical(df_employees, df_founders):
    """""A function that updates the feature "technical" of the full employee dataframe
    for founders so that technical founders are marked as a 1.
    This is then used to update the % of technical employees in the company
    dataframe (it can be very important for small companies where + or - 1
    technical employee can strongly impact the % of technical."""

    #new_features['No_people_input'] = new_features['id'].apply(nan_indicator)
    selection = ['profile-href','founder_from_institute',
       'founder_has_phd', 'founder_pat_pub', 'technical_founder']
    df_employees_full = df_employees.merge(df_founders[selection], how='left', on='profile-href')

    #The 'technical' feature is updated with a 1 if either the technical or technical
    #founder features are equal to 1:
    df_employees_full['technical'] = df_employees_full[["technical", 'technical_founder']].max(axis=1)

    return df_employees_full

def companies_technical_stats_with_founders_features(df_employees_full):
    #Group by company to generate stats
    df_companies_stats_with_founders_features = df_employees_full.groupby('linkedin_url', as_index = False ).agg(
                {'technical': 'mean',
                'phd': 'sum',
                'title':'count',
                'founder_from_institute':'sum',
                'founder_has_phd':'sum',
                'founder_pat_pub':'sum',
                'technical_founder':'sum',
                }).rename(columns={'title':'employee__linkedin_count','phd':'phd_found_linkedin'})
    return df_companies_stats_with_founders_features

def merge_initial_companies_with_founder(deal_room_df, df_companies_stats_with_founders_features):
    """"A function that creates and save a dataframe with all features from dealroom
    and from employee profile scraping."""
    df_companies_with_employee_features = deal_room_df.merge(df_companies_stats_with_founders_features, on='linkedin_url', how='left')
    # If used for a jupyter notebook, use "path = r'../bpideep/scraping_data/result_files/'""
    path = os.path.join(os.path.dirname(__file__),'scraping_data/result_files/')
    df_companies_with_employee_features.to_csv(path + 'companies_with_employee_features.csv')
    return df_companies_with_employee_features



