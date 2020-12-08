"""This module contains function used to process data exctracted on companies 
and employees to generate features useful to predict the start-up category."""

import glob

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
    technical_words = ['engineer', 'scientist', 'researcher', 'r&d', 'phd', 'technician', 'technical',
                    'manufacturing','process','ingénieur', 
                   "scientifique", "chercheur", "recherche et développement", "doctorant", 'cto ', 'cso ',
                   "technicien", "technique"]
    excluded_words = ['developer', 'web', 'stack', 'développeur', 'software', 'consultant']
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
    return 0

def strip_people_from_url(url):
    return url.strip('people').strip('/')
    
def process_company_people():
    #loading the data, all csvs must be with same columns/headers.
    #The routine below will load all the csvs and load data into a dataframe
    path = r'../bpideep/scraping_data/companies_people/' 
    all_files = glob.glob(path + "/*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)
    df_people = pd.concat(li, axis=0, ignore_index=True)
    df_people['title'] = df_people['title'].apply(strip_at_chez)
    df_people['title'] = df_people['title'].apply(lower)
    df_people['company_url'] = df_people['web-scraper-start-url'].apply(strip_people_from_url)
    df_people.drop(columns=['web-scraper-order','profile', 'web-scraper-start-url'], inplace=True)

    #Generates technical and phd binary columns in the dataframe.
    df_people['technical'] = df_people['title'].apply(technical_people)
    df_people['phd'] = df_people['name'].apply(phd)
    df_people['phd'] = df_people['title'].apply(phd)

    #Generate csv
    df_people.to_csv('../bpideep/scraping_data/result_files/companies_people.csv')
    
    return df_people

def companies_technical_stats():
    df_people = process_company_people()
    #Group by company to generate stats
    companies_info = df_people.groupby('company_url').agg(
                {'technical': 'mean',
                'phd': 'sum',
                'company_url':'count'})
    companies_info.rename(columns={'company_url':'count_technical'}, inplace = True)
        #Generate csv
    companies_stats.to_csv('../bpideep/scraping_data/result_files/companies_stats.csv')
    return companies_info


def make_technical_profile_lists(X):
    """Function that generates batches of 80 profile URLs for scrapping via phantombuster"""
    df_technical = X[X['technical']==1]
    df_technical = df_technical[~df_technical['profile-href'].isnull()]
    df_technical.reset_index(inplace = True)
    df_technical.drop(columns=['index'], inplace=True)
    counts = df_technical.shape[0]
    batches = int(counts /80)
    list_of_lists = []
    profile_url_list = df_technical['profile-href']
    for i in range(0, batches+1):
        path = f"../bpideep/scraping_data/scraping_lists_phantom/"
        if (i+1)*80>counts:
            #this route handles the end of the table
            #First generate list of profile urls 
            profile = profile_url_list[i*80:]
            profile.to_csv(path + f"profile_list_{i}.csv" , index = False, header=False)
            #Then save tuples with profile and company url to use as a key if needed.
            tupple_list=[]
            for j in range(counts - i*80):
                url_tuple = (df_technical.iloc[i*80+j,2], df_technical.iloc[i*80+j,3])
                tupple_list.append(url_tuple)
            list_of_lists.append(url_tuple)
            break
        else:
            profile = profile_url_list[i*80:(i+1)*80]
            profile.to_csv(path + f"profile_list_{i}.csv", index = False, header=False)
            tupple_list=[]
            for j in range(80):
                url_tuple = (df_technical.iloc[i*80+j,2], df_technical.iloc[i*80+j,3])
                tupple_list.append(url_tuple)
            list_of_lists.append(tupple_list)    
    return list_of_lists
