
import pandas as pd
import numpy as np
import json
import re



def impute_missing_launch_year(data):
    """
    function that replaces the missing launch year value
    for 33 companies which do not have growth_stage either
    """

    names = ['Amypore',
             'Kinnov Therapeutics',
             'Lipofabrik',
             'Step pharma',
             'LiMM Therapeutics',
             'Ilek',
             'LysPackaging',
             'TexiSense',
             "Institut de Prise en Charge de l'Obésité",
             'Izi Family',
             'Arthur Dupuy',
             'Gen.Orph',
             'Uniris',
             'NANOZ',
             'Akwatyx',
             'Black-line',
             'Eyye',
             "O'Sol",
             'Treenox',
             'Aqualeg',
             'Co-assit',
             'Wind my roof',
             'APPARTOO',
             'BimBamJob',
             'Buddytherobot.com',
             'Bcm',
             'CCI Paris Ile de France',
             'Datarocks',
             'EFFICIENCIA',
             'EONEF',
             'FEALINX',
             'INERIS',
             'Marguerite',
             'TokTokDoc',
             'Novaquark',
             'Peopeo',
             'Sloclap',
             'Swift',
             'Sword',
             'XT-VISION',
             'Ryax',
             'Sylha',
             'Opta LP']

    launch_years = [2018, 2015, 2012, 2014, 2018, 2016, 2015, 2010, 2011, 2016, 2015,
                   2012, 2017, 2012, 2015, 2017, 2016, 2016, 2018, 2011, 2016, 2018, 2015, 2015, 2014, 2014, 2013,
                   2014, 2012, 2016, 1997, 1990, 2012, 2016, 2014, 2017, 2015, 1973, 2000, 2011, 2017, 2019, 2013]

    zipbObj = zip(names, launch_years)
    launch_year_dict = dict(zipbObj)

    data["launch_year_clean"] = data['launch_year']

    for name, year in launch_year_dict.items() :
        data.loc[data.name == name, "launch_year_clean"] = year
    return data

def get_growth_dict(df):
    """
    function that creates a dictionary
    giving the mode of the growth_stage
    according to the launch_year of all companies in the dataset
    """

    table = df[["growth_stage", "launch_year_clean", "id"]].groupby(by=["launch_year_clean", "growth_stage"]).count()
    column = list(table.unstack(level = -1).columns)
    growth_dict = table.unstack(level = -1).fillna(0).apply(lambda x: column[x.argmax()][1], axis = 1).to_dict()

    return growth_dict


def fill_missing_growth(growth_dict, growth_stage, launch_year_clean):
    """
    function that fills all the missing growth stage
    according to the company's launch year
    based on the dictionary created thanks to the function get_growth_dict()
    """

    if type(growth_stage) == str:
        return growth_stage
    elif launch_year_clean in growth_dict:
        return growth_dict[launch_year_clean]
    else:
        return growth_stage


def load_json_field(bad_json):
    """
    function that transforms the type of the cells
    from string to dictionary as the pd.read_csv converts
    the dictionary to string (use it only in notebooks)
    """

    regex = r"\w\'\w"
    subst = ""
    bad_json = re.sub(regex, subst, bad_json)
    bad_json = bad_json.replace("d' Arrouzat", "darrouzat")
    good_json = bad_json.replace("\'", "\"").replace("None", "null").replace("True", "true").replace("False", "false")

    return json.loads(good_json)


def get_industries(x):
    '''
    function that extracts info from 'industries' column through mapping
    data['industries_list'] = data['industries'].map(lambda x: industries(x))
    '''
    industries_list = []

    industries = x.apply(load_json_field)

    for u in range(len(industries)):
        if len(industries[u]) > 0:
            industries_list.append(industries[u][0]['name'])
        else:
            industries_list.append("")
    return industries_list

def get_health(x):
    '''
    function that encodes :
    - 0 if "health" is not part of the industry list created by the function "get_industries"
    - 1 if "health" is part of the industry list
    '''
    industries = get_industries(x) # list of industries

    health_industry = []

    for element in industries:
        if element == 'health':
            health_industry.append(1)
        else:
            health_industry.append(0)
    return health_industry

def investors_type(x) :
    '''
    function that extracts info from 'investors' column
    '''
    investors_list = []
    investors = x
    if investors['total'] > 0 :
        for y in range(len(investors['items'])):
                investors_list.append(investors['items'][y]['type'])
    return investors_list


def fund_investors(x):
    """
    function that encodes :
    - 1 if the selected investors are part of the list created by the function investors_type()
    - 0 if not

    If needed, the selected investors can be modified. You can choose among the following :
        list_investor_type = ['fund',
                             'investor',
                             'corporate',
                             'government_nonprofit',
                             'service_provider',
                             'company',
                             'crowdfunding',
                             'workspace']
    """


    for row in range(len(x)):
        if "fund" in x["investors_type"][row] or "investors" in x["investors_type"][row] :
            x.loc[row,"investors_type"] = 1

        else :
            x.loc[row,"investors_type"] = 0
    return x

def simple_fund_investors(investors_type_list):
    '''
    returns :
    - 1 if the selected investors are part of the list created by the function investors_type()
    - 0 if not
    see fund_investors function documentation for full list of investors
    '''
    if "fund" in investors_type_list or "investors" in investors_type_list :
        return 1
    return 0


def get_clean_data():
    """
    GetData.get_data() is a function that returns a clean dataset which is then saved in the folder 'rawdata'
    (imputing missing values thanks to manual imputing, LinkedIn, and creating new features)
    Note: To use the function GetCleanData.get_clean_data(), don't forget to save the csv files
    (for the patents and LinkedIn data) in the folder "data", and replace the name of the csv
    if different from the name written in the function.
    """

    #1. Use the function Getfulldata to get an updated dataset from Dealroom
    #2. Save the updated dataset in the folder 'rawdata'
    #3. Load the dataset (please rename the csv file below)
    data = pd.read_csv("../bpideep/rawdata/data2020-12-03_with_corrections.csv")

    #4. Select only the firms labeled 'deeptech' or 'non_deeptech'
    data = data[data.deep_or_not != "almost_deeptech"]


    #5. Select the needed columns
    data = data[["id", "name", "target", "deep_or_not", "total_funding_source", "employees",
                 "employees_latest", "launch_year", "growth_stage", "linkedin_url", "industries", "investors"]]


    #6. Drop 2 duplicated companies Lalilo and Pixyl (cf. the explanation below)
    data.drop(data[(data.id == 1787891) | (data.id == 1893232)].index, inplace = True)

    #7. Harmonizing the growth stage : change the "not meaningful" growth stage status of 15789 Insoft to mature
    data.loc[data.id == 15789, "growth_stage"] = "mature"

    #8. Imputing the missing launch year (only 33 done manually -> to be automatized if a new source displays it)
    impute_missing_launch_year(data)

    #9. Imputing missing growth_stage (with the mode of the launch year)
    growth_table = get_growth_dict(data)
    data['growth_stage_imputed'] = data.apply(lambda row: fill_missing_growth(
            growth_table,
            row['growth_stage'],
            row['launch_year_clean']),
        axis=1
    )

    #10. Imputing missing employees values from LinkedIn scraping
    missing = pd.read_csv("../bpideep/data/missing_employee_count.csv")
    data["employees_clean"] = data.employees_latest
    data.loc[data.name == "CCI Paris Ile de France", "employees_clean"] = 1793

    for url in missing.linkedin_url:
        replace_value = missing[missing.linkedin_url == url]["check"].iloc[0]
        data.loc[data.linkedin_url == url, "employees_clean"] = replace_value

    #11. Computing the age of companies
    data["age"] = 2020 - data.launch_year_clean


    #12. Get the number of patents (from Google Patents)
    patent = pd.read_csv("../bpideep/data/patents.csv")
    data = pd.merge(data, patent, on="id", how = "left")

    #13. Create a new feature "investors_type"
    data["investors"] = data["investors"].apply(load_json_field)
    data["investors_type"] = pd.DataFrame(data["investors"].apply(lambda row: investors_type(row)))
    data["investors_type"] = fund_investors(data[["investors_type"]])

    #14. Create a new feature "health_industry"
    data["health_industry"] = pd.DataFrame(get_health(data["industries"]))

    #15. Get if the compagny has a doctor or no + proportion of technical among the employees (from Linkedin)
    #the column "No_people_input" tells the model whether the info was present on LinkedIn
    doctors = pd.read_csv("../bpideep/data/extra_features.csv").drop(columns = "Unnamed: 0")[["id", "company_has_phd"]]
    doctors2 = pd.read_csv("../bpideep/data/extra_features_v2.csv").drop(columns = "Unnamed: 0")[["id",
                                                                                                  "proportion_technical",
                                                                                                  "founder_from_institute",
                                                                                                  "founder_has_phd"]]

    data = data.merge(doctors, on = "id", how = "left")
    data = data.merge(doctors2, on = "id", how = "left")

    data["No_people_input"] = 0
    data.loc[data.proportion_technical.isna(), "No_people_input"] = 1


    data[["proportion_technical", "founder_from_institute","founder_has_phd"]] = data[[
        "proportion_technical", "founder_from_institute","founder_has_phd"]].fillna(0)

    return data

#         *
#                 + 3 duplicated names but with different id :
#                     1/ Lalilo : 926521 (http://www.lalilo.com/) vs. 1787891 (http://lalilo.fr)
#             -> same launch date, french website no longer exists + observation  almost filled by NAN + same obs as the .com
#             -> drop the french Lalilo (1787891)
#                     2/ Pixyl : 892048 vs 1893232 (different websites mentioned but same website page when launched)
#             -> kept  892048 because more info + the Dealroom profile was verified by Dealroom team on Sept, 1st 2020 vs. pending verification
#                     3/ NANOZ : 1836121 vs 1660543 -> kept both as different companies but the second one is German
