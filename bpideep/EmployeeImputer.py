import re
import math
from sklearn.base import BaseEstimator, TransformerMixin


class EmployeeImputer(BaseEstimator, TransformerMixin):

    """
    Customized imputer : impute the missing values in the column "employees_latest" in 3 steps
        1. impute the missing employees number by the mean of the range if the range is indicated (column "employees")
        2. for the companies launched after 2010
            -> impute according to the median of the companies launched after 2010
        3. for the companies launched before 2010 or which do not have a launch year
            -> impute according to the median of the dataset
    """


    def average_list(self, range_list):
        """
        function that returns the mean of employees
        it is used to compute the value of the dictionary with the range as a key (column "employees")
        """
        return sum(range_list)/len(range_list)

    def compute_employees_mean(self, data):
        """
        function that creates a dictionary
        which is used for imputing missing employees number
        according the employees range of the company if it is mentioned
        column "employees"
        """

        range_list = list(data.employees.unique())
        try:
            range_list.remove("n.a.")
        except:
            pass
        keys_list = []
        means_list = []

        for i in range(len(range_list)) :
            if type(range_list[i]) == str :
                temp = re.findall(r'\d+', range_list[i])
                res = list(map(int, temp))
                mean = self.average_list(res)
                means_list.append(mean)
                keys_list.append(range_list[i])
            else:
                pass

        zip_iterator = zip(keys_list, means_list)
        range_dict = dict(zip_iterator)

        return range_dict

    def replace_employees(self, df):
        """
        function that replaces the missing employees value
        by the mean of the range indicated in the column "employees"
        """

        dictionary = self.compute_employees_mean(df)
        for key, value in dictionary.items():
            df.loc[(df.employees == key) & (df.employees_latest.isna()), "employees_clean"] = value
        return df



    def fit(self, X, y=None):

        # create the dictionary to impute the missing employees number according to the range mean
        data = X.copy()
        print(data.shape)
        data = self.replace_employees(data)

        self.yg_median = data[["employees_clean", "launch_year_clean"]].groupby(by=["launch_year_clean"]).median()
        self.years = data.launch_year_clean.unique()
        self.years = [nb for nb in self.years if nb >= 2010]

        return self

    def transform(self, X, y=None):
        # 1. impute the missing employees number by the mean of the range if the range is indicated (column "employees")
        X = self.replace_employees(X)

        # 2. for the companies launched after 2010 : impute according to the median of the companies launched after 2010
        for year in self.years:
            replace_value = self.yg_median.loc[(year)][0]
            boolean_condition = ((X.launch_year_clean == year) & (X.employees_clean.isna()))
            X.loc[boolean_condition, "employees_clean"] = replace_value

        # 3. for the companies launched before 2010 or the company that has no launch year : impute according to the median of the dataset
        median_all_dataset = X[X.launch_year_clean.notna()][["employees_clean"]].median()
        X.loc[X.employees_clean.isna(), "employees_clean"] = replace_value

        return X
