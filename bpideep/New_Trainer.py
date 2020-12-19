import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.compose import ColumnTransformer
from bpideep.EmployeeImputer import EmployeeImputer
from sklearn.impute import SimpleImputer
from bpideep.NoTransformer import NoTransformer
from sklearn.ensemble import RandomForestClassifier
from bpideep.GetCleanData import get_clean_data
import joblib
# import pickle
import dill as pickle

def compute_growth_age_ratio(arr):
    arr[arr[:,0]== 0, 0] = 1
    return np.expand_dims(arr[:,1]/arr[:,0], -1)

def set_pipeline():

    # age imputer
    age_imputer = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'))

    # Growth stage Transformer (1-4)
    dictionary = {'mature' : 4, 'late growth' : 3,'early growth' : 2, 'seed' : 1}


    growth_stage_transformer = FunctionTransformer(
        lambda df: df[['growth_stage_imputed']].applymap(lambda x: dictionary[x]))


    growth_stage_age_preparator = ColumnTransformer([
        ["age_imputer", age_imputer, ["age"]],
        ["growth_stage_transformer", growth_stage_transformer, ["growth_stage_imputed"]],
    ])


    #Growth_stage_aaage_ratio
    growth_stage_age_ratio_constructor = FunctionTransformer(compute_growth_age_ratio)


    growth_transformer = Pipeline(steps = [
        ["growth_stage_age_preparator", growth_stage_age_preparator],
        ["growth_stage_age_ratio_constructor", growth_stage_age_ratio_constructor]
    ])

    # Ratio_transformer  : funding / employees ratio
    funding_employees_ratio_constructor = FunctionTransformer(
        lambda df: pd.DataFrame(df["total_funding_source"] / df["employees_clean"]))

    ratio_transformer = Pipeline(steps = [
        ("imputer1", EmployeeImputer()),
        ("ratio", funding_employees_ratio_constructor),
        ("scaler",  RobustScaler())
    ])

    # patent transformer
    patent_transformer = make_pipeline(
                                    SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0),
                                    RobustScaler())

    # Health industry, investors_type, company_has_phd, proportion_technical, No_people_input, founder_from_institute, founder_has_phd
    No_Transformer = ColumnTransformer([
        ["health_notransformer", NoTransformer(), ["health_industry"]],
        ["fund_investor_notransformer", NoTransformer(), ["investors_type"]],
        ["doctor_notransformer", NoTransformer(), ["company_has_phd"]],
        ["technical_notransformer", NoTransformer(), ["proportion_technical"]],
        ["no_people_input_notransformer", NoTransformer(), ["No_people_input"]],
        ["founder_from_institute_notransformer", NoTransformer(), ["founder_from_institute"]],
        ["founder_has_phd_notransformer", NoTransformer(), ["founder_has_phd"]]
    ])

    # PREPROCESSOR
    preprocessor = ColumnTransformer([
        ("growth_transformer", growth_transformer, ["growth_stage_imputed", "age"]),
        ("ratio_transformer", ratio_transformer, ["employees_clean", "employees", "launch_year_clean", "employees_latest", "total_funding_source"]),
        ("patent_transformer", patent_transformer, ["nb_patents"]),
        ("identity", No_Transformer, ["health_industry",
                                     "investors_type",
                                     "company_has_phd",
                                     "proportion_technical",
                                     "No_people_input",
                                    "founder_from_institute",
                                    "founder_has_phd"
                                    ]),
        ])

    #PIPELINE
    pipemodel = Pipeline(steps=[
                                ('features', preprocessor),
                                ('model', RandomForestClassifier(n_jobs = -1))
                                 ])

    return pipemodel


class NewTrainer():

    def __init__(self, X, y):
        '''
        instantiate trainer object with X and y
        '''
        self.pipeline = None
        self.X = X
        self.y = y


    def train(self):
        self.pipeline = set_pipeline()
        self.pipeline.fit(self.X, self.y)


    def save_model(self):
        '''
        Save the model into a .pickle
        '''
        # Export pipeline as pickle file
        with open("bpideepmodelnew.pkl", "wb") as file:
            pickle.dump(self.pipeline, file)


        # Export the model to a file
        # model_name = "bpideepmodelnew.joblib"
        # joblib.dump(self.pipeline, model_name)
        # Use it for later prediction
        # loaded_model = joblib.load(model_name)
        # y_pred = loaded_model.predict(X_test)

        print("bpideepmodelnew.pickle saved locally")


if __name__ == "__main__":

    # importing data
    data = get_clean_data()
    X = data.drop(columns = ["target", "deep_or_not"])
    y = data["target"]

    t = NewTrainer(X, y)
    t.train()
    t.save_model()
