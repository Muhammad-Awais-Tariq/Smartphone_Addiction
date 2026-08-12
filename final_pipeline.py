import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

addiction_df = pd.read_csv("F://Smartphone_Addiction//Data//train.csv")
X = addiction_df.drop(columns=["addicted_label"])
y = addiction_df["addicted_label"]



