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

numeric_coloumns_median = ["age" , "daily_screen_time_hours" , "sleep_hours" , "notifications_per_day" , "app_opens_per_day" , "weekend_screen_time" , "work_study_hours"]
numeric_column_zero = ["social_media_hours" , "gaming_hours"]
categorical_columns = ["gender", "stress_level", "academic_work_impact"]

