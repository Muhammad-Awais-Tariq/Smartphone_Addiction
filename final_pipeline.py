import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib as jb

addiction_df = pd.read_csv("F://Smartphone_Addiction//Data//train.csv")
X = addiction_df.drop(columns=["addicted_label"])
y = addiction_df["addicted_label"]

numeric_coloumns_median = ["age" , "daily_screen_time_hours" , "sleep_hours" , "notifications_per_day" , "app_opens_per_day" , "weekend_screen_time" , "work_study_hours"]
numeric_column_zero = ["social_media_hours" , "gaming_hours"]
categorical_columns = ["gender", "stress_level", "academic_work_impact"]

numeric_pipeline_1 = Pipeline(
    [
        ("Imputer" , SimpleImputer(strategy="median")),
        ("Scaler" , StandardScaler())
    ]
)

numeric_pipeline_2 = Pipeline(
    [
        ("Imputer" , SimpleImputer(strategy='constant', fill_value=0)),
        ("Scaler" , StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    [
        ("Imputer" , SimpleImputer(strategy='most_frequent')),
        ("Encoder" , OneHotEncoder(handle_unknown="ignore"))        
    ]
)

preprocessor = ColumnTransformer(
    [
        ("num1" , numeric_pipeline_1 , numeric_coloumns_median),
        ("num2" , numeric_pipeline_2 , numeric_column_zero),
        ("categorical" , categorical_pipeline , categorical_columns)
    ]
)

boosted_forest_pipeline = Pipeline(
    [
        ("Preprocessing" , preprocessor),
        ("Model" , XGBClassifier(subsample = 0.8, n_estimators = 200, min_child_weight = 15, max_depth = 5, learning_rate = 0.1, gamma = 2, colsample_bytree = 0.8 ,objective='binary:logistic', eval_metric='auc', random_state=42))
    ]
)

boosted_forest_pipeline.fit(X , y)

jb.dump(boosted_forest_pipeline , "F://Smartphone_Addiction//smartphone_addiction.joblib")