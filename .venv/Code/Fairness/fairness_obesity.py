import pandas as pd
import os
from fairmlhealth import report, measure
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

os.chdir("/Users/felixdiederichs/PycharmProjects/Results_Analysis/.venv")

# Analyse fairness of aids
obesity = pd.read_csv("data/obesity.csv")

X = obesity.drop("NObeyesdad", axis=1)
X["Gender"] = X["Gender"].apply(lambda x: 0 if x=="Female" else 1)

X = pd.get_dummies(X)

y = obesity["NObeyesdad"].apply(lambda x: 0 if x=="Normal_Weight" else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)

fairness_report = report.compare(test_data=X_test,
                                 targets=y_test,
                                 protected_attr=X_test["Gender"],
                                 models= model,
                                 skip_performance=True
                                 )
fairness_report.to_html("Reports/Fairness_Reports/fairness_report_obesity_original.html")
