"""Classification with a formula: Wisconsin breast cancer data (scikit-learn).

The response is a factor; ``FormulaEstimator`` passes the labels to the classifier and every
transformation (logs, interactions) is learned on the training folds only.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statsformula.sklearn import FormulaEstimator, FormulaTransformer

raw = load_breast_cancer(as_frame=True)
df = raw.frame.drop(columns="target")
df.columns = [c.replace(" ", "_") for c in df.columns]
df["diagnosis"] = pd.Categorical.from_codes(raw.target, list(raw.target_names))
cv = StratifiedKFold(5, shuffle=True, random_state=0)

formula = "diagnosis ~ log(mean_area) + mean_concave_points * mean_texture + worst_smoothness"
clf = FormulaEstimator(formula, LogisticRegression(C=10.0, max_iter=5000), remove_intercept=True)
auc = cross_val_score(clf, df, df["diagnosis"], cv=cv, scoring="roc_auc")
print("logistic regression, CV ROC AUC: %.3f (+/- %.3f)" % (auc.mean(), auc.std()))

clf.fit(df)
print("classes:", list(clf.classes_))
print(clf.predict_proba(df.head(3)).round(3))
print(clf.summary())

# the same design as a transformer step, followed by scaling
pipe = make_pipeline(FormulaTransformer("~ log(mean_area) + mean_concave_points * mean_texture + "
                                        "worst_smoothness", remove_intercept=True),
                     StandardScaler(), LogisticRegression(max_iter=5000))
auc = cross_val_score(pipe, df, df["diagnosis"], cv=cv, scoring="roc_auc")
print("scaled pipeline, CV ROC AUC: %.3f" % auc.mean())
