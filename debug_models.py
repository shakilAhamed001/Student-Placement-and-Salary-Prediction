import joblib

sm = joblib.load('linear_regression_model.joblib')
pm = joblib.load('logistic_regression_placement_model.joblib')

print('Salary model type:', type(sm))
print('Placement model type:', type(pm))

if hasattr(sm, 'feature_names_in_'):
    print('Salary features:', list(sm.feature_names_in_))
else:
    print('Salary model has no feature_names_in_')

if hasattr(pm, 'feature_names_in_'):
    print('Placement features:', list(pm.feature_names_in_))
else:
    print('Placement model has no feature_names_in_')

print('Salary n_features_in_:', sm.n_features_in_)
print('Placement n_features_in_:', pm.n_features_in_)
