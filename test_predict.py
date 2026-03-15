import warnings
warnings.filterwarnings('ignore')
import joblib
import pandas as pd

sm = joblib.load('linear_regression_model.joblib')
pm = joblib.load('logistic_regression_placement_model.joblib')

test = {
    'cgpa': 8.5, 'coding_skills': 7.0, 'dsa_score': 7.0, 'aptitude_score': 7.0,
    'communication_skills': 7.0, 'ml_knowledge': 6.0, 'system_design': 6.0,
    'internships': 1, 'projects_count': 2, 'certifications': 2,
    'hackathons': 1, 'backlogs': 0, 'open_source_contributions': 1,
    'extracurriculars': 1, 'branch': 'CSE', 'college_tier': 'Tier-1'
}

non_cat = list(pm.feature_names_in_)
df = pd.DataFrame([test])
proc = df[non_cat].copy()
for b in ['CSE', 'Chemical', 'ECE', 'EE', 'IT', 'ME']:
    proc[f'branch_{b}'] = (df['branch'] == b).astype(int)
for t in ['Tier-2', 'Tier-3']:
    proc[f'college_tier_{t}'] = (df['college_tier'] == t).astype(int)
proc = proc[list(sm.feature_names_in_)]

print('Salary prediction:', round(float(sm.predict(proc)[0]), 2), 'LPA')

p_df = df[list(pm.feature_names_in_)]
print('Placement prediction:', int(pm.predict(p_df)[0]))
