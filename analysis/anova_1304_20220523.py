lg = """test_list_lg= [71.6, 71.65, 71.63, 71.52, 71.53, 71.54, 71.6, 71.56, 71.55, 71.52] 71.57
val_list_lg= [67.49, 67.52, 67.56, 68.7, 68.77, 68.77, 68.87, 68.91, 68.95, 68.95] 68.449
train_list_lg= [72.01, 71.84, 71.75, 74.71, 74.97, 74.38, 75.36, 75.36, 75.29, 75.15] 74.082"""

svm = """test_list_svm= [71.2, 70.85, 71.5, 71.35, 71.27, 72.03, 72.02, 72.16, 69.61, 69.73] 71.172
val_list_svm= [72.36, 72.47, 73.25, 73.58, 73.65, 74.61, 74.68, 74.75, 75.78, 75.93] 74.106
train_list_svm= [76.63, 78.62, 76.77, 77.13, 78.85, 78.85, 78.84, 79.71, 84.03, 84.06] 79.349"""

mlp = """test_list_mlp= [71.33, 71.18, 69.51, 69.22, 69.8, 71.51, 70.35, 70.9, 71.72, 70.78] 70.63
val_list_mlp= [69.16, 69.44, 69.73, 69.94, 70.19, 70.66, 70.9, 72.15, 72.26, 73.43] 70.78599999999999
train_list_mlp= [74.88, 76.53, 72.22, 71.41, 73.24, 78.28, 74.86, 78.2, 81.94, 82.21] 76.377"""

rf = """test_list_rf= [71.45, 71.0, 70.87, 71.51, 70.95, 71.36, 71.12, 71.99, 71.95, 72.03] 71.423
val_list_rf= [71.44, 71.6, 72.04, 72.12, 72.26, 73.01, 73.04, 73.93, 73.97, 74.66] 72.807
train_list_rf= [74.5, 73.76, 74.7, 74.68, 74.34, 74.69, 74.67, 75.16, 75.17, 75.12] 74.679"""

gb = """test_list_gb= [69.41, 69.37, 69.65, 69.45, 69.41, 69.79, 69.56, 69.53, 69.7, 69.71] 69.558
val_list_gb= [67.36, 67.57, 67.81, 67.95, 68.27, 68.27, 68.38, 68.87, 69.16, 69.71] 68.33500000000001
train_list_gb= [76.86, 77.04, 77.27, 77.41, 77.63, 77.83, 77.96, 78.19, 78.32, 78.48] 77.699"""

xgb = """test_list [70.59, 70.59, 71.15, 71.15, 71.15, 71.15, 70.08, 70.08, 70.08, 70.08]
val_list [71.08, 71.08, 71.15, 71.15, 71.15, 71.15, 71.19, 71.19, 71.19, 71.19]
train_list [76.88, 76.88, 76.95, 76.95, 76.95, 76.95, 76.14, 76.14, 76.14, 76.14]"""

import numpy as np
from scipy import stats

model_list = [lg, svm, mlp, rf, gb, xgb]
model_name = ['lg', 'svm', 'mlp', 'rf', 'gb', 'xgb']

testing_results = []
for m, model in enumerate(model_list):
    local_train_results = []
    # print(model_name[m])
    folds = model.split('\n')

    for j, fold in enumerate(folds):
        if j == 1 or j == 2:
            continue  # skip the val folder
        fold = fold[fold.find('[')+1:]
        fold = fold[:fold.find(']')]
        tt = fold.split(', ')  # here is 10 numbers
        testing_results.append(tt)
    print(model_name[m], np.mean(local_train_results))

# transfer string results to list
for i, model_result in enumerate(testing_results):
    for fold_result in model_result:
        print("('"+model_name[i]+"', " + str(fold_result) + "),")


data = np.rec.array([
    ('lg', 71.6),
    ('lg', 71.65),
    ('lg', 71.63),
    ('lg', 71.52),
    ('lg', 71.53),
    ('lg', 71.54),
    ('lg', 71.6),
    ('lg', 71.56),
    ('lg', 71.55),
    ('lg', 71.52),
    ('svm', 71.2),
    ('svm', 70.85),
    ('svm', 71.5),
    ('svm', 71.35),
    ('svm', 71.27),
    ('svm', 72.03),
    ('svm', 72.02),
    ('svm', 72.16),
    ('svm', 69.61),
    ('svm', 69.73),
    ('mlp', 71.33),
    ('mlp', 71.18),
    ('mlp', 69.51),
    ('mlp', 69.22),
    ('mlp', 69.8),
    ('mlp', 71.51),
    ('mlp', 70.35),
    ('mlp', 70.9),
    ('mlp', 71.72),
    ('mlp', 70.78),
    ('rf', 71.45),
    ('rf', 71.0),
    ('rf', 70.87),
    ('rf', 71.51),
    ('rf', 70.95),
    ('rf', 71.36),
    ('rf', 71.12),
    ('rf', 71.99),
    ('rf', 71.95),
    ('rf', 72.03),
    ('gb', 69.41),
    ('gb', 69.37),
    ('gb', 69.65),
    ('gb', 69.45),
    ('gb', 69.41),
    ('gb', 69.79),
    ('gb', 69.56),
    ('gb', 69.53),
    ('gb', 69.7),
    ('gb', 69.71),
    ('xgb', 70.59),
    ('xgb', 70.59),
    ('xgb', 71.15),
    ('xgb', 71.15),
    ('xgb', 71.15),
    ('xgb', 71.15),
    ('xgb', 70.08),
    ('xgb', 70.08),
    ('xgb', 70.08),
    ('xgb', 70.08)], dtype=[('model', '|U5'), ('auc', 'f8')])

# print(data[data['model'] == 'lg'].auc)
f, p = stats.f_oneway(data[data['model'] == 'lg'].auc,
                      data[data['model'] == 'svm'].auc,
                      data[data['model'] == 'mlp'].auc,
                      data[data['model'] == 'rf'].auc,
                      data[data['model'] == 'gb'].auc,
                      data[data['model'] == 'xgb'].auc
                      )
print('One-way ANOVA')
print('=============')

print('F value:', f)
print('P value:', p, '\n')

from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison

mc = MultiComparison(data['auc'], data['model'])
result = mc.tukeyhsd()

print(result)
