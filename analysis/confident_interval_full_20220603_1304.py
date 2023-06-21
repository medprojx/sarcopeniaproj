import numpy as np
from scipy import stats


def stat(list):
    mean = np.average(list)
    std = np.std(list)
    ci = stats.norm.interval(0.95, loc=mean, scale=std)
    return mean, std, ci


test_list_lg= [71.6, 71.51, 71.58, 71.6, 71.65, 71.64, 71.54, 71.56, 71.59, 71.59] #71.58600000000001
val_list_lg= [69.16, 69.16, 69.2, 69.84, 69.84, 70.01, 70.16, 70.19, 70.19, 70.19] #69.79400000000001
train_list_lg= [75.13, 75.03, 74.86, 76.87, 76.43, 77.23, 77.7, 77.73, 77.63, 77.47] #76.608

test_list_svm= [70.9, 71.76, 71.52, 71.28, 69.78, 72.2, 72.35, 72.17, 70.03, 69.74] #71.173
val_list_svm= [72.86, 74.0, 74.18, 74.43, 75.11, 75.61, 75.61, 75.64, 77.64, 77.67] #75.275
train_list_svm= [78.61, 76.54, 77.02, 78.88, 84.28, 78.59, 79.5, 78.61, 83.48, 83.7] #79.921

test_list_mlp= [71.58, 71.59, 71.85, 71.56, 71.29, 71.45, 71.45, 71.83, 71.25, 70.99] #71.48400000000001
val_list_mlp= [66.84, 66.99, 67.06, 67.59, 69.2, 69.34, 69.34, 69.66, 72.04, 72.04] #69.00999999999999
train_list_mlp= [71.16, 70.64, 71.3, 71.21, 75.76, 76.05, 76.05, 76.31, 82.0, 81.55] #75.203

test_list_rf= [68.7, 69.08, 68.17, 69.62, 68.28, 68.77, 69.84, 69.52, 69.79, 69.18] # 69.095
val_list_rf= [70.28, 70.78, 70.94, 71.3, 71.35, 71.35, 71.67, 71.81, 71.96, 72.08] #71.352
train_list_rf= [73.68, 72.84, 74.03, 73.43, 74.57, 74.6, 74.91, 74.67, 74.76, 73.07] # 74.056

test_list_gb= [69.41, 69.37, 69.65, 69.45, 69.41, 69.79, 69.56, 69.53, 69.7, 69.71] # 69.558
val_list_gb= [67.36, 67.57, 67.81, 67.95, 68.27, 68.27, 68.38, 68.87, 69.16, 69.71] # 68.33500000000001
train_list_gb= [76.86, 77.04, 77.27, 77.41, 77.63, 77.83, 77.96, 78.19, 78.32, 78.48] # 77.699

test_list_xgb= [71.07, 71.07, 69.67, 69.67, 69.67, 69.67, 70.26, 70.26, 70.26, 70.26]
val_list_xgb= [69.96, 69.96, 70.05, 70.05, 70.05, 70.05, 70.76, 70.76, 70.76, 70.76]
train_list_xgb= [75.2, 75.2, 73.43, 73.43, 73.43, 73.43, 73.18, 73.18, 73.18, 73.18]


all_data = [[[test_list_lg, val_list_lg, train_list_lg]],
            [[test_list_svm, val_list_svm, train_list_svm]],
            [[test_list_mlp, val_list_mlp, train_list_mlp]],
            [[test_list_rf, val_list_rf, train_list_rf]],
            [[test_list_gb, val_list_gb, train_list_gb]],
            [[test_list_xgb, val_list_xgb, train_list_xgb]]]

line1, line2, line3, line4, line5, line6 = [], [], [], [], [], []
all_line = [line1, line2, line3, line4, line5, line6]
name_list = ['lg', 'svm', 'mlp', 'rf', 'gb', 'xgb']
for m, model in enumerate(all_data):

    name = name_list[m]
    method_list = ['full']

    for mt, method in enumerate(model):

        print(name + '-' + method_list[mt])
        for data_set in method:
            mean, std, ci = stat(data_set)
            mean = np.round(mean, 2)
            print(mean, ',', std, ',', [ci[0], ci[1]])
            # all_line[m].append(mean)
            # all_line[m].append(std)
            ci_left = np.round(ci[0], 2)
            ci_right = np.round(ci[1], 2)
            all_line[m].append(str(mean) + ' ' + str([ci_left, ci_right]))

import csv

# w21 = csv.writer(open('/home/resadmin/haoran/BiLSTM/data_1119/excell_output_1128.csv', "a"))
for line in all_line:
    line = ['& ' + str(item) for item in line]
    line = ' '.join(line)
    print(line)
    # w21.writerow(line)
