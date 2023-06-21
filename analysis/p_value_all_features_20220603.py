import numpy as np

np.set_printoptions(threshold=100000)
import csv
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from scipy.stats import ttest_ind, norm, f, pearsonr

"""
Update 1029:
Change KS-test to mannwhitneyu test as the default test tool.
Use 0809 data"""


def ftest(s1, s2):
    '''F-test: samples distribution various equal or not'''
    print("Null Hypothesis:var(s1)=var(s2)，α=0.05")
    F = np.var(s1) / np.var(s2)
    v1 = len(s1) - 1
    v2 = len(s2) - 1
    p_val = 1 - 2 * abs(0.5 - f.cdf(F, v1, v2))
    print(p_val)
    if p_val < 0.05:
        equal_var = False
    else:
        equal_var = True
    return equal_var


def ttest_ind_fun(s1, s2):
    # t-test: samples distribution means equal or not
    equal_var = ftest(s1, s2)
    print("Null Hypothesis:mean(s1)=mean(s2)，α=0.05")
    ttest, pval = ttest_ind(s1, s2, equal_var=equal_var)
    return pval


def kstest(array_1, array_2):
    e0 = np.mean(array_1)
    std0 = np.std(array_1)
    d0, p0 = stats.kstest(array_1, 'norm', (e0, std0))
    e1 = np.mean(array_2)
    std1 = np.std(array_2)
    d1, p1 = stats.kstest(array_2, 'norm', (e1, std1))
    if p0 > 0.05 and p1 > 0.05:
        p = ttest_ind_fun(array_1, array_2)
    else:
        d, p = stats.kstest(array_1, array_2)
    return p


def test(array_1, array_2):
    d, p = stats.mannwhitneyu(array_1, array_2)
    return p


root_path = '/home/resadmin/haoran/BiLSTM/'
data_path = root_path+'data_20220603_1131/id_label_bi_20220603_digital_1131.csv'

total_features = []
# ----------------------------------------------------------------------------
# load pharmacy (drug) records
pharmacy_records = {}
with open(root_path + 'data_20220603_1304/processed_encounter_pharmacy_digital_20220603.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if i == 0:  # just read row[0] which is the col names
            # print(row)
            pharmacy_features_len = len(row[1:])  # first col is 'id', do not need it
            pharmacy_features = row[1:]
            total_features += pharmacy_features
        # pharmacy_records[row[0]] = row[1:]

print('drug and diagnosis features:', len(total_features))

# load test records

test_records = {}
# loinc_dict=[]
with open(root_path + 'data_20220523_rebuild_1304/yellow_loinc_personal_digital_20220523.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if i == 0:
            # print(row)
            test_features_len = len(row[1:])
            loinc_dict = row[1:]
            total_features += loinc_dict
        # test_records[row[0]] = row[1:]

print('total features:', len(total_features))  # 349

# load encounter (diagnosis standard name) dict Code -- Name
disease_dict = {}
with open(root_path + 'data_1203/Integer_ICD10_code_1203_cancer.csv', newline='') as csvfile:
    # 3-dig code ; name ; range-code
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if i == 0:
            continue
        if row:
            # print(row)
            disease_dict[row[-1]] = row[-2]

# check label-tf_feature record
x = []
y = []

feature_names = []
for item in total_features:
    local_feature_name = item
    if local_feature_name in disease_dict.keys():
        local_feature_name = disease_dict[local_feature_name]
    # if local_feature_name in loinc_dict:
    # local_feature_name = local_feature_name
    # print(item, local_feature_name)
    feature_names.append(local_feature_name)

print(len(feature_names))  # 349

with open(data_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if row:
            row = [float(item) for item in row]
            y.append(row[0])
            x.append(row[1:])

x = np.array(x)
y = np.array(y)
X_transpose = x.transpose()

print(len(total_features))  # 349
print(np.shape(x))  # (1131, 349)
print(np.shape(X_transpose))  # (349, 1131)
print(len(y))  # 1131"""

pos_patient_index = []
for p in range(len(y)):
    if y[p] == 1:
        pos_patient_index.append(p)
print(len(pos_patient_index))  # 76

# save path
w20 = csv.writer(open(root_path + 'data_20220603_1131/feature_p_value_20220603.csv', "a"))
w20.writerow(['feature_name', 'p_value'])

issue = 0  # count problem features

special_loinc = loinc_dict

for feature_index in range(len(total_features)):
    local_feature_name = feature_names[feature_index]

    if local_feature_name in special_loinc:  # loinc terms need two special process

        continue  # separate into two code files

    else:  # for diagnosis and pharmacy terms

        pos_patient_values = []
        neg_patient_values = []

        for patient_index in range(len(y)):  # 1131

            patient_level = X_transpose[feature_index][patient_index]  # some patient's some feature's term level

            if patient_index in pos_patient_index:
                pos_patient_values.append(patient_level)
            else:
                neg_patient_values.append(patient_level)

        try:
            p_value = test(pos_patient_values, neg_patient_values)
            print([local_feature_name, p_value])

        except:
            p_value = kstest(pos_patient_values, neg_patient_values)
            print('use kstest', [local_feature_name, p_value])
            issue += 1

        # save the p-value
        w20.writerow([local_feature_name, p_value])

print('problem features:', issue)  # problem features: 6 (20220603)