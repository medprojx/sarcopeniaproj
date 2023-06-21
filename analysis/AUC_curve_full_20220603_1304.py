from sklearn import metrics
import pylab as plt
import csv
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)

root_path = '/home/resadmin/haoran/BiLSTM/'

big_brier_list = [
    [[], []],
    [[], []],
    [[], []],
    [[], []],
    [[], []],
    [[], []]
]  # 6 models, each has train scores and test scores

prob_pos_list_total_10 = []
y_test_total_10 = []

fold2 = '/home/resadmin/haoran/BiLSTM/data_20220603_1304/AUC_Calibration_712_full/'
tails2 = [' (Training set, no Oversampling)', ' (Test set, no Oversampling)']

fold3 = ''
tails3 = [' ']

plots = ['', 'full_oversampling', 'full_no_oversampling']

fold = fold2
tails =tails2
plot= plots[2]

date = '_20220603'

save_path = fold

for top in range(10):
    top = str(int(top+1))
    print(top)

    local_model = 'rf'
    with open(fold+local_model+'_top'+top+date+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or not row:
                continue
            else:
                # print(row, row[0])
                # print(row[1:])
                name, score_list = row[0], row[1:]
                score_list = [float(item) for item in score_list]
                # print(name)
                # print(local_model, len(score_list))
                # print('haha')
                if name == 'y_test1=':
                    y_test1 = score_list
                if name == 'y_test2=':
                    y_test2 = score_list

                if name == 'auc_'+local_model+'_over1=':
                    auc_rf_over1 = score_list
                if name == 'auc_'+local_model+'_over2=':
                    auc_rf_over2 = score_list
                if name == 'auc_'+local_model+'_over3=':
                    auc_rf_over3 = score_list
                if name == 'auc_'+local_model+'_over4=':
                    auc_rf_over4 = score_list

    local_model = 'lg'
    with open(fold+local_model+'_top'+top+date+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or not row:
                continue
            name, score_list = row[0], row[1:]
            # print(list)
            score_list = [float(item) for item in score_list]
            # print(local_model, len(score_list))

            if name == 'auc_'+local_model+'_over1=':
                auc_logistic_over1 = score_list
            if name == 'auc_'+local_model+'_over2=':
                auc_logistic_over2 = score_list
            if name == 'auc_'+local_model+'_over3=':
                auc_logistic_over3 = score_list
            if name == 'auc_'+local_model+'_over4=':
                auc_logistic_over4 = score_list

    local_model = 'svm'
    with open(fold+local_model+'_top'+top+date+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or not row:
                continue
            name, score_list = row[0], row[1:]
            score_list = [float(item) for item in score_list]
            # print(local_model, len(score_list))
            # print(list)
            if name == 'auc_'+local_model+'_over1=':
                auc_svm_over1 = score_list
            if name == 'auc_'+local_model+'_over2=':
                auc_svm_over2 = score_list
            if name == 'auc_'+local_model+'_over3=':
                auc_svm_over3 = score_list
            if name == 'auc_'+local_model+'_over4=':
                auc_svm_over4 = score_list

    local_model = 'mlp'
    with open(fold+local_model+'_top'+top+date+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or not row:
                continue
            name, score_list = row[0], row[1:]
            score_list = [float(item) for item in score_list]
            # print(local_model, len(score_list))
            if name == 'auc_'+local_model+'_over1=':
                auc_mlp_over1 = score_list
            if name == 'auc_'+local_model+'_over2=':
                auc_mlp_over2 = score_list
            if name == 'auc_'+local_model+'_over3=':
                auc_mlp_over3 = score_list
            if name == 'auc_'+local_model+'_over4=':
                auc_mlp_over4 = score_list

    local_model = 'gb'
    with open(fold+local_model+'_top'+top+date+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or not row:
                continue
            name, score_list = row[0], row[1:]
            score_list = [float(item) for item in score_list]
            # print(local_model, len(score_list))
            if name == 'auc_'+local_model+'_over1=':
                auc_gb_over1 = score_list
            if name == 'auc_'+local_model+'_over2=':
                auc_gb_over2 = score_list
            if name == 'auc_'+local_model+'_over3=':
                auc_gb_over3 = score_list
            if name == 'auc_'+local_model+'_over4=':
                auc_gb_over4 = score_list

    local_model = 'xgb'
    with open(fold+local_model+'_top'+top+date+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(spamreader):
            if i == 0 or not row:
                continue
            name, score_list = row[0], row[1:]
            score_list = [float(item) for item in score_list]
            # print(local_model, len(score_list))
            if name == 'auc_'+local_model+'_over1=':
                auc_xgb_over1 = score_list
            if name == 'auc_'+local_model+'_over2=':
                auc_xgb_over2 = score_list
            if name == 'auc_'+local_model+'_over3=':
                auc_xgb_over3 = score_list
            if name == 'auc_'+local_model+'_over4=':
                auc_xgb_over4 = score_list

    # print(y_test1)
    y_test1 = [int(item) for item in y_test1]
    y_test2 = [int(item) for item in y_test2]
    # 1 = training_full  # 2 = testing_full
    prob_pos_list1 = [auc_logistic_over1,auc_svm_over1,auc_mlp_over1,auc_rf_over1,auc_gb_over1,auc_xgb_over1]
    prob_pos_list2 = [auc_logistic_over2,auc_svm_over2,auc_mlp_over2,auc_rf_over2,auc_gb_over2,auc_xgb_over2]

    name_list = ['Logistic Regression', "SVM","MLP","Random Forest", "GradientBoost", "XGBoost"]

    prob_pos_list_total = [prob_pos_list1,prob_pos_list2]
    y_test_total = [y_test1,y_test2]

    prob_pos_list1_array = [np.array(item) for item in prob_pos_list1]
    prob_pos_list2_array = [np.array(item) for item in prob_pos_list2]
    y_test1_array = np.array(y_test1)
    y_test2_array = np.array(y_test2)
    prob_pos_list_total_array = [prob_pos_list1_array, prob_pos_list2_array]
    y_test_total_array = [y_test1_array, y_test2_array]

    prob_pos_list_total_10.append(np.array(prob_pos_list_total_array))
    y_test_total_10.append(np.array(y_test_total_array))

    # --------------------------------
    for set_index in range(2):
        if set_index == 0:
            data_set = 'train'
        else:
            data_set = 'test'

        tail = tails[set_index]
        prob_pos_list = prob_pos_list_total[set_index]
        y_test = y_test_total[set_index]

        for n in range(6):
            fpr, tpr, threshold = metrics.roc_curve(y_test, prob_pos_list[n])
            roc_auc = metrics.auc(fpr, tpr)
            # if n == 2:
                # print(roc_auc)  # this part is also match

            model_index = n
            big_brier_list[n][set_index].append(roc_auc)

print(len(big_brier_list))
print(len(big_brier_list[0]))
print(len(big_brier_list[0][0]))
print(big_brier_list[2][0])
print(big_brier_list[2][1])


from scipy import stats


def stat(list):
    mean = np.average(list)
    std = np.std(list)
    ci = stats.norm.interval(0.95, loc=mean, scale=std)
    return np.round(mean*100,2), ci


prob_pos_list_total_10 = np.array(prob_pos_list_total_10)
y_test_total_10 = np.array(y_test_total_10)
print(prob_pos_list_total_10.shape)
new = []
for i in range(10):
    new.append(prob_pos_list_total_10[:, [0][0]][i][0][0])
prob_pos_list_total_10 = np.mean(prob_pos_list_total_10, axis=0)
print('?',prob_pos_list_total_10.shape)
print(np.mean(new))
print(prob_pos_list_total_10[0][0][0])


for set_index in range(2):
    if set_index == 0:
        data_set = 'train'
    else:
        data_set = 'test'

    print(data_set+'-'*20)

    # Plot auc plots
    plt.figure(figsize=(5, 5))
    plt.cla()

    tail = tails[set_index]
    prob_pos_list_total = prob_pos_list_total_10
    y_test_total = [y_test1, y_test2]

    plt.title('Validation ROC' + tail)

    prob_pos_list = prob_pos_list_total[set_index]
    y_test = y_test_total[set_index]

    for n in range(6):
        mean, ci = stat(big_brier_list[n][set_index])  # data inside already normed
        print(name_list[n], '@', mean, '@', (np.round(ci[0]*100, 2), np.round(ci[1]*100, 2)))
        # print(data_set+'-'*20)
        fpr, tpr, threshold = metrics.roc_curve(y_test, prob_pos_list[n])
        # roc_auc = metrics.auc(fpr, tpr)  # we don't computer auc here using averaged data
        plt.plot(fpr, tpr, label=name_list[n] + ' (%0.2f)' % mean)

    plt.legend(loc='lower right', )
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.tight_layout()
    # plt.show()
    # plt.savefig('/home/resadmin/haoran/BiLSTM/data_20220603_1304/AUC_curve_'+
                # data_set+'_'+plot+'_20220603.png', dpi=300)

"""train--------------------
Logistic Regression @ 76.61 @ (74.42, 78.8)
SVM @ 79.92 @ (74.65, 85.19)
MLP @ 75.2 @ (67.42, 82.99)
Random Forest @ 74.06 @ (72.65, 75.46)
GradientBoost @ 77.63 @ (76.67, 78.59)
XGBoost @ 73.68 @ (72.18, 75.18)
test--------------------
Logistic Regression @ 71.59 @ (71.51, 71.66)
SVM @ 71.17 @ (69.28, 73.07)
MLP @ 71.48 @ (71.0, 71.97)
Random Forest @ 69.09 @ (67.97, 70.22)
GradientBoost @ 69.56 @ (69.28, 69.83)
XGBoost @ 70.19 @ (69.18, 71.19)"""