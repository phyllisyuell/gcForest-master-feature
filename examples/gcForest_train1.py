from lib.gcforest.cascade.cascade_classifier import final_feature_xgb_importance_list, final_feature_rf_importance_list
import matplotlib.pyplot as plt
from sklearn import  metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from lib.gcforest.gcforest import GCForest
from sklearn.metrics import accuracy_score,classification_report
import numpy as np
import pandas as pd
def evaluate(y_label, pred_label,title):
    print("AUC Score(" + title + "):%f" % metrics.roc_auc_score(y_label, pred_label))
    accuracy = accuracy_score(y_label, pred_label)
    print("Accuracy(" + title + "): %.4f%%" % (accuracy * 100.0))
    print("classification_report(" + title + ")：")
    print(classification_report(y_label, pred_label, digits=4))
# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, labels,title):
    cmap = plt.cm.binary
    cm = confusion_matrix(y_true, y_pred)
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(2, 2), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    intFlag = 0  # 标记在图片中对文字是整数型还是浮点型
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if (intFlag):
            c = cm[y_val][x_val]
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=23, va='center', ha='center')
        else:
            c = cm_normalized[y_val][x_val]
            if (c > 0.01):
                # 这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=20, va='center', ha='center')
            else:
                plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=20, va='center', ha='center')
    if (intFlag):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    else:
        plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    font2 = {'size': 12}
    plt.title('xgboost Confusion Matrix', font2)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    # plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    font3 = {'size': 7}
    plt.ylabel('Index of True Classes', font3)
    plt.xlabel('Index of Predict Classes', font3)
    plt.savefig('../ConMatrix/'+title+'1.jpg', dpi=300)
def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 3
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 60, "max_depth": 6,
         "objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": 0.1,"random_state":66})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 60, "max_depth": 6,"random_state":66, "n_jobs": -1})
    config["cascade"] = ca_config
    return config
if __name__ == '__main__':
    train = pd.read_csv("../data/train1.csv")
    print(train.shape)
    test = pd.read_csv("../data/test1.csv")
    y_train = train["label2"]
    x_train = train.ix[:, 4:]
    y_test = test["label2"]
    x_test = test.ix[:,4:]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    gc = GCForest(get_toy_config())

    X_train_enc = gc.fit_transform(x_train, y_train)
    pre_train = gc.predict(x_train)
    pro_train = gc.predict_proba(x_train)
    pre_test = gc.predict(x_test)
    pro_test = gc.predict_proba(x_test)
    #评价
    evaluate(y_train,pre_train,"train")
    evaluate(y_test,pre_test,"test")
    print("xgboost特征重要性列表： ")
    print(final_feature_xgb_importance_list)
    print("随机森林特征重要性列表： ")
    print(final_feature_rf_importance_list)
