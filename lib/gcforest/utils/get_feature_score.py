#--------------------------------------------------
#@Description: get_feature_score
#@Author: Phyllis
#@Create: 2018-12-24-15:51
#--------------------------------------------------
# import xgboost as xgb
# from sklearn.ensemble import RandomForestClassifier as rf
# [(0.0096, 'x3'), (0.0007, 'x6'), (0.0005, 'x5'),
# (0.0005, 'x4'), (0.0, 'x2'), (0.0, 'x1')]
def get_rf_feature_importance(rf, feature_names):
    names = feature_names
    rf_feature_importance_list = []
    feature_weight = rf.feature_importances_ * rf.get_params()['n_estimators']
    rf_feature_importance_list = zip(names, map(lambda x: round(x, 4), feature_weight))
    rf_feature_importance_dict = dict(rf_feature_importance_list)
    rf_feature_importance_dict = sorted(rf_feature_importance_dict.items(), key=lambda item:item[1], reverse=True)
    return rf_feature_importance_dict

# {'x3': 2002, 'x5': 1059, 'x1': 44, 'x2': 55, 'x4': 36}
def get_xgb_feature_importance(gbm):
    return gbm.get_fscore()

# features scores add
'''def feature_score_add(model_name, model):
    final_feature_score_dict = {}
    if (model_name=='RandomForestClassifier'):
        for i in range(rf):
            final_feature_score_dict = get_rf_feature_importance(model, feature_names)
    else if(mode_name=='XGBClassifier'):
        for i in range():
            get_xgb_feature_importance(model)


    return final_feature_score_list'''

