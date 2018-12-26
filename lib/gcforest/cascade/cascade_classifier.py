# -*- coding:utf-8 -*-
"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets.
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng.
"""
import numpy as np
import os
import os.path as osp
import pickle

from ..estimators import get_estimator_kfold
from ..utils.config_utils import get_config_value
from ..utils.log_utils import get_logger
from ..utils.metrics import accuracy_pb
from ..utils.get_feature_score import get_xgb_feature_importance, get_rf_feature_importance

LOGGER = get_logger('gcforest.cascade.cascade_classifier')
#111
final_feature_xgb_importance_list = []
final_feature_rf_importance_list = []


def check_dir(path):
    d = osp.abspath(osp.join(path, osp.pardir))
    if not osp.exists(d):
        os.makedirs(d)


def calc_accuracy(y_true, y_pred, name, prefix=""):
    acc = 100. * np.sum(np.asarray(y_true) == y_pred) / len(y_true)
    LOGGER.info('{}Accuracy({})={:.2f}%'.format(prefix, name, acc))
    return acc


def get_opt_layer_id(acc_list):
    """ Return layer id with max accuracy on training data """
    opt_layer_id = np.argsort(-np.asarray(acc_list), kind='mergesort')[0]
    return opt_layer_id


class CascadeClassifier(object):
    def __init__(self, ca_config):
        """
        Parameters (ca_config)
        ----------
        early_stopping_rounds: int
            when not None , means when the accuracy does not increase in early_stopping_rounds, the cascade level will stop automatically growing
        max_layers: int
            maximum number of cascade layers allowed for exepriments, 0 means use Early Stoping to automatically find the layer number
        n_classes: int
            Number of classes
        est_configs:
            List of CVEstimator's config
        look_indexs_cycle (list 2d): default=None
            specification for layer i, look for the array in look_indexs_cycle[i % len(look_indexs_cycle)]
            defalut = None <=> [range(n_groups)]
            .e.g.
                look_indexs_cycle = [[0,1],[2,3],[0,1,2,3]]
                means layer 1 look for the grained 0,1; layer 2 look for grained 2,3; layer 3 look for every grained, and layer 4 cycles back as layer 1
        data_save_rounds: int [default=0]
        data_save_dir: str [default=None]
            each data_save_rounds save the intermidiate results in data_save_dir
            if data_save_rounds = 0, then no savings for intermidiate results
        """
        self.ca_config = ca_config
        self.early_stopping_rounds = self.get_value("early_stopping_rounds", None, int, required=True)
        self.max_layers = self.get_value("max_layers", 0, int)
        self.n_classes = self.get_value("n_classes", None, int, required=True)
        self.est_configs = self.get_value("estimators", None, list, required=True)
        self.look_indexs_cycle = self.get_value("look_indexs_cycle", None, list)
        self.random_state = self.get_value("random_state", None, int)
        # self.data_save_dir = self.get_value("data_save_dir", None, basestring)
        self.data_save_dir = ca_config.get("data_save_dir", None)
        self.data_save_rounds = self.get_value("data_save_rounds", 0, int)
        if self.data_save_rounds > 0:
            assert self.data_save_dir is not None, "data_save_dir should not be null when data_save_rounds>0"
        self.eval_metrics = [("predict", accuracy_pb)]
        self.estimator2d = {}
        self.opt_layer_num = -1
        # LOGGER.info("\n" + json.dumps(ca_config, sort_keys=True, indent=4, separators=(',', ':')))

    @property
    def n_estimators_1(self):
        # estimators of one layer
        return len(self.est_configs)

    def get_value(self, key, default_value, value_types, required=False):
        return get_config_value(self.ca_config, key, default_value, value_types,
                required=required, config_name="cascade")

    def _set_estimator(self, li, ei, est):
        if li not in self.estimator2d:
            self.estimator2d[li] = {}
        self.estimator2d[li][ei] = est

    def _get_estimator(self, li, ei):
        return self.estimator2d.get(li, {}).get(ei, None)

    def _init_estimators(self, li, ei):
        est_args = self.est_configs[ei].copy()
        est_name = "layer_{} - estimator_{} - {}_folds".format(li, ei, est_args["n_folds"])
        # n_folds
        n_folds = int(est_args["n_folds"])
        est_args.pop("n_folds")
        # est_type
        est_type = est_args["type"]
        est_args.pop("type")
        # random_state
        if self.random_state is not None:
            random_state = (self.random_state + hash("[estimator] {}".format(est_name))) % 1000000007
        else:
            random_state = None
        return get_estimator_kfold(est_name, n_folds, est_type, est_args, random_state=random_state)

    def _check_look_indexs_cycle(self, X_groups, is_fit):
        # check look_indexs_cycle
        n_groups = len(X_groups)
        if is_fit and self.look_indexs_cycle is None:
            look_indexs_cycle = [list(range(n_groups))]
        else:
            look_indexs_cycle = self.look_indexs_cycle
            for look_indexs in look_indexs_cycle:
                if np.max(look_indexs) >= n_groups or np.min(look_indexs) < 0 or len(look_indexs) == 0:
                    raise ValueError("look_indexs doesn't match n_groups!!! look_indexs={}, n_groups={}".format(
                        look_indexs, n_groups))
        if is_fit:
            self.look_indexs_cycle = look_indexs_cycle
        return look_indexs_cycle

    def _check_group_dims(self, X_groups, is_fit):
        if is_fit:
            group_starts, group_ends, group_dims = [], [], []
        else:
            group_starts, group_ends, group_dims = self.group_starts, self.group_ends, self.group_dims
        n_datas = X_groups[0].shape[0]
        X = np.zeros((n_datas, 0), dtype=X_groups[0].dtype)
        for i, X_group in enumerate(X_groups):
            assert(X_group.shape[0] == n_datas)
            X_group = X_group.reshape(n_datas, -1)
            if is_fit:
                group_dims.append( X_group.shape[1] )
                group_starts.append(0 if i == 0 else group_ends[i - 1])
                group_ends.append(group_starts[i] + group_dims[i])
            else:
                assert(X_group.shape[1] == group_dims[i])
            X = np.hstack((X, X_group))
        if is_fit:
            self.group_starts, self.group_ends, self.group_dims = group_starts, group_ends, group_dims
        return group_starts, group_ends, group_dims, X

    def fit_transform(self, X_groups_train, y_train, X_groups_test, y_test, stop_by_test=False, train_config=None):
        """
        fit until the accuracy converges in early_stop_rounds
        stop_by_test: (bool)
            When X_test, y_test is validation data that used for determine the opt_layer_id,
            use this option
        """
        if train_config is None:
            from ..config import GCTrainConfig
            train_config = GCTrainConfig({})
        data_save_dir = train_config.data_cache.cache_dir or self.data_save_dir

        is_eval_test = "test" in train_config.phases
        if not type(X_groups_train) == list:
            X_groups_train = [X_groups_train]
        if is_eval_test and not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_train.shape={},y_train.shape={},X_groups_test.shape={},y_test.shape={}".format(
            [xr.shape for xr in X_groups_train], y_train.shape,
            [xt.shape for xt in X_groups_test] if is_eval_test else "no_test", y_test.shape if is_eval_test else "no_test"))

        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_train, True)
        if is_eval_test:
            self._check_look_indexs_cycle(X_groups_test, False)

        # check groups dimension
        group_starts, group_ends, group_dims, X_train = self._check_group_dims(X_groups_train, True)
        if is_eval_test:
            _, _, _, X_test = self._check_group_dims(X_groups_test, False)
        else:
            X_test = np.zeros((0, X_train.shape[1]))
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("group_starts={}".format(group_starts))
        LOGGER.info("group_ends={}".format(group_ends))
        LOGGER.info("X_train.shape={},X_test.shape={}".format(X_train.shape, X_test.shape))

        n_trains = X_groups_train[0].shape[0]
        n_tests = X_groups_test[0].shape[0] if is_eval_test else 0

        n_classes = self.n_classes
        assert n_classes == len(np.unique(y_train)), "n_classes({}) != len(unique(y)) {}".format(n_classes, np.unique(y_train))
        train_acc_list = []
        test_acc_list = []
        # X_train, y_train, X_test, y_test
        opt_datas = [None, None, None, None]
        try:
            # probability of each cascades's estimators
            X_proba_train = np.zeros((n_trains, n_classes * self.n_estimators_1), dtype=np.float32)
            X_proba_test = np.zeros((n_tests, n_classes * self.n_estimators_1), dtype=np.float32)
            X_cur_train, X_cur_test = None, None
            layer_id = 0
            # 111
            every_layer_feature_xgb_importance_dict = {}
            every_model_feature_xgb_importance_dict_1 = {}
            ##
            every_layer_feature_rf_importance_dict = {}
            every_model_feature_rf_importance_dict_1 = {}
            while 1:
                # 111
                if (len(every_model_feature_xgb_importance_dict_1)>0):
                    every_layer_feature_xgb_importance_dict_tmp = every_model_feature_xgb_importance_dict_1
                    for key in every_layer_feature_xgb_importance_dict:
                        if (every_layer_feature_xgb_importance_dict_tmp.get(key)):
                            every_layer_feature_xgb_importance_dict[key] = every_layer_feature_xgb_importance_dict[key] + every_layer_feature_xgb_importance_dict_tmp[key]
                        else:
                            every_layer_feature_xgb_importance_dict[key] = every_layer_feature_xgb_importance_dict[key]
                    for key in every_layer_feature_xgb_importance_dict_tmp:
                        if (every_layer_feature_xgb_importance_dict.get(key)):
                            pass
                        else:
                            every_layer_feature_xgb_importance_dict[key] = every_layer_feature_xgb_importance_dict_tmp[key]
                #elif (len(every_model_feature_rf_importance_dict_1)>0):
                    # 111
                if (len(every_model_feature_rf_importance_dict_1) > 0):
                    every_layer_feature_rf_importance_dict_tmp = every_model_feature_rf_importance_dict_1
                    for key in every_layer_feature_rf_importance_dict:
                        if (every_layer_feature_rf_importance_dict_tmp.get(key)):
                            every_layer_feature_rf_importance_dict[key] = every_layer_feature_rf_importance_dict[key] + every_layer_feature_rf_importance_dict_tmp[key]
                        else:
                            every_layer_feature_rf_importance_dict[key] = every_layer_feature_rf_importance_dict[key]
                    for key in every_layer_feature_rf_importance_dict_tmp:
                        if (every_layer_feature_rf_importance_dict.get(key)):
                            pass
                        else:
                            every_layer_feature_rf_importance_dict[key] = every_layer_feature_rf_importance_dict_tmp[key]
                                
                                

                if self.max_layers > 0 and layer_id >= self.max_layers:
                    break
                # Copy previous cascades's probability into current X_cur
                if layer_id == 0:
                    # first layer not have probability distribution
                    X_cur_train = np.zeros((n_trains, 0), dtype=np.float32)
                    X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
                else:
                    X_cur_train = X_proba_train.copy()
                    X_cur_test = X_proba_test.copy()
                # Stack data that current layer needs in to X_cur
                look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
                for _i, i in enumerate(look_indexs):
                    X_cur_train = np.hstack((X_cur_train, X_train[:, group_starts[i]:group_ends[i]]))
                    X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
                LOGGER.info("[layer={}] look_indexs={}, X_cur_train.shape={}, X_cur_test.shape={}".format(
                    layer_id, look_indexs, X_cur_train.shape, X_cur_test.shape))
                # Fit on X_cur, predict to update X_proba
                y_train_proba_li = np.zeros((n_trains, n_classes))
                y_test_proba_li = np.zeros((n_tests, n_classes))
                # 111
                every_model_feature_xgb_importance_dict = {}
                every_model_feature_rf_importance_dict = {}
                for ei, est_config in enumerate(self.est_configs):
                    # 模型判断
                    est = self._init_estimators(layer_id, ei)
                    # fit_trainsform
                    test_sets = [("test", X_cur_test, y_test)] if n_tests > 0 else None
                    est_, y_probas = est.fit_transform(X_cur_train, y_train, y_train,
                            test_sets=test_sets, eval_metrics=self.eval_metrics,
                            keep_model_in_mem=train_config.keep_model_in_mem)
                    # train
                    X_proba_train[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[0]
                    y_train_proba_li += y_probas[0]
                    # test
                    if n_tests > 0:
                        X_proba_test[:, ei * n_classes: ei * n_classes + n_classes] = y_probas[1]
                        y_test_proba_li += y_probas[1]
                    if train_config.keep_model_in_mem:
                        self._set_estimator(layer_id, ei, est)
                    # 每个层级中对应每个模型的权重
                    # 111
                    #rf_feature_name = []
                    if (est_config['type']=='XGBClassifier'):
                        #every_model_feature_xgb_importance_dict_tmp = est.est_class.get_fscore()  # est 中模型的类别
                        every_model_feature_xgb_importance_dict_tmp = get_xgb_feature_importance(est_._Booster)
                        #rf_feature_name = est_._Booster.feature_names
                        # print (rf_feature_name)
                        for key in every_model_feature_xgb_importance_dict:
                            if(every_model_feature_xgb_importance_dict_tmp.get(key)):
                                every_model_feature_xgb_importance_dict[key] = every_model_feature_xgb_importance_dict[key] + every_model_feature_xgb_importance_dict_tmp[key]
                            else:
                                every_model_feature_xgb_importance_dict[key] = every_model_feature_xgb_importance_dict[key]
                        for key in every_model_feature_xgb_importance_dict_tmp:
                            if (every_model_feature_xgb_importance_dict.get(key)):
                                pass
                            else:
                                every_model_feature_xgb_importance_dict[key] = every_model_feature_xgb_importance_dict_tmp[key]
                        every_model_feature_xgb_importance_dict_1 = every_model_feature_xgb_importance_dict

                    if(est_config['type']=='RandomForestClassifier'):
                        feature_name = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32', 'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48', 'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', 'f80', 'f81', 'f82', 'f83', 'f84', 'f85', 'f86', 'f87', 'f88', 'f89', 'f90', 'f91', 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105', 'f106', 'f107', 'f108', 'f109', 'f110', 'f111', 'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118', 'f119', 'f120', 'f121', 'f122', 'f123', 'f124', 'f125', 'f126', 'f127', 'f128', 'f129', 'f130', 'f131', 'f132', 'f133', 'f134', 'f135', 'f136', 'f137', 'f138', 'f139', 'f140', 'f141', 'f142', 'f143', 'f144', 'f145', 'f146', 'f147', 'f148', 'f149', 'f150', 'f151', 'f152', 'f153', 'f154', 'f155', 'f156', 'f157', 'f158', 'f159', 'f160', 'f161', 'f162', 'f163', 'f164', 'f165', 'f166', 'f167', 'f168', 'f169', 'f170', 'f171', 'f172', 'f173', 'f174', 'f175', 'f176', 'f177', 'f178', 'f179', 'f180', 'f181', 'f182', 'f183', 'f184', 'f185', 'f186', 'f187', 'f188', 'f189', 'f190', 'f191', 'f192', 'f193', 'f194', 'f195', 'f196', 'f197', 'f198', 'f199', 'f200', 'f201', 'f202', 'f203', 'f204', 'f205', 'f206', 'f207', 'f208', 'f209', 'f210', 'f211', 'f212', 'f213', 'f214', 'f215', 'f216', 'f217', 'f218', 'f219', 'f220', 'f221', 'f222', 'f223', 'f224', 'f225', 'f226', 'f227', 'f228', 'f229', 'f230', 'f231', 'f232', 'f233', 'f234', 'f235', 'f236', 'f237', 'f238', 'f239', 'f240', 'f241', 'f242', 'f243', 'f244', 'f245', 'f246', 'f247', 'f248', 'f249', 'f250', 'f251', 'f252', 'f253', 'f254', 'f255', 'f256', 'f257', 'f258', 'f259', 'f260', 'f261', 'f262', 'f263', 'f264', 'f265', 'f266', 'f267', 'f268', 'f269', 'f270', 'f271', 'f272', 'f273', 'f274', 'f275', 'f276', 'f277', 'f278', 'f279', 'f280', 'f281', 'f282', 'f283', 'f284', 'f285', 'f286', 'f287', 'f288', 'f289', 'f290', 'f291', 'f292', 'f293', 'f294', 'f295', 'f296', 'f297', 'f298', 'f299', 'f300', 'f301', 'f302', 'f303', 'f304', 'f305', 'f306', 'f307', 'f308', 'f309', 'f310', 'f311', 'f312', 'f313', 'f314', 'f315', 'f316', 'f317', 'f318', 'f319', 'f320', 'f321', 'f322', 'f323', 'f324', 'f325', 'f326', 'f327', 'f328', 'f329', 'f330', 'f331', 'f332', 'f333', 'f334', 'f335', 'f336', 'f337', 'f338', 'f339', 'f340', 'f341', 'f342', 'f343', 'f344', 'f345', 'f346', 'f347', 'f348', 'f349', 'f350', 'f351', 'f352', 'f353', 'f354', 'f355', 'f356', 'f357', 'f358', 'f359', 'f360', 'f361', 'f362', 'f363', 'f364', 'f365', 'f366', 'f367', 'f368', 'f369', 'f370', 'f371', 'f372', 'f373', 'f374', 'f375', 'f376', 'f377', 'f378', 'f379', 'f380', 'f381', 'f382', 'f383', 'f384', 'f385', 'f386', 'f387', 'f388', 'f389', 'f390', 'f391', 'f392', 'f393', 'f394', 'f395', 'f396', 'f397', 'f398', 'f399', 'f400', 'f401', 'f402', 'f403', 'f404', 'f405', 'f406', 'f407', 'f408', 'f409', 'f410', 'f411', 'f412', 'f413', 'f414', 'f415', 'f416', 'f417', 'f418', 'f419', 'f420', 'f421', 'f422', 'f423', 'f424', 'f425', 'f426', 'f427', 'f428', 'f429', 'f430', 'f431', 'f432', 'f433', 'f434', 'f435', 'f436', 'f437', 'f438', 'f439', 'f440', 'f441', 'f442', 'f443', 'f444', 'f445', 'f446', 'f447', 'f448', 'f449', 'f450', 'f451', 'f452', 'f453', 'f454', 'f455', 'f456', 'f457', 'f458', 'f459', 'f460', 'f461', 'f462', 'f463', 'f464', 'f465', 'f466', 'f467', 'f468', 'f469', 'f470', 'f471', 'f472', 'f473', 'f474', 'f475', 'f476', 'f477', 'f478', 'f479', 'f480', 'f481', 'f482', 'f483', 'f484', 'f485', 'f486', 'f487', 'f488', 'f489', 'f490', 'f491', 'f492', 'f493', 'f494', 'f495', 'f496', 'f497', 'f498', 'f499', 'f500', 'f501', 'f502', 'f503', 'f504', 'f505', 'f506', 'f507', 'f508', 'f509', 'f510', 'f511', 'f512', 'f513', 'f514', 'f515', 'f516', 'f517', 'f518', 'f519', 'f520', 'f521', 'f522', 'f523', 'f524', 'f525', 'f526', 'f527', 'f528', 'f529', 'f530', 'f531', 'f532', 'f533', 'f534', 'f535', 'f536', 'f537', 'f538', 'f539', 'f540', 'f541', 'f542', 'f543', 'f544', 'f545', 'f546', 'f547', 'f548', 'f549', 'f550', 'f551', 'f552', 'f553', 'f554', 'f555', 'f556', 'f557', 'f558', 'f559', 'f560', 'f561', 'f562', 'f563', 'f564', 'f565', 'f566', 'f567', 'f568', 'f569', 'f570', 'f571', 'f572', 'f573', 'f574', 'f575', 'f576', 'f577', 'f578', 'f579', 'f580', 'f581', 'f582', 'f583', 'f584', 'f585', 'f586', 'f587', 'f588', 'f589', 'f590', 'f591', 'f592', 'f593', 'f594', 'f595', 'f596', 'f597', 'f598', 'f599', 'f600', 'f601', 'f602', 'f603', 'f604', 'f605', 'f606', 'f607', 'f608', 'f609', 'f610', 'f611', 'f612', 'f613', 'f614', 'f615', 'f616', 'f617', 'f618', 'f619', 'f620', 'f621', 'f622', 'f623', 'f624', 'f625', 'f626', 'f627', 'f628', 'f629', 'f630', 'f631', 'f632', 'f633', 'f634', 'f635', 'f636', 'f637', 'f638', 'f639', 'f640', 'f641', 'f642', 'f643', 'f644', 'f645', 'f646', 'f647', 'f648', 'f649', 'f650', 'f651', 'f652', 'f653', 'f654', 'f655', 'f656', 'f657', 'f658', 'f659', 'f660', 'f661', 'f662', 'f663', 'f664', 'f665', 'f666', 'f667', 'f668', 'f669', 'f670', 'f671', 'f672', 'f673', 'f674', 'f675', 'f676', 'f677', 'f678', 'f679', 'f680', 'f681', 'f682', 'f683', 'f684', 'f685', 'f686', 'f687', 'f688', 'f689', 'f690', 'f691', 'f692', 'f693', 'f694', 'f695', 'f696', 'f697', 'f698', 'f699', 'f700', 'f701', 'f702', 'f703', 'f704', 'f705', 'f706', 'f707', 'f708', 'f709', 'f710', 'f711', 'f712', 'f713', 'f714', 'f715', 'f716', 'f717', 'f718', 'f719', 'f720', 'f721', 'f722', 'f723', 'f724', 'f725', 'f726', 'f727', 'f728', 'f729', 'f730', 'f731', 'f732', 'f733', 'f734', 'f735', 'f736', 'f737', 'f738', 'f739', 'f740', 'f741', 'f742', 'f743', 'f744', 'f745', 'f746', 'f747', 'f748', 'f749', 'f750', 'f751', 'f752', 'f753', 'f754', 'f755', 'f756', 'f757', 'f758', 'f759', 'f760', 'f761', 'f762', 'f763', 'f764', 'f765', 'f766', 'f767', 'f768', 'f769', 'f770', 'f771', 'f772', 'f773', 'f774', 'f775', 'f776', 'f777', 'f778', 'f779', 'f780', 'f781', 'f782', 'f783']
                        every_model_feature_rf_importance_dict_tmp = get_rf_feature_importance(est_, feature_name)  # est 中模型的类别
                        every_model_feature_rf_importance_dict_tmp = dict(every_model_feature_rf_importance_dict_tmp)
                        for key in every_model_feature_rf_importance_dict:
                            if (every_model_feature_rf_importance_dict_tmp.get(key)):
                                every_model_feature_rf_importance_dict[key] = every_model_feature_rf_importance_dict[key] +  every_model_feature_rf_importance_dict_tmp[key]
                            else:
                                every_model_feature_rf_importance_dict[key] = every_model_feature_rf_importance_dict[key]
                        for key in every_model_feature_rf_importance_dict_tmp:
                            if (every_model_feature_rf_importance_dict.get(key)):
                                pass
                            else:
                                every_model_feature_rf_importance_dict[key] = every_model_feature_rf_importance_dict_tmp[key]
                        every_model_feature_rf_importance_dict_1 = every_model_feature_rf_importance_dict



                y_train_proba_li /= len(self.est_configs)
                train_avg_acc = calc_accuracy(y_train, np.argmax(y_train_proba_li, axis=1), 'layer_{} - train.classifier_average'.format(layer_id))
                train_acc_list.append(train_avg_acc)
                if n_tests > 0:
                    y_test_proba_li /= len(self.est_configs)
                    test_avg_acc = calc_accuracy(y_test, np.argmax(y_test_proba_li, axis=1), 'layer_{} - test.classifier_average'.format(layer_id))
                    test_acc_list.append(test_avg_acc)
                else:
                    test_acc_list.append(0.0)

                opt_layer_id = get_opt_layer_id(test_acc_list if stop_by_test else train_acc_list)
                # set opt_datas
                if opt_layer_id == layer_id:
                    opt_datas = [X_proba_train, y_train, X_proba_test if n_tests > 0 else None, y_test]
                # early stop
                if self.early_stopping_rounds > 0 and layer_id - opt_layer_id >= self.early_stopping_rounds:
                    # log and save final result (opt layer)
                    LOGGER.info("[Result][Optimal Level Detected] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                        opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
                    if data_save_dir is not None:
                        self.save_data( data_save_dir, opt_layer_id, *opt_datas)
                    # remove unused model
                    if train_config.keep_model_in_mem:
                        for li in range(opt_layer_id + 1, layer_id + 1):
                            for ei, est_config in enumerate(self.est_configs):
                                self._set_estimator(li, ei, None)
                    self.opt_layer_num = opt_layer_id + 1
                    return opt_layer_id, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
                # save opt data if needed
                if self.data_save_rounds > 0 and (layer_id + 1) % self.data_save_rounds == 0:
                    self.save_data(data_save_dir, layer_id, *opt_datas)
                # inc layer_id
                layer_id += 1
                # 111
                #final_feature_importance_dict = every_layer_feature_xgb_importance_dict
            final_feature_xgb_importance_list.append(every_layer_feature_xgb_importance_dict)
            every_layer_feature_rf_importance_dict = sorted(every_layer_feature_rf_importance_dict.items(), key=lambda item:item[1], reverse=True)
            final_feature_rf_importance_list.append(every_layer_feature_rf_importance_dict)


            LOGGER.info("[Result][Reach Max Layer] opt_layer_num={}, accuracy_train={:.2f}%, accuracy_test={:.2f}%".format(
                opt_layer_id + 1, train_acc_list[opt_layer_id], test_acc_list[opt_layer_id]))
            if data_save_dir is not None:
                self.save_data(data_save_dir, self.max_layers - 1, *opt_datas)
            self.opt_layer_num = self.max_layers
            return self.max_layers, opt_datas[0], opt_datas[1], opt_datas[2], opt_datas[3]
        except KeyboardInterrupt:
            pass

    def transform(self, X_groups_test):
        if not type(X_groups_test) == list:
            X_groups_test = [X_groups_test]
        LOGGER.info("X_groups_test.shape={}".format([xt.shape for xt in X_groups_test]))
        # check look_indexs_cycle
        look_indexs_cycle = self._check_look_indexs_cycle(X_groups_test, False)
        # check group_dims
        group_starts, group_ends, group_dims, X_test = self._check_group_dims(X_groups_test, False)
        LOGGER.info("group_dims={}".format(group_dims))
        LOGGER.info("X_test.shape={}".format(X_test.shape))

        n_tests = X_groups_test[0].shape[0]
        n_classes = self.n_classes

        # probability of each cascades's estimators
        X_proba_test = np.zeros((X_test.shape[0], n_classes * self.n_estimators_1), dtype=np.float32)
        X_cur_test = None
        for layer_id in range(self.opt_layer_num):
            # Copy previous cascades's probability into current X_cur
            if layer_id == 0:
                # first layer not have probability distribution
                X_cur_test = np.zeros((n_tests, 0), dtype=np.float32)
            else:
                X_cur_test = X_proba_test.copy()
            # Stack data that current layer needs in to X_cur
            look_indexs = look_indexs_cycle[layer_id % len(look_indexs_cycle)]
            for _i, i in enumerate(look_indexs):
                X_cur_test = np.hstack((X_cur_test, X_test[:, group_starts[i]:group_ends[i]]))
            LOGGER.info("[layer={}] look_indexs={}, X_cur_test.shape={}".format(
                layer_id, look_indexs, X_cur_test.shape))
            for ei, est_config in enumerate(self.est_configs):
                est = self._get_estimator(layer_id, ei)
                if est is None:
                    raise ValueError("model (li={}, ei={}) not present, maybe you should set keep_model_in_mem to True".format(
                        layer_id, ei))
                y_probas = est.predict_proba(X_cur_test)
                X_proba_test[:, ei * n_classes:ei * n_classes + n_classes] = y_probas
        return X_proba_test

    def predict_proba(self, X):
        # n x (n_est*n_classes)
        y_proba = self.transform(X)
        # n x n_est x n_classes
        y_proba = y_proba.reshape((y_proba.shape[0], self.n_estimators_1, self.n_classes))
        y_proba = y_proba.mean(axis=1)
        return y_proba

    def save_data(self, data_save_dir, layer_id, X_train, y_train, X_test, y_test):
        for pi, phase in enumerate(["train", "test"]):
            data_path = osp.join(data_save_dir, "layer_{}-{}.pkl".format(layer_id, phase))
            check_dir(data_path)
            data = {"X": X_train, "y": y_train} if pi == 0 else {"X": X_test, "y": y_test}
            LOGGER.info("Saving Data in {} ... X.shape={}, y.shape={}".format(data_path, data["X"].shape, data["y"].shape))
            with open(data_path, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
