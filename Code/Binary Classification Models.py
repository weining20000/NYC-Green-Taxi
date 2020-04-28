############################################################################
###                          Import Packages                              ##
############################################################################
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.options.mode.chained_assignment = None

############################################################################
##                          Data Pre-processing                           ##
############################################################################

# Load data, select valid observations, and create target variable

def pre_processing():
    df = pd.read_csv('cleaned_NYC_taxi_df_version_3.csv', header=0)
    df.head(2)
    df['tip_pct'] = round(100 * df['tip_amount'] / df['total_amount'], 4)

    # select data for prediction model
    # Filter 1:
    # Only observations whose payment_type is Credit Card (1) are included to build a predictive model.
    # Observations whose payment types that are Cash (2), No charge (3), Dispute (4), Unkonwn (5), and Voided trip (6) were exlcuded,
    # because tips were not automatically populated for those payment types.

    tip_filter1 = df['payment_type'] == 1

    # Filter 2:
    # trips with negotiated fare were excluded, becasue tip was likely included in the negotiated fare
    tip_filter2 = (df['ratecodeid'] != 5)

    # subset data
    dfc = df[tip_filter1 & tip_filter2]

    print('A total of', len(df) - len(dfc), 'records were non-credict card transactions, and', len(dfc),
          'were included in the predictive model. The selected observations took up,',
          round(100 * (len(dfc) / int(471052)), 2), '% of the original dataset.')

    return dfc

# Reduce input dimension

def feature_engineering(dfc):
    # Create weekday / weekend variable
    dfc['weekend'] = 'weekday'

    # row_index_weekend = list(dfc['pickup_day'].isin(['Saturday', 'Sunday']).index.values)
    # dfc.loc[row_index_weekend, ['weekend']] = 'weekend'
    dfc['weekend'].loc[dfc['pickup_day'].isin(['5', '6'])] = 'weekend'

    # convert continuous passenger count variable to categorical passenger range variable
    dfc['passenger_range'] = pd.Series(pd.cut(dfc['passenger_count'], [0, 2, 5, 9],
                                              labels=['1-2', '3-5', '5+'], retbins=False, right=True))

    # convert distance to distance range
    dfc['distance_range'] = pd.Series(pd.cut(dfc['trip_distance'], [0, 1.60, 3.17, 10.64, 40],
                                             labels=['q1', 'q2', 'q3', 'q4'],
                                             retbins=False, right=False))

    # convert time to time range
    dfc['time_range'] = pd.Series(pd.cut(dfc['trip_time'], [0, 9, 17, 44, 150],
                                         labels=['q1', 'q2', 'q3', 'q4'],
                                         include_lowest=True, retbins=False, right=False))

    # convert pick up hour and drop off hour to 2-hour intervals
    labels = ['0-1', '2-3', '4-5', '6-7', '8-9', '10-11', '12-13', '14-15', '16-17', '18-19', '20-21', '22-23']
    dfc['pickuphour_range'] = pd.Series(pd.cut(dfc['pickup_hour'], list(np.arange(0, 25, 2)),
                                               labels=labels, right=False))
    dfc['dropoffhour_range'] = pd.Series(pd.cut(dfc['dropoff_hour'], list(np.arange(0, 25, 2)),
                                                labels=labels, right=False))

    return dfc


def drop_na(df):
    df = df.drop(columns=['ehail_fee'], axis=1)
    df = df.dropna(axis='index', how='any', inplace=False)
    return df


def feature_encoding(X):

    # change cerntain int type to object type
    X['vendorid'] = X['vendorid'].astype(object)
    X['ratecodeid'] = X['ratecodeid'].astype(object)
    X['trip_type'] = X['trip_type'].astype(object)
    X['passenger_range'] = X['passenger_range'].astype(object)
    X['distance_range'] = X['distance_range'].astype(object)
    X['time_range'] = X['time_range'].astype(object)
    X['pickuphour_range'] = X['pickuphour_range'].astype(object)
    X['congestion_surcharge'] = X['congestion_surcharge'].astype(object)

    df_cat = cat_var_checker(X)

    X = pd.get_dummies(X, columns=df_cat['var'])
    return X


def get_features_and_targets(dfc):

    # remove highly correlated trip distance; dropoff_day is almost the same as pickup_day, dropoffhour_range
    predictors = ['vendorid', 'ratecodeid', 'fare_amount', 'extra',
                  'mta_tax', 'tolls_amount', 'improvement_surcharge', 'trip_type',
                  'congestion_surcharge', 'pickup_boro', 'pickup_servicezone',
                  'dropoff_boro', 'dropoff_servicezone', 'pickup_day',
                  'weekend', 'passenger_range', 'distance_range',
                  'time_range', 'pickuphour_range']

    X = dfc[predictors]
    X = feature_encoding(X)

    # Start extracting Y
    dfc['cat_target'] = 'non_tipper'
    dfc.loc[dfc['tip_pct'] > 0, 'cat_target'] = 'tipper'

    # count the number of tipper
    print('The percentage of tipper is',
          (len(dfc.loc[dfc['cat_target'] == 'tipper', 'cat_target']) / len(dfc)) * 100, '%.')

    # The labelEncoder
    le = LabelEncoder()

    # get label name mapping
    le.fit(dfc['cat_target'])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    # Encode the categorical target in the data
    dfc['cat_target'] = le.fit_transform(dfc['cat_target'].astype(str))
    y = dfc['cat_target']

    return X, y

# identify the categorical variables
def cat_var_checker(df):
    # get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           for var in df.columns if (df[var].dtype == 'object')],
                          columns=['var', 'nunique'])
    # sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    return df_cat

def generate_small_samples(X, y, sample_size):
    X_shuffled, y_shuffled = shuffle(X, y)
    X = X_shuffled[:sample_size]
    y = y_shuffled[:sample_size]
    X = X.values
    y = y.values
    return X, y


############################################################################
##                                Modeling                                ##
############################################################################


def split_training_testing_sets(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test


def model_selection(X_train, X_test, y_train, y_test):

    # create the dictionary of the models
    models = {'dtc': DecisionTreeClassifier(class_weight='balanced', random_state=42),
              'rfc': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
              'svm': svm.SVC(class_weight='balanced', random_state=42),
              'abc': AdaBoostClassifier(random_state=42),
              'xgbc': XGBClassifier(seed=42)}

    # create the dictionary of the pipelines
    pipes = {}

    for acronym, model in models.items():
        pipes[acronym] = Pipeline([('StandardScaler', StandardScaler()), ('model', model)])

    # create the dictionary of the parameters grid
    param_grids = {}

    # Update param_grids
    ## The parameter grid for decision tree
    param_grids['dtc'] = [{'model__min_samples_split': [2, 100, 200],
                           'model__min_samples_leaf': [1, 10, 20]}]

    ## the parameter grid for random forest
    param_grids['rfc'] = [{'model__min_samples_split': [2, 10, 100],
                           'model__min_samples_leaf': [1, 2, 5]}]

    ## the parameter grid for svm
    param_grids['svm'] = [{'model__C': [20, 100, 160],
                           'model__gamma': [0.00001, 0.0001, 0.001],
                           'model__kernel': ['rbf']}]

    ## The parameter grid for Adaboost
    param_grids['abc'] = [{'model__learning_rate': [1, 1.2]}]

    ##The parameter grid for xgboost
    param_grids['xgbc'] = [{'model__eta': [0.00000001, 0.0000001],
                            'model__gamma': [0.001, 0.01, 0.1],
                            'model__lambda': [0.00000001, 0.0000001]}]

    # run GridSearchCV
    best_score_param_estimator_gs = []
    for acronym in pipes.keys():
        # GridSearchCV
        gs = GridSearchCV(estimator=pipes[acronym],
                          param_grid=param_grids[acronym],
                          scoring='f1',
                          n_jobs=-1,
                          cv=StratifiedKFold(n_splits=3,  # can adjust kfolds
                                             shuffle=True,
                                             random_state=0),
                          return_train_score=True)

        # Fit the pipeline
        gs = gs.fit(X_train, y_train)

        # Update best_score_param_estimator_gs
        best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

    # save and print results
    # Sort best_score_param_estimator_gs in descending order of the best_score_
    best_score_param_estimator_gs = sorted(best_score_param_estimator_gs, key=lambda x: x[0], reverse=True)

    # save best_score_param_estimator_gs
    df_results = pd.DataFrame(best_score_param_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])
    df_results.to_csv('Sunday1_binary_classification_model_selection_results.csv', index=False)
    print(df_results)

    df_prediction_results = pd.DataFrame(y_test, columns=['y_test_orig'])
    df_prediction_evaluation = pd.DataFrame({"Estimator": [],
                                             'accuracy': [],
                                             "precision": [],
                                             "recall": [],
                                             "f1-score": []})

    # making predictions on X_test
    for i in range(5):
        best_score_gs, best_param_gs, best_estimator_gs = best_score_param_estimator_gs[i]
        # get the prediction on the testing data using the best model of each classifier
        y_test_pred = best_estimator_gs.predict(X_test)

        # save prediction results to df_prediction_results
        df_prediction_results[str(i)] = y_test_pred

        # save accuracy, precision, recall and f1-score to df_evaluation
        accuracy = round(accuracy_score(y_test, y_test_pred), 4)
        precision = round(precision_score(y_test, y_test_pred), 4)
        recall = round(recall_score(y_test, y_test_pred), 4)
        f1 = round(f1_score(y_test, y_test_pred), 4)

        df_prediction_evaluation = df_prediction_evaluation.append({'Estimator': best_estimator_gs,
                                                                    'accuracy': accuracy,
                                                                    'precision': precision,
                                                                    'recall': recall,
                                                                    'f1-score': f1},
                                                                   ignore_index=True)

        # print classification report
        print("--------------Best Model", str(i + 1),
              "----------------------------------------------------------------")
        print(best_estimator_gs.get_params()['model'])
        print("")
        print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))
        print("")
        print(classification_report(y_test, y_test_pred))
        print("")
        print(confusion_matrix(y_test, y_test_pred))

        # plot
        cm_plot = confusion_matrix(y_test, y_test_pred)
        plot_confusion_matrix(cm_plot, title=str(i))

    # save results
    df_prediction_results.to_csv('Sunday1_y_test_results_for_top_binary_classification_model.csv')
    df_prediction_evaluation = df_prediction_evaluation.sort_values(by='precision', ascending=False)
    df_prediction_evaluation.to_csv('Sunday1_y_test_evaluation_results_for_top_binary_classification_model.csv')

    print('Finished')

    return best_score_param_estimator_gs

def Model_Ensemble(best_score_param_estimator, xtrain, xtest, ytrain, ytest):
        from sklearn.ensemble import StackingClassifier
        # the base estimators
        best_score_1, best_param_1, estimator_1 = best_score_param_estimator[0]
        best_score_2, best_param_2, estimator_2 = best_score_param_estimator[1]
        best_score_3, best_param_3, estimator_3 = best_score_param_estimator[2]
        best_score_4, best_param_4, estimator_4 = best_score_param_estimator[3]
        best_score_5, best_param_5, estimator_5 = best_score_param_estimator[4]

        estimators = [('estimator_5', estimator_5.get_params()['model']),
                      ('estimator_4', estimator_4.get_params()['model']),
                      ('estimator_3', estimator_3.get_params()['model']),
                      ('estimator_2', estimator_2.get_params()['model'])]
        # the stacking classifer
        sc = StackingClassifier(estimators=estimators, final_estimator=estimator_1.get_params()['model'])

        # train the stacking classifier on the training data
        sc.fit(xtrain, ytrain)
        y_test_pred = sc.predict(xtest)

        print("--------------Model Ensemble----------------------------------------------------------------")
        print("Accuracy:", '{:1.4f}'.format(accuracy_score(ytest, y_test_pred)))
        print("")
        print("Precision:", round(precision_score(ytest, y_test_pred), 4))
        print("")
        print("Recall:", round(recall_score(ytest, y_test_pred), 4))
        print("")
        print("f1-score:", round(f1_score(ytest, y_test_pred), 4))
        print("")
        print(classification_report(ytest, y_test_pred))
        print("")
        print(confusion_matrix(ytest, y_test_pred))

############################################################################
##                       Evaluation Results Plots                         ##
############################################################################

def plot_confusion_matrix(cm, classes=["Non-Tipper", "Tipper"], title="", cmap=plt.cm.Blues):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
    figure_title = title + '.png'
    plt.savefig(figure_title)


## Plotting Feature Importance

def random_forest(X_features, xtrain, ytrain):
    plt.clf()
    # get feature names
    features = X_features.columns

    # set decision tree classifier
    rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                 class_weight='balanced',
                                 criterion='gini', max_depth=None,
                                 max_features='auto',
                                 max_leaf_nodes=None, max_samples=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 min_samples_leaf=5, min_samples_split=2,
                                 min_weight_fraction_leaf=0.0,
                                 n_estimators=100, n_jobs=None,
                                 oob_score=False, random_state=42,
                                 verbose=0, warm_start=False)
    # train the decision tree classifier on the training data
    rfc.fit(xtrain, ytrain)

    # plot feature importance
    # Get the dataframe of feature and importance
    f_importances = pd.Series(rfc.feature_importances_, features)

    # Sort the array in descending order of the importances
    f_importances = f_importances.sort_values(ascending=False)[:3]
    print(f_importances)
    f_importances = f_importances.sort_values(ascending=True)
    h = f_importances.plot(kind='barh', figsize=(10, 4), rot=0, fontsize=12,
                           color='cornflowerblue', edgecolor='white', linewidth=1.2)

    h.set_xlabel('Importance (Random Forest Classifier)', fontsize=12)
    h.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('Sunday_feature_importance_rfc.png')
    # reference: https://stackoverflow.com/questions/44511636/matplotlib-plot-feature-importance-with-feature-names


def decision_tree_feature_importance(X_features, xtrain, xtest, ytrain, ytest):
    plt.clf()
    # get feature names
    features = X_features.columns

    # set decision tree classifier
    dtc = DecisionTreeClassifier(ccp_alpha=0.0, class_weight='balanced',
                                 criterion='gini', max_depth=None,
                                 max_features=None, max_leaf_nodes=None,
                                 min_impurity_decrease=0.0,
                                 min_impurity_split=None,
                                 min_samples_leaf=20,
                                 min_samples_split=200,
                                 min_weight_fraction_leaf=0.0,
                                 presort='deprecated', random_state=42,
                                 splitter='best')
    # train the decision tree classifier on the training data
    dtc.fit(xtrain, ytrain)

    # plot feature importance
    # Get the dataframe of feature and importance
    f_importances = pd.Series(dtc.feature_importances_, features)

    f_importances = f_importances.sort_values(ascending=False)[:3]
    print(f_importances)
    f_importances = f_importances.sort_values(ascending=True)
    h = f_importances.plot(kind='barh', figsize=(10, 4), rot=0, fontsize=12,
                           color='lightseagreen', edgecolor='white', linewidth=1.2)

    h.set_xlabel('Importance (Decision Tree Classifier)', fontsize=12)
    h.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('Sunday_feature_importance_dtc.png')

    return f_importances


def XGBoost_feature_importance(X_features, xtrain, ytrain):
    plt.clf()
    # get feature names
    features = X_features.columns

    # set decision tree classifier
    xgbc = XGBClassifier(base_score=0.5, booster='gbtree',
                         colsample_bylevel=1, colsample_bynode=1,
                         colsample_bytree=1, eta=1e-08, gamma=0.001,
                         learning_rate=0.1,
                         max_delta_step=0, max_depth=3,
                         min_child_weight=1, missing=None,
                         n_estimators=100, n_jobs=1, nthread=None,
                         objective='binary:logistic', random_state=0,
                         reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                         seed=42, silent=None, subsample=1,
                         verbosity=1)
    # train the decision tree classifier on the training data
    xgbc.fit(xtrain, ytrain)

    # plot feature importance
    # Get the dataframe of feature and importance
    f_importances = pd.Series(xgbc.feature_importances_, features)

    f_importances = f_importances.sort_values(ascending=False)[:3]
    print(f_importances)
    f_importances = f_importances.sort_values(ascending=True)
    h = f_importances.plot(kind='barh', figsize=(10, 4), rot=0, fontsize=12,
                           color='lightcoral', edgecolor='white', linewidth=1.2)

    h.set_xlabel('Importance (XGBoost Classifier)', fontsize=12)
    h.set_ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('Sunday_feature_importance_xgbc.png')

    return f_importances


if __name__ == '__main__':
    # Classification Model
    dfc = pre_processing()
    dfc = feature_engineering(dfc)
    dfc = drop_na(dfc)
    X, y = get_features_and_targets(dfc)
    # X_samples, y_samples = generate_small_samples(X, y, sample_size = 20000)
    X_train, X_test, y_train, y_test = split_training_testing_sets(X.values, y.values)
    best_score_param_estimator_gs = model_selection(X_train, X_test, y_train, y_test)
    Model_Ensemble(best_score_param_estimator = best_score_param_estimator_gs,
                   xtrain = X_train, xtest = X_test, ytrain = y_train, ytest = y_test)

    # feature importance of decision tree
    f_importance = decision_tree_feature_importance(X_features=X, xtrain=X_train, xtest=X_test, ytrain=y_train,
                                                    ytest=y_test)
    random_forest(X_features=X, xtrain=X_train, ytrain=y_train)
    XGBoost_feature_importance(X_features=X, xtrain=X_train, ytrain=y_train)
