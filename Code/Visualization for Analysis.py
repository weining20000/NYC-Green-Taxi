############################################################################
#                            Import Packages                               #
############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import Model_Classification_3 as A

############################################################################
#                            Data Pre-processing                           #
############################################################################

def feature_encoding(X):
    # change cerntain int type to object type
    X['vendorid'] = X['vendorid'].astype(object)
    X['ratecodeid'] = X['ratecodeid'].astype(object)
    X['trip_type'] = X['trip_type'].astype(object)
    X['passenger_range'] = X['passenger_range'].astype(object)
    X['distance_range'] = X['distance_range'].astype(object)
    X['time_range'] = X['time_range'].astype(object)
    X['pickuphour_range'] = X['pickuphour_range'].astype(object)
    X['dropoffhour_range'] = X['dropoffhour_range'].astype(object)
    X['congestion_surcharge'] = X['congestion_surcharge'].astype(object)

    df_cat = cat_var_checker(X)

    # create categorical target variable
    X['tip_cat'] = 'non_tipper'
    X.loc[X['tip_pct'] > 0, 'tip_cat'] = 'tipper'

    return X, df_cat

# identify the categorical variables
def cat_var_checker(df):
    # get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           for var in df.columns if (df[var].dtype == 'object')],
                          columns=['var', 'nunique'])
    # sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    return df_cat

############################################################################
#                  Plots for Modeling Results and Analysis                 #
############################################################################

def feature_relationship_plot(dfc):
    # continuous variables
    fig, ax = plt.subplots(3, 3, figsize=(8, 8), sharey=True)
    fig.subplots_adjust(hspace=.4)

    cont_list = ['trip_distance', 'trip_time', 'trip_speed', 'passenger_count',
                 'fare_amount', 'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge']

    for i, var in enumerate(cont_list):
        c = int(i / 3)
        r = (i + 3) % 3
        ax[c][r].scatter(x=var, y="tip_pct", data=dfc.sample(1000), color='lightskyblue', alpha=.4)
        ax[c][r].set(title=var, ylabel='Tip%');

    fig.savefig('barplot_matrix_of_continous_variables.png', bbox_inches='tight')

    # categorical variables
    fig, ax = plt.subplots(3, 3, figsize=(8, 8), sharey=True)
    fig.subplots_adjust(wspace=.2, hspace=.8)

    cat_list = ['ratecodeid', 'trip_type', 'pickup_boro', 'dropoff_boro',
                'weekend', 'passenger_range', 'distance_range', 'time_range',
                'pickuphour_range']

    for i, var in enumerate(cat_list):
        c = int(i / 3)
        r = (i + 3) % 3
        sns.barplot(x=var, y="tip_pct", data=dfc.sample(10000), color='lightskyblue',
                    ax=ax[c][r])  # sample to reduce computation time
        ax[c][r].set(title=var, ylabel='Tip%', xlabel='')
        ax[c][r].set_xticklabels(ax[c][r].get_xticklabels(), rotation=90);
    fig.savefig('barplot_matrix_of_categorical_variables.png', bbox_inches='tight')

def correlation_heatmap(df):
    sns.set(style="white")
    # select varibles for modeling
    variables = ['tip_cat', 'tip_pct', 'vendorid', 'ratecodeid', 'trip_distance', 'fare_amount', 'extra',
                  'mta_tax', 'tolls_amount', 'improvement_surcharge', 'trip_type',
                  'congestion_surcharge', 'pickup_boro', 'pickup_servicezone',
                  'dropoff_boro', 'dropoff_servicezone', 'pickup_day',
                  'dropoff_day', 'weekend', 'passenger_range', 'distance_range',
                  'time_range', 'pickuphour_range', 'dropoffhour_range']

    df_modeling = df[variables]

    # # convert categorical variables to numeric type
    cat_columns = df_modeling.select_dtypes(['object']).columns
    for col_name in cat_columns:
        df_modeling[col_name] = df_modeling[col_name].astype('category')
    df_modeling[cat_columns] = df_modeling[cat_columns].apply(lambda x: x.cat.codes)

    # plot correlation heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(df_modeling.corr(), annot = True, cmap = 'RdBu_r')
    plt.savefig('correlation_heatmap.png', bbox_inches='tight')

    df = df_modeling.corr()
    remove_features = df.loc[(df['tip_cat'] < 0.03) & (df['tip_cat'] > -0.03)]
    selected_features = df.drop(remove_features.index)
    corr_features = selected_features.index.tolist()
    corr_features.remove('tip_cat')
    corr_features.remove('tip_pct')

    # reference: https://stackoverflow.com/questions/32011359/convert-categorical-data-in-pandas-dataframe/32011969
    return df_modeling, corr_features

def boxplot_categorical_fare_amount(df):
    # plot fare amount boxplot by tipper and non-tipper
    plt.clf()
    df.boxplot(by = 'tip_cat', column = ['fare_amount'])
    plt.ylabel('Fare Amount (USD)')
    plt.title('Boxplot of Fare Amount Grouped by Tipping Behavior')
    plt.suptitle("")
    plt.savefig('boxplot_fare_amount_vs_y.png', bbox_inches='tight')
    print("mean fare amount by tipping type:\n", df[['fare_amount', 'tip_cat']].groupby('tip_cat').mean())

def bar_plot_categorical_congestion_surcharge(df):
    plt.clf()
    # bar plot
    # numbers of pairs of bars
    N = len(df['congestion_surcharge'].unique())

    # Specify the values (count) of the tipper bars (height)
    tipper_counts = df.loc[df['tip_cat'] == 'tipper']['congestion_surcharge'].value_counts().sort_index()
    tipper_bar = tuple(tipper_counts.values.reshape(1, -1)[0])

    # Specify the values (counts) of the non-tipper bars (height)
    non_tipper_counts = df.loc[df['tip_cat'] == 'non_tipper']['congestion_surcharge'].value_counts().sort_index()
    non_tipper_counts = non_tipper_counts.append(pd.Series([0], index=[0.75])).sort_index()
    non_tipper_bar = tuple(non_tipper_counts.values.reshape(1, -1)[0])

    # position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # width of a bar
    width = 0.3

    # plotting
    plt.bar(ind, tipper_bar, width, label = 'Tipper',
            color=['palevioletred', 'palevioletred', 'palevioletred', 'palevioletred'],
            edgecolor='white', linewidth=1.2)
    plt.bar(ind + width, non_tipper_bar, width, label = 'Non-tipper',
            color=['navajowhite', 'navajowhite', 'navajowhite', 'navajowhite'],
            edgecolor='white', linewidth=1.2)

    plt.ylabel('Count')
    plt.title('Distribution of Tipper and Non-tipper by Congestion Surcharge')
    plt.xticks(ind + width / 2, ('$0.00', '$0.75', '$2.50', '$2.75'))
    plt.legend(loc='best')
    plt.savefig('Barplot_congestion_surcharge_vs_y.png', bbox_inches='tight')

def boxplot_categorical_extra(df):
    # plot fare amount boxplot by tipper and non-tipper
    plt.clf()
    df.boxplot(by = 'tip_cat', column = ['extra'])
    plt.ylabel('Extra (USD)')
    plt.title('Boxplot of Extra Fees Grouped by Tipping Behavior')
    plt.suptitle("")
    plt.savefig('boxplot_extra_fee_vs_y.png', bbox_inches='tight')
    print("mean extra by tipping type:\n", df[['extra', 'tip_cat']].groupby('tip_cat').mean())

def bar_plot_continuous_pickup_boro(df):
    plt.clf()
    # bar plot
    # numbers of pairs of bars
    N = len(df['pickup_boro'].unique())

    # Specify the values (count) of the tipper bars (height)
    tipper_counts = df.loc[df['tip_cat'] == 'tipper']['pickup_boro'].value_counts().sort_index()
    tipper_bar = tuple(tipper_counts.values.reshape(1, -1)[0])
    # Specify the values (counts) of the non-tipper bars (height)
    non_tipper_counts = df.loc[df['tip_cat'] == 'non_tipper']['pickup_boro'].value_counts().sort_index()
    non_tipper_bar = tuple(non_tipper_counts.values.reshape(1, -1)[0])

    # position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # width of a bar
    width = 0.3

    # plotting
    plt.bar(ind, tipper_bar, width, label = 'Tipper',
            color=['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'cornflowerblue'],
            edgecolor='white', linewidth=1.2)
    plt.bar(ind + width, non_tipper_bar, width, label = 'Non-tipper',
            color=['lightseagreen', 'lightseagreen', 'lightseagreen', 'lightseagreen', 'lightseagreen'],
            edgecolor='white', linewidth=1.2)

    plt.ylabel('Count')
    plt.title('Distribution of Tipper and Non-tipper by Pickup Borough')
    plt.xticks(ind + width / 2, ('Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Unknown'))
    plt.legend(loc='best')
    plt.savefig('Barplot_pickup_boro_vs_y.png', bbox_inches='tight')


def bar_plot_continuous_pickup_servicezone(df):
    plt.clf()
    # bar plot
    # numbers of pairs of bars
    N = len(df['pickup_servicezone'].unique())

    # Specify the values (count) of the tipper bars (height)
    tipper_counts = df.loc[df['tip_cat'] == 'tipper']['pickup_servicezone'].value_counts().sort_index()
    tipper_bar = tuple(tipper_counts.values.reshape(1, -1)[0])

    # Specify the values (counts) of the non-tipper bars (height)
    non_tipper_counts = df.loc[df['tip_cat'] == 'non_tipper']['pickup_servicezone'].value_counts().sort_index()
    non_tipper_bar = tuple(non_tipper_counts.values.reshape(1, -1)[0])


    # position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # width of a bar
    width = 0.3

    # plotting
    plt.bar(ind, tipper_bar, width, label = 'Tipper',
            color=['palevioletred', 'palevioletred', 'palevioletred'],
            edgecolor='white', linewidth=1.2)
    plt.bar(ind + width, non_tipper_bar, width, label = 'Non-tipper',
            color=['navajowhite', 'navajowhite', 'navajowhite'],
            edgecolor='white', linewidth=1.2)

    plt.ylabel('Count')
    plt.title('Distribution of Tipper and Non-tipper by Pickup Servicezone')
    plt.xticks(ind + width / 2, ('Airports', 'Boro Zone', 'Yellow Zone'))
    plt.legend(loc='best')
    plt.savefig('Barplot_pickup_servicezone_vs_y.png', bbox_inches='tight')

def bar_plot_continuous_trip_distance(df):
    plt.clf()
    # bar plot
    # numbers of pairs of bars
    N = len(df['distance_range'].unique())

    # Specify the values (count) of the tipper bars (height)
    tipper_counts = df.loc[df['tip_cat'] == 'tipper']['distance_range'].value_counts().sort_index()
    tipper_bar = tuple(tipper_counts.values.reshape(1, -1)[0])

    # Specify the values (counts) of the non-tipper bars (height)
    non_tipper_counts = df.loc[df['tip_cat'] == 'non_tipper']['distance_range'].value_counts().sort_index()
    non_tipper_bar = tuple(non_tipper_counts.values.reshape(1, -1)[0])


    # position of bars on x-axis
    ind = np.arange(N)

    # Figure size
    plt.figure(figsize=(10, 5))

    # width of a bar
    width = 0.3

    # plotting
    plt.bar(ind, tipper_bar, width, label = 'Tipper',
            color=['teal', 'teal', 'teal'],
            edgecolor='white', linewidth=1.2)
    plt.bar(ind + width, non_tipper_bar, width, label = 'Non-tipper',
            color=['steelblue', 'steelblue', 'steelblue'],
            edgecolor='white', linewidth=1.2)

    plt.ylabel('Count')
    plt.title('Distribution of Tipper and Non-tipper by Distance Range')
    plt.xticks(ind + width / 2, ('q1', 'q2', 'q3', 'q4'))
    plt.legend(loc='best')
    plt.savefig('Barplot_pickup_distance_range_vs_y.png', bbox_inches='tight')


if __name__ == '__main__':
    # Classification Model
    dfc = A.pre_processing()
    dfc = A.feature_engineering(dfc)
    dfc = A.drop_na(dfc)
    X, df_cat = feature_encoding(dfc)
    boxplot_categorical_fare_amount(df = X)
    bar_plot_categorical_congestion_surcharge(df = X)
    boxplot_categorical_extra(df = X)
    bar_plot_continuous_pickup_boro(df = X)
    bar_plot_continuous_pickup_servicezone(df = X)
    bar_plot_continuous_trip_distance(df = X)
    #feature_relationship_plot(X)

    # correlation heatmap
    df_modeling, correlated_features = correlation_heatmap(X)

