import numpy as np
import seaborn as snb
import matplotlib.pyplot as plt
import graphviz as gv
import pandas as pd
from sklearn.tree import export_graphviz
from sklearn.preprocessing import MinMaxScaler
from prettytable import PrettyTable

def value_of_features(dataframe, head = False):
    data = []
    t = PrettyTable(['Feature Name', 'Count', 'Value', 'Null Count'])
    dataframer = dataframe.replace({'?': None})
    features_num = 1
    for column in dataframe:
        row = []
        null_value = dataframer[column].isnull().sum()
        unique_vals = np.unique(dataframe[column])
        if not head:
            row.append(features_num)
            features_num = features_num + 1
        else:
            row.append(column)
        row.append(len(unique_vals))
        if len(unique_vals) < 10:
            row.append(unique_vals)
        else:
            row.append('')
        row.append(null_value)
        data.append(row)
        t.add_row(row)
    return data, t

def scale_dataframe(dataframe, features):
    scaler = MinMaxScaler()
    dataframe[features] = scaler.fit_transform(dataframe[features])

def pairplot(df, hue, height = 1.5):
    snb.pairplot(df, hue = hue, height = height)
    plt.show()

def create_graphviz(df, dt, fetures_name):
    dot_data = export_graphviz(dt, out_file = None,
    feature_names = df.drop(fetures_name, axis = 1).columns,
    class_names=df[fetures_name].unique().astype(str),
    filled = True, rounded=True,
    special_characters=True)
    return gv.Source(dot_data)

#CONFUSION_MATRIX
def confusion_matrix(y_true, y_pred, norm = True):
    label = np.unique(y_true)

    dict_cm = {}
    for l in label:
        dict_cm[l] = {}
        for l2 in label:
            dict_cm[l][l2] = 0

    for y_t, y_p in zip(y_true, y_pred):
        dict_cm[y_t][y_p] = dict_cm[y_t][y_p] + 1

    n = len(label)
    cm = np.zeros((n, n))
    num_el = np.zeros(n)
    for i in range(n):
        for j in range(n):
            cm[i][j] = dict_cm[label[i]][label[j]]
            num_el[i] = num_el[i] + cm[i][j]

    if norm:
        for i in range(n):
            for j in range(n):
                cm[i][j] = cm[i][j] / num_el[i]

    return cm, label

def plot_confusion_matix(cm, classes = None, title = "Confusion Matrix"):
    if classes is None:
        snb.heatmap(cm, vmin=0, vmax=1, annot = True, annot_kws={'size':20})
    else:
        snb.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0, vmax=1, annot = True, annot_kws={'size':20})

    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

def split_x_y(dataframe, classname):
    x = dataframe.drop(classname, axis = 1).values
    y = dataframe[classname].values

    return x,y

def oneHotEncoding(df, label_name, categorical_column):
    prefix = [ label_name[indx] for indx in categorical_column ]
    for p in prefix:
        df_p = pd.get_dummies(df[p], prefix=p)
        df = df.drop(columns=[p])
        df = pd.concat([df, df_p], axis=1)

    return df
