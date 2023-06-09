import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

import mylib.datanalysis as da
from mylib.treelearning import RandomForestClassifier as MyRandomForestClassifier

########################### MAIN ############################
#Parametri da cambiare per utilizzare un database diverso
label_name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risknum']
dataframe_name = "/home/giannimoretti/Desktop/Unifi/AI/Foresta_a_caso/database/processed.cleveland.data"
classes_feature_name = 'risknum'
categorical_column = [2,6,10,11,12]
criterion_type = 'gini'

#read from UCI database with pandas
df = pd.read_csv(dataframe_name, names = label_name)

#Delete the row with missing value
for col in df.columns:
    df = df[df[col] != '?']

#split data in x and y
x, y = da.split_x_y(df, classes_feature_name)

#number of iteration per value
N_p_value = 1

my_accuracy_train = []
skl_accuracy_train = []
my_accuracy_test = []
skl_accuracy_test = []
values = []

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

for value in range(1, 30, 5):
    sum_train_my = 0
    sum_train_skl = 0
    sum_test_my = 0
    sum_test_skl = 0

    for i in range(N_p_value):

        #SKlearn random forest
        skrf = RandomForestClassifier(n_estimators=value, criterion=criterion_type, max_depth=4, min_samples_split=2, max_features=7) 
        skrf.fit(x_train, y_train)
        y_pred_train_skl = skrf.predict(x_train)
        y_pred_test_skl = skrf.predict(x_test)

        #My decision tree classifier
        mrf = MyRandomForestClassifier(tree_num=value, max_samples=1, criterion=criterion_type, min_sample_split=2, max_depth=4, num_of_feature_on_split=7) 
        mrf.fit(x_train, y_train, categorical_column = categorical_column)
        y_pred_train = mrf.predict(x_train)
        y_pred_test = mrf.predict(x_test)

        sum_train_my += accuracy_score(y_train, y_pred_train)
        sum_test_my += accuracy_score(y_test, y_pred_test)
        sum_train_skl += accuracy_score(y_train, y_pred_train_skl)
        sum_test_skl += accuracy_score(y_test, y_pred_test_skl)

    my_accuracy_train.append(sum_train_my/N_p_value)
    my_accuracy_test.append(sum_test_my/N_p_value)
    skl_accuracy_train.append(sum_train_skl/N_p_value)
    skl_accuracy_test.append(sum_test_skl/N_p_value)
    values.append(value)


plt.subplot(1, 2, 1)
plt.ylim([0, 1.01])
plt.title("Train")
plt.xlabel("tree number")
plt.ylabel("Accuracy")
plt.plot(values, my_accuracy_train, color='blue', label='My_RF')
plt.plot(values, skl_accuracy_train, color='red', label='Skl_RF')
plt.legend()

plt.subplot(1, 2, 2)
plt.ylim([0, 1.01])
plt.title("Test")
plt.xlabel("Tree number")
plt.ylabel("Accuracy")
plt.plot(values, my_accuracy_test, color='blue', label='My_RF')
plt.plot(values, skl_accuracy_test, color='red', label='Skl_RF')
plt.legend()

plt.show()