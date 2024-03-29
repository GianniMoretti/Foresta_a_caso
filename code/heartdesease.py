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
dataframe_name = r"C:\Users\jinnw\Desktop\Code\Foresta_a_caso\database\processed.cleveland.data"
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
N_p_value = 5

my_accuracy_train = []
skl_accuracy_train = []
my_accuracy_test = []
skl_accuracy_test = []
values = []

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

#One-hot-Encoding for sklearn
label_feature_name = label_name[1:]                    #Name of the attributes whitout class label name
tmp_train = pd.DataFrame(x_train, columns = label_feature_name)
x_train_skl = da.oneHotEncoding(tmp_train, label_name, categorical_column)
tmp_test = pd.DataFrame(x_test, columns = label_feature_name)
x_test_skl = da.oneHotEncoding(tmp_test, label_name, categorical_column)

while len(x_train_skl.columns) != len(x_test_skl.columns):
    #Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)
    label_feature_name = label_name[1:]                     #Name of the attributes whitout class label name
    tmp_train = pd.DataFrame(x_train, columns = label_feature_name)
    x_train_skl = da.oneHotEncoding(tmp_train, label_feature_name, categorical_column)
    tmp_test = pd.DataFrame(x_test, columns = label_feature_name)
    x_test_skl = da.oneHotEncoding(tmp_test, label_feature_name, categorical_column)

for value in range(1, 100, 10):
    sum_train_my = 0
    sum_train_skl = 0
    sum_test_my = 0
    sum_test_skl = 0

    for i in range(N_p_value):
        #SKlearn random forest
        skrf = RandomForestClassifier(n_estimators=value, criterion=criterion_type, max_depth=5, min_samples_split=2, max_features=3) 
        skrf.fit(x_train_skl, y_train)
        y_pred_train_skl = skrf.predict(x_train_skl)
        y_pred_test_skl = skrf.predict(x_test_skl)

        #My decision tree classifier
        mrf = MyRandomForestClassifier(tree_num=value, max_samples=1, criterion=criterion_type, min_sample_split=2, max_depth=5, num_of_feature_on_split=3) 
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
plt.ylim([0.4, 1.01])
plt.title("Train")
plt.xlabel("Tree number")
plt.ylabel("Accuracy")
plt.plot(values, my_accuracy_train, color='blue', label='My_RF')
plt.plot(values, skl_accuracy_train, color='red', label='Skl_RF')
plt.legend()

plt.subplot(1, 2, 2)
plt.ylim([0.4, 1.01])
plt.title("Test")
plt.xlabel("Tree number")
plt.ylabel("Accuracy")
plt.plot(values, my_accuracy_test, color='blue', label='My_RF')
plt.plot(values, skl_accuracy_test, color='red', label='Skl_RF')
plt.legend()

plt.show()