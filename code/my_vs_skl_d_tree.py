import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from mylib.treelearning import DecisionTreeClassifier as MyDecisionTreeClassifier, create_graphviz_tree
import mylib.datanalysis as da

########################### MAIN ############################
label_name = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
dataframe_name = r"C:\Users\jinnw\Desktop\Code\Foresta_a_caso\database\winequality-white.csv"
classes_feature_name = 'quality'
categorical_column = []
criterion_type = 'gini'

#read from UCI database with pandas
df = pd.read_csv(dataframe_name, sep = ';', names = label_name)

#Delete the row with missing value
for col in df.columns:
    df = df[df[col] != '?']

#split data in x and y
x, y = da.split_x_y(df, classes_feature_name)

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3)

#My decision tree classifier
mdt = MyDecisionTreeClassifier(criterion_type, 2, 10, -1)
mdt.fit(x_train, y_train, categorical_column = categorical_column)
y_pred_train = mdt.predict(x_train)
y_pred_test = mdt.predict(x_test)

#Graph
g = create_graphviz_tree(mdt.root, 'example', criterion_type, label_name)
g.view()

############################# TRAIN ####################################
#confusion Matrix and classification report
mc, lbl = da.confusion_matrix(y_train ,y_pred_train)
da.plot_confusion_matix(mc, lbl, "Cunfusion Matrix Train")

print("TRAIN DATA REPORT\n---------------------------------------------------------------------")
print(classification_report(y_train, y_pred_train, zero_division=0))

############################# TEST #####################################
#confusion Matrix and classification report
mc, lbl = da.confusion_matrix(y_test ,y_pred_test)
da.plot_confusion_matix(mc, lbl, "Cunfusion Matrix Test")

print("TEST DATA REPORT\n----------------------------------------------------------------------")
print(classification_report(y_test, y_pred_test, zero_division=0))

#DecisionTreeClassifier Sklearn
dt = DecisionTreeClassifier(criterion=criterion_type, max_depth=10, random_state = None)
dt.fit(x_train, y_train)
y_pred_train = dt.predict(x_train)
y_pred_test = dt.predict(x_test)

#Graph
graph = da.create_graphviz(df, dt, classes_feature_name)
graph.view()

############################# TRAIN ####################################
#confusion Matrix and classification report
mc, lbl = da.confusion_matrix(y_train ,y_pred_train)
da.plot_confusion_matix(mc, lbl, "Cunfusion Matrix Train")

print("TRAIN DATA REPORT\n---------------------------------------------------------------------")
print(classification_report(y_train, y_pred_train, zero_division=0))

############################# TEST ####################################
#confusion Matrix and classification report
mc, lbl = da.confusion_matrix(y_test ,y_pred_test)
da.plot_confusion_matix(mc, lbl, "Cunfusion Matrix Test")

print("TEST DATA REPORT\n---------------------------------------------------------------------")
print(classification_report(y_test, y_pred_test, zero_division=0))