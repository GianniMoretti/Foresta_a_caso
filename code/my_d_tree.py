import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from mylib.treelerning import DecisionTreeClassifier as MyDecisionTreeClassifier, create_graphviz_tree
import mylib.datanalysis as da

########################### MAIN ############################
label_name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risknum']
dataframe_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
classes_feature_name = 'risknum'
categorical_column = [2,5,6,8,10,11,12]
criterion_type = 'gini'

#read from UCI database with pandas
df = pd.read_csv(dataframe_name, names = label_name)

#Delete the row with missing value
for col in df.columns:
    df = df[df[col] != '?']

#split data in x and y
x, y = da.split_x_y(df, classes_feature_name)

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

#My decision tree classifier
mdt = MyDecisionTreeClassifier(criterion_type, 2, 3, 4)
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
