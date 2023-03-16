import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import datanalysis as da
from sklearn.metrics import classification_report
from treelerning import DecisionTreeClassifier as MyDecisionTreeClassifier, create_graphviz_tree

########################### MAIN ############################
label_name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risknum']
delete_this_row = [87,166,192,266,287,302]
dataframe_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
classes_feature_name = "risknum"

df = pd.read_csv(dataframe_name, names = label_name)

#Delete the row with missing value
df = df.drop(labels = delete_this_row, axis=0)

#Number of value per features
data, t = da.value_of_features(df, True)
print(t)

#split data in x and y
x, y = da.split_x_y(df, classes_feature_name)

#Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

#My decision tree classifier
mdt = MyDecisionTreeClassifier('entropy', 2, 2, 6)
mdt.fit(x_train, y_train, categorical_column = [2,5,6,8,10,11,12])
y_pred_train = mdt.predict(x_train)

g = create_graphviz_tree(mdt.root, "example")
g.view()

# #confusion Matrix
# mc, lbl = da.confusion_matrix(y_train ,y_pred_train)
# da.plot_confusion_matix(mc, lbl, "Cunfusion Matrix")

# print(classification_report(y_train, y_pred_train, zero_division=0))

#DecisionTreeClassifier Sklearn
# dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=1)
# dt.fit(x_train, y_train)
# y_pred_train = dt.predict(x_train)
# y_pred_test = dt.predict(x_test)

# #confusion Matrix
# mc, lbl = da.confusion_matrix(y_test ,y_pred_test)
# da.plot_confusion_matix(mc, lbl, "Cunfusion Matrix")



#PairPlot
#da.pairplot(df, classes_feature_name, height=1)

#Graph
# graph = da.create_graphviz(df, dt, classes_feature_name)
# graph.view()

# print(classification_report(y_test, y_pred_test, zero_division=0))



