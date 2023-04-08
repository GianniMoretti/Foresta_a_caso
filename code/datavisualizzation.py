import pandas as pd
import lib.datanalysis as da


############################## DATASET DATA ####################################################
label_name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'risknum']
delete_this_row = [87,166,192,266,287,302]
dataframe_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
classes_feature_name = "risknum"

#read from UCI database with pandas
df = pd.read_csv(dataframe_name, names = label_name)

#Delete the row with missing value
df = df.drop(labels = delete_this_row, axis=0)

#Number of value per features
data, t = da.value_of_features(df, True)
print(t)

#PairPlot
da.pairplot(df, classes_feature_name, height=1)
