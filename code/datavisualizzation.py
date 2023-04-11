import pandas as pd
import mylib.datanalysis as da


############################## DATASET DATA ####################################################
label_name = ['poisonous', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color', 'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring', 'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
dataframe_name = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
classes_feature_name = "poisonous"

#read from UCI database with pandas
df = pd.read_csv(dataframe_name, names = label_name)

#Delete the row with missing value
for col in df.columns:
    df = df[df[col] != '?']

print("\n\n")

#Number of value per features
data, t = da.value_of_features(df, True)
print(t)

#PairPlot
#da.pairplot(df, classes_feature_name, height=1)
