import numpy as np
import random as rn

class Node:
    def __init__(self, feature = None, feature_value = None, criterior_value = None):
        self.feature = feature
        self.feature_value = feature_value
        self.feature_type = ""
        self.children = []
        self.samples_count = 0
        self.samples_class_count = []
        self.criterior_value = None
        self.class_value = None

class DecisionTreeClassifier:
    def __init__(self, criterion = 'gini', min_sample_split = 2, max_depth = None, num_of_feature_on_split = None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.criterion_function = self.__get_criterion_function_from_string(criterion)
        self.num_of_feature_on_split = int(num_of_feature_on_split)
        self.root = Node()

    #devo riusare gli attributi che ho gia usato o toglierli?
    def fit(self, X, Y, categorical_column = []):
        pass

    def __decision_tree_learning(self, dataset, current_depth, categorical_column):
        #split in X, Y
        X = dataset[:,:-1]
        Y = dataset[:,-1]
        #value of this node
        criterion_value = self.criterion_function(Y)
        if len(X) >= self.min_sample_split and current_depth <= self.max_depth:
            #Select num_of_feature_on_split from the possible features
            number_of_features = len(X[0])
            feature_selected_indexes = self.__select_features(number_of_features)
            #Find the best split from the selected features (considering if they are categorical or numerical)
            best_split = self.__find_best_split(dataset, criterion_value, len(X), feature_selected_indexes, categorical_column)
            #split X, Y in different dataset with the different values of the selected feature
            #call recursively this method for each of the dataset
            #return a node with as children the different node from the different call
            pass
        elif len(X) == 0:
            #Devo tornare il valore del padre? Se sei qui vuol dire che non ci sono example per questo valore dell'attributo
            pass
        else:
            #return leaf node whit the plurality value
            class_value, counts = self.plurality_value(Y)
            return Node() 

    def __find_best_split(self, dataset, parent_criterion_value, num_of_parent_ex, selected_indexes, categorical_column):
        best_split = {}
        max_info_gain = -float('inf')

        for feature in selected_indexes:
            if feature in categorical_column:
                #categorical features
                possible_value = np.unique(dataset[:,feature])
                datasets = self.__split_dataset_categorical(dataset, feature, possible_value)
                info_gain = parent_criterion_value
                for dt in datasets:
                    #Take all the y for every dataset
                    y = dt[:,-1]
                    info_gain = -(len(y) / num_of_parent_ex) * self.criterion_function(y)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split['feature_index'] = feature
                    best_split['type'] = 'categorical'
                    best_split['info_gain'] = info_gain
                    best_split['datasets'] = dataset
            else:
                #numerical features
                possible_value = np.unique(dataset[:,feature])
                np.sort(possible_value)
                threshold_value = self.__find_threshold_value(possible_value)
                for value in possible_value:
                    dt_left, dt_right = self.__find_best_split(dataset, feature, value)
                    w_l = len(dt_left) / num_of_parent_ex
                    w_r = len(dt_right) / num_of_parent_ex
                    info_gain = parent_criterion_value - (w_l * self.criterion_function(dt_left[:,-1]) + w_r * self.criterion_function(dt_right[:,-1]))
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split['feature_index'] = feature
                        best_split['threshold'] = value
                        best_split['type'] = 'numerical'
                        best_split['info_gain'] = info_gain
                        best_split['datasets'] = [dt_left, dt_right]

        return best_split

    def __find_threshold_value(self, values):
        possible_values = []
        for i in range(0, len(values) - 1):
            possible_values.append((values[i] + values[i + 1]) / 2)
        return possible_values

    def __split_dataset_categorical(self, dataset, feature, possible_value):
        result = []
        for value in possible_value:
            result.append(np.array([row for row in dataset if row[feature] == value]))
        return result
    
    def __split_dataset_numerical(self, dataset, feature, threshold_value):
        dataset_left = [row for row in dataset if dataset[feature] <= threshold_value]
        dataset_right = [row for row in dataset if dataset[feature] > threshold_value]
        return dataset_left, dataset_right

    def __select_features(self, num_of_features):
        indexes = range(0, num_of_features)
        if self.num_of_feature_on_split == 0 or self.num_of_feature_on_split < 0 or self.num_of_feature_on_split > num_of_features:
            return indexes
        return rn.sample(indexes, self.num_of_feature_on_split)

    def __get_criterion_function_from_string(self, criterion):
        if criterion == 'gini':
            return gini
        elif criterion == 'entropy':
            return entropy
        else:
            return gini
    
    def plurality_value(self, Y):
        classes, counts = np.unique(Y, return_counts=True)
        return classes[np.where(counts == np.amax(counts))], counts

    def predict():
        pass

#Split criterions
def gini(y):
    gini_index = 0
    classes = np.unique(y)
    for cls in classes:
        p_cls = len(y[y == cls]) / len(y)    #check if is ok
        gini_index += p_cls * (1 - p_cls)
    return gini_index

def entropy(y):
    entropy = 0
    classes = np.unique(y)
    for cls in classes:
        p_cls = len(y[y == cls]) / len(y)    #check if is ok
        entropy += -p_cls * np.log2(p_cls)
    return entropy


