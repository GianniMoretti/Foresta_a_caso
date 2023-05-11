import numpy as np
import random as rn
import graphviz as gv
from sklearn.utils import resample
from tqdm import tqdm
import multiprocessing

class Node:
    def __init__(self, feature = None, feature_type = 'numerical', feature_value = None, criterior_value = None):
        self.feature = feature
        self.feature_value = feature_value
        self.feature_type = feature_type
        self.children = {}
        self.samples_count = 0
        self.samples_class_count = []
        self.criterion_value = criterior_value
        self.class_value = None

class DecisionTreeClassifier:
    def __init__(self, criterion = 'gini', min_sample_split = 2, max_depth = None, num_of_feature_on_split = None):
        #self.ccp_alpha = ccp_alpha
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.criterion_type = criterion
        self.criterion_function = self.__get_criterion_function_from_string(criterion)
        self.num_of_feature_on_split = int(num_of_feature_on_split)
        self.root = None

    #devo riusare gli attributi che ho gia usato o toglierli?
    def fit(self, X, Y, categorical_column = []):
        Y = np.resize(Y, (len(Y), 1))
        dataset = np.append(X, Y, axis=1)
        all_classes_value = np.unique(Y)
        self.root = self.__decision_tree_learning(dataset, 0, categorical_column, all_classes_value)

    def __decision_tree_learning(self, dataset, current_depth, categorical_column, all_classes_value):
        #split in X, Y
        X = dataset[:,:-1]
        Y = dataset[:,-1]
        #criterion value of this node
        criterion_value = self.criterion_function(Y)
        #value for this node
        class_value, counts = self.__plurality_value(Y, all_classes_value)
        #Select num_of_feature_on_split from the possible features
        if len(X) >= self.min_sample_split and current_depth < self.max_depth and criterion_value > 0:
            number_of_features = len(X[0])
            feature_selected_indexes = self.__select_features(number_of_features)
            #Find the best split from the selected features (considering if they are categorical or numerical)
            best_split = self.__find_best_split(dataset, criterion_value, len(X), feature_selected_indexes, categorical_column)
            while len(best_split) == 0:
                feature_selected_indexes = self.__select_features(number_of_features)
                #Find the best split from the selected features (considering if they are categorical or numerical)
                best_split = self.__find_best_split(dataset, criterion_value, len(X), feature_selected_indexes, categorical_column)
                #if the samples have the same value in all the column, this cycle go forever!!!! 
            #create this node
            if best_split['type'] == 'categorical':
                n = Node(best_split['feature_index'], 'categorical', None, criterion_value)
            else:
                n = Node(best_split['feature_index'], 'numerical', best_split['threshold'], criterion_value)
            n.class_value = class_value
            n.samples_class_count = counts
            n.samples_count = len(X)
            #call recursively this method for each of the dataset
            splitted_dataset = best_split['datasets']
            for feat_val in splitted_dataset.keys():
                if len(splitted_dataset[feat_val]) == 0:
                    #if there isn't any example for this split, return leaf node with as class, the class of this node
                    nn = Node(-1, 'leaf', -1, 0)
                    nn.samples_count = 0
                    nn.children = None
                    nn.samples_class_count = None
                    nn.class_value = class_value
                    #Aggiungere i campi che mancano
                    n.children[feat_val] = nn
                else:
                    nn = self.__decision_tree_learning(splitted_dataset[feat_val], current_depth + 1, categorical_column,all_classes_value)
                    n.children[feat_val] = nn
            #return a node with as children the different node from the different call
            return n
        else:
            nn = Node(-1, 'leaf', -1, criterion_value)
            nn.samples_count = len(X)
            nn.children = None
            nn.samples_class_count = counts
            nn.class_value = class_value
            return nn

    def __find_best_split(self, dataset, parent_criterion_value, num_of_parent_ex, selected_indexes, categorical_column):
        best_split = {}
        max_info_gain = -float('inf')
        for feature in selected_indexes:
            if feature in categorical_column:
                #categorical features
                possible_value = np.unique(dataset[:,feature])
                datasets = self.__split_dataset_categorical(dataset, feature, possible_value)
                info_gain = parent_criterion_value
                for dt in datasets.values():
                    #Take all the y for every dataset
                    y = dt[:,-1]
                    info_gain += -(len(y) / num_of_parent_ex) * self.criterion_function(y)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split['feature_index'] = feature
                    best_split['type'] = 'categorical'
                    best_split['info_gain'] = info_gain
                    best_split['datasets'] = datasets
            else:
                #numerical features
                possible_value = np.unique(dataset[:,feature])
                np.sort(possible_value)
                threshold_values = self.__find_threshold_value(possible_value)
                step = 1
                if len(threshold_values) > 100:
                    step = int(len(threshold_values) / 100)
                for i in range(0, len(threshold_values), step):
                    value = threshold_values[i]
                    dts = self.__split_dataset_numerical(dataset, feature, value)
                    dt_left = dts['left']
                    dt_right = dts['right']
                    w_l = len(dt_left) / num_of_parent_ex
                    w_r = len(dt_right) / num_of_parent_ex
                    info_gain = parent_criterion_value - (w_l * self.criterion_function(dt_left[:,-1]) + w_r * self.criterion_function(dt_right[:,-1]))
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split['feature_index'] = feature
                        best_split['threshold'] = value
                        best_split['type'] = 'numerical'
                        best_split['info_gain'] = info_gain
                        best_split['datasets'] = dts
        return best_split

    def __find_threshold_value(self, values):
        possible_values = []
        for i in range(0, len(values) - 1):
            possible_values.append((values[i] + values[i + 1]) / 2)
        return possible_values

    def __split_dataset_categorical(self, dataset, feature, possible_value):
        result = {}
        column = dataset[:,feature]
        for value in possible_value:            
            mask = column == value
            result[value] = dataset[mask]
            #result[value] = np.array([row for row in dataset if row[feature] == value])
        return result

    def __split_dataset_numerical(self, dataset, feature, threshold_value):
        column = dataset[:, feature]
        mask = column < threshold_value
        dataset_left = dataset[mask]
        dataset_right = dataset[~mask]
        return {'left': dataset_left, 'right': dataset_right}

    def __select_features(self, num_of_features):
        indexes = list(range(0, num_of_features))
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

    def __plurality_value(self, Y, all_classes_value):
        max = -1
        counts = []
        class_value = None
        for clc in all_classes_value:
            n = len([x for x in Y if x == clc])
            counts.append(n)
            if n >= max:
                max = n
                class_value = clc
        return class_value, counts

    def predict(self, X):
        if self.root == None:
            return [] 
        else:
            y_predict = []
            for row in X:
                y_predict.append(self.__predict_row(row))
            return y_predict

    def __predict_row(self, row):
        node = self.root
        while node.feature_type != 'leaf':
            if node.feature_type == 'categorical':
                feature_val = row[node.feature]
                if feature_val not in node.children.keys():
                    return node.class_value                                        #Non è del tutto giusto
                node = node.children[feature_val]
            else:
                feature_val = row[node.feature]
                if feature_val <= node.feature_value:
                    node = node.children['left']
                else:
                    node = node.children['right']
        return node.class_value

class RandomForestClassifier:
    def __init__(self, tree_num = 10, max_samples = None ,criterion = 'gini', min_sample_split = 2, max_depth = None, num_of_feature_on_split = None):
        self.tree_num = tree_num
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.min_sample_split = min_sample_split
        self.criterion_type = criterion
        self.num_of_feature_on_split = int(num_of_feature_on_split)
        self.forest = []
    
    def fit(self, X, Y, categorical_column = []):
        print('RANDOM FOREST\n----------------------------------------------------------')
        l = len(X)
        if self.max_samples != None:
            l = int(l * self.max_samples)
        print('Fertilizing the field...')
        for i in tqdm(range(0, self.tree_num), ncols = 100, desc ="Planting trees: "):    
            XB, YB = self.__resample(X, Y, True, l)
            mdt = DecisionTreeClassifier(self.criterion_type, self.min_sample_split, self.max_depth, self.num_of_feature_on_split)
            mdt.fit(XB, YB, categorical_column = categorical_column)
            self.forest.append(mdt)
        print('Forest done.\n')

    def predict(self, X):
        y_pred = []
        for row in X:
            y_values = []
            for t in self.forest:
                li = [row]
                y_values.append(t.predict(li))
            y_pred.append(self.__most_frequent(y_values))
        return y_pred
    
    def __most_frequent(self, classes):
        values, counts = np.unique(classes, return_counts=True)
        ind = list(counts).index(max(counts))
        return values[ind]
    
    def __resample(self, X, Y, replace = True, samples_num = -1):
        indexes = list(range(0, len(X)))
        if samples_num == -1:
            samples_num = len(X)
        if replace:
            selected_index = rn.choices(indexes, k = samples_num)
        else:
            selected_index = rn.sample(indexes, k = samples_num)
        
        return X[selected_index], Y[selected_index]


#Split criterions
def gini(y):
    gini_index = 0
    classes = np.unique(y)
    for cls in classes:
        p_cls = len(y[y == cls]) / len(y)
        gini_index += p_cls * (1 - p_cls)
    return gini_index

def entropy(y):
    entropy = 0
    classes = np.unique(y)
    for cls in classes:
        p_cls = len(y[y == cls]) / len(y)
        entropy += -p_cls * np.log2(p_cls)
    return entropy

def create_graphviz_tree(root, title, criterion_name, features_name):
    dot = gv.Digraph(comment=title, node_attr={ 'shape':'box', 'style':"filled, rounded", 'color':"lightblue", 'fontname':"helvetica" })
    recursive_graph(dot, root, 0, criterion_name, features_name)
    return dot

def recursive_graph(dot, root, myindex, criterion_name, features_name):
    if root.feature_type == 'leaf':
        myindex += 1
        s = 'type={type}\n\n{criterion}={criterion_value}\nsample={sample}\ncount={count}\nclass={c}'.format(type = root.feature_type, criterion = criterion_name,criterion_value = str(round(root.criterion_value, 3)), sample = root.samples_count, count = root.samples_class_count,c=str(root.class_value))
        dot.node(str(myindex),s)
        return myindex, myindex
    else:
        myindex += 1
        if root.feature_type == 'categorical':
            s = '{feature_name} = ?\n\ntype={type}\n{criterion}={criterion_value}\nsample={sample}\ncount={count}\nclass={c}'.format(type = root.feature_type,criterion = criterion_name, feature_name = features_name[root.feature], criterion_value =str(round(root.criterion_value, 3)), sample = root.samples_count, count = root.samples_class_count,c=str(root.class_value))
            dot.node(str(myindex),s)
        elif root.feature_type == 'numerical':
            s = '{feature_name}<={feature_value}\n\ntype={type}\n{criterion}={criterion_value}\nsample={sample}\ncount={count}\nclass={c}'.format(type = root.feature_type, feature_name = features_name[root.feature], criterion = criterion_name, feature_value = root.feature_value,criterion_value = str(round(root.criterion_value, 3)), sample = root.samples_count, count = root.samples_class_count,c=str(root.class_value))
            dot.node(str(myindex),s)
        else:
            s = 'type={type}\n\n{criterion}={criterion_value}\nsample={sample}\ncount={count}\nclass={c}'.format(type = root.feature_type, criterion = criterion_name,criterion_value = str(round(root.criterion_value, 3)), sample = root.samples_count, count = root.samples_class_count,c=str(root.class_value))
            dot.node(str(myindex),s)
        lastindex = myindex
        for node in root.children.keys():
            index, lastindex = recursive_graph(dot, root.children[node], lastindex, criterion_name, features_name)
            dot.edge(str(myindex), str(index))
        return myindex, lastindex

