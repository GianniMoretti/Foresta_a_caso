import numpy as np
import random as rn
import graphviz as gv
from tqdm import tqdm
import cProfile
from math import floor

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

    def fit(self, X, Y, categorical_column = [], performance = False):
        if performance:
            profiler = cProfile.Profile()
            profiler.enable()
        Y = np.resize(Y, (len(Y), 1))
        dataset = np.append(X, Y, axis=1)
        all_classes_value = np.unique(Y)
        self.root = self.__decision_tree_learning(dataset, 0, categorical_column, all_classes_value)
        if performance:
            # Ferma il profiler
            profiler.disable()
            # Visualizza i risultati del profiler
            profiler.print_stats()

    def __decision_tree_learning(self, dataset, current_depth, categorical_column, all_classes_value):
        #split in X, Y
        X = dataset[:,:-1]
        Y = dataset[:,-1]
        #value for this node
        class_value, counts = self.__plurality_value(Y, all_classes_value)
        #criterion value of this node
        criterion_value = self.criterion_function(counts)
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
                    classes, counts = np.unique(y, return_counts=True)
                    #devi continuare qua
                    info_gain += -(len(y) / num_of_parent_ex) * self.criterion_function(counts)
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_split['feature_index'] = feature
                    best_split['type'] = 'categorical'
                    best_split['info_gain'] = info_gain
                    best_split['datasets'] = datasets
            else:
                #numerical features
                index = dataset[:, feature].argsort()
                sorted_dataset = dataset[index]
                split_index = 0
                dt_lenght = len(dataset)   #forse non serve
                classes, right_counts = np.unique(sorted_dataset[:,-1], return_counts=True)
                left_count = np.zeros(len(classes), dtype=int)
                encoded_cls = encodeLabels(sorted_dataset[:,-1], classes)

                while split_index <= dt_lenght - 2:
                    index = encoded_cls[split_index]
                    left_count[index] += 1
                    right_counts[index] -= 1
                    split_index += 1
                    #Da cambiare l'== con un controllo di precisione
                    while sorted_dataset[split_index - 1, feature] == sorted_dataset[split_index, feature] and split_index <= dt_lenght - 2:
                        index = encoded_cls[split_index]
                        left_count[index] += 1
                        right_counts[index] -= 1
                        split_index += 1

                    w_l = split_index / num_of_parent_ex
                    w_r = (dt_lenght - split_index) / num_of_parent_ex
                    info_gain = parent_criterion_value - (w_l * self.criterion_function(left_count) + w_r * self.criterion_function(right_counts))

                    if info_gain >= max_info_gain:
                        max_info_gain = info_gain
                        best_split['feature_index'] = feature
                        best_split['threshold'] = floor(((sorted_dataset[split_index - 1, feature] + sorted_dataset[split_index, feature]) / 2) * 100) / 100
                        best_split['type'] = 'numerical'
                        best_split['info_gain'] = max_info_gain
                        #Il problema è qui, l'index dovrebbe essere uno in più ma non puoi fare questo caso perche ti va in index out of bound
                        best_split['datasets'] = {'left': sorted_dataset[:split_index], 'right': sorted_dataset[split_index:]}
        return best_split

    def __split_dataset_categorical(self, dataset, feature, possible_value):
        result = {}
        column = dataset[:,feature]
        for value in possible_value:            
            mask = column == value
            result[value] = dataset[mask]
        return result

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
            mask = Y == clc
            n = len(Y[mask])
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

def gini(counts):
    tot = np.sum(counts, dtype=int)
    probabilities = counts / tot
    gini_index = np.sum(probabilities * (1 - probabilities), dtype=float)
    return gini_index

def entropy(counts):
    tot = np.sum(counts, dtype=int)
    probabilities = counts / tot
    entropy = np.sum(-probabilities * np.log2(probabilities), dtype=float)
    return entropy

def encodeLabels(labels, classes):
    class_index = {}
    index = 0
    for c in classes:
        class_index[c] = index
        index += 1
    
    return [class_index[l] for l in labels]

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

