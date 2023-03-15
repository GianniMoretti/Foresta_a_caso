import numpy as np

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
        self.criterion_function = get_criterion_function_from_string(criterion)
        self.num_of_feature_on_split = int(num_of_feature_on_split)
        self.root = Node()

    #devo riusare gli attributi che ho gia usato o toglierli?
    def fit(self, X, Y, categorical_column = []):
        pass

    def __decision_tree_learning(self, X, Y, current_depth, categorical_column):
        if len(X) >= self.min_sample_split and current_depth <= self.max_depth:
            #Select num_of_feature_on_split from the possible features
            #Find the best split from the selected features (considering if they are categorical or numerical)
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
    
    def plurality_value(self, Y):
        classes, counts = np.unique(Y, return_counts=True)
        return classes[np.where(counts == np.amax(counts))], counts

    def predict():
        pass

#Split criterions

def gini():
    pass

def entropy():
    pass

def log_loss():
    pass

def get_criterion_function_from_string(criterion):
    if criterion == 'gini':
        return gini
    elif criterion == 'entropy':
        return entropy
    elif criterion == 'log_loss':
        return log_loss
    else:
        return gini
