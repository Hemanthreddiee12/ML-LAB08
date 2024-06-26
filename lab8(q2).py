import numpy as np
from collections import Counter
import math

class DecisionTree:
    def __init__(self):
        pass
    
    def entropy(self, labels):
        label_counts = Counter(labels)
        num_labels = len(labels)
        entropy = 0
        for count in label_counts.values():
            prob = count / num_labels
            entropy -= prob * math.log(prob, 2)
        return entropy
    
    def information_gain(self, data, labels, feature_idx):
        # Calculate parent entropy
        parent_entropy = self.entropy(labels)
        
        # Calculate weighted sum of child entropies
        unique_values = np.unique(data[:, feature_idx])
        children_entropy = 0
        for value in unique_values:
            child_indices = np.where(data[:, feature_idx] == value)[0]
            child_labels = labels[child_indices]
            children_entropy += (len(child_labels) / len(labels)) * self.entropy(child_labels)
        
        # Calculate information gain
        information_gain = parent_entropy - children_entropy
        
        return information_gain
    
    def find_root_node(self, data, labels, feature_names):
        num_features = data.shape[1]
        best_feature_idx = None
        max_information_gain = -float('inf')
        
        for feature_idx in range(num_features):
            information_gain = self.information_gain(data, labels, feature_idx)
            if information_gain > max_information_gain:
                max_information_gain = information_gain
                best_feature_idx = feature_idx
                
        return best_feature_idx, feature_names[best_feature_idx]
    
    def equal_width_binning(self, feature_values, num_bins):
        min_val = min(feature_values)
        max_val = max(feature_values)
        bin_width = (max_val - min_val) / num_bins
        bins = [min_val + i * bin_width for i in range(num_bins)]
        binned_values = np.digitize(feature_values, bins)
        return binned_values
    
    def frequency_binning(self, feature_values, num_bins):
        sorted_values = np.sort(feature_values)
        bin_size = len(feature_values) // num_bins
        bins = [sorted_values[i * bin_size] for i in range(num_bins)]
        binned_values = np.digitize(feature_values, bins)
        return binned_values
    

# Example usage:
def load_data(dataset_path):
    data = np.genfromtxt(dataset_path, delimiter=',', dtype=str)
    features = data[0, :-1]
    labels = data[1:, -1]
    data = data[1:, :-1]
    return data, labels, features

def binning_decision_tree(dataset_path, binning_type='equal_width', num_bins=5):
    dt = DecisionTree()
    data, labels, feature_names = load_data(dataset_path)
    
    if binning_type == 'equal_width':
        binning_function = dt.equal_width_binning
    elif binning_type == 'frequency':
        binning_function = dt.frequency_binning
    else:
        print("Invalid binning type. Using default equal width binning.")
        binning_function = dt.equal_width_binning
        
    binned_data = []
    for feature_idx in range(data.shape[1]):
        feature_values = data[:, feature_idx]
        
        # Skip binning for non-numeric features
        if not np.issubdtype(feature_values.dtype, np.number):
            binned_data.append(feature_values)
            continue
        
        feature_values = feature_values.astype(float)
        
        binned_values = binning_function(feature_values, num_bins)
        binned_data.append(binned_values)
    
    binned_data = np.array(binned_data).T
    root_node_idx, root_node_name = dt.find_root_node(binned_data, labels, feature_names)
    print("Root node of the Decision Tree after binning:")
    print("Feature index:", root_node_idx)
    print("Feature name:", root_node_name)

# Example usage:
dataset_path = r"C:\Users\heman\OneDrive\Documents\SEM_4\ML\Lab\8\Unemployment_in_India.csv"  # Replace "your_dataset.csv" with the path to your dataset file
binning_decision_tree(dataset_path, binning_type='equal_width', num_bins=5)
