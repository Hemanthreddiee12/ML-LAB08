import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DecisionTreeRootNodeDetector:
    def __init__(self):
        pass
    
    def calculate_entropy(self, y):
        # Calculate entropy
        classes, counts = np.unique(y, return_counts=True)
        entropy = 0
        total_samples = len(y)
        for count in counts:
            probability = count / total_samples
            entropy -= probability * np.log2(probability)
        return entropy
    
    def calculate_information_gain(self, X, y, feature_idx):
        # Calculate information gain for a specific feature
        total_entropy = self.calculate_entropy(y)
        unique_values, value_counts = np.unique(X[:, feature_idx], return_counts=True)
        weighted_entropy = 0
        for value, count in zip(unique_values, value_counts):
            subset_indices = np.where(X[:, feature_idx] == value)[0]
            subset_y = y[subset_indices]
            subset_entropy = self.calculate_entropy(subset_y)
            weighted_entropy += (count / len(X)) * subset_entropy
        information_gain = total_entropy - weighted_entropy
        return information_gain
    
    def find_root_node(self, X, y):
        num_features = X.shape[1]
        best_information_gain = -1
        best_feature_idx = None
        for i in range(num_features):
            information_gain = self.calculate_information_gain(X, y, i)
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_idx = i
        return best_feature_idx

# Function to load dataset from path
def load_dataset(dataset_path, target_column):
    data = pd.read_csv(dataset_path)
    # Assuming 'date' is not relevant for decision making, dropping it
    data = data.drop(columns=['Date'])
    
    # Binning numerical features 'e', 'f', 'g', and 'h' if needed
    
    # Example of binning 'e'
    data['Region'] = pd.qcut(data['Region'], q=4, labels=False)
    
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    return X, y

# Replace 'dataset_path' with the actual path of your dataset CSV file
dataset_path = r"C:\Users\heman\OneDrive\Documents\SEM_4\ML\Unemployment_in_India.csv"
target_column = 'Region'

# Load dataset
X, y = load_dataset(dataset_path, target_column)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the root node detector
root_node_detector = DecisionTreeRootNodeDetector()

# Find the root node feature index
root_node_feature_idx = root_node_detector.find_root_node(X_train, y_train)
print("Root node feature index:", root_node_feature_idx)
