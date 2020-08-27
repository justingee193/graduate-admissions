from sklearn.model_selection import train_test_split

def split(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
            features, 
            target, 
            train_size = 0.8, 
            random_state = 444)
    
    return X_train, X_test, y_train, y_test