from model import performance_metric
import matplotlib.pyplot as plt
from seaborn import heatmap, pairplot

def correlation_plot(corr):
    return heatmap(corr)

def pairplot(data, y_vars, x_vars, hue):
    return pairplot(data, y_vars = ['Chance of Admit '], x_vars, 
                    hue='University Rating')

def plot_optimization(regressor, parameter, X_train, X_test, y_train, y_test): 
    train_results = []
    test_results = []
    
    for key in parameter.keys():
        temp_key = key
    
    for i in range(len(parameter[key])):
        new_dict = {key : parameter[key][i]}
        
        model = regressor.set_params(**new_dict) 
        model = model.fit(X_train, y_train)
    
        train_pred = model.predict(X_train)
        train_results.append(performance_metric(y_train, train_pred))
    
        test_pred = model.predict(X_test)
        test_results.append(performance_metric(y_test, test_pred))

    line1, = plt.plot(parameter[key], train_results, 'b', label='Train R2')
    line2, = plt.plot(parameter[key], test_results, 'r', label='Test R2')
    plt.ylabel('R^2 score')
    plt.xlabel(key)
    plt.legend()
    plt.grid()
    plt.show()
    
def feature_importance(features, model_importance):
    plt.bar(features.columns, model_importance)
    plt.xticks(rotation='vertical')
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
