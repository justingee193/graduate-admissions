import split
import model
import visuals
import numpy as np
import pandas as pd

def main():
    data1 = pd.read_csv('Admission_Predict.csv')
    data2 = pd.read_csv('Admission_Predict_Ver1.1.csv')
    
    data = pd.concat([data1, data2]).drop('Serial No.', axis = 1)
    target = data['Chance of Admit ']
    features = data.drop('Chance of Admit ', axis = 1)
    
    X_train, X_test, y_train, y_test = split(features, target)
    
    gb = model.base_model()
    pred = gb.predict(X_test)
    
    print("Our baseline model without tuning gave an R2 of {}".format(
            model.performance_metric(y_test, pred)))
    
    # Tune 1
    model.optimize(X_train, y_train, regressor=gb, 
         parameter={'n_estimators' : [1, 2, 4, 8, 16, 32, 64, 100]})  
    visuals.plot_optimization(regressor=gb, 
                              parameter={'n_estimators' : [1, 2, 4, 8, 16, 32, 64, 100]})
    gb = gb.set_params(n_estimators=50)
    
    # Tune 2
    model.optimize(X_train, y_train, regressor=gb, 
                   parameter={'max_depth' : range(2, 12, 2),
                              'min_samples_split': range(6,18,2)})
    visuals.plot_optimization(regressor=gb, 
                              parameter={'max_depth' : range(2, 20, 2)})
    gb = gb.set_params(max_depth=10)
    
    # Tune 3
    model.optimize(X_train, y_train, regressor=gb,
                   parameter={'min_samples_split': range(6,18,2),
                              'min_samples_leaf': [3,5,7,9,12,15]})
    visuals.plot_optimization(regressor=gb,
                              parameter={'min_samples_split':range(6,18,2)})
    gb = gb.set_params(min_samples_split=6)
    visuals.plot_optimization(regressor=gb,
                              parameter={'min_samples_leaf': [3,5,7,9,12,15]})
    gb = gb.set_params(min_samples_leaf=3)
    
    # Tune 4
    model.optimize(X_train, y_train, regressor=gb,
                   parameter={'max_features' : range(1, 8)})
    visuals.plot_optimization(regressor=gb, 
                              parameter={'max_features' : range(1, 8)})
    gb = gb.set_params(max_features=3)
    
    # Tune 5
    model.optimize(X_train, y_train, regressor=gb, 
             parameter={'subsample' : [0.7,0.75,0.8,0.85,0.9,0.95]})
    visuals.plot_optimization(regressor=gb, 
                      parameter={'subsample' : [0.7,0.75,0.8,0.85,0.9,0.95]})
    gb = gb.set_params(subsample=0.95)
    
    # Tune 6
    model.robust_model(gb, rates=[0.05, 0.01, 0.005, 0.005], 
                       trees=[100, 500, 1000, 1500])
    gb = gb.set_params(learning_rate=0.005, n_estimators=1500)
    
    gb = gb.fit(X_train, y_train)

    pred = gb.predict(X_test)
    print('Rsquare score of {}'.format(
            np.round(model.performance_metric(y_test, pred), decimals=5)))
    
    visuals.feature_importance(features.columns, gb.feature_importances_)
    
if __name__ == "__main__":
    main()
    
