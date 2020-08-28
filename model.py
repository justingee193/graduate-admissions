from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

def performance_metric(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    
    return score

def base_model(learning_rate = 0.1, n_estimators = 100, max_depth = 3, 
               min_samples_split = 9, min_samples_leaf = 3, 
               max_features = 'sqrt',subsample = 0.8, X_train, y_train):
    
    gb = GradientBoostingRegressor(learning_rate = learning_rate, 
                                   n_estimators = n_estimators,
                                   max_depth = max_depth, 
                                   min_samples_split = min_samples_split,
                                   min_samples_leaf = min_samples_leaf,
                                   max_features = max_features,
                                   random_state = 444)  
    gb = gb.fit(X_train, y_train)
    
    return gb

def optimize(X, y, regressor, parameter, metric=performance_metric, n_jobs=-1):
    scorer = make_scorer(metric)
    cv = ShuffleSplit(X.shape[0], train_size=0.8, random_state=444)
    grid = GridSearchCV(regressor, parameter, scorer, n_jobs=n_jobs, cv=cv)
    grid = grid.fit(X, y)
    return grid.best_params_, grid.best_score_
    
def robust_model(regressor, rates, trees):
    if len(rates) != len(trees):
        return 'invalid lengths'
    else:
        num = len(rates)
    
    results = []
    
    for i in range(num):
        regressor = regressor.set_params(learning_rate=rates[i], n_estimators=trees[i])
        regressor = regressor.fit(X_train, y_train)

        pred = regressor.predict(X_test)
        results.append({np.round(performance_metric(y_test, pred), decimals=5) : [rates[i], trees[i]]})
        
    return results
