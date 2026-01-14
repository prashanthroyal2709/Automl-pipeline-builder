log_reg_params = {
    "model__C":[0.001,0.01,0.1,1,10,100],
    "model__solver":['lbfgs','liblinear','saga','sag'],
    "model__penalty":['l1','l2','elasticnet'],
    "model__max_iter":[100,1000,10000]
}

dt_params = {
    "model__criterion":['gini','entropy','log_loss'],
    "model__splitter":['best','random'],
    "model__max_depth":[5,10,20,25,30],
    "model__min_samples_split":[2,5,10],
    "model__min_samples_leaf":[1,2,3,4,5]
}

svc_params = {
    "model__C":[0.001,0.01,0.1,1,10,100],
    "model__kernel":['linear','rbf','poly'],
    "model__degree":[1,2,3,4],
    "model__gamma":['scale','auto'],
    "model__max_iter":[100,1000,10000]
}

knc_params = {
    "model__n_neighbors":[3,5,7,9,11],
    "model__weights":['uniform','distance'],
    "model__algorithm":['auto','kd_tree','ball_tree','brute'],
    "model__metric":['euclidean','manhattan'],
}

random_params = {
    "model__n_estimators":[100,200,300,400,500],
    "model__criterion":['gini','entropy','log_loss'],
    "model__max_depth":[None,5,10,15,20,25,30],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}

naive_params = {
    "model__var_smoothing": [1e-9, 1e-8, 1e-7]
}


linreg_params = {
    "model__fit_intercept": [True, False],
    "model__positive": [True, False]
}

dtr_params = {
    "model__criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "model__splitter": ["best", "random"],
    "model__max_depth": [5, 10, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

rfr_params = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__criterion": ["squared_error", "absolute_error"],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["auto", "sqrt", "log2"]
}

svr_params = {
    "model__C": [0.1, 1, 10, 100],
    "model__kernel": ["linear", "rbf", "poly"],
    "model__degree": [2, 3, 4],
    "model__gamma": ["scale", "auto"]
}

knr_params = {
    "model__n_neighbors": [3, 5, 7, 9],
    "model__weights": ["uniform", "distance"],
    "model__algorithm": ["auto", "kd_tree", "ball_tree", "brute"],
    "model__metric": ["euclidean", "manhattan"]
}
