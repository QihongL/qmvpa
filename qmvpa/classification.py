from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def tune_svm(X_train, y_train, Cs, kernel='linear'):
    # tune SVM
    tuning_svm = SVC(class_weight='balanced', kernel=kernel)
    tuning_grid = GridSearchCV(
        estimator=tuning_svm, param_grid=dict(C=Cs), scoring='accuracy')
    # fit the tuning_grid parameters
    tuning_grid.fit(X_train, y_train)
    # fit final model
    final_svm = SVC(C=tuning_grid.best_estimator_.C,
                    class_weight='balanced', kernel=kernel)
    return final_svm, tuning_grid
