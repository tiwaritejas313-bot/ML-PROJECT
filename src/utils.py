import os
import sys
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
def evaluate_models(X_train, y_train, X_test, y_test, models, param=None, params=None):
    try:
        report = {}
        hyperparameters = param if param is not None else (params if params is not None else {})

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            para = hyperparameters.get(model_name, {})

            if para:
                gs = GridSearchCV(model, para, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)