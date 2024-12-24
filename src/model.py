# Remember the model is being trained on the training data set first 

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np

def eval_model(X,y,model,task="r"):
    if task=="r":
        model.fit(X,y)
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y,y_pred))
        mae  = mean_absolute_error(y,y_pred)
        r2 = r2_score(y,y_pred)

        print(f"Model {model.__class__.__name__}(Regression)")
        print(f"Predictions {y_pred}")
        print(f"Root Mean Squared Error {rmse}")
        print(f"Mean Absolute Error {mae}")
        print(f"R^2 {r2}")

    elif task=="c":
        model.fit(X,y)
        y_pred = model.predict(X)
        acc = accuracy_score(y,y_pred)
        pre = precision_score(y,y_pred)
        rec = recall_score(y,y_pred)
        f1 = f1_score(y,y_pred)

        print(f"Model {model.__class__.__name__}(Classification)")
        print(f"Accuracy {round(acc,2)}%")
        print(f"Precision {pre}")
        print(f"Recall Score {rec}")
        print(f"F1 Score {f1}")

    else: 
        raise ValueError("Invalid task. Put c for classification or r for regression")


