from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    iris_data = load_iris()
    X_iris = iris_data.data
    y_iris = iris_data.target

    rf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_iris.fit(X_iris, y_iris)

    iris_model_path = "data/iris_model.pkl"
    joblib.dump(rf_iris, iris_model_path)

    return rf_iris, iris_data

if __name__ == '__main__':
    train_and_save_model()