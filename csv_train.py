import model_trainer
import pandas as pd


def load_data(data_file):
    df = pd.read_csv(data_file, skiprows=1, header=None)
    Xindex = [2, 3]
    Xindex += range(6, 36)
    X = df.iloc[:, Xindex].values
    y = df.iloc[:, 0].values
    return X, y


if __name__ == "__main__":
    data_file = "data.csv"
    X, y = load_data(data_file)
    mt = model_trainer.ModelTrainer()
    mt.load_X_y(X, y)

    # 采用MLPClassifier训练
    # MLP可选solver 有 ["lbfgs", "adam", "sgd"]
    MLP_accuracies = mt.train_and_evaluate_classifier(
        "csv",
        "MLP",
        solver="adam",
        hidden_layer_sizes=(100, 200),
        max_iter=10000,
        alpha=0.99,
    )

    # # 采用SVC训练
    # # SVC可选kernel 有 ["linear", "poly", "rbf", "sigmoid"]
    # SVC_accuracies = mt.train_and_evaluate_classifier("SVC", kernel="rbf", C=10)

    # # 采用RandomForestClassifier训练
    # RF_accuracies = mt.train_and_evaluate_classifier("RF", n_estimators=200)

    # # 采用KNeighborsClassifier训练
    # KNN_accuracies = mt.train_and_evaluate_classifier("KNN", n_neighbors=3)

    # # 评估模型参数取值
    # mt.evaluate_params(MLPClassifier(solver="adam", hidden_layer_sizes=(100, 200), max_iter=10000), "alpha", [0.001,0.005,0.01,0.05,0.1,0.5,0.999])
