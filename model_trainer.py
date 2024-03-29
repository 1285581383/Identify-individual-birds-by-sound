from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import validation_curve


class ModelTrainer:
    def __init__(
        self, clf_random: int = 1, split_random: int = 1, test_size: float = 0.3
    ):
        """模型训练器

        Args:
            clf_random (int, optional): 训练模型随机种子. Defaults to 1.

            split_random (int, optional): 数据集划分随机种子,用于改变数据集划分为训练集和测试集的方法. Defaults to 1.

            test_size (float, optional): 划分的测试集占数据集比. Defaults to 0.3.
        """
        self.clf_random = clf_random
        self.split_random = split_random
        self.test_size = test_size
        self.output_log("=" * 8 + time.ctime(time.time()) + "=" * 8)

    # 直接读取X,y数据
    def load_X_y(self, X, y):
        self.X = X
        self.y = y

    def train_and_evaluate_classifier(
        self, X_name: str, classifier_type: str, **classifier_params
    ):
        """该函数用于训练模型, 可以选择不同的分类器类型, 以及不同的分类器参数.

        Args:
            classifier_type (str): 可选择的分类器类型:'MLP', 'SVC', 'RF', 'KNN'

            **classifier_params: 分类器的参数

        Returns:
            float: 训练集准确率及测试集准确率
        """
        # 标准化输入数据
        self.scaler = StandardScaler()
        # X = scaler.fit_transform(self.X)
        # 划分训练集和测试集
        if self.test_size == 0:
            X_train = self.X
            y_train = self.y
        elif 0 < self.test_size < 1:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.split_random
            )
        else:
            raise ValueError("Invalid test set size.")
        X_train = self.scaler.fit_transform(X_train)
        if self.test_size != 0:
            X_test = self.scaler.transform(X_test)

        if classifier_type == "MLP":
            # 使用MLPClassifier
            classifier = MLPClassifier(
                random_state=self.clf_random, **classifier_params
            )
        elif classifier_type == "SVC":
            # 使用SVC
            classifier = SVC(random_state=self.clf_random, **classifier_params)
        elif classifier_type == "RF":
            # 使用RandomForestClassifier
            classifier = RandomForestClassifier(
                random_state=self.clf_random, **classifier_params
            )
        elif classifier_type == "KNN":
            # 使用KNeighborsClassifier
            classifier = KNeighborsClassifier(**classifier_params)
        else:
            raise ValueError("Invalid classifier type.")

        print(f"分类器:{classifier_type}")
        self.output_log(f"分类器:{classifier_type}")
        clf_attributes = classifier.__dict__
        # 打印属性名和取值
        for attr_name, attr_value in clf_attributes.items():
            self.output_log(f"{attr_name}: {attr_value}")

        # 训练分类器
        start1 = time.time()
        print("训练分类器中...")
        classifier.fit(X_train, y_train)
        self.classifier = classifier
        print("训练分类器完成")
        end1 = time.time()

        # 保存分类器和标准化器
        joblib.dump(classifier, f"{X_name}_{classifier_type}_model.pkl")
        joblib.dump(self.scaler, f"{X_name}_{classifier_type}_scalar")

        start2 = time.time()
        if self.test_size != 0:
            # 预测及评估分类器 测试集
            with tqdm(total=len(X_test), desc="预测及评估分类器 测试集") as pbar_test:
                y_pred_test = []
                for x_test_sample in X_test:
                    prediction = classifier.predict([x_test_sample])[0]
                    y_pred_test.append(prediction)
                    pbar_test.update(1)

            accuracy_test = accuracy_score(y_test, y_pred_test)
        elif self.test_size == 0:
            accuracy_test = None

        # 预测及评估分类器 训练集
        with tqdm(total=len(X_train), desc="预测及评估分类器 训练集") as pbar_train:
            y_pred_train = []
            for x_train_sample in X_train:
                prediction = classifier.predict([x_train_sample])[0]
                y_pred_train.append(prediction)
                pbar_train.update(1)

        accuracy_train = accuracy_score(y_train, y_pred_train)
        end2 = time.time()

        print(f"测试集准确率:{accuracy_test}")
        print(f"训练集准确率:{accuracy_train}")
        print(f"训练时间:{f'{end1 - start1:.2f}' if end1 - start1 >= 0.01 else '<0.01'}s")
        print(
            f"预测及评估时间:{f'{end2 - start2:.2f}' if end2 - start2 >= 0.01 else '<0.01'}s"
        )
        print(f"训练模型保存为{X_name}_{classifier_type}_model.pkl")
        print(f"标准化器保存为{X_name}_{classifier_type}_scalar\n")
        self.output_log(f"测试集准确率:{accuracy_test}")
        self.output_log(f"训练集准确率:{accuracy_train}")
        self.output_log(
            f"训练时间:{f'{end1 - start1:.2f}' if end1 - start1 >= 0.01 else '<0.01'}s"
        )
        self.output_log(f"训练模型保存为{X_name}_{classifier_type}_model.pkl")
        self.output_log(f"标准化器保存为{X_name}_{classifier_type}_scalar\n")

        return accuracy_test, accuracy_train

    def predit_X(self, X):
        X = self.scaler.transform(X)
        y = self.classifier.predict(X)
        return y

    # 评估模型参数取值
    def evaluate_params(self, clf, param_name, param_range):
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(self.X)
        start = time.time()
        # 测试参数取值
        train_score, test_score = validation_curve(
            clf,
            X,
            self.y,
            param_name=param_name,
            param_range=param_range,
            cv=10,
            scoring="accuracy",
        )
        end = time.time()
        print(f"评估时间:{end - start:.1f}s")
        train_score = np.mean(train_score, axis=1)
        test_score = np.mean(test_score, axis=1)
        plt.plot(param_range, train_score, "o-", color="r", label="training")
        plt.plot(param_range, test_score, "o-", color="g", label="testing")
        plt.legend(loc="best")
        plt.xlabel(param_name)
        plt.ylabel("accuracy")
        plt.show()

    def output_log(self, str: str):
        with open("ModelTrainer.log", "a", encoding="utf-8") as file:
            file.write(str)
            file.write("\n")
