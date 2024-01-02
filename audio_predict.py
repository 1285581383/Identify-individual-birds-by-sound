import joblib
import audio_train
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    # 读取新的声音文件做预处理, 以得到训练模型的标准化输入数据
    X_test = audio_train.AudioTrain.preprocess_audio("3-1.wav", n_mfcc=20)
    # 加载训练好的模型
    mlp = joblib.load("audio_SVC_model.pkl")
    # 加载标准化器
    scaler = joblib.load("audio_SVC_scalar")
    # 标准化输入数据
    X_test = scaler.transform(X_test)
    # 预测新的声音文件是否含鸟叫声
    y_pred_test = mlp.predict(X_test)
    # []这里填鸟的编号
    y_test = [3] * len(y_pred_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    # 打印每一帧声音预测的鸟编号
    for i in range(len(y_pred_test)):
        print(y_pred_test[i], end=" ")
