import joblib
import audio_train


if __name__ == "__main__":
    # 读取新的声音文件做预处理, 以得到训练模型的标准化输入数据
    new_X = audio_train.AudioTrain.preprocess_audio("2-1.wav", n_mfcc=40)
    # 加载训练好的模型
    mlp = joblib.load("audio_MLP_model.pkl")
    # 加载标准化器
    scaler = joblib.load("audio_MLP_scalar")
    # 标准化输入数据
    new_X = scaler.transform(new_X)
    # 预测新的声音文件是否含鸟叫声
    new_y = mlp.predict(new_X)
    lists = [0] * 20
    for i in new_y:
        lists[i - 1] += 1
    for i in range(len(lists)):
        print(f"判断为第{i+1}只鸟的帧: {lists[i]}, {lists[i]/sum(lists)*100:.2f}%")
