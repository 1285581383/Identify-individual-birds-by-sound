import librosa
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import model_trainer
import tqdm
import soundfile as sf
from joblib import Parallel, delayed


class AudioTrain:
    def __init__(
        self,
        sr: int = 22050,
        win_length: int = 4096,
        hop_length: int = 1024,
        n_fft: int = 4096,
        n_mels: int = 128,
        n_mfcc: int = 20,
        top_db: int = 20,
        alpha_s: float = 4,
        beta_s: float = 0.001,
        gamma_s: float = 1,
        k: int = 1,
        alpha_w: float = 1,
        beta_w: float = 2,
        save_fig_flag: bool = False,
        save_remix_audio_flag: bool = False,
        n_jobs: int = -1,
    ):
        """_音频文件特征提取，调用model_trainer使用这些特征值来训练模型_

        Args:
            sr (int, optional): _采样率_. Defaults to 22050.
            win_length (int, optional): _帧长（单位：采样个数），win_length/sr = 帧长的时间（单位：s）_. Defaults to 4096.
            hop_length (int, optional): _帧移，单位同帧长_. Defaults to 1024.
            n_fft (int, optional): _短时傅里叶变换的采样点数_. Defaults to 4096.
            n_mels (int, optional): _梅尔滤波器的数量_. Defaults to 128.
            n_mfcc (int, optional): _梅尔频率倒谱系数的数量_. Defaults to 20.
            top_db (int, optional): _能量阈值，声音文件中低于该能量(-top_db)的帧将会被移除_. Defaults to 20.
            save_fig_flag (bool, optional): _是否保存mfcc和mel图片，位置为脚本同级目录下的img文件夹_. Defaults to False.
            save_remix_audio_flag (bool, optional): _是否保存重新组合的声音文件，位置为脚本同级目录下的audio文件夹_. Defaults to False.
            n_jobs (int, optional): _并行处理音频文件的进程数_. Defaults to -1.

        """
        self.sr = sr
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.top_db = top_db
        # 存放声音的特征值
        self.combined_features = np.array([])
        # 存放声音的类别标签
        self.y = np.array([])
        self.save_fig_flag = save_fig_flag
        self.save_remix_audio_flag = save_remix_audio_flag
        self.n_jobs = n_jobs
        self.alpha_s = alpha_s
        self.beta_s = beta_s
        self.gamma_s = gamma_s
        self.k = k
        self.alpha_w = alpha_w
        self.beta_w = beta_w

    def load_files(self, audio_folder: str):
        """_加载训练声音文件_

        Args:
            sound_folder (_type_): _包含训练声音的文件夹路径_
        """
        """使用joblib并行加载训练声音文件"""
        sound_files = [
            f for f in os.listdir(audio_folder) if f.lower().endswith(".wav")
        ]
        pattern = r"^(\d+)"  # 匹配开头的数字部分作为鸟编号

        # 定义并行处理的函数
        def process_file(sound_file):
            match = re.match(pattern, sound_file)
            bird_index = int(match.group(1))
            sound_file = os.path.join(audio_folder, sound_file)
            # 调用预处理声音文件的方法
            combined_features = self.preprocess_audio(
                audio_file=sound_file,
                sr=self.sr,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
                top_db=self.top_db,
                alpha_s=self.alpha_s,
                beta_s=self.beta_s,
                gamma_s=self.gamma_s,
                k=self.k,
                alpha_w=self.alpha_w,
                beta_w=self.beta_w,
                save_fig_flag=self.save_fig_flag,
                save_remix_audio_flag=self.save_remix_audio_flag,
            )
            y = np.array([bird_index] * combined_features.shape[0])
            return combined_features, y

        print("加载并预处理声音文件中...")
        # 并行处理
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_file)(sound_file) for sound_file in tqdm.tqdm(sound_files)
        )

        # 拼接各个声音文件的combined_features和y
        for combined_features, y in results:
            if self.combined_features.shape[0] == 0 and self.y.shape[0] == 0:
                self.combined_features = combined_features
                self.y = y
            else:
                self.combined_features = np.vstack(
                    (self.combined_features, combined_features)
                )
                self.y = np.concatenate((self.y, y))
        # 以下是串行读取声音文件的代码
        # sound_files = [
        #     f for f in os.listdir(audio_folder) if f.lower().endswith(".wav")
        # ]

        # pattern = r"^(\d+)"  # 匹配开头的数字部分作为鸟编号
        # print("加载并预处理声音文件中...")
        # for sound_file in tqdm.tqdm(sound_files):
        #     match = re.match(pattern, sound_file)
        #     bird_index = int(match.group(1))
        #     sound_file = os.path.join(audio_folder, sound_file)
        #     combined_features = self.preprocess_audio(
        #         audio_file=sound_file,
        #         sr=self.sr,
        #         win_length=self.win_length,
        #         hop_length=self.hop_length,
        #         n_fft=self.n_fft,
        #         n_mels=self.n_mels,
        #         n_mfcc=self.n_mfcc,
        #         top_db=self.top_db,
        #         save_fig_flag=self.save_fig_flag,
        #         save_remix_audio_flag=self.save_remix_audio_flag,
        #     )
        #     y = np.array([bird_index] * combined_features.shape[0])

        #     if self.combined_features.shape[0] == 0 and self.y.shape[0] == 0:
        #         self.combined_features = combined_features
        #         self.y = y
        #     else:
        #         self.combined_features = np.vstack(
        #             (self.combined_features, combined_features)
        #         )
        #         self.y = np.concatenate((self.y, y))

    @staticmethod
    def preprocess_audio(
        audio_file: str,
        sr: int = 22050,
        win_length: int = 4096,
        hop_length: int = 1024,
        n_fft: int = 4096,
        n_mels: int = 128,
        n_mfcc: int = 20,
        top_db: int = 20,
        alpha_s: float = 4,
        beta_s: float = 0.001,
        gamma_s: float = 1,
        k: int = 1,
        alpha_w: float = 1,
        beta_w: float = 2,
        save_fig_flag: bool = False,
        save_remix_audio_flag: bool = False,
    ) -> np.array:
        """_预处理声音文件,提取特征_

        Args:
            audio_file (_type_): _声音文件路径_
            sr (int, optional): _采样率_. Defaults to 22050.
            win_length (int, optional): _帧长（单位：采样个数），win_length/sr = 帧长的时间（单位：s）_. Defaults to 4096.
            hop_length (int, optional): _帧移，单位同帧长_. Defaults to 1024.
            n_fft (int, optional): _短时傅里叶变换的采样点数_. Defaults to 4096.
            n_mels (int, optional): _梅尔滤波器的数量_. Defaults to 128.
            n_mfcc (int, optional): _梅尔频率倒谱系数的数量_. Defaults to 13.
            top_db (int, optional): _能量阈值，声音文件中低于该能量(-top_db)的帧将会被移除_. Defaults to 20.
            save_fig_flag (bool, optional): _是否保存mfcc和mel图片,位置为脚本同级目录下的img文件夹_. Defaults to False.
            save_remix_audio_flag (bool, optional): _是否保存分割后的声音文件，位置为脚本同级目录下的remix_audio文件夹_. Defaults to False.
        Returns:
            np.array: _声音文件的特征值_

        """
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        sig, sr = librosa.load(audio_file, sr=sr)
        # 分割声音文件，帧能量大于-top_db的索引数组
        intervals = librosa.effects.split(
            sig, top_db=top_db, frame_length=win_length, hop_length=hop_length
        )
        # 分割声音文件，帧能量小于-top_db的索引数组
        inverse_intervals = AudioTrain.get_inverse_intervals(intervals, len(sig))

        sig_remix = librosa.effects.remix(sig, intervals)
        sig_noise_remix = librosa.effects.remix(sig, inverse_intervals)

        sig_subtraction = AudioTrain.spec_subtract(
            noisy=sig_remix,
            noise=sig_noise_remix,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            alpha_s=alpha_s,
            beta_s=beta_s,
            gamma_s=gamma_s,
            k=k,
        )
        sig_wiener = AudioTrain.wiener_filter(
            noisy=sig_remix,
            clean=sig_subtraction,
            noise=sig_noise_remix,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            alpha_w=alpha_w,
            beta_w=beta_w,
        )
        # 保存分割后的声音文件，位置为脚本同级目录下的remix_audio文件夹
        if save_remix_audio_flag:
            remix_audio_folder = "subtraction_remix_audio"
            if not os.path.exists(remix_audio_folder):
                os.makedirs(remix_audio_folder)
            remix_audio_path = os.path.join(
                remix_audio_folder, f"{file_name}.remix_wiener.wav"
            )
            sf.write(remix_audio_path, sig_wiener, sr)
        # 预加重
        sig_emphasis = librosa.effects.preemphasis(sig_wiener)
        # 短时傅里叶变换
        stft = librosa.stft(
            y=sig_emphasis, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        # freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        # 对stft求模平方
        power, phase = librosa.magphase(stft, power=2)
        # 计算梅尔滤波器组
        melfb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        # 计算翻转梅尔滤波器组
        imelfb = np.flip(np.flip(melfb, axis=1), axis=0)
        # 计算翻转梅尔频谱图
        imel_spectrogram = np.matmul(imelfb, power)
        imelspec_db = librosa.power_to_db(imel_spectrogram)
        # 计算IMFCC特征
        imfcc = librosa.feature.mfcc(S=imelspec_db, n_mfcc=n_mfcc)

        # 计算MFCC特征
        mfcc = librosa.feature.mfcc(
            y=sig_emphasis,
            sr=sr,
            n_mfcc=n_mfcc,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
        )
        # 一阶差分
        mfcc_deta = librosa.feature.delta(mfcc)
        # 二阶差分
        mfcc_deta2 = librosa.feature.delta(mfcc, order=2)
        # 梅尔频谱图
        mel_spectrogram = librosa.feature.melspectrogram(
            y=sig_emphasis,
            sr=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        # 画图并保存
        if save_fig_flag:
            img_folder = "img"
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
            MFCC_fig_path = os.path.join(img_folder, f"{file_name}.MFCC.png")
            mel_fig_path = os.path.join(img_folder, f"{file_name}.Mel.png")
            imel_fig_path = os.path.join(img_folder, f"{file_name}.IMel.png")
            AudioTrain.save_fig(
                MFCC_fig_path=MFCC_fig_path,
                mel_fig_path=mel_fig_path,
                imel_fig_path=imel_fig_path,
                mfcc=mfcc,
                mfcc_deta=mfcc_deta,
                mfcc_deta2=mfcc_deta2,
                mel_spectrogram=mel_spectrogram,
                imelspec_db=imel_spectrogram,
                sr=sr,
                win_length=win_length,
                hop_length=hop_length,
                n_fft=n_fft,
            )

        # 特征拼接
        combined_features = np.concatenate([mfcc, mfcc_deta, mfcc_deta2], axis=0)
        # 转置
        combined_features = combined_features.T
        return combined_features

    @staticmethod
    def get_inverse_intervals(intervals: np.ndarray, maxlen: int) -> np.ndarray:
        inverse_intervals = []
        first = 0
        for interval in intervals:
            if interval[0] <= first:
                first = interval[1]
                continue
            else:
                inverse_intervals.append([first, interval[0]])
                first = interval[1]
        if first < maxlen:
            inverse_intervals.append([first, maxlen])
        inverse_intervals = np.array(inverse_intervals)
        return inverse_intervals

    @staticmethod
    def spec_subtract(
        noisy: np.ndarray,
        noise: np.ndarray,
        win_length: int = 4096,
        hop_length: int = 1024,
        n_fft: int = 4096,
        alpha_s: float = 5,
        beta_s: float = 0.001,
        gamma_s: float = 1,
        k: int = 1,
    ) -> np.ndarray:
        """
        超减法去噪
        noisy: 含噪声声音
        noise: 噪声
        win_length: 窗口大小
        hop_length: 窗口移动步长
        n_fft: FFT大小
        alpha_s: 超减法参数
        beta_s: 超减法参数
        gamma_s: 超减法参数
        k: 超减法参数
        return: 去噪后的信号
        """

        # 计算noisy的频谱
        S_noisy = librosa.stft(
            noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )  # D x T
        D, T = np.shape(S_noisy)
        Mag_noisy = np.abs(S_noisy)
        Phase_noisy = np.angle(S_noisy)
        Power_noisy = Mag_noisy**2

        # 计算noise的频谱
        S_noise = librosa.stft(
            noise, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        Mag_noise = np.mean(np.abs(S_noise), axis=1, keepdims=True)
        Power_noise = Mag_noise**2
        Power_noise = np.tile(Power_noise, [1, T])

        ## 方法3 引入平滑
        Mag_noisy_new = np.copy(Mag_noisy)
        for t in range(k, T - k):
            Mag_noisy_new[:, t] = np.mean(Mag_noisy[:, t - k : t + k + 1], axis=1)
        Power_noisy = Mag_noisy_new**2

        # 超减法去噪
        Power_enhance = np.power(Power_noisy, gamma_s) - alpha_s * np.power(
            Power_noise, gamma_s
        )
        Power_enhance = np.power(Power_enhance, 1 / gamma_s)

        # 对于过小的值用 beta* Power_noise 替代
        mask = (Power_enhance >= beta_s * Power_noise) - 0
        Power_enhance = mask * Power_enhance + beta_s * (1 - mask) * Power_noise
        Mag_enhance = np.sqrt(Power_enhance)

        Mag_enhance_new = np.copy(Mag_enhance)
        # 计算最大噪声残差
        maxnr = np.max(np.abs(S_noise) - Mag_noise, axis=1)

        for t in range(k, T - k):
            index = np.where(Mag_enhance[:, t] < maxnr)[0]
            temp = np.min(Mag_enhance[:, t - k : t + k + 1], axis=1)
            Mag_enhance_new[index, t] = temp[index]

        # 对信号进行恢复
        S_enhance = Mag_enhance_new * np.exp(1j * Phase_noisy)
        enhance = librosa.istft(
            stft_matrix=S_enhance,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
        )
        return enhance

    @staticmethod
    def wiener_filter(
        noisy: np.ndarray,
        clean: np.ndarray,
        noise: np.ndarray,
        win_length: int = 4096,
        hop_length: int = 1024,
        n_fft: int = 4096,
        alpha_w: float = 1,
        beta_w: float = 2,
    ):
        """_使用维纳滤波对声音进行增强_

        Args:
            noisy (np.ndarray): _含噪声声音_
            clean (np.ndarray): _清晰声音_
            noise (np.ndarray): _噪声_
            win_length (int, optional): _窗口大小. Defaults to 4096.
            hop_length (int, optional): _步长. Defaults to 1024.
            n_fft (int, optional): _FFT大小. Defaults to 4096.
            alpha_w (float, optional): _权重. Defaults to 1.
            beta_w (float, optional): _权重. Defaults to 2.

        Returns:
            np.ndarray: _增强后的声音_
        """
        # 计算noisy的频谱
        S_noisy = librosa.stft(
            y=noisy, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )  # DxT
        S_noise = librosa.stft(
            y=noise, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        S_clean = librosa.stft(
            y=clean, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

        Pxx = np.mean((np.abs(S_clean)) ** 2, axis=1, keepdims=True)  # Dx1
        Pnn = np.mean((np.abs(S_noise)) ** 2, axis=1, keepdims=True)

        H = (Pxx / (Pxx + alpha_w * Pnn)) ** beta_w

        S_enhec = S_noisy * H

        wiener_enhance = librosa.istft(
            S_enhec, hop_length=hop_length, win_length=win_length, n_fft=n_fft
        )

        return wiener_enhance

    # 保存图片的函数
    @staticmethod
    def save_fig(
        MFCC_fig_path: str,
        mel_fig_path: str,
        imel_fig_path: str,
        mfcc,
        mfcc_deta,
        mfcc_deta2,
        mel_spectrogram,
        imelspec_db,
        win_length,
        n_fft,
        hop_length: int,
        sr: int,
    ):
        """_保存图片_

        Args:
            MFCC_fig_path (str): _MFCC图片保存路径_
            mel_fig_path (str): _梅尔频谱图保存路径_
            imel_fig_path (str): _翻转梅尔频谱图保存路径_
            mfcc (np.array): _MFCC特征_
            mfcc_deta (np.array): _MFCC一阶差分_
            mfcc_deta2 (np.array): _MFCC二阶差分_
            mel_spectrogram (np.array): _梅尔频谱图_
            imelspec_db (np.array): _翻转梅尔频谱图_
            win_length (int): _窗口长度_
            n_fft (int): _FFT点数_
            hop_length (int): _帧移长度_
            sr (int): _采样率_
        """
        # 绘制翻转梅尔频谱图
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(imelspec_db, ref=np.max),
            y_axis="mel",
            x_axis="time",
            sr=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Inverse Mel Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        # 调整布局，防止重叠
        plt.tight_layout()
        plt.figure(figsize=(18, 6))
        # 第一个子图 - MFCC
        plt.subplot(1, 3, 1)
        plt.title("MFCC")
        librosa.display.specshow(mfcc, x_axis="time", hop_length=hop_length, sr=sr)
        plt.colorbar(format="%+2.0f")
        # 第二个子图 - MFCC Deta
        plt.subplot(1, 3, 2)
        plt.title("MFCC Deta")
        librosa.display.specshow(mfcc_deta, x_axis="time", hop_length=hop_length, sr=sr)
        plt.colorbar(format="%+2.0f")
        # 第三个子图 - MFCC Deta2
        plt.subplot(1, 3, 3)
        plt.title("MFCC Deta2")
        librosa.display.specshow(
            mfcc_deta2, x_axis="time", hop_length=hop_length, sr=sr
        )
        plt.colorbar(format="%+2.0f")
        # 调整布局，防止重叠
        plt.tight_layout()
        # 保存图片
        plt.savefig(MFCC_fig_path)
        # 关闭图片
        plt.close()

        # 绘制梅尔频谱图
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(mel_spectrogram, ref=np.max),
            y_axis="mel",
            x_axis="time",
            sr=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        # 保存图片
        plt.savefig(mel_fig_path)
        # 关闭图片
        plt.close()

        # 绘制翻转梅尔频谱图
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            librosa.power_to_db(imelspec_db, ref=np.max),
            y_axis="mel",
            x_axis="time",
            sr=sr,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("IMel Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Mel Frequency")
        # 保存图片
        plt.savefig(imel_fig_path)
        # 关闭图片
        plt.close()

    def train(
        self,
        X_name: str,
        classifier_type: str,
        clf_random: int = 1,
        split_random: int = 1,
        test_size: float = 0.3,
        **classifier_params,
    ):
        """训练模型

        Args:
            classifier_type (_str_): _可选择的分类器类型:'MLP', 'SVC', 'RF', 'KNN'_
            **classifier_params: 分类器的参数
            X_name (_str_): _训练集名称_
            clf_random (int, optional): 训练模型随机种子. Defaults to 1.
            split_random (int, optional): 数据集划分随机种子,用于改变数据集划分为训练集和测试集的方法. Defaults to 1.
            test_size (float, optional): 划分的测试集占数据集比. Defaults to 0.3.

        Returns:
            accuracy_test (_float_): _测试集准确率_
            accuracy_train (_float_): _训练集准确率_

        """
        mt = model_trainer.ModelTrainer(clf_random, split_random, test_size)
        mt.load_X_y(self.combined_features, self.y)
        accuracy_test, accuracy_train = mt.train_and_evaluate_classifier(
            X_name, classifier_type, **classifier_params
        )
        self.mt = mt
        return accuracy_test, accuracy_train


if __name__ == "__main__":
    at = AudioTrain(
        n_mfcc=20, save_fig_flag=True, save_remix_audio_flag=True, n_jobs=4, top_db=20
    )
    at.load_files("audio")

    at.train(
        "audio",
        "MLP",
        solver="adam",
        hidden_layer_sizes=(465, 465),
        max_iter=10000,
        alpha=0.0001,
        verbose=True,
    )
    at.train(
        X_name="audio",
        classifier_type="SVC",
        test_size=0.3,
        kernel="rbf",
        C=10,
        verbose=False,
    )
