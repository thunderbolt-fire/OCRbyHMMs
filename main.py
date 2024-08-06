
"""
OCR Evaluation
This script evaluates the accuracy of Optical Character Recognition (OCR) models by comparing the expected text with the recognized text. It uses simple models and Hidden Markov Models (HMM) to perform the OCR.
Parameters:
- train_img (str): The file path of the training image.
- train_txt (str): The file path of the training text.
- test_img (str): The file path of the test image.
Returns:
- None
Prints:
- The recognized text for each test case.
- The accuracy of the simple model, Forward HMM, and Viterbi HMM for each test case.
- The mean accuracy of the simple model, Forward HMM, and Viterbi HMM.
"""
from PIL import Image
import sys
from hmm_model import SimpleModel, HMM
width = 14
height = 25

def evaluation(expected, actual):
    acc = 0
    total = len(expected)
    for i in range(total):
            acc += 1 if expected[i] == actual[i] else 0
    acc = acc / total
    return acc

if(__name__ == "__main__"):
    def load_chars(name):
        """
        根据给定的图像文件名，加载图像并转换为字符表示。

        参数:
        name: str - 图像文件的名称。

        返回值:
        list - 包含图像字符表示的列表，其中每个元素代表一列字符。
        """
        # 打开图像文件
        img = Image.open(name)
        # 加载图像像素数据
        pix = img.load()
        # 初始化结果列表
        result = []
        # 获取图像宽度
        (X, _) = img.size
        # 遍历图像宽度方向上的每个像素块
        for x in range(0, X // width * width, width):
            # 初始化当前像素块的字符表示
            current_column = []
            # 遍历图像高度方向上的每个像素块
            for yi in range(0, height):
                # 初始化当前行的字符表示
                current_row = ""
                # 遍历当前像素块的每个像素
                for xi in range(x, x + width):
                    # 根据像素值设置字符，如果像素值小于1，使用'*'表示，否则使用' '表示
                    current_row += "*" if pix[xi, yi] < 1 else " "
                # 将当前行的字符表示添加到当前像素块的字符表示中
                current_column.append(current_row)
            # 将当前像素块的字符表示添加到结果列表中
            result.append(current_column)
        # 返回图像的字符表示
        return result

    def load_training_chars(name):
        """
        根据指定名称加载训练字符集。

        该函数将加载特定训练集的字符，并将字符映射到对应的图像上。
        这对于训练字符识别模型非常有用，特别是在处理手写体或特定字体时。

        参数:
        name (str): 训练集的名称，用于加载相应的字符图像。

        返回:
        dict: 一个字典，将字符映射到对应的图像上。
        """
        # 定义字符集，包含大小写字母、数字及常见标点符号
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        # 加载字符图像
        char_images = load_chars(name)
        # 创建字符与图像的映射
        return {letters[i]: char_images[i] for i in range(0, len(letters))}

    (train_img, train_txt, test_img) = sys.argv[1:]
    train_letters = load_training_chars(train_img)
    test_letters = load_chars(test_img)
    test_simple_model = SimpleModel(train_letters, test_letters)
    test_hmm_model = HMM(train_letters, test_letters, train_txt)

    testcase_answer = ["SUPREME COURT OF THE UNITED STATES", 
    "Certiorari to the United States Court of appeals for the Sixth Circuit", 
    "Nos. 14-556. Argued April 28, 2015 - Decided June 26, 2015",
    "Together with No. 14–562, Tanco et al. v. Haslam, Governor of",
    "Tennessee, et al., also on certiorari to the same court.",
    "Opinion of the Court",
    "As some of the petitioners in these cases demonstrate, marriage",
    "embodies a love that may endure even past death.",
    "It would misunderstand these men and women to say they disrespect",
    "the idea of marriage.",
    "Their plea is that they do respect it, respect it so deeply that",
    "they seek to find its fulfillment for themselves.",
    "Their hope is not to be condemned to live in loneliness,",
    "excluded from one of civilization's oldest institutions.",
    "They ask for equal dignity in the eyes of the law.",
    "The Constitution grants them that right.",
    "The judgement of the Court of Appeals for the Sixth Circuit is reversed.",
    "It is so ordered.",
    "KENNEDY, J., delivered the opinion of the Court, in which",
    "GINSBURG, BREYER, SOTOMAYOR, and KAGAN, JJ., joined."]

    n = len(testcase_answer)

    simple_mean_accuracy = 0
    hmm_mean_accuracy = 0
    fd_mean_accuracy = 0


    # 遍历测试用例，对每个用例进行字符识别和模型测试
    for i in range(n):
        # 构造测试图像文件名
        test_img_fname = './test_images/test-' + str(i) + '-0.png'
        # 加载测试图像中的字符
        test_letters = load_chars(test_img_fname)
        # 构造训练图像文件名
        train_img_fname = './test_images/references.png'
        #train_img_fname = './test_images/courier-train.png'
        # 加载训练图像中的字符
        train_letters = load_training_chars(train_img_fname)
        # 使用简单的字符识别模型进行测试
        result = test_simple_model.simple_model(train_letters, test_letters)
        # 使用隐马尔可夫模型进行测试，返回前向算法和维特比算法的结果
        fd_result, hmm_result = test_hmm_model.hidden_markov_model(train_letters, test_letters)
        
        # 打印测试结果，包括正确答案、简单模型结果、前向算法HMM结果和维特比算法HMM结果
        print(f"Answer: {testcase_answer[i]}\nSimple: {result}\nFD HMM: {fd_result}\nVi HMM: {hmm_result}")
        # 计算并打印简单模型的准确率
        simple_accuracy = evaluation(testcase_answer[i], result) * 100
        # 计算并打印前向算法HMM的准确率
        hmm_accuracy = evaluation(testcase_answer[i], hmm_result) * 100
        # 计算并打印维特比算法HMM的准确率
        fd_accuracy = evaluation(testcase_answer[i], fd_result) * 100
        print(f"Accuracy for simple on case number {i}: {round(simple_accuracy, 3)} %")
        print(f"Accuracy for Forward HMM on case {i}: {round(fd_accuracy, 3)} %")
        print(f"Accuracy for Viterbi HMM on case {i}: {round(hmm_accuracy, 3)} %")
        
        # 累加准确率，用于后续计算平均准确率
        simple_mean_accuracy += simple_accuracy
        hmm_mean_accuracy += hmm_accuracy
        fd_mean_accuracy += fd_accuracy
    
    print(f"Simple mean accuracy: {round(simple_mean_accuracy / n, 3)} %")
    print(f"Forward HMM mean accuracy: {round(fd_mean_accuracy / n, 3)} %")
    print(f"Viterbi HMM mean accuracy: {round(hmm_mean_accuracy / n, 3)} %")