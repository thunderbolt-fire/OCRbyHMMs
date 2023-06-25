from hmm_model import SimpleModel, HMM
from PIL import Image, ImageDraw, ImageFont
import sys

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
        img = Image.open(name)
        pix = img.load()
        result = []
        (X, _) = img.size
        for x in range(0, X // width * width, width):
            result += [["".join([ '*' if pix[xi, yi] < 1 else ' ' for xi in range(x, x + width) ]) for yi in range(0, height)],]
        return result

    def load_training_chars(name):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
        char_images = load_chars(name)
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


    for i in range(n):
        test_img_fname = './test_images/test-' + str(i) + '-0.png'
        test_letters = load_chars(test_img_fname)
        train_img_fname = './test_images/courier-train.png'
        train_letters = load_training_chars(train_img_fname)
        result = test_simple_model.simple_model(train_letters, test_letters)
        fd_result, hmm_result = test_hmm_model.hidden_markov_model(train_letters, test_letters)
        
        print(f"Answer: {testcase_answer[i]}\nSimple: {result}\nFD HMM: {fd_result}\nVi HMM: {hmm_result}")
        simple_accuracy = evaluation(testcase_answer[i], result) * 100
        hmm_accuracy = evaluation(testcase_answer[i], hmm_result) * 100
        fd_accuracy = evaluation(testcase_answer[i], fd_result) * 100
        print(f"Accuracy for simple on case number {i} is: {round(simple_accuracy, 4)} %")
        print(f"Accuracy for Forward HMM on case {i} is: {round(fd_accuracy, 3)} %")
        print(f"Accuracy for Viterbi HMM on case {i} is: {round(hmm_accuracy, 3)} %")
        
        simple_mean_accuracy += simple_accuracy
        hmm_mean_accuracy += hmm_accuracy
        fd_mean_accuracy += fd_accuracy
    
    print(f"Simple mean accuracy: {round(simple_mean_accuracy / n, 4)} %")
    print(f"Forward HMM mean accuracy: {round(fd_mean_accuracy / n, 4)} %")
    print(f"Viterbi HMM mean accuracy: {round(hmm_mean_accuracy / n, 4)} %")