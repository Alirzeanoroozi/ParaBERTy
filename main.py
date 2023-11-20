import random
from os.path import isfile
import numpy as np
import torch
from matplotlib import pyplot as plt
from data_loader import get_cv_dataloaders, get_single_dataloaders
from model import Parabert, train, evaluate


def cv_run(device, num_iters=10):
    for i in range(num_iters):
        print("Cross_validation run", i + 1)
        train_dataloader, test_dataloader, valid_dataloader = get_cv_dataloaders(i, max_len=150)

        model = Parabert().to(device)

        weights = "precomputed/weights/fold-{}.pth".format(i + 1)

        if not isfile(weights):
            history, tresh = train(model, train_dl=train_dataloader, val_dl=valid_dataloader, device=device, cv=i)
            torch.save(model, weights)

            f = open("precomputed/weights/threshold-{}.txt".format(i + 1), "w")
            f.write(str(tresh))
            f.close()

            plt.plot(history['train_loss'], 'r', history['val_loss'], 'b')
            plt.savefig("results/{}.png".format(i + 1))
            plt.show()

        else:
            model = torch.load(weights)
            tresh = float(open("precomputed/weights/threshold-{}.txt".format(i + 1), "r").read())

            evaluate(model, test_dataloader, device, epoch="test", threshold=tresh, cv=i)


def single_run(device):
    train_dataloader, test_dataloader, valid_dataloader = get_single_dataloaders(max_len=150)

    model = Parabert().to(device)

    model_weight = "precomputed/single_weight.pth"

    if not isfile(model_weight):
        history, tresh = train(model, train_dl=train_dataloader, val_dl=valid_dataloader, device=device)
        torch.save(model, model_weight)

        f = open("precomputed/threshold.txt", "w")
        f.write(str(tresh))
        f.close()

        plt.plot(history['train_loss'], 'r', history['val_loss'], 'b')
        plt.savefig("results/single.png")
    else:
        model = torch.load(model_weight).to(device)
        tresh = float(open("precomputed/threshold.txt", "r").read())

        evaluate(model, test_dataloader, device, epoch="test", threshold=tresh)


def report_cv():
    recall_values = []
    precision_values = []
    f1_values = []
    roc_values = []
    pr_values = []
    mcc_values = []

    for i in range(10):
        file_path = f"results/{i}/model_test.txt"

        with open(file_path, 'r') as file:
            lines = file.readlines()

            recall_line = lines[4]  # Assuming Precision is in the 7th line
            recall_values.append(float(recall_line.split()[2]))  # Extracting the second value

            precision_line = lines[5]  # Assuming Precision is in the 7th line
            precision_values.append(float(precision_line.split()[2]))  # Extracting the second value

            f1_line = lines[6]  # Assuming Precision is in the 7th line
            f1_values.append(float(f1_line.split()[2]))  # Extracting the second value

            roc_line = lines[7]  # Assuming Precision is in the 7th line
            roc_values.append(float(roc_line.split()[3]))  # Extracting the second value

            pr_line = lines[8]  # Assuming Precision is in the 7th line
            pr_values.append(float(pr_line.split()[2]))  # Extracting the second value

            mcc_line = lines[9]  # Assuming Precision is in the 7th line
            mcc_values.append(float(mcc_line.split()[2]))  # Extracting the second value

    with open("results/cv.txt", "w") as f:
        f.writelines([
            "recall: " + str(sum(recall_values) / 10) + "\n",
            "precision: " + str(sum(precision_values) / 10) + "\n",
            "F1-Score:" + str(sum(f1_values) / 10) + "\n",
            "ROC:" + str(sum(roc_values) / 10) + "\n",
            "PR-score:" + str(sum(pr_values) / 10) + "\n",
            "MCC:" + str(sum(mcc_values) / 10) + "\n",
        ])


def initiate_system_device():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    device = initiate_system_device()

    # single_run(device)
    cv_run(device)
    report_cv()
