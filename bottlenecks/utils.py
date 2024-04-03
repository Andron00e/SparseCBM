import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import ast
from copy import deepcopy
import json


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def plot_training(
    model, optimizer, n_epochs, lr, train_loader, test_loader, log_interval
):
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    print("End of test")

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.plot(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples")
    plt.ylabel("loss function")
    fig.show()


def plot_averaged_training(
    model,
    optimizer,
    n_epochs,
    lr,
    train_loader,
    test_loader,
    log_interval,
    number_of_iterations,
):
    train_losses_avg = []
    test_losses_avg = []

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    for i in range(number_of_iterations):
        print(color.BOLD + "Training iteration: " + color.END, i + 1)
        model.apply(weights_init)

        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

        def train(epoch):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    train_losses.append(loss.item())
                    train_counter.append(
                        (batch_idx * len(data))
                        + ((epoch - 1) * len(train_loader.dataset))
                    )

        def test():
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    test_loss += F.nll_loss(output, target, size_average=False).item()
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print(
                "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )

        test()
        for epoch in range(1, n_epochs + 1):
            train(epoch)
            test()

        train_losses_avg.append(train_losses)
        test_losses_avg.append(test_losses)

    print(color.BOLD + "End of test" + color.END)
    train_losses_avg = list(np.mean(train_losses_avg, axis=0))
    test_losses_avg = list(np.mean(test_losses_avg, axis=0))

    fig = plt.figure()
    plt.plot(train_counter, train_losses_avg, color="blue")
    plt.plot(test_counter, test_losses_avg, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples")
    plt.ylabel("loss function")
    fig.show()


def group_uniques_full(hist, losses_to_average, verbose=False, group_norm_diffs=False):
    grouped_hist = {}
    unique_tried = {}
    for one_optim_hist in hist:
        unique_tried[one_optim_hist["name"]] = False

    for one_optim_hist in hist:
        label = one_optim_hist["name"]
        if not unique_tried[label]:
            unique_tried[label] = True
            grouped_hist[label] = {
                "hist": deepcopy(one_optim_hist),
                "repeats": {
                    loss_name: [1] * len(one_optim_hist[loss_name])
                    for loss_name in losses_to_average
                },
            }
            if group_norm_diffs:
                grouped_hist[label]["hist"]["norm_diffs"] = [
                    [np.array(x)] for x in grouped_hist[label]["hist"]["norm_diffs"]
                ]
            continue

        for loss_name in losses_to_average:
            losses = one_optim_hist[loss_name]

            for i, loss_elem in enumerate(losses):
                if i < len(grouped_hist[label]["hist"][loss_name]):
                    grouped_hist[label]["hist"][loss_name][i] += loss_elem
                else:
                    grouped_hist[label]["hist"][loss_name].append(loss_elem)

                if i >= len(grouped_hist[label]["repeats"][loss_name]):
                    grouped_hist[label]["repeats"][loss_name].append(0)
                grouped_hist[label]["repeats"][loss_name][i] += 1

        if group_norm_diffs:
            if len(grouped_hist[label]["hist"]["norm_diffs"]) == 0:
                if "norm_diffs_x" in one_optim_hist:
                    grouped_hist[label]["hist"]["norm_diffs_x"] = one_optim_hist[
                        "norm_diffs_x"
                    ]
                grouped_hist[label]["hist"]["norm_diffs"] = [
                    [np.array(x)] for x in one_optim_hist["norm_diffs"]
                ]
            else:
                for x, y in zip(
                    grouped_hist[label]["hist"]["norm_diffs"],
                    one_optim_hist["norm_diffs"],
                ):
                    x.append(np.array(y))

    for key in grouped_hist:
        one_optim_hist = grouped_hist[key]
        if verbose and len(one_optim_hist["repeats"][losses_to_average[0]]) > 0:
            repeats_1 = float(one_optim_hist["repeats"][losses_to_average[0]][0])
            print(
                "Repeats_1 = {}, Name = {}".format(
                    repeats_1, one_optim_hist["hist"]["name"]
                )
            )
        for loss_name in losses_to_average:
            for i in range(len(one_optim_hist["hist"][loss_name])):
                repeats = one_optim_hist["repeats"][loss_name][i]
                one_optim_hist["hist"][loss_name][i] /= repeats

        if group_norm_diffs:
            for i, group in enumerate(one_optim_hist["hist"]["norm_diffs"]):
                means = []
                stds = []
                for j, elem in enumerate(group):
                    means.append(elem.mean())
                    stds.append(elem.std())
                    group[j] = (elem - elem.mean()) / elem.std()
                mean = np.mean(means)
                std = np.mean(stds)
                one_optim_hist["hist"]["norm_diffs"][i] = (
                    np.concatenate(group) * std + mean
                )

    grouped_hist = [grouped_hist[x]["hist"] for x in grouped_hist]

    return grouped_hist


def load_hist_jsons(hists_names_list, path="./models"):
    hists = []
    for hist_name in hists_names_list:
        with open(r"{}/{}.json".format(path, hist_name), "r") as read_file:
            hist = json.load(read_file)
            hists += hist
    return hists


def rec_hist_from_json(h, key):
    for i in range(len(h)):
        if key == "val_norm_diffs":
            h[i] = ast.literal_eval(h[i])
            for j in range(len(h[i])):
                h[i][j] = float(h[i][j])
        elif isinstance(h[i], list):
            rec_hist_from_json(h[i], key)
        else:
            h[i] = float(h[i])


def hist_from_json(hists):
    for h in hists:
        for key in h:
            if isinstance(h[key], list):
                rec_hist_from_json(h[key], key)
    return hists
