import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import bisect
import scipy
from scipy import stats
import numpy as np


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def smooth(a, eps=0.01):
    b = [a[0]]
    for e in a[1:]:
        b.append(b[-1] * (1 - eps) + e * eps)
    return b


colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def group_uniques(hist, one_optim_hist, group_unique, loss_name):
    if not group_unique:
        return one_optim_hist[loss_name]
    else:
        res = [0] * len(one_optim_hist[loss_name])
        unique_name = one_optim_hist["name"]
        k = 0
        for one_optim_hist in hist:
            if one_optim_hist["name"] == unique_name:
                k += 1
                for i, elem in enumerate(one_optim_hist[loss_name]):
                    res[i] += elem
        res = [elem / k for elem in res]
        return res


def make_loss_plot(
    ax,
    hist,
    loss_name,
    eps=0.01,
    alpha=0.5,
    make_train=True,
    make_val=True,
    starting_epoch=0,
    group_unique=False,
):  # loss_name="logloss"
    if len(hist) < 7:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 0.9, len(hist)))
    unique_tried = {}
    for one_optim_hist in hist:
        unique_tried[one_optim_hist["name"]] = False

    for i, one_optim_hist in enumerate(hist):
        label = one_optim_hist["name"]

        if group_unique and unique_tried[label]:
            continue
        unique_tried[label] = True

        if len(one_optim_hist["train_x"]) == 0:
            continue

        epochs_x = one_optim_hist["epochs_x"][starting_epoch:]
        start = int(epochs_x[0])

        if make_train:
            train_y = group_uniques(hist, one_optim_hist, group_unique, "train_loss")
            smoothed_train_y = smooth(train_y, eps=eps)[start:]
            train_x = one_optim_hist["train_x"][start:]
            ax.plot(
                train_x,
                smoothed_train_y,
                label=label + " (train)",
                alpha=alpha,
                color=colors[i],
                linestyle="-",
            )

        if make_val and len(one_optim_hist["val_x"]) > 0:
            val_y = group_uniques(hist, one_optim_hist, group_unique, "val_loss")
            val_x = one_optim_hist["val_x"]
            ind = bisect.bisect_left(val_x, start)
            ax.plot(
                val_x[ind:],
                val_y[ind:],
                label=label + " (val)",
                alpha=alpha,
                color=colors[i],
                linestyle="--",
            )

        # epoch sep lines
        for x in epochs_x:
            ax.axvline(x, linestyle="--", color=colors[i], alpha=0.2)

    if make_train and make_val:
        ax.set_title("{} on Train/Validation".format(loss_name))
    elif make_train:
        ax.set_title("{} on Train".format(loss_name))
    elif make_val:
        ax.set_title("{} on Validation".format(loss_name))
    ax.set_ylabel("{}".format(loss_name))
    ax.set_xlabel("Iterations")
    ax.grid(True)
    ax.legend()

    return ax


def make_metrics_plot(
    ax,
    hist,
    eps=0.01,
    alpha=0.5,
    make_train=True,
    make_val=True,
    starting_epoch=0,
    metric_name="acc_top_1",
    title="top-1 accuracy",
):
    if len(hist) < 7:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 0.9, len(hist)))

    for i, one_optim_hist in enumerate(hist):
        label = one_optim_hist["name"]

        if len(one_optim_hist["train_x"]) == 0:
            continue

        epochs_x = one_optim_hist["epochs_x"][starting_epoch:]
        start = int(epochs_x[0])

        if make_train:
            train_acc_top = one_optim_hist["train_{}".format(metric_name)]
            smoothed_train_acc_top = smooth(train_acc_top, eps=eps)[start:]
            train_x = one_optim_hist["train_x"][start:]
            ax.plot(
                train_x,
                smoothed_train_acc_top,
                label=label + " (train)",
                alpha=alpha,
                color=colors[i],
                linestyle="-",
            )

        if make_val and len(one_optim_hist["val_x"]) > 0:
            val_acc_top = one_optim_hist["val_{}".format(metric_name)]
            val_x = one_optim_hist["val_x"]
            ind = bisect.bisect_left(val_x, start)
            ax.plot(
                val_x[ind:],
                val_acc_top[ind:],
                label=label + " (val)",
                alpha=alpha,
                color=colors[i],
                linestyle="--",
            )

        # epoch sep lines
        for x in epochs_x:
            ax.axvline(x, linestyle="--", color=colors[i], alpha=0.2)

    if make_train and make_val:
        ax.set_title("{} on Train/Validation".format(title))
    elif make_train:
        ax.set_title("{} on Train".format(title))
    elif make_val:
        ax.set_title("{} on Validation".format(title))

    ax.set_ylabel("{}".format(title))
    ax.set_xlabel("Iterations")
    ax.grid(True)

    return ax


def make_accuracy_plot(
    ax,
    hist,
    eps=0.01,
    alpha=0.5,
    top_k=1,
    make_train=True,
    make_val=True,
    starting_epoch=0,
):
    return make_metrics_plot(
        ax,
        hist,
        eps,
        alpha,
        make_train,
        make_val,
        starting_epoch,
        metric_name="acc_top_{}".format(top_k),
        title="top-{} accuracy".format(top_k),
    )


def make_plot(
    ax,
    hist,
    y_name,
    x_name,
    title="Loss with different optimizers",
    y_label="Loss",
    x_label="Iterations",
    eps=0.01,
    alpha=0.5,
    starting_epoch=0,
    draw_epoch_lines=True,
    grid=True,
    legend=True,
    mark_every_count=15,
    title_fontsize=14,
    xy_label_fontsize=10,
    line_params={},
):
    if len(hist) < 7:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        cmap = plt.get_cmap("hsv")
        colors = cmap(np.linspace(0, 0.9, len(hist)))

    for i, one_optim_hist in enumerate(hist):
        label = one_optim_hist["name"]
        if label in line_params:
            line_style = line_params[label]["line_style"]
            line_color = line_params[label]["line_color"]
            line_marker = line_params[label]["line_marker"]
            mark_every = line_params[label]["mark_every"]
            label = line_params[label]["name"]
        else:
            line_style = "-"
            line_color = colors[i]
            line_marker = None
            mark_every = None

        if len(one_optim_hist["epochs_x"]) <= starting_epoch:
            continue

        epochs_x = one_optim_hist["epochs_x"][starting_epoch:]
        start = int(epochs_x[0])

        if len(one_optim_hist[y_name]) == 0:
            continue

        smoothed_y = smooth(one_optim_hist[y_name], eps=eps)
        x = one_optim_hist[x_name]

        ind = bisect.bisect_left(x, start)
        smoothed_y = smoothed_y[ind:]
        x = x[ind:]

        if mark_every == "auto":
            mark_every = len(x) // (mark_every_count + 1)
            mark_every = np.arange((i * mark_every) // (len(hist)), len(x), mark_every)

        ax.plot(
            x,
            smoothed_y,
            label=label,
            alpha=alpha,
            color=line_color,
            linestyle=line_style,
            marker=line_marker,
            markevery=mark_every,
        )

        # epoch sep lines
        if draw_epoch_lines:
            for x in epochs_x:
                ax.axvline(x, linestyle="--", color=line_color, alpha=0.2)

    plt.rc("font", size=xy_label_fontsize)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=xy_label_fontsize)
    ax.set_xlabel(x_label, fontsize=xy_label_fontsize)
    ax.grid(grid)
    if legend:
        ax.legend()

    return ax


def draw_norm_hist(ax, norm_diffs, bins_n=100, draw_normal=True):
    counts, bins = np.histogram(norm_diffs, bins_n, density=True)

    if draw_normal:
        mu = np.mean(norm_diffs)
        sigma = np.sqrt(np.mean((norm_diffs - mu) ** 2))
        temp2 = np.linspace(bins[0], bins[-1], bins_n)
        y = scipy.stats.norm.pdf(temp2, mu, sigma)
        temp2 = temp2[y > (counts[counts > 0]).min()]

        ax.plot(temp2, scipy.stats.norm.pdf(temp2, mu, sigma), linewidth=2, color="red")

    ax.hist(bins[:-1], bins, weights=counts)

    return ax


def draw_norm_hists_for_different_models(
    fig, subplotspec_outer, hist, bins_n=100, draw_normal=True, number=-1
):
    h = (len(hist) + 2) // 3
    sgs = subplotspec_outer.subgridspec(h, 3, wspace=0.15, hspace=0.25)
    for i, one_optim_hist in enumerate(hist):
        ax = fig.add_subplot(sgs[i // 3, i % 3])
        if len(one_optim_hist["norm_diffs"]) > 0:
            ax = draw_norm_hist(
                ax,
                one_optim_hist["norm_diffs"][number],
                bins_n=bins_n,
                draw_normal=draw_normal,
            )
            label = one_optim_hist["name"]
            ax.set_title(
                "{},\n batch_count={}, skew={:0.2f}".format(
                    label,
                    len(one_optim_hist["norm_diffs"][number]),
                    stats.skew(one_optim_hist["norm_diffs"][number]),
                ),
                fontsize=11,
            )
        ax.set_ylabel("Density")
        ax.set_xlabel("Noise norm")
        ax.grid(True)

    ax = plt.Subplot(fig, subplotspec_outer)
    ax.set_title(
        "Distribution of the gradient noise for different optimizers",
        y=1.15,
        fontsize=14,
    )
    ax.set_frame_on(False)
    ax.axis("off")
    fig.add_subplot(ax)

    return ax


def draw_norm_hists_for_one_model(
    ax,
    hist,
    title,
    y_label="density",
    x_label="noise norm",
    bins_n=100,
    draw_normal=True,
    number=-1,
    grid=True,
    add_batch_count=True,
    round_batch_count=True,
    title_fontsize=14,
    xy_label_fontsize=10,
):
    one_optim_hist = hist[0]
    ax = draw_norm_hist(
        ax, one_optim_hist["norm_diffs"][number], bins_n=bins_n, draw_normal=draw_normal
    )

    batch_count = len(one_optim_hist["norm_diffs"][number])
    if round_batch_count:
        batch_count = "~{}k".format(batch_count // 1000)
    if add_batch_count:
        title = "{}, batch count={}".format(title, batch_count)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(y_label, fontsize=xy_label_fontsize)
    ax.set_xlabel(x_label, fontsize=xy_label_fontsize)
    ax.grid(grid)

    return ax


def draw_norm_hists_evolution(
    fig, subplotspec_outer, one_optim_hist, bins_n=100, draw_normal=True, w=None
):
    if w is None:
        w = len(one_optim_hist["norm_diffs"])

    label = one_optim_hist["name"]

    h = (len(one_optim_hist["norm_diffs"]) + w - 1) // w
    sgs = subplotspec_outer.subgridspec(h, w, wspace=0.15, hspace=0.05)
    for i, norm_diffs in enumerate(one_optim_hist["norm_diffs"]):
        ax = fig.add_subplot(sgs[i // w, i % w])
        if len(norm_diffs) > 0:
            ax = draw_norm_hist(ax, norm_diffs, bins_n=bins_n, draw_normal=draw_normal)
            step = 500 * i
            if "norm_diffs_x" in one_optim_hist:
                step = one_optim_hist["norm_diffs_x"][i]
            ax.set_title(
                "step = {},\n batch_count={}, skew={:0.2f}".format(
                    step, len(norm_diffs), stats.skew(norm_diffs)
                ),
                fontsize=11,
            )
        ax.set_ylabel("density")
        ax.set_xlabel("noise norm")
        ax.grid(True)

    ax = plt.Subplot(fig, subplotspec_outer)
    ax.set_title(
        "Evolution of the gradient noise for {}".format(label), y=1.15, fontsize=14
    )
    ax.set_frame_on(False)
    ax.axis("off")
    fig.add_subplot(ax)

    return ax
