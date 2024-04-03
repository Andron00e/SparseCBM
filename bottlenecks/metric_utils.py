import sklearn
import warnings
from cbm import *
from configs import *
from trainer_utils import *
from typing import List, Dict, Optional


def plot_trainer_metrics(hist: List[Dict]):
    """
    Function that plots metrics in metrics_to_draw dict for trainer.hist
    Example of usage:
        trainer.train()
        plot_trainer_metrics(trainer.hist)
    """
    num_rows = 3
    num_cols = 4
    for metrics_dict in hist:
        print_centered_text(metrics_dict["name"])
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
        metrics_to_draw = [
            "train_loss",
            "train_cbl_loss",
            "train_acc_top_1",
            "train_acc_top_5",
            "val_loss",
            "val_cbl_loss",
            "val_acc_top_1",
            "val_acc_top_5",
            "val_precision",
            "val_recall",
            "val_f1",
            "train_x",
        ]
        drawn_metrics_dict = {
            k: metrics_dict[k] for k in metrics_to_draw if k in metrics_dict
        }
        for idx, (metric_name, metric_values) in enumerate(drawn_metrics_dict.items()):
            if metric_name in metrics_to_draw:
                row_idx = idx // num_cols
                col_idx = idx % num_cols
                ax = axs[row_idx, col_idx] if len(metrics_dict.keys()) > 1 else axs
                ax.plot(metric_values)
                ax.set_title(metric_name)

        plt.tight_layout()
        plt.grid(True)
        plt.show()


@torch.no_grad()
def draw_confusion_matrix(
    model,
    loader,
    device,
    save_drawing: bool = False,
):
    """
    Draws a confusion matrix for model prediction on the selected dataloader
    If you use this function after training with BottleneckTrainer type:
    draw_confusion_matrix(trainer.nets[0], loader)
    it will visualize a confusion matrix for the first trained model
    """
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for step, batch in enumerate(loader, 0):
            warnings.filterwarnings("ignore")
            inputs, labels = batch
            inputs, targets = inputs.to(device), torch.LongTensor(labels).to(device)
            cbl_logits, logits = model(**inputs)
            all_predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    confusion_matrix = sklearn.metrics.confusion_matrix(
        y_true=all_targets, y_pred=all_predictions
    )
    plt.figure(figsize=(10, 8), dpi=300)
    plt.imshow(confusion_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    if save_drawing:
        plt.savefig("confusion_matrix.jpg")
    plt.show()


@torch.no_grad()
def cbm_interpretability_scores(model, processor, concepts, image, device):
    """
    Draws an interpretability bars showing which concepts are highly activated for CBM
    Args:
        model: refers to the trained Concept Bottleneck Model
        processor: makes a preprocessing of both images and texts in order to make them acceptable by model
    Return:
        cbl_logits: outputs of the Concept Bottleneck Layer, each component of the output corresponds to some concept
    """
    with torch.no_grad():
        model.to(device)
        inputs = processor(
            text=concepts, images=image, return_tensors="pt", padding=True
        ).to(device)
        cbl_logits, logits = model(**inputs)
    return cbl_logits


@torch.no_grad()
def backbone_interpretability_scores(model, processor, concepts, image, device):
    """
    Draws an interpretability bars showing which concepts are highly activated for backbone model
    Args:
        model: refers to the backbone model, i.e., multi-modal encoder NN
        processor: makes a preprocessing of both images and texts in order to make them acceptable by model
    """
    with torch.no_grad():
        model.to(device)
        inputs = processor(
            text=concepts, images=image, return_tensors="pt", padding=True
        ).to(device)
        logits_per_image = model(**inputs).logits_per_image
    return logits_per_image
