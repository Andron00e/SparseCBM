import torch
import transformers
import torch.nn as nn
import seaborn as sns
from configs import *
import matplotlib.pyplot as plt
import torch.nn.functional as F


class BaseCBModel(nn.Module):
    """
    Basic architecture: backbone model + two layers: CBL and FC (head)
    Backbone model must be implemented with correct .logits_per_image method
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        backbone_name: str = "openai/clip-vit-base-patch32",
        train_backbone: bool = False,
    ):
        super().__init__()
        if backbone_name == Constants.altclip_link:
            self.backbone = transformers.AltCLIPModel.from_pretrained(backbone_name)
            self.processor = transformers.AltCLIPProcessor.from_pretrained(
                backbone_name
            )
        elif backbone_name == Constants.align_link:
            self.backbone = transformers.AlignModel.from_pretrained(backbone_name)
            self.processor = transformers.AlignProcessor.from_pretrained(backbone_name)
        elif backbone_name in [
            Constants.siglip_so_link,
            Constants.siglip_base_link,
            Constants.siglip_large_link,
            Constants.siglip_large_256_link,
        ]:
            self.backbone = transformers.AutoModel.from_pretrained(backbone_name)
            self.processor = transformers.AutoProcessor.from_pretrained(backbone_name)
        else:
            self.backbone = transformers.CLIPModel.from_pretrained(backbone_name)
            self.processor = transformers.CLIPProcessor.from_pretrained(backbone_name)
        for param in self.backbone.parameters():
            param.requires_grad = train_backbone
        self.cbl = nn.Linear(num_concepts, num_concepts, bias=False)
        self.head = nn.Linear(num_concepts, num_classes, bias=False)

    def forward(self, **batch):
        cbl_out = self.cbl(self.backbone(**batch).logits_per_image)
        return cbl_out, self.head(cbl_out)


class BaseCBModelWithLora(torch.nn.Module):
    """
    Base class for introducing CBM with LoRA adapters.
    Args:
        connect_to: can be either vit self attn, text self attn or last linear layer of a backbone model, i.e., projection
    """

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        backbone_name: str = "openai/clip-vit-base-patch32",
        train_backbone: bool = False,
        num_loras: int = 1,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        connect_to: str = "last",
    ):
        super().__init__()
        if backbone_name == Constants.altclip_link:
            backbone = transformers.AltCLIPModel.from_pretrained(backbone_name)
            self.processor = transformers.AltCLIPProcessor.from_pretrained(
                backbone_name
            )
        elif backbone_name == Constants.align_link:
            backbone = transformers.AlignModel.from_pretrained(backbone_name)
            self.processor = transformers.AlignProcessor.from_pretrained(backbone_name)
        elif backbone_name in [
            Constants.siglip_so_link,
            Constants.siglip_base_link,
            Constants.siglip_large_link,
            Constants.siglip_large_256_link,
        ]:
            backbone = transformers.AutoModel.from_pretrained(backbone_name)
            self.processor = transformers.AutoProcessor.from_pretrained(backbone_name)
        else:
            backbone = transformers.CLIPModel.from_pretrained(backbone_name)
            self.processor = transformers.CLIPProcessor.from_pretrained(backbone_name)
        for param in backbone.parameters():
            param.requires_grad = train_backbone
        self.num_loras = num_loras
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.connect_to = connect_to
        if self.connect_to == "last":
            target_modules = ["visual_projection"]
        elif self.connect_to == "self-attn":
            target_modules = [
                "k_proj",
                "q_proj",
                "v_proj",
                "out_proj",
            ]
        elif self.connect_to == "vit-self-attn":
            target_modules = []
        elif self.connect_to == "text-self-attn":
            target_modules = []
        peft_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
        )
        for i in range(self.num_loras):
            self.backbone = get_peft_model(
                backbone, peft_config, adapter_name=f"lora{i}"
            )
        for name, p in self.backbone.named_parameters():
            if "lora" in name:
                p.requires_grad_(True)

        self.cbl = nn.Linear(num_concepts, num_concepts, bias=False)
        self.head = nn.Linear(num_concepts, num_classes, bias=False)

    def forward(self, **batch):
        cbl_out = self.cbl(self.backbone(**batch).logits_per_image)
        return cbl_out, self.head(cbl_out)


class BaseCBModelForSegmentation(nn.Module):
    pass


def contrastive_loss(logits, dim: int):
    """
    Contrastive loss, adapted from https://sachinruk.github.io/blog/2021-03-07-clip.html
    """
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()


def criterion_contrastive(similarity: torch.Tensor) -> torch.Tensor:
    """
    Args:
        similarity: is equal to logits_per_image
    """
    caption_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (caption_loss + image_loss) / 2.0


def gumbel_contrastive_loss(logits, tau: float = 1.0, hard: bool = False, dim: int = 0):
    """
    Simple contrastive loss based on Gumbel-Softmax distribution
    """
    gumbel_softmax_samples = F.gumbel_softmax(logits, tau=tau, dim=dim, hard=hard)
    neg_ce = -torch.log(gumbel_softmax_samples)
    return neg_ce.mean()


def criterion_gumbel(
    similarity: torch.Tensor, tau: float = 1.0, hard: bool = False
) -> torch.Tensor:
    """
    Self-supervised contrastive loss for CBL training
    """
    first_loss = gumbel_contrastive_loss(similarity, tau=tau, hard=hard, dim=0)
    second_loss = gumbel_contrastive_loss(similarity, tau=tau, hard=hard, dim=1)
    return (first_loss + second_loss) / 2.0


def criterion_l1(model, l1_lambda=1e-3):
    """
    Implements L1 regularization.
    During training should be scaled by len(concepts)
    """
    l1_loss = 0.0
    for param in model.cbl.parameters():
        l1_loss += torch.norm(param, p=1)

    return l1_loss * l1_lambda


def criterion_similarity(
    logits_per_image: torch.Tensor, cbl_logits: torch.Tensor, is_cubed: bool = False
) -> torch.Tensor:
    """
    Implementation of cosine similarity loss between .logits_per_image and outputs of CBL layer
    CLIP-like models normalize its logits_per_image outputs, in this case, only the normalization of
    CBL layer outputs is necessary. But we are trying to normalize both inputs.
    """
    logits_per_image = logits_per_image - torch.mean(
        logits_per_image, dim=0, keepdim=True
    )
    cbl_logits = cbl_logits - torch.mean(cbl_logits, dim=0, keepdim=True)
    if is_cubed == True:
        logits_per_image = logits_per_image**3
        cbl_logits = cbl_logits**3
    logits_per_image = logits_per_image / torch.norm(
        logits_per_image, p=2, dim=0, keepdim=True
    )
    cbl_logits = cbl_logits / torch.norm(cbl_logits, p=2, dim=0, keepdim=True)
    similarities = torch.sum(cbl_logits * logits_per_image, dim=0)
    return torch.mean(similarities)


def draw_bottleneck(
    image, cbl_logits, k: int, concepts: list, draw_probs: bool = False
):
    """
    Having a CBL outputs draws a bottleneck scores for a single PIL image
    """
    top_values, top_indices = torch.topk(cbl_logits, k)

    if draw_probs == True:
        top_values = torch.nn.functional.softmax(top_values, dim=-1)

    import pandas as pd

    data = pd.DataFrame(
        {
            "Concepts": [concepts[i] for i in top_indices.squeeze().tolist()],
            "Probability": top_values.squeeze().tolist(),
        }
    )

    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x="Probability", y="Concepts", data=data)
    plt.xlabel("Weight", fontsize=16)
    plt.ylabel("Concepts", fontsize=16)
    plt.title("Top {} Logits".format(k), fontsize=16)

    for i, value in enumerate(top_values.squeeze().tolist()):
        plt.text(value + 0.01, i, f"{value:.2f}", va="center")

    plt.show()
