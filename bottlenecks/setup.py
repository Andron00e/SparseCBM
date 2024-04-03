from configs import *
import clip
import open_clip
from open_clip import tokenizer
from PIL import Image
from sklearn.manifold import TSNE
from bokeh.io import output_notebook
from sklearn.decomposition import PCA
import bokeh.models as bm, bokeh.plotting as pl
from sklearn.preprocessing import StandardScaler


def get_image_features(
    path_to_im: str,
    classes: dict,
    N: int,
    model_name="ViT-B-16",
    model_author="openai",
    device="cuda",
):
    """
    Args:
        path_to_im: path to the directory with images data
        classes: dict of class idx, class label pairs
        N: how many images of every class to take
        model_name: name of CLIP model's Image Encoder
        model_author: pretrained name of CLIP model from open_clip
        device: device
    Return: torch.Tensor of encoded by CLIP model images
    """
    image_dirs = [path_to_im + "/{}/".format(cls) for cls in classes.values()]

    selected_images = []
    for direc in image_dirs:
        filenames = os.listdir(direc)
        upd_filenames = [direc + f for f in filenames]
        selected_images.append(sorted(upd_filenames[:N]))

    selected_images = [item for sublist in selected_images for item in sublist]

    image_inputs = []
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=model_author, device=device
    )
    for name in tqdm(selected_images):
        im = Image.open(name)
        im_input = preprocess(im).unsqueeze(0).to(device)
        image_inputs.append(im_input)

    image_inputs = torch.stack(image_inputs, dim=0)

    image_encodings = []
    with torch.no_grad():
        for image in tqdm(image_inputs):
            image_feature = clip_model.encode_image(image)
            image_encodings.append(image_feature)

    image_features = torch.stack(image_encodings, dim=0)

    return image_features


def remove_prefixes(strings):
    prefixes = ["a", "an", "the"]
    result = []
    for string in strings:
        words = string.split()
        if words[0].lower() in prefixes:
            result.append(" ".join(words[1:]))
        else:
            result.append(string)

    return result


def get_text_features(
    path_to_text: str, model_name="ViT-B-16", model_author="openai", device="cuda"
):
    """
    Args:
        path_to_text: path to the directory with texts data
        model_name: name of CLIP model's Image Encoder
        model_author: pretrained name of CLIP model from open_clip
        device: device
    Return: torch.Tensor of encoded by CLIP model texts
    """
    with open(path_to_text, "r") as f:
        texts = f.read().lower().split("\n")
        texts = remove_prefixes(texts)

    text_encodings = []
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=model_author, device=device
    )

    with torch.no_grad():
        for c in tqdm(texts):
            text_input = clip.tokenize(c).to(device)
            text_feature = clip_model.encode_text(text_input)
            text_encodings.append(text_feature)

    text_features = torch.stack(text_encodings, dim=0)

    return text_features


def similarity(a: torch.Tensor, b: torch.Tensor):
    nom = a @ b.T
    denom = a.norm(dim=-1) * b.norm(dim=-1)
    return nom / denom


def cubed_similarity(a: torch.Tensor, b: torch.Tensor):
    nom = a**3 @ (b**3).T
    denom = (a**3).norm(dim=-1) * (b**3).norm(dim=-1)
    return nom / denom


def get_dot_prods_matrix(
    image_features: torch.Tensor, text_features: torch.Tensor, eps=1e-8
):
    """
    Args:
        image_features: tensor of shape [num_images, dim]
        text_features: tensor of shape [num_texts, dim]
        eps: to avoid division by zero
    Return: images-texts matrix with normalized rows (sum of each row == 1.)
    """
    image_features = image_features.squeeze(dim=1)
    text_features = text_features.squeeze(dim=1)
    image_norms = image_features.norm(dim=-1, keepdim=True) ** 2
    text_norms = text_features.norm(dim=-1, keepdim=True) ** 2
    matrix = image_features @ text_features.T
    matrix /= torch.sqrt(image_norms @ text_norms.T)
    row_sum = torch.sum(matrix, dim=1, keepdim=True)
    matrix /= row_sum + eps
    return matrix


def draw_vectors(
    x,
    y,
    radius=10,
    alpha=0.25,
    color="blue",
    width=600,
    height=400,
    show=True,
    **kwargs,
):
    """draws an interactive plot for data points with auxiliary info on hover"""
    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource({"x": x, "y": y, "color": color, **kwargs})

    fig = pl.figure(active_scroll="wheel_zoom", width=width, height=height)
    fig.scatter("x", "y", size=radius, color="color", alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def generate_slices_for_classes(classes: dict, V_matrix: torch.Tensor):
    """
    classes: dict of cls_idx: class_name
    V_matrix: image-concepts matrix
    return: slices for every class
    """
    slice_size = len(V_matrix) // len(classes)
    slices = {}
    for i, class_idx in enumerate(classes):
        start = i * slice_size
        end = (i + 1) * slice_size
        slices[classes[class_idx]] = slice(start, end)

    return slices


def calculate_similarity_score(
    classes: dict, V_rows: torch.Tensor, T_matrix: torch.Tensor, sim: str = "sim"
):
    """
    V_rows: a rows of V_matrix for the same class
    T_matrix: classes-concepts matrix
    return: a dictionary with similarity scores for each class
    """
    scores = {}

    for v_row in V_rows:
        for i in range(len(T_matrix)):
            t_row = T_matrix[i]

            if sim == "sim":
                sim = similarity(v_row, t_row)
            else:
                sim = cubed_similarity(v_row, t_row)

            class_name = classes[i]

            if class_name in scores:
                scores[class_name] += sim.item()
            else:
                scores[class_name] = sim.item()

    return scores


def get_scores_dict(classes: dict, V_matrix: torch.Tensor, T_matrix: torch.Tensor):
    """
    Args:
        classes: classes dict
        V_matrix:
        T_matrix:
    Return: scores_dict for drawing similarity scores
    """
    scores_dict = {}
    slices = generate_slices_for_classes(classes, V_matrix)

    for class_name, slice_range in slices.items():
        V_rows = V_matrix[slice_range]
        scores = calculate_similarity_score(classes, V_rows, T_matrix)
        scores_dict[class_name] = scores

    return scores_dict


def draw_similarity_scores(scores_dict: dict, true_class: str):
    """
    scores_dict: a nested dictionary with similarity scores
    true_class: the true image class for which scores should be plotted
    """

    if true_class not in scores_dict:
        print(f"True class '{true_class}' not found in the scores dictionary.")
        return

    scores = scores_dict[true_class]
    df = pd.DataFrame(list(scores.items()), columns=["Class", "Total Similarity Score"])

    plt.figure(figsize=(6, 3))  # 12 6
    sns.scatterplot(data=df, x="Class", y="Total Similarity Score")
    plt.title(f"Similarity Scores for True Class: {true_class}")
    plt.xticks(rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Total Similarity Score")
    plt.tight_layout()
    plt.show()


def calculate_similarity_score_accuracy_for_class(
    class_name: str, V_matrix: torch.Tensor, T_matrix: torch.Tensor, classes: dict
):
    """
    Test the accuracy of the hypothesis
    return: accuracy score for the class_name label by similarity method
    """
    slices = generate_slices_for_classes(classes, V_matrix)
    slice_range = slices[class_name]
    V_rows = V_matrix[slice_range]
    sim_matrix = torch.zeros((V_rows.shape[0], T_matrix.shape[0]))

    correct, total = 0, 0

    for i, v_row in enumerate(V_rows):
        for j, t_row in enumerate(T_matrix):
            sim_matrix[i, j] = similarity(v_row, t_row).item()

    for idx in range(sim_matrix.shape[0]):
        pred_idx = torch.argmax(sim_matrix[idx])

        if pred_idx == list(classes.values()).index(class_name):
            correct += 1.0
        total += 1.0

    return 100 * correct / total


def similarity_score_accuracy(
    classes: dict, V_matrix: torch.Tensor, T_matrix: torch.Tensor
):
    mean = np.mean(
        [
            calculate_similarity_score_accuracy_for_class(
                class_name, V_matrix, T_matrix, classes
            )
            for class_name in classes.values()
        ]
    )
    return "Similarity Score accuracy: {}%".format(mean)


def calculate_max_score_accuracy_for_class(
    class_name: str, V_matrix: torch.Tensor, T_matrix: torch.Tensor, classes: dict
):
    """
    Args:
        class_name: name of class from classes dict
        V_matrix: yeap
        T_matrix: yeap
        classes: classes dict
    Return:
    """
    slices = generate_slices_for_classes(classes, V_matrix)
    slice_range = slices[class_name]
    V_rows = V_matrix[slice_range]

    correct, total = 0, 0

    for v_row in V_rows:
        max_elem_idx = torch.argmax(v_row).item()
        t_matrix_column = T_matrix[:, max_elem_idx]
        t_max_elem_idx = torch.argmax(t_matrix_column).item()

        if classes[t_max_elem_idx] == class_name:
            correct += 1
        total += 1

    return 100 * correct / total


def max_score_accuracy(classes: dict, V_matrix: torch.Tensor, T_matrix: torch.Tensor):
    mean = np.mean(
        [
            calculate_max_score_accuracy_for_class(
                class_name, V_matrix, T_matrix, classes
            )
            for class_name in classes.values()
        ]
    )
    return "Max Score accuracy: {}%".format(mean)
