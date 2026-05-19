import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from get_dataset import FineTuneDataset_prepared

def scatter(features, targets, subtitle=None, n_classes=10, save_path="./visualization.png"):
    palette = np.array(sns.color_palette("hls", n_classes))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(features[:, 0], features[:, 1], lw=0, s=40, c=palette[targets, :])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    for i in range(n_classes):
        xtext, ytext = np.median(features[targets == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()

def obtain_embedding_feature_map(online_network, classifier, test_dataloader, device):
    online_network.eval()
    classifier.eval()
    feature_map = []
    target_output = []
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            output = classifier(online_network(data)[0])
            output = F.log_softmax(output, dim=1)
            feature_map.extend(output.cpu().tolist())
            target_output.extend(target.tolist())
    feature_map = torch.Tensor(feature_map)
    target_output = np.array(target_output)
    return feature_map, target_output

def main():
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    config_ft = config['finetune']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train, X_test, Y_train, Y_test = FineTuneDataset_prepared(
        config_ft['ft'],
        range(config_ft['class_start'], config_ft['class_end'] + 1),
        config_ft['k_shot'],
        seed=0
    )

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=config_ft['test_batch_size'], shuffle=False)

    online_network = torch.load('./online_network.pth', map_location=device)
    classifier = torch.load('./classifier.pth', map_location=device)

    X_embed, targets = obtain_embedding_feature_map(online_network, classifier, test_dataloader, device)

    tsne = TSNE(n_components=2, random_state=0)
    tsne_embeds = tsne.fit_transform(X_embed.cpu().numpy())
    scatter(tsne_embeds, targets.astype(np.int64), n_classes=config_ft['class_end'] - config_ft['class_start'] + 1)

    score = sm.silhouette_score(X_embed.cpu().numpy(), targets, metric='euclidean')
    print(f"Silhouette score: {score:.4f}")

if __name__ == "__main__":
    main()