from umap import UMAP
import plotly.express as px
from sklearn.manifold import TSNE

def vis_distribution(feats, tags):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)

    tsne = TSNE(n_components=2, random_state=0)
#     proj_2d = tsne.fit_transform(feats)
    proj_2d = umap_2d.fit_transform(feats)

    return px.scatter(
        proj_2d, x=0, y=1,
        color=tags, labels={'color': 'classes'}
    )