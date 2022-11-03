import numpy as np
import glob
import os
embeddings_paths = glob.glob(os.path.join("preprocessed_data\\AISHELL3\\ge2e\\*", '*.npy'), recursive=True)
n_speakers=218
utterances_per_speaker=20
embeds=[]
speakers=[]
speaker=''
speakerid=-1
now_utterances=0
# speaker=range(len(embeddings_path))
for embedding_path in embeddings_paths: 
    if speaker!=embedding_path.replace("\\", " ").split()[-2]:
        speaker=embedding_path.replace("\\", " ").split()[-2]
        speakerid+=1
        now_utterances=0
    if now_utterances==20:
        continue
    now_utterances+=1    
    speakers.append(speakerid)
    embeds.append(np.load(embedding_path))
embeds=np.array(embeds)    
print(embeds.shape)
# colormap = np.array([
#     [76, 255, 0],
#     [0, 127, 70],
#     [255, 0, 0],
#     [255, 217, 38],
#     [0, 135, 255],
#     [165, 0, 165],
#     [255, 167, 255],
#     [0, 255, 255],
#     [255, 96, 38],
#     [142, 76, 0],
#     [33, 0, 127],
#     [0, 0, 0],
#     [183, 183, 183],
# ], dtype=np.float) / 255
colormap=np.random.rand(n_speakers,3)
colors = [colormap[i] for i in speakers]
import matplotlib.pyplot as plt
import umap 
reducer = umap.UMAP()
projected = reducer.fit_transform(embeds)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.scatter(projected[:, 0], projected[:, 1], c=colors)
plt.gca().set_aspect("equal", "datalim")
plt.title("说话人聚类")
plt.show()
