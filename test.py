
# from speechbrain.lobes.models.ECAPA_TDNN import Classifier
# import torch
# classify = Classifier(input_size=192, lin_neurons=2, out_neurons=218)
# outputs = torch.randn(4,192)
# cos = classify(outputs)
# print(cos)

# import numpy as np
# a=np.load("usr_embed/pjt_embedding.npy").squeeze(0).squeeze(0)
# # a=np.load("preprocessed_data\AISHELL3\spker_embed\ECAPA-TDNN\SSB0005-spker_embed.npy")
# print(a.shape)

# from preprocessor.preprocessor import Preprocessor
# import tgt
# import yaml
# from praatio import textgrid
# tg_path1="F:/dataset/BZNSYP/PhoneLabeling/000001.interval"

# tg_path2="preprocessed_data\AISHELL3\TextGrid\SSB0005\SSB00050001.TextGrid"
# # textgrid = tgt.io.read_textgrid(tg_path1)
# textgrid=textgrid.openTextgrid(
#             tg_path1, includeEmptyIntervals=True)
# config = yaml.load(open("config\AISHELL3\preprocess.yaml", "r"), Loader=yaml.FullLoader)
# preprocessor = Preprocessor(config)
# phone, duration, start, end = Preprocessor.get_alignment(preprocessor,textgrid.tierDict[textgrid.tierNameList[0]].entryList)
# print(phone, duration, start, end)


# alignment = textgrid.openTextgrid(
#             tg_path1, includeEmptyIntervals=True)
# # only with baker's annotation
# utt_id = alignment.tierNameList[0].split(".")[0]
# intervals = alignment.tierDict[alignment.tierNameList[0]].entryList
# phones = []
# for interval in intervals:
#     label = interval.label
#     phones.append(label)
# print(phones)

# from matplotlib.pyplot import text
# import numpy as np
# import yaml
# from synthesize import preprocess_mandarin
# preprocess_config = yaml.load(
#         open("config/baker/preprocess.yaml", "r",encoding='UTF-8'), Loader=yaml.FullLoader
#     )
# text="你好，我们是哥们儿。"
# texts = np.array([preprocess_mandarin(text, preprocess_config)])

#####################################################
#测试
server = "http://150.158.212.116:5000/TTS"
import requests
# from play import playwav
import time
def TTS(text):
    # data={"text":text,"speaker":0,"language":"zh","speed":"标准","high":"标准"}
    data={"text":text}
    #     data={"text":"","speaker":"","language":"","speed":"","high":""}
    request = requests.post(server, data=data)
    flow=request.content
    f=open("./demo/server_output/"+text+".wav","wb")
    f.write(flow)
texts=["啊啊啊啊啊啊啊1234，这是什么呀woasds"]
texts=["英伟达开源的自然语音处理开发套件",
    "仅通过几行代码",
    "就可以实现语音识别",
    "包含很多最新的端到端语音识别算法",
    "侧重算法研究与实现",
    "但不提供部署方案",
    "非流式语音识别常用的端到端语音识别框架",
    "语音合成模型及部署套件",
    "声纹技术：从核心算法到工程实践",
    "智能语音时代",
    "以下视频资料可能不是最新的",
    ]
n=len(texts)
start_time=time.time()
for i in texts:
    TTS(i)
end_time=time.time()
print((end_time-start_time)/n)


# from string import ascii_letters
# tr_table = str.maketrans({c:None for c in ascii_letters})
# return text.translate(tr_table)

############################################
# 内存图
# import matplotlib.pyplot as plt
# import numpy as np
# memory=[535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7, 535.7,535.7, 535.7, 535.7, 536.1, 536.1, 536.1, 651.2, 651.2, 651.2, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.3, 651.9, 651.9, 651.9, 652.2, 652.2, 652.2, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 652.3, 667.2, 667.2, 667.2, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 666.9, 667.2, 667.2, 667.2, 775.1, 775.1, 775.1, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 730.2, 732.5, 732.5, 732.5, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.7, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 732.9, 803.2, 803.2, 803.2, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.3, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 803.6, 810.8, 810.8, 810.8, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 810.9, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.2, 811.4, 811.4, 811.4]

# # memory=[786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.6,786.6,786.6,1022.0,1022.0,1022.0,1022.0,1030.6,1030.6,793.9,  793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,793.9,794.6,794.6,794.6,1041.6,1041.6,1041.6,1041.6,1041.6,1041.6,786.1,  786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,786.1,1002.6,1011.1,1011.1,1011.1,1015.2,815.8,  815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,815.8,819.6,819.6,819.6,1003.8,1003.8,1003.8,1003.8,1006.6,1006.6,981.5,  977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,977.3,986.1,986.1,986.1,1050.6,1050.6,1050.6,1050.6,1054.5,806.2,  806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.2,806.7,806.7,806.7,960.2,960.2,960.2,960.2,963.3,963.3,1038.2,  1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1038.2,1049.0,1049.0,1049.0,925.0,925.0,925.0,925.0,925.0,925.0,986.5,  986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,986.5,989.5,989.5,989.5,986.7,986.7,986.7,986.7,986.7,986.7,1036.2,  1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1036.2,1031.8,1041.8,1041.8,1041.8,947.5,947.5,947.5,947.5,956.2,956.2,919.9,  919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,919.9,960.5,960.5,960.5,960.5,960.5,960.5,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,936.9,911.7,911.7,911.7,911.7,911.7,911.7]
# x=np.arange(len(memory))

# plt.xticks(range(0, len(memory)+1, 50))
# # plt.yticks(range(500, 1051, 50))
# plt.yticks(range(500, 851, 50))
# # plt.title('CER-epoch',fontsize=20)
# plt.xlabel('time')
# plt.ylabel('memory/MB')
# plt.plot(x,memory,'.-') # 302.01ms
# plt.legend(loc="best",fontsize=14)
# plt.grid(axis='y',ls="--",c='black',)#打开坐标网格

# plt.show()