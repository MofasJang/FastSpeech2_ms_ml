from email.policy import default
import tkinter as tk
from tkinter import ttk
# from PIL import ImageTk
from tkinter.filedialog import askopenfilename
from click import option
import requests
from play import playwav

class App:
    def __init__(self):
        window.title('语音合成GUI')
        window.geometry('800x600')
        window.resizable(0,0)   # 阻止窗口的大小调整
        window.mainloop()

window = tk.Tk()
canvas = tk.Canvas(window,height=600, width=1000)
# image_file = ImageTk.PhotoImage(file='bg3.jpg')
# image = canvas.create_image(0,0,anchor='center',image=image_file)
tk.Label(canvas,).place(x=650,y=0)
canvas.pack(side='top')


tk.Label(window,text='端到端语音合成系统',font=('华文行楷',35),width=30).place(x=80,y=150)
tk.Label(window,text='请输入',font=('黑体',15)).place(x=190,y=250)
L1=tk.Entry(window,font=('黑体',15),width=30)
L1.place(x=270,y=250)
# tk.Label(window,text='语言：',font=('黑体',15)).place(x=190,y=280)
# value1 = tk.StringVar()
# value1.set('中文')
# values1 = ["中文","英文"]
# combobox1 = ttk.Combobox(
#         master=window,  # 父容器
#         height=10,  # 高度,下拉显示的条目数量
#         width=5,  # 宽度
#         state='readonly',  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
#         cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
#         font=('', 15),  # 字体
#         textvariable=value1,  # 通过StringVar设置可改变的值
#         values=values1,  # 设置下拉框的选项
#         )
# combobox1.place(x=270,y=280)
# lan_dict={"中文":"zh","英文":"en"}

tk.Label(window,text='说话人',font=('黑体',15)).place(x=190,y=310)
value2 = tk.StringVar()
value2.set('0')
values2 = [i for i in range(218)]
combobox2 = ttk.Combobox(
        master=window,  # 父容器
        height=10,  # 高度,下拉显示的条目数量
        width=5,  # 宽度
        state='readonly',  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
        cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
        font=('', 15),  # 字体
        textvariable=value2,  # 通过StringVar设置可改变的值
        values=values2,  # 设置下拉框的选项
        )
combobox2.place(x=270,y=310)

tk.Label(window,text='语速',font=('黑体',15)).place(x=190,y=370)
value3 = tk.StringVar()
value3.set('标准')
values3 = ["慢速","标准","快速"]
combobox3 = ttk.Combobox(
        master=window,  # 父容器
        height=10,  # 高度,下拉显示的条目数量
        width=5,  # 宽度
        state='readonly',  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
        cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
        font=('', 15),  # 字体
        textvariable=value3,  # 通过StringVar设置可改变的值
        values=values3,  # 设置下拉框的选项
        )
combobox3.place(x=270,y=370)

tk.Label(window,text='音高',font=('黑体',15)).place(x=190,y=430)
value4 = tk.StringVar()
value4.set('标准')
values4 = ["低音","标准","高亢"]
combobox4 = ttk.Combobox(
        master=window,  # 父容器
        height=10,  # 高度,下拉显示的条目数量
        width=5,  # 宽度
        state='readonly',  # 设置状态 normal(可选可输入)、readonly(只可选)、 disabled
        cursor='arrow',  # 鼠标移动时样式 arrow, circle, cross, plus...
        font=('', 15),  # 字体
        textvariable=value4,  # 通过StringVar设置可改变的值
        values=values4,  # 设置下拉框的选项
        )
combobox4.place(x=270,y=430)
server = "http://150.158.212.116:5000/TTS"
# server = "http://localhost:5000/TTS"

def TTS():    # 合成 
    L2=tk.Label(window,text='',font=('黑体',15)).place(x=340,y=470)
#     data={"text":L1.get(),"speaker":combobox2.get(),"language":lan_dict[combobox1.get()]}
    # data={"text":L1.get(),"speaker":combobox2.get(),"language":"zh","speed":combobox3.get(),"high":combobox4.get()}
#     data={"text":"","speaker":"","language":"","speed":"","high":""}
    data={"text":L1.get()}
    request = requests.post(server,data=data)
    text=request.content
    f=open("test.wav","wb")
    f.write(text)
    playwav("test.wav")
    L2=tk.Label(window,text='合成完成',font=('黑体',15)).place(x=340,y=470)

# def selectPath():  #选择音频文件
#     L1['text']='识别中...'
#     path_ = askopenfilename(initialdir='E:\毕业设计\项目资源\data_aishell\wav\dev\S0729')
#     f = open(path_ , "rb")
#     files = {"file": f}
#     r = requests.post(server, files=files)
#     L1['text']='识别完成'
#     L2['text']='识别结果：'+r.text

rec = tk.Button(window,text='合成',font=('黑体',15),activebackground='red',command=TTS).place(x=350,y=500)
# rec = tk.Button(window,text='本地音频',font=('黑体',15),activebackground='red',command=selectPath).place(x=450,y=400)
App()