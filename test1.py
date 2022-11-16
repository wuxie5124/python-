# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk

class Index(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)    

#        self.pack(expand=1,fill="both")
#        self.frame_left = tk.Frame(self)
#        self.frame_left.pack(side='left',expand=1,fill="x",padx=5,pady=5)
        #三个按钮用于切换页面
        mainmenu = tk.Menu(root)
        menuFile = tk.Menu(mainmenu)  # 菜单分组 menuFile
        mainmenu.add_cascade(label="灾害点",menu=menuFile)
        menuFile.add_command(label="灾害点分类预测",command=self.change)
        menuFile.add_command(label="灾害点回归预测",command=self.change)
        menuFile.add_separator()  # 分割线
#        menuFile.add_command(label="退出",command=self.destroy)
         
        menuEdit =tk.Menu(mainmenu)  # 菜单分组 menuEdit
        mainmenu.add_cascade(label="区域",menu=menuEdit)
        menuEdit.add_command(label="区域灾害分类预测",command=self.change)
        menuEdit.add_command(label="区域灾害回归预测",command=self.change)
        
        
#        for i in ["增加","删除","撤销"]:
#            but = tk.Button(self.frame_left,text=i)
#            but.pack(side='top',expand=1,fill="y")
#            but.bind("<Button-1>",self.change) 
         #用于承载切换的页面内容      
#        self.frame_right=tk.Frame(self)
#        self.frame_right.pack(side='left',expand=1,fill="both",padx=5,pady=5)
#        lab = tk.Label(self.frame_right,text="我是第一个页面")
#        lab.pack() 
        #根据鼠标左键单击事件，切换页面
#    def change(self,event):
#        res = event.widget["label"]
#        for i in self.frame_right.winfo_children():
#            i.destroy()
#        if res == "灾害点分类预测":
#            Page1(self.frame_right)
#        elif res == "灾害点回归预测":
#            Page2(self.frame_right)
#        elif res == "区域灾害分类预测":
#            Page3(self.frame_right)
#        elif res == "区域灾害分类预测":
#            Page4(self.frame_right)
            
class Page1(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pack(expand=1,fill="both")
        tk.Label(self,text="我是page1").pack()
    
class Page2(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pack(expand=1,fill="both")
        tk.Label(self,text="我是page2").pack()
        
class Page3(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pack(expand=1,fill="both")
        tk.Label(self,text="我是page3").pack()
class Page4(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pack(expand=1,fill="both")
        tk.Label(self,text="我是page4").pack()
class Page5(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pack(expand=1,fill="both")
        tk.Label(self,text="我是page5").pack()
class Page6(tk.Frame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pack(expand=1,fill="both")
        tk.Label(self,text="我是page6").pack()        
def form1():  
    def ok():
        if user_name_entry.get()=='admin' and password_entry.get() == '123':
            ntk_loge.destroy() # 关闭登录窗体
            form2() # 进入第2个窗体:主窗体
        else:
            tk.messagebox.showwarning("警告：","密码错!")
    ntk_loge = tk.Tk();
    ntk_loge.title('登陆界面')
    ntk_loge.geometry('300x150+600+200')
 
    frame_user_name = Frame( ntk_loge, borderwidth=5)
    frame_user_name.pack(fill='x')
    user_name = Label(frame_user_name, text='用户名')
    user_name.pack(side='left', anchor='nw')
    user_name_entry = Entry(frame_user_name, bd=5)
    user_name_entry.pack(side='right', anchor='nw')
    
    frame_password  = Frame(ntk_loge,  borderwidth=5)
    frame_password.pack(fill='x')
    password = Label(frame_password, text='密码')
    password.pack(side='left')
    password_entry = Entry(frame_password, bd=5)
    
    password_entry.pack(side='right')


    but1=tk.Button(ntk_loge,text=" 确 定 ",command=ok) # 判断密码是否正确
    but1.pack(padx=50, side = 'left')
    but2=tk.Button(ntk_loge,text=" 退 出 ",command=ntk_loge.destroy) # 关闭登录窗体
    but2.pack(padx=50,side = 'right')
    
    ntk_loge.mainloop()
    
def form2():
    root = tk.Tk()
    root.title("融雪洪灾预测软件")
    root.geometry('800x600+600+200')
    Index(root)
    root.mainloop()

if __name__ == "__main__":
    form2()
