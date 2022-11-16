# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 01:03:52 2022

@author: A
"""


import tkinter as tk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk
#import pyimage2

from STACKING1 import calculate as  cal1
from STACKING2 import calculate as  cal2


#def Dialog():
#    tk.messagebox.showinfo('puttext',textField1.get())


 
 

def form2(): 
    def Cal1():
        
        cal1(infilepath,outFirstFilePath,outSecondFilePath)
    
    def Cal2():
        cal2(infilepath,outFirstFilePath,outSecondFilePath)
    ntk = tk.Tk();  
    ntk.title('融雪洪灾预测软件')
    ntk.geometry('600x600')
    infilepath = 'Normalized_primitive'
    outFirstFilePath = 'firstlevelresult'
    outSecondFilePath = 'secondlevelresult'
    
    button_1 = tk.Button(ntk,text = "回归计算",command = Cal1)
    button_2 = tk.Button(ntk,text = "分类计算",command = Cal2)
    
#    button_1 = tk.Button(ntk,text = "回归计算")
#    button_2 = tk.Button(ntk,text = "分类计算")
    
    button_1.grid(row = 0, column = 1,padx =10)
    button_2.grid(row = 0, column = 2,padx =10)
    
def form3():
    def new():
         s = '灾害点分类预测'
         lb1.config(text=s)
 
    def ope():
        s = '灾害点回归预测'
        lb1.config(text=s)
     
    def sav():
        s = '区域灾害分类预测'
        lb1.config(text=s)
     
    def cut():
         s = '区域灾害回归预测'
         lb1.config(text=s)
         
    def cop():
         s = '复制'
         lb1.config(text=s)
     
    def pas():
        s = '粘贴'
        lb1.config(text=s)
     
    def popupmenu(event):
        mainmenu.post(event.x_root,event.y_root)
        
        
    root = Tk()
    root.title('融雪洪灾预测软件')
    root.geometry('320x240')
     
    lb1 = Label(root,text='显示信息',font=('黑体',32,'bold'))
    lb1.place(relx=0.2,rely=0.2)
    
    mainmenu = Menu(root)
    menuFile = Menu(mainmenu)  # 菜单分组 menuFile
    mainmenu.add_cascade(label="灾害点",menu=menuFile)
    menuFile.add_command(label="灾害点分类预测",command=new)
    menuFile.add_command(label="灾害点回归预测",command=ope)
    menuFile.add_command(label="复制",command=sav)
    menuFile.add_separator()  # 分割线
    menuFile.add_command(label="退出",command=root.destroy)
     
    menuEdit = Menu(mainmenu)  # 菜单分组 menuEdit
    mainmenu.add_cascade(label="区域",menu=menuEdit)
    menuEdit.add_command(label="区域灾害分类预测",command=cut)
    menuEdit.add_command(label="区域灾害回归预测",command=cop())
    menuEdit.add_command(label="粘贴",command=pas())
    
    root.config(menu=mainmenu)
    root.bind('Button-3',popupmenu)

     
    
        
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

if __name__ == '__main__':
    form3()

#    image = Image.open("1.tif") 
    
#    photo = ImageTk.PhotoImage(image)
#    
#    imageLabel = tk.Label(ntk,image =photo,height = 300,width = 300)
#       
#    imageLabel.grid(row = 1, column = 0)
    
#    textField1 = tk.Entry(ntk)
#    
#    textField1.grid(row = 0, column = 0)  

#    ntk.mainloop()