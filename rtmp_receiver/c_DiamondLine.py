# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 17:30:54 2022

@author: g107904
"""

import numpy as np
import matplotlib.pyplot as plt
import diamand_vanish
import os
import cv2
import math

d = 5

def S_point(point):
    x = point[0]
    y = point[1]
    w = point[2]
    return [-x+y,-d*w,d*x]

def S_line(line):
    a = line[0]
    b = line[1]
    c = line[2]
    return [d*b,-c,a+b]

def T_point(point):
    x = point[0]
    y = point[1]
    w = point[2]
    return [x+y,-d*w,d*x]

def T_line(line):
    a = line[0]
    b = line[1]
    c = line[2]
    return [d*b,-c,a-b]

def SS(line):
    return S_point(S_line(line))

def ST(line):
    return S_point(T_line(line))

def TS(line):
    return T_point(S_line(line))

def TT(line):
    return T_point(T_line(line))

def sgn(val):
    if val >= 0:
        return 1
    else:
        return -1
    
def ST_line(line):
    x = np.arange(-d,d,0.1)
    y = -(line[2]+line[0]*x)/line[1]
    plt.plot(x,y)
    
def polygon(line):
    a = line[0]
    b = line[1]
    c = line[2]
    alpha = sgn(a*b)
    beta = sgn(b*c)
    gamma = sgn(a*c)
    res = []
    space_c = d
    a_x = alpha*a*space_c*space_c / (c + gamma*a*space_c)
    b_x = -alpha*c*space_c / (c + gamma*a*space_c)
    
    
    res.append(a_x)
    res.append(b_x)
    res.append(b*space_c*space_c / (c + beta*b*space_c) )
    res.append(0)
    res.append(0)
    res.append(b * space_c / (a + alpha*b))
    res.append(-a_x)
    res.append(-b_x)
    return res

def diamond_space():
    ax = plt.gca()
 
    ax.set_xlim(-170,170)
    ax.set_ylim(-170,170)
    # 去掉上、右二侧的边框线
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
     
    # 将左侧的y轴，移到x=0的位置
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    #ax获得y轴，并将y轴与x中的0对齐
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    
    #隐藏掉y轴的0坐标，不然和x轴重了，不好看，0位于从下到上第6个
    plt.xticks([])
    plt.yticks([])
    
    plt.plot([-160,-160],[170,-170],'k--',linewidth = 1)
    plt.arrow(-160,-165,0,-0.01,color='k' ,head_length=5,head_width=5)
    plt.text(-150,-165,'-y')
    
    plt.text(-80,160,'T')
    
    plt.plot([160,160],[170,-170],'k--',linewidth = 1)
    plt.arrow(160,165,0,0,color='k' ,head_length=5,head_width=5)
    plt.text(150,165,'y')
    
    plt.text(80,160,'S')
    
    plt.text(-8,-15,'0')
    
    plt.arrow(165,0,0.1,0,color='k' ,head_length=5,head_width=5)
    plt.arrow(0,165,0,0,color='k' ,head_length=5,head_width=5)
    plt.text(165,-20,'u')
    plt.text(-20,165,'v(x)')
    
    plt.plot([0,160],[40,20],color='r')
    plt.plot([0,-160],[40,-20],color='g')
    
    plt.text(-10,40,'A')
    plt.text(-20,25,'[0,p]')
    plt.text(162,20,'B')
    plt.text(160,8,'[d,q]')
    plt.text(-155,-20,'C')
    plt.text(-155,-35,'[-d,-q]')
    
def line_trans():
    ax = plt.gca()
 
    ax.set_xlim(-170,170)
    ax.set_ylim(-100,100)
    # 去掉上、右二侧的边框线
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    
    
    plt.plot([-160,-160],[-100,100],'k',linewidth=1)
    plt.plot([-170,-40],[-90,-90],'k',linewidth=1)
    
    plt.arrow(-45,-90,0.1,0,color='k' ,head_length=5,head_width=5)
    plt.arrow(-159,95,0,0,color='k' ,head_length=5,head_width=5)
    plt.text(-45,-85,'x')
    plt.text(-155,95,'y')
    plt.text(-185,-99,'[0,0]')
    
    plt.plot([-170,-50],[80,-100],'k--',linewidth=1) 
    #-3/2 x - 25 + 90
    # -70,-70 -80 -55
    #-90,-40
    plt.plot([-70],[-70],'r.') #90,20
    plt.text(-65,-70,'A')
    plt.plot([-110],[-10],'r.')#50,80
    plt.text(-105,-10,'C')
    plt.plot([-90],[-40],'r.')#70,50
    plt.text(-85,-40,'B')
    
    plt.text(-175,80,'l')

    
    plt.plot([0,170],[-90,-90],'k',linewidth=1)
    plt.plot([10,10],[-100,100],'k',linewidth=1)
    plt.plot([160,160],[-100,100],'k',linewidth=1)
    plt.arrow(165,-90,0.1,0,color='k' ,head_length=5,head_width=5)
    plt.arrow(161,95,0,0,color='k' ,head_length=5,head_width=5)
    plt.arrow(10,95,0,0,color='k' ,head_length=5,head_width=5)
    plt.text(165,-85,'u')
    plt.text(15,95,'v(x)')
    plt.text(165,95,'y')
    plt.text(1,-100,'0')
    plt.text(150,-100,'0')
    
    plt.text(-25,5,'S')
    plt.plot([-35,-15],[0,0],'k',linewidth=1)
    plt.arrow(-15,-1,0.01,0,color='k' ,head_length=5,head_width=5)
    
    plt.plot([10,160],[0,-70],'k--',linewidth=1) #0,90   ,20
    plt.text(-6,0,'A.x')
    plt.text(165,-70,'A.y')
    
    plt.plot([10,160],[-40,-10],'k--',linewidth=1)#0,50    80
    plt.text(-6,-40,'C.x')
    plt.text(165,-10,'C.y')        
    plt.plot([10,160],[-20,-40],'k--',linewidth=1)#0,70  50
    plt.text(-6,-20,'B.x')
    plt.text(165,-40,'B.y')
    
    
    
    plt.plot([70],[-28],'r.')
    plt.text(70,-23,'L')
    
    
    plt.xticks([])
    plt.yticks([])

def draw_diamand():
    line = [1,2,-3]
    res = polygon(line)
    
    plt.subplot(121)
    
    ax = plt.gca()
     
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    ax.spines['bottom'].set_position(('data',0))
    
    
    ax.spines['left'].set_position(('data',0))
    plt.xticks([0])
    plt.yticks([])
    
    plt.arrow(0,4.8,0,0,color='k' ,head_length=0.2,head_width=0.2)
    plt.arrow(4.8,0,0.01,0,color='k' ,head_length=0.2,head_width=0.2)
    
    
    plt.plot([5,3],[-1,0],'k',linewidth=1)
    plt.plot([3,0],[0,1.5],'k',linewidth=1)
    plt.plot([0,-5],[1.5,4],'k',linewidth=1)
    plt.text(2,1,'ab')
    plt.text(5,-0.5,'cd')
    plt.text(-2,3,'bc')
    plt.plot(3,0,color="black", marker='o', markerfacecolor='white')
    plt.plot(0,1.5,color="black", marker='o', markerfacecolor='white')
    plt.text(5,0.2,'x')
    plt.text(0.2,5,'y')
    
    
    plt.subplot(122)
    
    ax = plt.gca()
     
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    
    plt.xticks([0])
    plt.yticks([])
    plt.arrow(0,4.8,0,0,color='k' ,head_length=0.2,head_width=0.2)
    plt.arrow(4.8,0,0.01,0,color='k' ,head_length=0.2,head_width=0.2)
    
    plt.plot([-5,0],[0,-5],'k--',linewidth=1)
    plt.plot([5,0],[0,5],'k--',linewidth=1)
    plt.plot([0,-5],[5,0],'k--',linewidth=1)
    plt.plot([0,5],[-5,0],'k--',linewidth=1)
    
    plt.text(res[0],res[1],'A')
    plt.text(res[2],res[3]+0.5,'B')
    plt.text(res[4],res[5],'C')
    plt.text(res[6],res[7],'D')
    plt.text(5,0.2,'u')
    plt.text(0.2,5,'v')
    
    for i in range(0,6,2):
        plt.plot([res[i],res[i+2]],[res[i+1],res[i+3]],'k',linewidth=1)
        
K = np.zeros((3,3))
K[0][0] = 525
K[0][2] = 320
K[1][1] = 525	
K[1][2] = 240
K[2][2] = 1
K_inv = np.linalg.inv(K) 


def jacbian(x,w):
    res = np.zeros((4,1))
    res[0] = 2*w[0]*x[0] + w[1] * x[1]
    res[1] = w[1] * x[0] + 2 * x[1]
    res[2] = 2*w[2]*x[2] + w[3] * x[3]
    res[3] = w[3] * x[3] + 2 * x[3]
    return res

def residual(x,w):
    res = 0
    res += w[0] * x[0] * x[0]
    res += w[1] * x[0] * x[1]
    res += x[1] * x[1]
    res += w[2] * x[2] * x[2]
    res += w[3] * x[2] * x[3]
    res += x[3] * x[3] + 1
    res = res * res
    return res

def jacbian_k(x,w):
    res = np.zeros((4,1))
    res[0] = -2/x[0]*(x[1]*x[1] - w[1]*x[1] + w[0]) / x[0] / x[0]
    res[1] = (2*x[1]-w[1])/x[0]/x[0]
    res[2] = -2/x[2]*(x[3]*x[3] - w[3]*x[3] + w[2]) / x[2] / x[2]
    res[3] = (2*x[3]-w[3])/x[2]/x[2]
    return res

def residual_k(x,w):
    res = 0
    res += (x[1]*x[1] - w[1]*x[1]+w[0]) / x[0]/x[0]
    res += (x[3]*x[3] - w[3]*x[3]+w[2])/ x[2]/x[2]
    res = (res+1)*(res+1)
    return res

def LM(x,w):
    lam = 0
    r0 = residual(x,w)
    for it in range(100):
        J = jacbian(x, w)
        A = np.matmul(J,J.transpose())
        b = -J * r0
        pow_of_lam = 0
        while True:
            tmp_A = A
            for j in range(4):
                tmp_A[j][j] *= (1+lam)
            A_inv = np.linalg.pinv(A)
            increase = np.matmul(np.linalg.pinv(A),b)
            increase = np.resize(increase,4)
            pow_of_lam += 1
            r_candidate = residual(x+increase,w)
            print(r_candidate)
            if(r_candidate < r0):
                x = x+increase
                r0 = r_candidate
                if r_candidate / r0 > 0.99:
                    it = 100
                if lam <= 0.2:
                    lam = 0
                else:
                    lam *= 0.5
                break
            else:
                if np.dot(increase,increase) < 1e-8:
                    it = 100
                    break
                if lam == 0:
                    lam = 0.2
                else:
                    lam *= pow(2,pow_of_lam)
    return x

def residual_k_all(x,points,point_weight):
    res = 0
    n = points.shape[0]
    for i in range(n):
        w  = np.zeros(4)
        w[0] = points[i][0][0]*points[i][1][0]
        w[1] = points[i][0][0]+points[i][1][0]
        w[2] = points[i][0][1]*points[i][1][1]
        w[3] = points[i][0][1]+points[i][1][1]
        res += residual_k(x,w) * point_weight[i]
    return res

def jacobian_k_all(x,points,point_weight):
    res = np.zeros((4,1))
    n = points.shape[0]
    for i in range(n):
        w  = np.zeros(4)
        w[0] = points[i][0][0]*points[i][1][0]
        w[1] = points[i][0][0]+points[i][1][0]
        w[2] = points[i][0][1]*points[i][1][1]
        w[3] = points[i][0][1]+points[i][1][1]
        res += jacbian_k(x,w) * point_weight[i]
    return res
def LM_k(x,points,point_weight):
    lam = 0
    r0 = residual_k_all(x,points,point_weight) 
    for it in range(100):
        J = jacobian_k_all(x, points,point_weight)
        A = np.matmul(J,J.transpose())
        b = -J * r0
        pow_of_lam = 0
        while True:
            tmp_A = A
            for j in range(4):
                tmp_A[j][j] *= (1+lam)
            A_inv = np.linalg.pinv(A)
            increase = np.matmul(np.linalg.pinv(A),b)
            increase = np.resize(increase,4)
            pow_of_lam += 1
            r_candidate = residual_k_all(x+increase,points,point_weight)
            if(r_candidate < r0):
                x = x+increase
                last_r = r0
                r0 = r_candidate
                if r_candidate / last_r > 0.99:
                    it = 100
                if lam <= 0.2:
                    lam = 0
                else:
                    lam *= 0.5
                break
            else:
                if np.dot(increase,increase) < 1e-6:
                    it = 100
                    break
                if lam == 0:
                    lam = 0.2
                else:
                    lam *= pow(2,pow_of_lam)
    return x

def LM_t(x_t,w):
    def jacbian_t(x,w):
            res = np.zeros((3,1))
            res[0] = 2*w[0]*x[0]+2*w[2]*x[0]+x[1]*w[1]+x[2]*w[3] 
            res[1] = w[1] * x[0] + 2 * x[1]
            res[2] = w[3] * x[2] + 2 * x[2]
            return res
    def residual_t(x,w):
        res = 0
        res += w[0] * x[0] * x[0]
        res += w[1] * x[0] * x[1]
        res += x[1] * x[1]
        res += w[2] * x[0] * x[0]
        res += w[3] * x[0] * x[2]
        res += x[2] * x[2] + 1
        res = res * res
        return res
    x_t = np.array([x_t[0],x_t[1],x_t[3]])
    lam = 0
    r0 = residual_t(x_t,w)
    for it in range(100):
        J = jacbian_t(x_t, w)
        A = np.matmul(J,J.transpose())
        b = -J * r0
        pow_of_lam = 0
        while True:
            tmp_A = A
            for j in range(3):
                tmp_A[j][j] *= (1+lam)
            A_inv = np.linalg.pinv(A)
            increase = np.matmul(np.linalg.pinv(A),b)
            increase = np.resize(increase,3)
            pow_of_lam += 1
            r_candidate = residual_t(x_t+increase,w)
            print(increase)
            if(r_candidate < r0):
                x_t = x_t+increase
                r0 = r_candidate
                if r_candidate / r0 > 0.99:
                    it = 100
                if lam <= 0.2:
                    lam = 0
                else:
                    lam *= 0.5
                    break
            else:
                if np.dot(increase,increase) < 1e-8:
                    it = 100
                    break
                if lam == 0:
                    lam = 0.2
                else:
                    lam *= 2
        x = np.array([x_t[0],x_t[1],x_t[0],x_t[2]])
        return x
    
def check_f(x,w):
    r0 = residual(x, w)
    for it in range(100):
        J = 2 * w[0] * x[0] + w[1] * x[1] + 2 * w[2] * x[2] + w[3] * x[3]
        A = J*J
        b = - J*r0
        increase = b / A
        tmp_x = x
        tmp_x[0] = x[0]+increase
        tmp_x[2] = tmp_x[0]
        print(x,tmp_x)
        r_candidate = residual(tmp_x,w)
        if r_candidate < r0:
            x = tmp_x
            r0 = r_candidate
        else:
            break
    return x

def check_c(x,w):
    r0 = residual(x,w)
    lam= 0
    for it in range(10):
        J = np.zeros((2,1))
        J[0] = w[1] * x[0] + 2 * x[1]
        J[1] = w[3] * x[3] + 2 * x[3]
        
        A = np.matmul(J,J.transpose())
        b = -J * r0
        pow_of_lam = 0
        while True:
            tmp_A = A
            for j in range(2):
                tmp_A[j][j] *= (1+lam)
            A_inv = np.linalg.pinv(A)
            increase = np.matmul(np.linalg.pinv(A),b)
            increase = np.resize(increase,2)
            pow_of_lam += 1
            tmp_x = x
            tmp_x[1] += increase[0]
            tmp_x[3] += increase[1]
            r_candidate = residual(tmp_x,w)
            if(r_candidate < r0):
                x = tmp_x
                r0 = r_candidate
                if r_candidate / r0 > 0.99:
                    it = 100
                if lam <= 0.2:
                    lam = 0
                else:
                    lam *= 0.5
                    break
            else:
                if np.dot(increase,increase) < 1e-8:
                    it = 100
                    break
                if lam == 0:
                    lam = 0.2
                else:
                    lam *= 2        

    return x

def dot_mul(line1,line2):
    res = 0
    n = line1.shape[0]
    for i in range(n):
        res += line1[i]*line2[i]
    return abs(res)
    
def check(path,x,init):
    w = np.zeros(4)
    res = 0
    c = 0
    num = init
    points = np.zeros((num,2,3))
    point_weight = np.zeros(num)
    for file in os.listdir(path):
        filename = os.path.join(path,file)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        lsd = cv2.createLineSegmentDetector(0)
        # 执行检测结果
        dlines = lsd.detect(img)
        # 绘制检测结果
        point,_,_,_,result_line = diamand_vanish.run_on_dataset(img.shape,dlines[0])
        pos1 = 0
        pos2 = 1
        minm = dot_mul(result_line[pos1], result_line[pos2])
        if dot_mul(result_line[1],result_line[2]) < minm:
            pos1 = 1
            pos2 = 2
            minm = dot_mul(result_line[1], result_line[2])
        if dot_mul(result_line[0],result_line[2]) < minm:
            pos1 = 0
            pos2 = 2
            minm = dot_mul(result_line[0],result_line[2])
        points[c,0,:] = point[pos1]
        points[c,1,:] = point[pos2]  
        if minm < 0.08:
            point_weight[c] = (0.08-minm) / 0.08
        c += 1
        if c >= num:
            break
    print(point_weight)
    x_res = LM_k(x,points,point_weight)
    print(x_res)
    # K_res = np.zeros((3,3))
    # K_res[0][0] = x_res[0]
    # K_res[0][2] = x_res[1]
    # K_res[1][1] = x_res[2]
    # K_res[1][2] = x_res[3]
    # K_res[2][2] = 1
    # K_res = np.linalg.inv(K_res) 
    # p1 = points[num-1,0,:]
    # p2 = points[num-1,1,:]
    # p1 = np.resize(p1,(3,1))
    # p2 = np.resize(p2,(3,1))
    # print(dot_mul(np.dot(K_inv, p1), np.dot(K_inv,p2)))
    # print(dot_mul(np.dot(K_res, p1), np.dot(K_res,p2 )))    
    
    return x_res
def test_diamond():
    plt.rcParams['figure.dpi'] = 1000
            
    w = np.zeros(4)
    x = np.array([K[0][0],K[0][2],K[1][1],K[1][2]]).transpose()
    path = 'D:\\download\\TUM\\rgbd_dataset_freiburg3_long_office_household\\rgbd_dataset_freiburg3_long_office_household\\rgb'
    #	520.9	521.0	325.1	249.7	
    groudtruth = [535.4	,320.1	,539.2,	247.6]
    max_num = 15
    u = [i for i in range(max_num+1)]
    v = [i+1 for i in range(max_num+1)]
    tmp_init = 0
    for j in range(4):
        tmp_init += (x[j] - groudtruth[j]) * (x[j] - groudtruth[j])  / groudtruth[j]
    tmp_init /= 4
    v[0] = tmp_init 
    for i in range(1,max_num+1):
        x_res = check(path,x,u[i])
        tmp_v = 0
        for j in range(4):
            tmp_v += (x_res[j] - groudtruth[j]) * (x_res[j] - groudtruth[j]) / groudtruth[j]
        tmp_v /= 4
        v[i] = tmp_v
    plt.plot(u,v,'k',linewidth = 1)
    plt.plot([0,max_num],[v[0],v[0]],'k--',linewidth=1)
    plt.xlabel('number of pictures')
    plt.ylabel('MSE')
    

m_file_img = []
m_K = []

def receive_img(img):

    m_file_img.append(img)

def receive_K(fx,fy,cx,cy):
    m_K.append(float(fx))
    m_K.append(float(fy))
    m_K.append(float(cx))
    m_K.append(float(cy))
    m_K = np.array(m_K).transpose()

def do_diamond_line():
    w = np.zeros(4)
    max_num = len(m_file_img)
    init = max_num
    res = 0
    c = 0
    num = init
    points = np.zeros((num,2,3))
    point_weight = np.zeros(num)
    for img in m_file_img:
        lsd = cv2.createLineSegmentDetector(0)
        # 执行检测结果
        dlines = lsd.detect(img)
        # 绘制检测结果
        point,_,_,_,result_line = diamand_vanish.run_on_dataset(img.shape,dlines[0])
        pos1 = 0
        pos2 = 1
        minm = dot_mul(result_line[pos1], result_line[pos2])
        if dot_mul(result_line[1],result_line[2]) < minm:
            pos1 = 1
            pos2 = 2
            minm = dot_mul(result_line[1], result_line[2])
        if dot_mul(result_line[0],result_line[2]) < minm:
            pos1 = 0
            pos2 = 2
            minm = dot_mul(result_line[0],result_line[2])
        points[c,0,:] = point[pos1]
        points[c,1,:] = point[pos2]  
        if minm < 0.08:
            point_weight[c] = (0.08-minm) / 0.08
        c += 1
        if c >= num:
            break
    print(point_weight)
    x_res = LM_k(m_K,points,point_weight)
    print(x_res)
    # K_res = np.zeros((3,3))
    # K_res[0][0] = x_res[0]
    # K_res[0][2] = x_res[1]
    # K_res[1][1] = x_res[2]
    # K_res[1][2] = x_res[3]
    # K_res[2][2] = 1
    # K_res = np.linalg.inv(K_res) 
    # p1 = points[num-1,0,:]
    # p2 = points[num-1,1,:]
    # p1 = np.resize(p1,(3,1))
    # p2 = np.resize(p2,(3,1))
    # print(dot_mul(np.dot(K_inv, p1), np.dot(K_inv,p2)))
    # print(dot_mul(np.dot(K_res, p1), np.dot(K_res,p2 )))    
    
    return x_res.tolist()



    
    