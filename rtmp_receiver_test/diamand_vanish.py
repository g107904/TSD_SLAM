# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 01:12:27 2022

@author: g107904
"""

import math
import numpy as np


class line_param:
    def __init__(self,a,b,c,w):
        self.a = a
        self.b = b
        self.c = c
        self.w = w

class coords:
    def __init__(self,x,y,d):
        self.x = x
        self.y = y
        self.d = d
    

def sgn(val):
    if val >= 0:
        return 1
    else:
        return -1

def lines_end_points(line,endpoints,space_c,numlines):
    center = round(space_c)
    
    for i in range(numlines):
        a = line[i][0]
        b = line[i][1]
        c = line[i][2]
        
        alpha = float(sgn(a*b))
        beta = float(sgn(b*c))
        gamma = float(sgn(a*c))
        
        a_x = alpha*a / (c + gamma*a)
        b_x = -alpha*c / (c + gamma*a)
        
        endpoints[i*8+1] = round((a_x + 1) * space_c)
        endpoints[i*8+0] = round((b_x + 1) * space_c)
        
        endpoints[i*8+3] = round((b / (c + beta*b) + 1) * space_c)
        endpoints[i*8+2] = center
        
        endpoints[i*8+5] = center
        endpoints[i*8+4] = round((b / (a + alpha*b) + 1) * space_c)
        
        endpoints[i*8+7] = round((-a_x + 1) * space_c)
        endpoints[i*8+6] = round((-b_x + 1) * space_c)
    return endpoints

        
        

def lineH(x0,y0,x1,y1,space,y_steps,weight):
    slope = float(y1-y0)/(x1-x0+1e-8)
    
    y_start = y0+0.5
    y_iter = y_start
    
    step = 1
    if(x0 >= x1):
        step = -1
    slope *= step
    
    x = x0
    c = 1
    while x != x1:
        space[y_steps[int(y_iter)]+int(x)] += weight
        y_iter = y_start + c * slope
        c += 1
        x += step

def lineV(x0,y0,x1,y1,space,y_steps,weight):
    slope = (x1-x0) / float(y1-y0+1e-8)
    
    x_start = x0+0.5
    x_iter = x_start
    
    step = 1
    if y0 >= y1:
        step = -1
    
    slope *= step
    
    y = y0
    c = 1
    while y != y1:
        space[y_steps[y]+int(x_iter)] += weight
        x_iter = x_start+c*slope
        c += 1
        y += step
        
    

def rasterize_lines(line,endpoints,space,spacesize,numlines):
    v_steps = [i*spacesize for i in range(spacesize)]
    
    for i in range(numlines):
        weight = int(line[i][3])
        
        for j in range(0,6,2):
            if abs(endpoints[i*8+j+3] - endpoints[i*8+j+1]) > abs(endpoints[i*8+j+2] - endpoints[i*8+j]):
                lineV(endpoints[i*8+j],endpoints[i*8+j+1],endpoints[i*8+j+2],endpoints[i*8+j+3],space,v_steps,weight)
            else:
                lineH(endpoints[i*8+j],endpoints[i*8+j+1],endpoints[i*8+j+2],endpoints[i*8+j+3],space,v_steps,weight)
        space[v_steps[endpoints[i*8+7]]+endpoints[i*8+6]] += weight

def mx_raster_space(space_size,lines_data,numlines):
    space = [0 for i in range(space_size*space_size)]
    space_c = (space_size-1.0)/2
    
    endpoints = np.array([0 for i in range(8*numlines)],dtype = np.int32)
    
    endpoints = lines_end_points(lines_data, endpoints, space_c, numlines)
    
    rasterize_lines(lines_data, endpoints, space, space_size, numlines)
    return space,endpoints

def find_maxm(space,subpixel_radius,space_size):
    np_space = np.array(space)
    
    pos = np.argmax(np_space)
    r = pos // space_size
    c = pos % space_size
    cc = 0
    rr = 0
    np_space = np.resize(np_space,(space_size,space_size))
    O = np_space[r:r+subpixel_radius*2+1,c:c+subpixel_radius*2+1]
    grid = np.array([i for i in range(-subpixel_radius,subpixel_radius+1)])
    c += np.sum(O*grid.transpose()) / np.sum(O)
    r += np.sum(grid*O) / np.sum(O)
    
    vanPC = np.array([r,c])
    return vanPC

    

def norm_PC_points(vanP,spacesize):
    normP = (2 * vanP - (spacesize+1)) / (spacesize-1)
    return normP

def point_to_lines_list(Point,Lines):
    x = Point[0]
    y = Point[1]
    T = [0,-1,1,1,-1,0,0,-1,0,
         0,-1,1,1,1,0 ,0,-1,0,
         0,1,1 ,1,-1,0,0,-1,0,
         0,1,1,1,1,0 ,0,-1,0]
    T = np.array(T)
    T= np.resize(T,(12,3))
    T = T.transpose()
    L = np.dot(Lines[:,:3],T)
    
    P = np.zeros((Lines.shape[0],4))
    point = np.array([x,y,1]).transpose()
    
    for j in range(4):
        tmp_mid = L[:,j*3:j*3+3]
        tmp = np.dot(tmp_mid,point)
        tmp_sum = L[:,j*3]+L[:,j*3+1]
        tmp1 = np.array(np.sqrt(tmp_sum*tmp_sum))
        mid_res = tmp/tmp1
        P[:,j] = mid_res
    
    D = np.zeros(Lines.shape[0])
    D = np.min(abs(P),1)
    return D
    
    # res = np.zeros(Lines.shape[0])
    # tmp = abs(np.dot(Lines[:,0:3],point))
    # tmp1 = np.array(np.sqrt(Lines[:,0]*Lines[:,0]+Lines[:,1]*Lines[:,1]))
    # res = tmp/tmp1
    # return res
    

def PC_point_to_CC(normalization,point,imgsize):
    u = point[0]
    v = point[1]
    res = np.array([v,sgn(v)*v+sgn(u)*u-1,u])
    
    if abs(u) > 0.005:
        res = res / u
    else:
        tmp_sum = np.linalg.norm(res)
        res = res / tmp_sum
        res[2] = 0
    
    w = imgsize[0]
    h = imgsize[1]
    m = max(w,h)
    res[0] = (res[0] * (m-1)+w+1)/2
    res[1] = (res[1]*(m-1)+h+1)/2
    
    return res

    
    
    
    
    
def diamond_vanish(imgsize,normalization,spacesize,num_vanish,lines_data):
    subpixel_radius = 2
    threshold = 0.05
    
    numlines = lines_data.shape[0]
    Space = np.zeros(spacesize*spacesize)
    vanishPC = np.zeros((num_vanish,2))
    vanishPC_norm = np.zeros((num_vanish,2))
    Distance = np.zeros((3,numlines))
    endpoint = []
    point_CC = np.zeros((num_vanish,3))
    line_weight = np.zeros((numlines,3))
    for V in range(num_vanish):
        space,endpoint = mx_raster_space(spacesize, lines_data, numlines)
        vanishPC[V,:] = find_maxm(space,subpixel_radius,spacesize)
        if V == 1:
            Space = np.array(space)
        vanishPC_norm[V,:] = norm_PC_points(vanishPC[V,:], spacesize)
        
        point = PC_point_to_CC(normalization,vanishPC_norm[V,:],imgsize)
        
        point_CC[V,:] = point
        
        Distance[V,:] = point_to_lines_list(vanishPC_norm[V,:], np.array(lines_data))
        
        line_weight[:,V] = lines_data[:,3] * (Distance[V,:] <= 0.05)
        lines_data[:,3] *= (Distance[V,:] > 0.05)
    
    return point_CC,Distance,endpoint,line_weight


def run_on_dataset(imgsize,lines):
    tmp_lines = np.array(lines)
    out_lines = np.zeros((tmp_lines.shape[0],4))
    tmp_lines = np.resize(tmp_lines,(tmp_lines.shape[0],4))
    
    re_lines = tmp_lines
    w = imgsize[0]
    h = imgsize[1]
    tmp_lines[:,0] = tmp_lines[:,0]/(w/2)-1
    tmp_lines[:,1] = tmp_lines[:,1]/(h/2)-1
    tmp_lines[:,2] = tmp_lines[:,2]/(w/2)-1
    tmp_lines[:,3] = tmp_lines[:,3]/(h/2)-1
    out_lines[:,3] = 0
    out_lines[:,0] = tmp_lines[:,3] - tmp_lines[:,1]
    out_lines[:,1] = tmp_lines[:,0] - tmp_lines[:,2]
    out_lines[:,2] = tmp_lines[:,1] * tmp_lines[:,2] - tmp_lines[:,0] * tmp_lines[:,3]
    
    line_norm = np.linalg.norm(out_lines,axis=1)
    out_lines[:,2] = out_lines[:,2] / line_norm
    out_lines[:,1] = out_lines[:,1] / line_norm
    out_lines[:,0] = out_lines[:,0] / line_norm
    
    out_lines[:,3] = 1
    
    vanishPC,color,endpoint,line_weight = diamond_vanish(imgsize,1,1024,3,out_lines)
    
    tmp_vector = np.zeros((re_lines.shape[0],2))
    tmp_vector[:,0] = re_lines[:,2] - re_lines[:,0]
    tmp_vector[:,1] = re_lines[:,3] - re_lines[:,1]
    
    tmp_length = np.linalg.norm(tmp_vector,axis=1)
    tmp_vector[:,0] /= tmp_length
    tmp_vector[:,1] /= tmp_length
    
    
    
    result_line = np.zeros((3,2))
    cur_vector = np.zeros(tmp_vector.shape)
    for i in range(3):
        cur_length = tmp_length * line_weight[:,i]
        cur_vector[:,0] = tmp_vector[:,0] * line_weight[:,i]
        cur_vector[:,1] = tmp_vector[:,1] * line_weight[:,i]
        cur_sum = np.sum(cur_length)
        weight = (cur_length / cur_sum) * line_weight[:,i]
        pos = np.nonzero(cur_vector)[0][10]
        vector0 = cur_vector[pos,:]
        weight = weight * np.sign(cur_vector[:,0]*vector0[0]+cur_vector[:,1] * vector0[1])
        cur_vector[:,0] *= weight
        cur_vector[:,1] *= weight
        result_line[i,:] = np.sum(cur_vector,axis = 0)
        
    tmp_line_length = np.linalg.norm(result_line,axis=1)
    result_line[:,0] /= tmp_line_length
    result_line[:,1] /= tmp_line_length
    
    
    return vanishPC,color,endpoint,line_weight,result_line
    
    
    
    
            
        
        
        
        
    
                
            
       
        
