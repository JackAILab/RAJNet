import numpy as np
import torch
import h5py
from curses.ascii import isdigit
from locale import atoi
import cv2
import random
from scipy import ndimage
import pandas as pd  
import math

def scipy_rotate(volume):
    # define some rotation angles
    angles = [-20, -10, -5, 5, 10, 20]
    # pick angles at random
    angle = random.choice(angles)
    # rotate volume
    volume = ndimage.rotate(volume, angle, axes=(1, 2), reshape=False)
    volume[volume < 0] = 0 
    volume[volume > 1] = 1
    return volume



def py_cpu_nms(dets, thresh):

    y1 = dets[:, 1]
    x1 = dets[:, 0]
    y2 = y1 + dets[:, 3]
    x2 = x1 + dets[:, 2]
    
    scores = dets[:, 4]  # bbox打分
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    
    # keep为最后保留的边框
    keep = []
    
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds + 1]
        
    return keep

def calculate(bound, mask):  # 不需要去除重叠的框框

    x, y, w, h = bound

    area = mask[y:y+h, x:x+w]

    pos = area > 0 + 0

    score = np.sum(pos)/(w*h)

    return score

def nms_cnts(cnts, mask, min_area):

    nms_threshold=0.3  

    bounds = [cv2.boundingRect(
        c) for c in cnts if cv2.contourArea(c) > min_area]  # 筛掉面积过小的轮廓框 返回

    if len(bounds) == 0:
        return []

    scores = [calculate(b, mask) for b in bounds] 

    bounds = np.array(bounds)

    scores = np.expand_dims(np.array(scores), axis=-1)

    keep = py_cpu_nms(np.hstack([bounds, scores]), nms_threshold)

    return bounds[keep]

def pop(l, value):

    l.pop(0)
    l.append(value)

    return l

def frame_judge(IMG,data_path,k):

    NEW_IMG=[]
    # NEW_IMG=np.array(NEW_IMG)
    previous=[]
    IMG=ndimage.rotate(IMG,-90)

    for i in range(np.shape(IMG)[2]):
        NEW_IMG.append(IMG[:,:,i])
        
        threshold=50
        k_size=7
        iterations=3
        min_area=800
        es = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        previous.append(IMG[:,:,i])  # previous现存灰度图
        raw = IMG[:,:,i].copy()
        temp = cv2.absdiff(IMG[:,:,i], previous[0]) # 计算绝对值差   gray:array   Qimage[0]?
        temp = temp.astype(np.uint8)

        temp = cv2.medianBlur(temp, k_size)  # 中值滤波 bug:改uint

        ret, mask = cv2.threshold(
            temp, threshold, 255, cv2.THRESH_BINARY)  # 二值化

        mask = cv2.dilate(mask, es, iterations)  # 膨胀
        mask = cv2.erode(mask, es, iterations)  # 腐蚀

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

        #cnts为list结构，列表中每个元素代表一个边沿信息           

        bounds = nms_cnts(cnts, mask, min_area)

        if len(bounds)>=k:  # if len(bounds)>=4 矩形框个数大于等于4-130多  # 6:26  # 5:34  最开始用4；2.9试6
            print("New IMG Add!")
            from PIL import Image

            im = Image.fromarray(IMG[:,:,i])
            if im.mode == "F":
                im = im.convert('RGB')
            NEW_IMG.append(IMG[:,:,i])

               
        previous = pop(previous, raw)
        
    if NEW_IMG==[]:
        return "sorry"
    else:
        NEW_IMG=np.array(NEW_IMG)
        NEW_IMG=NEW_IMG.transpose(2,1,0)
        return NEW_IMG


def get_name_code(string):
    code=''
    if 'HC' in string:
        code+='1'
    elif 'PD' in string:
        code+='2'
    for i in range(0,len(string)):
        if isdigit(string[i]):
            code+=string[i]
    return code


section = 20 # 切片大小
h5f_input_section_size = 20 # 表示一次往h5f文件里面放20张堆叠的图

def prepareHDData(data_path,labels,data_type,k_threshold): # 传入 256 可以扩大 4 倍的图像数据
    
    # 240420
    train_save_data_path = './Data/h5f_data/train_small_data.he5'   
    train_save_label_path = './Data/h5f_data/train_small_data_label.he5'
    train_save_patient_name_path = './Data/h5f_data/train_small_name_label.he5'

    # 测试保存路径
    test_save_data_path = './Data/h5f_data/test_small_data.he5'
    test_save_label_path = './Data/h5f_data/test_small_data_label.he5'
    test_save_patient_name_path = './Data/h5f_data/test_small_name_label.he5'


    h5f_label = 0 # 控制h5f文件的写入
    if data_type=='train':
        data_h5f = h5py.File(train_save_data_path, 'w') # 存数据
        label_h5f = h5py.File(train_save_label_path,'w') # 存标签
        name_h5f = h5py.File(train_save_patient_name_path,'w') # 存名字

    elif data_type=='test':
        data_h5f = h5py.File(test_save_data_path, 'w') # 存数据
        label_h5f = h5py.File(test_save_label_path,'w') # 存标签
        name_h5f = h5py.File(test_save_patient_name_path,'w') # 存名字

    for i in range(len(data_path)): # len(data_path)):  # 这里是循环操作563张图
        image = np.load(data_path[i])
        cur_label = labels[i]
        name = data_path[i][43:-4]  # 患者名字

        #---------enhance---------
        image = np.transpose(image, (2, 1, 0)) # (512, 512, 20)  调换数组的行列值的索引值，类似于求矩阵的转置
        image_new = frame_judge(image,data_path[i],k_threshold) # image_new是增强之后的，由原来的90张，扩展为现在的130张
        if image_new=="sorry":
            continue
        image_new = np.transpose(image_new,(2,1,0))
        image = image_new
        #---------enhance---------

        image = image.astype(np.float32)

        times = math.ceil(image.shape[0] / section)  
        for j in range(0,times):
            image_sliced_ori = image[section*j:(section)*(j+1),:,:]
            # 假设image_sliced是原始数据
            target_shape = (20, 512, 512)  # 目标形状
            padded_image = np.zeros(target_shape, dtype=np.float32)
            # 将原始数据复制到新数组的前面部分
            padded_image[:image_sliced_ori.shape[0], :, :] = image_sliced_ori
            image_sliced = padded_image
            # 写死 -- 可以用循环优化一下  切
            img1 = image_sliced[:,0:256,0:256]
            img2 = image_sliced[:,256:512,0:256]
            img3 = image_sliced[:,0:256,256:512]
            img4 = image_sliced[:,256:512,256:512]
            # [20,512,512] --> [80,256,256]
            img_stacked = np.vstack((img1,img2,img3,img4))
            img_stacked = np.expand_dims(img_stacked, axis=0) # Jack处理图像时即扩展了Z轴
            # [1,80,256,256]
            cur_channal_size = int(img_stacked.shape[1]/h5f_input_section_size)
            for k in range(0,cur_channal_size):
                img_input = img_stacked[:,k*h5f_input_section_size:(k+1)*h5f_input_section_size,:,:]
                # cur_label 
                data_h5f.create_dataset(str(h5f_label),data = torch.from_numpy(img_input))
                label_h5f.create_dataset(str(h5f_label),data = torch.tensor(cur_label,dtype = torch.long))

                h5f_label += 1  
        print(f'write successful : {i}')
    print('successful prepareHDData')


if __name__ == '__main__':

    # load data 
    df_train = pd.read_csv('./Data/Data_train.csv') 
    df_test = pd.read_csv('./Data/Data_test.csv') 

    train_data_path = list(df_train['path'])   # 'data_path'-->path
    train_labels = list(df_train['label'])     # label_list-->label

    test_data_path = list(df_test['path'])
    test_labels = list(df_test['label'])

    prepareHDData(data_path = train_data_path,labels = train_labels,data_type='train',k_threshold=1)
    prepareHDData(data_path=test_data_path,labels=test_labels,data_type='test',k_threshold=1)
    print("H5fMake Successful!")