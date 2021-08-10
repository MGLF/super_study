# 该部分测试代码模拟superpoint解码器输出后到整张图像上映射，采用的Hc=H/4=16，D=16， 即从[H/8,H/8,16]->[H,W,1]
# cell的尺寸为4，其4*4代表了其在[H/8,H/8,16]时的深度，也代表了其在[H,W,1]的宽度
# 该部分代码功能是扁平化，将在[H/8,H/8,16]的一个像素（深度为16）转化为[H,W,1]的一块像素4*4，因此说一个cell里面有4*4个像素（论文是8*8）
from torch.autograd import Variable
import torch
import numpy as np
import cv2
cell=4   # cell尺寸
Hc=16    # 关键点解码器的输出尺寸
Wc=16
D=16

print(Hc)
# 生成模拟关键点解码器输出 尺寸为32*32*64
a=np.arange(0,Hc*Wc)  # 首先获取模拟输出列表
nodust = a
for i in range(D-1):
    nodust=np.append(nodust,a)
    print(i)

nodust = nodust.reshape((16,16,16))
print(np.shape(nodust))

nodust = nodust.transpose(1, 2, 0)  
heatmap = np.reshape(nodust, [Hc, Wc, cell, cell])
heatmap = np.transpose(heatmap, [0, 2, 1, 3])
heatmap = np.reshape(heatmap, [Hc*cell, Wc*cell])
print(np.shape(heatmap))
print(heatmap)
cv2.imwrite("result.png",heatmap) # 写入图像
