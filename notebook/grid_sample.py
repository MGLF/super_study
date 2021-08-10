import torch
from torch.nn import functional as F


inp = torch.ones(1, 1, 4, 4)

# 目的是得到一个 长宽为20的tensor
out_h = 20
out_w = 20
 # grid的生成方式等价于用mesh_grid
new_h = torch.linspace(-1, 1, out_h).view(-1, 1).repeat(1, out_w)
new_w = torch.linspace(-1, 1, out_w).repeat(out_h, 1)
grid = torch.cat((new_h.unsqueeze(2), new_w.unsqueeze(2)), dim=2)
grid = grid.unsqueeze(0)

outp = F.grid_sample(inp, grid=grid, mode='bilinear')
print(outp.shape)  #torch.Size([1, 1, 20, 20])

