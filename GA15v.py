# #!/usr/bin/env python
# # coding: utf-8
#
# import pandas as pd
# import numpy as np
# import os
#
# def GA15v(sequence):
#
#     GA15v_dict = {
#         'A': [-1.009,1.62,-4.94,-2.13,1.70,0.4,1.31,0.18,1.49,1.352,0.53,0.11,-2.51,1.285,-0.8426], # A
#         'C': [-1.419,-0.271,-3.54,1.50,-1.26,2.23,3.78,1.43,1.15,1.217,0.797,0.38,-2.98,-7.783,-2.3553], # C
#         'D': [-0.584,1.205,0.94,-3.04,-4.58,0.24,1.5,1.98,0.9,0.213,0.268,-1.38,0.14,-4.884,1.663], # D
#         'E': [-0.361,1.248,2.20,-3.59,-2.26,0,2.26,0.35,2.29,0.018,0.025,-0.74,-0.68,-3.097,1.8745], # E
#         'F': [1.336,-0.779,0.88,0.89,-1.12,1.62,0.73,2.14,0.68,0.887,0.093,1.36,1.13,0.317,1.5894], # F
#         'G': [-1.656,0.327,-9.72,4.18,-1.35,7.47,2.9,0.53,0.3,1.1,0.371,-1.3,-3.99,3.547,0.1484], # G
#         'H': [-0.024,0.831,2.14,1.20,0.71,0.41,1.52,4.18,3.37,0.234,0.137,0,-0.11,-1.84,-4.046], # H
#         'I': [0.809,-1.556,-1.73,-2.49,1.09,1.18,1.98,1.78,1.54,0.097,0.789,1.54,0.18,0.988,-4.3605], # I
#         'K': [0.285,0.689,5.00,0.70,3.00,0.1,1.66,2.71,0.96,0.72,0.002,-0.67,2.05,3.603,5.4347], # K
#         'L': [0.703,-1.062,-1.33,-1.71,0.63,0.09,1.2,0.8,0.63,0.711,0.928,1.33,0.05,1.276,-1.6107], # L
#         'M': [0.277,0.105,0.19,-1.02,0.15,2.07,0.78,1.39,4.32,0.579,0.423,1.27,-0.87,-2.617,1.5207], # M
#         'N': [-0.428,1.116,0.81,0.14,-0.14,2.08,7.02,3.13,1.54,0.014,0.565,-1.2,-1.11,1.081,1.1064], # N
#         'P': [-0.564,-1.041,-2.31,3.45,1.00,1.82,0.66,0.36,0.16,0.006,0.269,-1.28,1.21,5.468,-3.6024], # P
#         'Q': [-0.044,1.196,2.88,-0.83,0.52,0.6,1.67,6.22,2.72,0.304,0.483,-0.45,-0.84,0.728,2.3877], # Q
#         'R': [0.516,1.62,6.60,1.21,2.07,0.37,0.99,1.32,2.62,1.05,0.224,-0.47,6.71,-3.279,-1.3541], # R
#         'S': [-1.079,0.088,-2.31,3.45,1.00,3.36,0.39,2.42,2.77,0.809,0.276,-0.97,-2.42,1.73,1.4693], # S
#         'T': [-0.610,-0.363,-1.77,-0.70,1.02,2.45,0.24,0.2,3.73,0.514,0.783,-0.36,-2.12,1.71,2.9086], # T
#         'V': [0.029,-1.614,-2.90,-2.29,1.38,2.01,1.27,1.06,1.97,0.246,0.898,1.23,-1.41,-0.042,0.168], # V
#         'W': [2.069,-0.997,4.55,2.77,-2.41,0.86,0.66,0.72,2.73,1.556,0.379,1.19,4.27,0.631,-0.5572], # W
#         'Y': [1.753,-0.171,3.59,2.45,-1.27,0.53,0.95,0.76,2.56,1.52,0.299,0.43,3.32,1.179,-1.5419], # Y
#     }
#
#     # 如果输入是单个字符串，将其视为包含单个序列的列表
#     if isinstance(sequence, str):
#         sequence = [sequence]
#
#     encodings = []
#     for seq in sequence:
#         seq_encodings = []
#         for aa in seq:
#             if aa in GA15v_dict:
#                 seq_encodings.extend(GA15v_dict[aa])
#             else:
#                 seq_encodings.extend([0] * 15)  # 未知氨基酸使用0向量填充
#         encodings.append(seq_encodings)
#     return encodings
#             # a =GA15v[aa]
#             # encodings.append(a)
#
#         c = []
#         d = []
#         j = 1
#         for e in encodings:
#             for f in e:
#                 c.append(f)
#                 if j % (6*15) == 0:
#                     d.append(c)
#                     c = []
#                 j = j + 1
#     return d
#
# def data_read(dir):  #读取文件
#     data = open(dir)
#     s = data.read()
#     l = s.split()
#     return l
#
# def transdata(filename):
#     pos_hex = data_read(filename)
#     GA15v_pos_hex = GA15v(pos_hex)
#     GA15v_pos_hex = np.array(GA15v_pos_hex)
#
#     print(GA15v_pos_hex.shape)
#
#     colnum=[]
#     sequence = pos_hex
#     for i in range(len(GA15v_pos_hex)):
#         colnum.append(i)
#
#     pos_GA15v = pd.DataFrame(GA15v_pos_hex,index=sequence)
#     pos_GA15v.index.name = 'sequence'
#
#     tmp = []
#     tmp = filename.split('\\')
#     tmp = tmp[-1].split('.')
#     basepath = os.path.dirname(__file__)
#     generateddirpath = os.path.join(basepath[:len(basepath)], 'generated')
#     if not os.path.exists(generateddirpath):
#         os.makedirs(generateddirpath)
#     xlsxfilename = os.path.join(generateddirpath, tmp[0] + '_GA15.xlsx')
#     with pd.ExcelWriter(xlsxfilename) as writer:
#         pos_GA15v.to_excel(writer, sheet_name='sheet_1', header=True, index=True)
#     print(filename + '转换完成\n')
#     print('输出文件：'+ xlsxfilename + '\n')
#     return xlsxfilename
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os

def GA15v(input_data):
    GA15v_dict = {
        'A': [-1.009,1.62,-4.94,-2.13,1.70,0.4,1.31,0.18,1.49,1.352,0.53,0.11,-2.51,1.285,-0.8426], # A
        'C': [-1.419,-0.271,-3.54,1.50,-1.26,2.23,3.78,1.43,1.15,1.217,0.797,0.38,-2.98,-7.783,-2.3553], # C
        'D': [-0.584,1.205,0.94,-3.04,-4.58,0.24,1.5,1.98,0.9,0.213,0.268,-1.38,0.14,-4.884,1.663], # D
        'E': [-0.361,1.248,2.20,-3.59,-2.26,0,2.26,0.35,2.29,0.018,0.025,-0.74,-0.68,-3.097,1.8745], # E
        'F': [1.336,-0.779,0.88,0.89,-1.12,1.62,0.73,2.14,0.68,0.887,0.093,1.36,1.13,0.317,1.5894], # F
        'G': [-1.656,0.327,-9.72,4.18,-1.35,7.47,2.9,0.53,0.3,1.1,0.371,-1.3,-3.99,3.547,0.1484], # G
        'H': [-0.024,0.831,2.14,1.20,0.71,0.41,1.52,4.18,3.37,0.234,0.137,0,-0.11,-1.84,-4.046], # H
        'I': [0.809,-1.556,-1.73,-2.49,1.09,1.18,1.98,1.78,1.54,0.097,0.789,1.54,0.18,0.988,-4.3605], # I
        'K': [0.285,0.689,5.00,0.70,3.00,0.1,1.66,2.71,0.96,0.72,0.002,-0.67,2.05,3.603,5.4347], # K
        'L': [0.703,-1.062,-1.33,-1.71,0.63,0.09,1.2,0.8,0.63,0.711,0.928,1.33,0.05,1.276,-1.6107], # L
        'M': [0.277,0.105,0.19,-1.02,0.15,2.07,0.78,1.39,4.32,0.579,0.423,1.27,-0.87,-2.617,1.5207], # M
        'N': [-0.428,1.116,0.81,0.14,-0.14,2.08,7.02,3.13,1.54,0.014,0.565,-1.2,-1.11,1.081,1.1064], # N
        'P': [-0.564,-1.041,-2.31,3.45,1.00,1.82,0.66,0.36,0.16,0.006,0.269,-1.28,1.21,5.468,-3.6024], # P
        'Q': [-0.044,1.196,2.88,-0.83,0.52,0.6,1.67,6.22,2.72,0.304,0.483,-0.45,-0.84,0.728,2.3877], # Q
        'R': [0.516,1.62,6.60,1.21,2.07,0.37,0.99,1.32,2.62,1.05,0.224,-0.47,6.71,-3.279,-1.3541], # R
        'S': [-1.079,0.088,-2.31,3.45,1.00,3.36,0.39,2.42,2.77,0.809,0.276,-0.97,-2.42,1.73,1.4693], # S
        'T': [-0.610,-0.363,-1.77,-0.70,1.02,2.45,0.24,0.2,3.73,0.514,0.783,-0.36,-2.12,1.71,2.9086], # T
        'V': [0.029,-1.614,-2.90,-2.29,1.38,2.01,1.27,1.06,1.97,0.246,0.898,1.23,-1.41,-0.042,0.168], # V
        'W': [2.069,-0.997,4.55,2.77,-2.41,0.86,0.66,0.72,2.73,1.556,0.379,1.19,4.27,0.631,-0.5572], # W
        'Y': [1.753,-0.171,3.59,2.45,-1.27,0.53,0.95,0.76,2.56,1.52,0.299,0.43,3.32,1.179,-1.5419], # Y

    }

    if not input_data.startswith('>'):
        input_data = ">Converted\n" + input_data  # 将输入转化为FASTA格式

    lines = input_data.split('\n')
    sequence = ''.join(lines[1:])  # 跳过FASTA头部

    encodings = []
    for aa in sequence:
        encodings.append(GA15v_dict.get(aa, [0] * 15))

    c = []
    d = []
    j = 1
    for e in encodings:
        for f in e:
            c.append(f)
            if j % (6*15) == 0:
                d.append(c)
                c = []
            j += 1
    if c:
        d.append(c)

    return d

# def transdata(input_data):
#     GA15v_pos_hex = GA15v(input_data)
#     GA15v_pos_hex = np.array([item for sublist in GA15v_pos_hex for item in sublist])
#
#     print(GA15v_pos_hex.shape)
#
#     sequence = ['Converted_Sequence'] * len(GA15v_pos_hex)  # 为每个特征行创建一个虚拟的序列名称
#     pos_GA15v = pd.DataFrame(GA15v_pos_hex, index=sequence)
#     pos_GA15v.index.name = 'sequence'
#
#     basepath = os.path.dirname(__file__)
#     generateddirpath = os.path.join(basepath, 'generated')
#     if not os.path.exists(generateddirpath):
#         os.makedirs(generateddirpath)
#     xlsxfilename = os.path.join(generateddirpath, 'output_GA15.xlsx')
#     with pd.ExcelWriter(xlsxfilename) as writer:
#         pos_GA15v.to_excel(writer, sheet_name='Sheet1', header=True)
# 
#     print("转换完成\n")
#     print("输出文件：" + xlsxfilename + "\n")
#     return xlsxfilename
