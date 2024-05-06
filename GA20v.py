import pandas as pd
import numpy as np
import os


def GA20v(input_data):

    GA20v_dict = {
        'A': [0.821,-1.009,-0.600,-4.94,0.97,0.077,0.19,0.4,1.23,2.32,1.31,1.14,0.18,1.49,0.32,2.09,-2.88,-0.56,-0.5021,-0.8426],  # A
        'C': [0.021,-1.419,0.502,-3.54,-0.23,-1.084,-1.048,2.23,0.44,3.49,3.78,1.98,1.43,1.15,0.886,-0.33,1.12,3.42,-0.5394,-2.3553],# C
        'D': [-0.444,-0.584,-1.762,0.94,0.94,1.551,0.01,0.24,5.15,1.17,1.5,1.51,1.98,0.9,0.223,-0.07,3.32,1.61,6.1944,1.663],  # D
        'E': [1.388,-0.361,-1.303,2.20,-1.31,1.477,-0.075,0,5.66,0.11,2.26,1.62,0.35,2.29,0.147,1.01,0.65,-0.98,-0.4608,1.8745],  # E
        'F': [0.293,1.336,-0.015,0.88,1.05,-1.619,0.29,1.62,0.15,0.41,0.73,0.56,2.14,0.68,0.263,-0.49,0.84,2.22,-0.7836,1.5894],  # F
        'G': [-2.219,-1.656,-1.146,-9.72,-0.11,0.849,1.186,7.47,0.41,1.62,2.9,0.98,0.53,0.3,0.208,1.09,-5,-2.97,-4.9694,0.1484],  # G
        'H': [0.461,-0.024,0.169,2.14,-2.79,0.716,-0.785,0.41,1.61,0.6,1.52,2.28,4.18,3.37,0.016,-0.74,2.68,-0.66,-1.7119,-4.046],  # H
        'I': [0.536,0.809,0.427,-1.73,-0.92,-1.462,0.276,1.18,0.21,3.45,1.98,0.89,1.78,1.54,0.013,0.37,-3.13,0.01,4.8464,-4.3605],  # I
        'K': [0.572,0.285,1.157,5.00,0.87,1.135,0.097,0.1,4.01,0.01,1.66,5.86,2.71,0.96,0.159,-0.09,1.54,-4.28,4.1011,5.4347],  # K
        'L': [1.128,0.703,-0.141,-1.33,-0.51,-1.406,0.344,0.09,0.27,4.06,1.2,0.67,0.8,0.63,0.025,1.35,-2.57,0,-3.1443,-1.6107],  # L
        'M': [1.346,0.277,-0.265,0.19,0.50,-0.963,0.365,2.07,0.84,1.85,0.78,1.53,1.39,4.32,0.763,0.33,-0.01,1.21,-0.088,1.5207],  # M
        'N': [-0.93,-0.428,-0.605,0.81,-1.94,1.511,0.834,2.08,0.4,2.47,7.02,1.32,3.13,1.54,0.482,-0.39,1.86,0.38,-0.2461,1.1064],  # N
        'P': [-2.038,-0.564,-1.008,-2.31,-0.30,0.883,-2.023,1.82,0.12,1.18,0.66,0.64,0.36,0.16,2.482,-1.09,-2.89,1.77,4.7338,-3.6024],# P
        'Q': [0.381,-0.044,0.405,2.88,0.64,1.094,-0.08,0.6,0.25,2.11,1.67,0.7,6.22,2.72,0.026,-0.09,1.16,-0.57,-4.3069,2.3877],  # Q
        'R': [0.378,0.516,2.728,6.60,0.32,1.014,0.195,0.37,3.81,0.98,0.99,4.9,1.32,2.62,0.059,-0.92,4.13,-4.41,0.7243,-1.3541],  # R
        'S': [-0.847,-1.079,-0.068,-2.31,-0.30,0.844,0.541,3.36,1.39,1.21,0.39,2.92,2.42,2.77,0.531,0.52,-0.19,1.06,-1.4355,1.4693],  # S
        'T': [-0.450,-0.610,0.577,-1.77,1.65,0.188,0.378,2.45,0.65,3.43,0.24,0.53,0.2,3.73,0.404,0.31,-0.66,0.13,2.3161,2.9086],  # T
        'V': [0.545,0.029,1.038,-2.90,-0.38,-1.127,0.158,2.01,0.33,3.93,1.27,0.43,1.06,1.97,0.19,0.9,-3.02,-0.22,-2.3731,0.168],  # V
        'W': [-0.075,2.069,-0.380,4.55,0.59,-1.577,0.239,0.86,1.07,1.66,0.66,2.49,0.72,2.73,0.216,-1.89,1.78,1.68,-1.4521,-0.5572],  # W
        'Y': [-0.858,1.753,0.289,3.59,0.30,-1.142,-0.475,0.53,1.3,1.31,0.95,1.91,0.76,2.56,0.566,-1.87,1.26,1.15,-0.9027,-1.5419],
        # Y
    }

    if not input_data.startswith('>'):
        input_data = ">Converted\n" + input_data  # 将输入转化为FASTA格式

    lines = input_data.split('\n')
    sequence = ''.join(lines[1:])  # 跳过FASTA头部

    encodings = []
    for aa in sequence:
        encodings.append(GA20v_dict.get(aa, [0] * 15))

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



    # encodings = []
    # for i in fastas:
    #     for aa in i:
    #         a = GA20v[aa]
    #         encodings.append(a)
    #
    #     c = []
    #     d = []
    #     j = 1
    #     for e in encodings:
    #         for f in e:
    #             c.append(f)
    #             if j % (6 * 20) == 0:
    #                 d.append(c)
    #                 c = []
    #             j = j + 1
    # return d


# def data_read(dir):  # 读取文件
#     data = open(dir)
#     s = data.read()
#     l = s.split()
#     return l
#
#
# def transdata(filename):
#     pos_hex = data_read(filename)
#     GA20v_pos_hex = GA20v(pos_hex)
#     GA20v_pos_hex = np.array(GA20v_pos_hex)
#
#     print(GA20v_pos_hex.shape)
#
#     colnum = []
#     sequence = pos_hex
#     for i in range(len(GA20v_pos_hex)):
#         colnum.append(i)
#
#     pos_GA20v = pd.DataFrame(GA20v_pos_hex, index=sequence)
#     pos_GA20v.index.name = 'sequence'
#
#     tmp = []
#     tmp = filename.split('\\')
#     tmp = tmp[-1].split('.')
#     basepath = os.path.dirname(__file__)
#     generateddirpath = os.path.join(basepath[:len(basepath)], 'generated')
#     if not os.path.exists(generateddirpath):
#         os.makedirs(generateddirpath)
#     xlsxfilename = os.path.join(generateddirpath, tmp[0] + '_GA20.xlsx')
#     with pd.ExcelWriter(xlsxfilename) as writer:
#         pos_GA20v.to_excel(writer, sheet_name='sheet_1', header=True, index=True)
#     print(filename + '转换完成\n')
#     print('输出文件：' + xlsxfilename + '\n')
#     return xlsxfilename