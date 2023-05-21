import pandas as pd
import numpy as np
import os


def PCA20v(fastas):

    PCA20v = {
        'A': [0.207,0.063,-0.600,1.568,0.219,-4.940,0.970,0.077,0.240,1.230,1.310,0.349,2.090,-1.940,-3.738,-1.933,1.285,-0.887,-0.502,2.888],  # A
        'C': [-1.229,-0.052,2.728,1.568,0.219,6.600,0.320,1.014,1.277,3.810,0.990,1.085,-0.920,4.410,9.935,-5.359,-3.279,0.434,0.724,0.651 ],# C
        'D': [-1.009,-0.539,-0.605,-1.267,-0.716,0.810,-1.940,1.511,0.120,0.400,7.020,0.113,-0.390,1.730,-0.717,0.269,1.081,1.777,-0.246,0.616 ],  # D
        'E': [-1.298,-0.259,-1.762,-1.579,-0.857,0.940,0.940,1.551,-0.991,5.150,1.500,1.466,-0.070,1.240,-2.635,6.278,-4.884,2.019,6.194,-1.071],  # E
        'F': [0.997,-0.799,0.502,-0.768,-2.428,-3.540,-0.230,-1.084,1.828,0.440,3.780,0.125,-0.330,-1.700,-4.088,-3.663,-7.783,0.458,-0.539,-2.142],  # F
        'G': [-0.880,-0.040,0.405,-0.388,0.038,2.880,0.640,1.094,0.500,0.250,1.670,0.071,-0.090,1.930,-0.483,0.127,0.728,-1.533,-4.307,-2.186],  # G
        'H': [-1.349,0.424,-1.303,-0.650,-0.328,2.200,-1.310,1.477,-0.284,5.660,2.260,1.613,1.010,1.410,-3.013,5.833,-3.097,-1.603,-0.461,-3.965],  # H
        'I': [-0.205,-1.115,-1.146,2.448,-0.407,-9.720,-0.110,0.849,-0.548,0.410,2.900,0.117,1.090,-2.160,-5.684,-4.217,3.547,-1.405,-4.969,1.571],  # I
        'K': [-0.270,0.001,0.169,-0.536,0.983,2.140,-2.790,0.716,0.350,1.610,1.520,0.458,-0.740,0.440,1.990,-1.436,-1.840,-0.827,-1.712,5.797],  # K
        'L': [1.524,-0.196,0.427,-0.363,0.018,-1.730,-0.920,-1.462,-0.599,0.210,1.980,0.061,0.370,-1.100,0.808,0.998,0.988,3.108,4.846,-1.183],  # L
        'M': [1.200,0.536,-0.141,0.271,-0.987,-1.330,-0.510,-1.406,0.032,0.270,1.200,0.077,1.350,-1.030,0.092,-0.278,1.276,-0.591,-3.144,-0.788],  # M
        'N': [-1.387,-0.169,1.157,1.809,-0.666,5.000,0.870,1.135,0.819,4.010,1.660,1.142,-0.090,2.190,4.118,-1.872,3.603,0.818,4.101,4.323],  # N
        'P': [0.886,0.007,-0.265,-0.172,1.741,0.190,0.500,-0.963,0.236,0.840,0.780,0.238,0.330,-0.990,0.232,-1.628,-2.617,-0.070,-0.088,-0.997],# P
        'Q': [1.247,0.012,-0.015,0.224,-0.549,0.880,1.050,-1.619,-0.549,0.150,0.730,0.043,-0.490,-1.400,2.755,0.834,0.317,1.672,-0.784,1.789],  # Q
        'R': [-0.407,3.847,-1.008,0.042,-0.371,-2.310,-0.300,0.883,-1.243,0.120,0.660,0.033,-1.090,-1.670,-1.032,3.440,5.468,2.273,4.734,-0.906],  # R
        'S': [-0.495,0.035,-0.068,-0.327,1.755,-2.310,-0.300,0.844,0.200,1.390,0.390,0.396,0.520,-0.130,-3.304,-2.798,1.730,0.353,-1.436,-1.149],  # S
        'T': [-0.032,0.117,0.577,-0.873,1.179,-1.770,1.650,0.188,0.022,0.650,0.240,0.187,0.310,0.170,-2.440,-1.886,1.710,-0.463,2.316,-0.346],  # T
        'V': [0.844,-0.810,-0.380,0.268,-0.301,4.550,0.590,-1.577,0.122,1.070,0.660,0.304,-1.890,-0.230,4.383,4.168,0.631,-1.799,-1.452,-0.456],  # V
        'W': [0.329,-0.835,0.289,0.001,0.481,3.590,0.300,-1.142,-0.119,1.300,0.950,0.369,-1.870,0.250,4.247,4.441,1.179,-2.916,-0.903,-2.097],  # W
        'Y': [1.332,-0.229,1.038,-0.333,0.305,-2.900,-0.380,-1.127,-0.596,0.330,1.270,0.093,0.900,-1.440,-1.429,-1.319,-0.042,-0.816,-2.373,-0.349]
        # Y
    }

    encodings = []
    for i in fastas:
        for aa in i:
            a = PCA20v[aa]
            encodings.append(a)

        c = []
        d = []
        j = 1
        for e in encodings:
            for f in e:
                c.append(f)
                if j % (6 * 20) == 0:
                    d.append(c)
                    c = []
                j = j + 1
    return d


def data_read(dir):  # 读取文件
    data = open(dir)
    s = data.read()
    l = s.split()
    return l


def transdata(filename):
    pos_hex = data_read(filename)
    PCA20v_pos_hex = PCA20v(pos_hex)
    PCA20v_pos_hex = np.array(PCA20v_pos_hex)

    print(PCA20v_pos_hex.shape)

    colnum = []
    sequence = pos_hex
    for i in range(len(PCA20v_pos_hex)):
        colnum.append(i)

    pos_PCA20v = pd.DataFrame(PCA20v_pos_hex, index=sequence)
    pos_PCA20v.index.name = 'sequence'

    tmp = []
    tmp = filename.split('\\')
    tmp = tmp[-1].split('.')
    basepath = os.path.dirname(__file__)
    generateddirpath = os.path.join(basepath[:len(basepath)], 'generated')
    if not os.path.exists(generateddirpath):
        os.makedirs(generateddirpath)
    xlsxfilename = os.path.join(generateddirpath, tmp[0] + '_PCA20.xlsx')
    with pd.ExcelWriter(xlsxfilename) as writer:
        pos_PCA20v.to_excel(writer, sheet_name='sheet_1', header=True, index=True)
    print(filename + '转换完成\n')
    print('输出文件：' + xlsxfilename + '\n')
    return xlsxfilename