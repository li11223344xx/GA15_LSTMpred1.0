import streamlit as st  #st用于访问 Streamlit 的各种功能，如创建网页、控件和展示数据。
from annotated_text import annotated_text
from millify import millify
import plotly.express as px
# OS and file management
import os
import pickle
from PIL import Image
import zipfile
# Data analysis
import pandas as pd
import base64
from prediction import prediction1
from io import BytesIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np
# Importing the descriptor modules
import GA15v
import PCA15v
import GA20v
import PCA20v
import tempfile


def insert_active_peptide_example():
    st.session_state.peptide_input = 'GTFFIN'

def insert_inactive_peptide_example():
    st.session_state.peptide_input = 'AALQSS'

def clear_peptide():
    st.session_state.peptide_input = ''

def get_descriptor_module(descriptor_type):
    return {
        'GA15v': GA15v,
        'PCA15v': PCA15v,
        'GA20v': GA20v,
        'PCA20v': PCA20v
    }[descriptor_type]


# General options 一般选项
im = Image.open("favicon.ico")
st.set_page_config(
    page_title="GA15_LSTMpred",
    page_icon=im,
    layout="wide",
)# 这里设置了 Streamlit 页面的一些配置项：
# 页面标题、页面图标和布局设置为宽屏模式。
# Image.open("favicon.ico") 用于打开一个图标文件，这个图标被设置为网页的 favicon

# Attach customized ccs style 附加定制的 CCS 样式
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# # Function load the best ML model 函数加载最佳ML模型
# @st.cache_resource
# def load_model(model_file):
#     with open(model_file, 'rb') as f_in:
#         model = pickle.load(f_in)
#     return model

# Add a title and info about the app 添加有关应用程序的标题和信息
st.title('GA15_LSTMpred: Self-assembling peptides prediction streamlit app')
"""
[![](https://img.shields.io/github/stars/sayalaruano/ML_AMPs_prediction_streamlitapp?style=social)](https://github.com/dbt-963/GA15_LSTMpred) &nbsp; [![](https://img.shields.io/twitter/follow/sayalaruano?style=social)](https://twitter.com/lxing121193)
"""

#[Self-assembling peptide](https://en.wikipedia.org/wiki/Antimicrobial_peptides)
with st.expander('About this app'):
    st.write('''
    This model was constructed by deep learning and used amino acid descriptor as the characrization method. This model can predict the self-assembling ablity of hexpeptides.It is helpful to explore the sequence-aggregation relationship of peptides and easy to recognize the assemble-prone sequence in peptides and proteins.

    The GA15_LSTMpred package consists of two main modules: One is the data characterization module. Txt file of peptide sequences was used as input and descriptors were used to characterize peptides. Characteristic matrix was returned as Excel file. The second is the prediction module. The Excel document of descriptor matrix was used as input. GA15-LSTM model was used for prediction, and the prediction label and prediction score of peptide sequences were returned. Users can only use the data characterization module and obtain the descriptor Excel document for constructing and training their own model. Users can also import two modules at the same time to get the prediction results and scores directly.

      ''')

# 添加空白间隔
st.markdown('<div style="margin: 20px;"></div>', unsafe_allow_html=True)

# Set the session state to store the peptide sequence 设置会话状态以存储肽序列
if 'peptide_input' not in st.session_state:
    st.session_state.peptide_input = ''

# st.sidebar.subheader('Input peptide sequence')
peptide_seq = st.sidebar.text_input(
    'Input',  # 这里应始终是字段的名称或提示信息
    st.session_state.peptide_input,
    placeholder='Enter peptide sequence',  # 设置占位符为 "Enter peptide sequence"
    key='peptide_input_unique',
    help='Be sure to enter a valid peptide sequence'
)

# 初始化 session_state 中的变量
if 'confirm_clicked' not in st.session_state:
    st.session_state.confirm_clicked = False


st.sidebar.button('Self-assembling peptide sequence', on_click=insert_active_peptide_example)
st.sidebar.button('Non-self-assembling peptide sequence', on_click=insert_inactive_peptide_example)
st.sidebar.button('Clear input', on_click=clear_peptide)


# Input peptide
descriptor_type = st.sidebar.selectbox('Select Descriptor Type', ('GA15v', 'PCA15v', 'GA20v', 'PCA20v'))

if 'download_clicked' not in st.session_state:
    st.session_state.download_clicked = False
# 打印调试信息
print(f"Descriptor Type: {descriptor_type}")
print(f"Peptide Sequence: {peptide_seq}")


# 初始化确认和预测状态
if 'confirm_clicked' not in st.session_state:
    st.session_state.confirm_clicked = False
if 'prediction_clicked' not in st.session_state:
    st.session_state.prediction_clicked = False

# 添加一个确定按钮
if st.sidebar.button('Confirm'):
    # 标记确认按钮已被点击
    st.session_state.confirm_clicked = True

# 控制信息显示逻辑
if not st.session_state.confirm_clicked:
    st.subheader('Welcome to the app!')
    st.info('Enter peptide sequence in the sidebar to proceed', icon='👈')
else:
    # 确认所选描述符类型为GA15v，并且输入了序列
    if descriptor_type == 'GA15v' and peptide_seq:
        # 调用GA15v模块中的GA15v函数
        result = GA15v.GA15v(peptide_seq)
        df = pd.DataFrame(result)
        st.dataframe(df)
        print(df)
        st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)
        if st.button("进行预测"):
            # 在这里执行预测操作
            st.write("按钮已按下，正在预测...")
            # 创建临时文件并将 DataFrame 写入 Excel 文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_filename = temp_file.name
                print(temp_filename)
                df.to_excel(temp_filename, index=False)
                # 调用模型预测函数
                result4 = prediction1(temp_filename)
                print(result4)
                # 输出预测结果
                st.write("预测结果:")
                st.write(result4)
    elif descriptor_type == 'PCA20v' and peptide_seq:
        result1 = PCA20v.PCA20v(peptide_seq)
        df1 = pd.DataFrame(result1)
        st.dataframe(df1)
        print(df1)
        st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)
        # 当预测按钮未被点击时，显示原有提示信息
        if not st.session_state.prediction_clicked:
            st.dataframe(df1)
            st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)

        # 显示“进行预测”按钮
        if st.button("进行预测"):
            st.session_state.prediction_clicked = True
            st.write("按钮已按下，正在预测...")
            # 创建临时文件并将 DataFrame 写入 Excel 文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_filename = temp_file.name
                df1.to_excel(temp_filename, index=False)

            # 调用模型预测函数
            result5 = prediction1(temp_filename)

            # 输出预测结果
            st.write("预测结果:")
            st.write(result5)
    elif descriptor_type == 'PCA15v' and peptide_seq:
        result2 = PCA15v.PCA15v(peptide_seq)
        df2 = pd.DataFrame(result2)
        st.dataframe(df2)
        print(df2)
        st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)
        # 当预测按钮未被点击时，显示原有提示信息
        if not st.session_state.prediction_clicked:
            st.dataframe(df2)
            st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)

        # 显示“进行预测”按钮
        if st.button("进行预测"):
            st.session_state.prediction_clicked = True
            st.write("按钮已按下，正在预测...")
            # 创建临时文件并将 DataFrame 写入 Excel 文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_filename = temp_file.name
                df2.to_excel(temp_filename, index=False)

            # 调用模型预测函数
            result6 = prediction1(temp_filename)

            # 输出预测结果
            st.write("预测结果:")
            st.write(result6)
    elif descriptor_type == 'GA20v' and peptide_seq:
        result3 = GA20v.GA20v(peptide_seq)
        df3 = pd.DataFrame(result3)
        st.dataframe(df3)
        print(df3)
        st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)
        # 当预测按钮未被点击时，显示原有提示信息
        if not st.session_state.prediction_clicked:
            st.dataframe(df3)
            st.markdown("<small>鼠标滑过表格右上方可下载对应的CSV文件</small>", unsafe_allow_html=True)

        # 显示“进行预测”按钮
        if st.button("进行预测"):
            st.session_state.prediction_clicked = True
            st.write("按钮已按下，正在预测...")
            # 创建临时文件并将 DataFrame 写入 Excel 文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as temp_file:
                temp_filename = temp_file.name
                df3.to_excel(temp_filename, index=False)

            # 调用模型预测函数
            result7 = prediction1(temp_filename)

            # 输出预测结果
            st.write("预测结果:")
            st.write(result7)



st.sidebar.header('Code availability')

st.sidebar.write('The code for this project is available under the [MIT License](https://mit-license.org/) in this [GitHub repo](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp). If you use or modify the source code of this project, please provide the proper attributions for this work.')

st.sidebar.header('Support')

st.sidebar.write('If you like this project, please give it a star on the [GitHub repo](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp) and share it with your friends. Also, you can support me by [buying me a coffee](https://www.buymeacoffee.com/sayalaruano).')

st.sidebar.header('Contact')

st.sidebar.write('If you have any comments or suggestions about this work, please DM by [twitter](https://twitter.com/sayalaruano) or [create an issue](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp/issues/new) in the GitHub repository of this project.')

# # 控制信息的显示
# if not st.session_state.confirm_clicked:
#     st.subheader('Welcome to the app!')
#     st.info('Enter peptide sequence in the sidebar to proceed', icon='👈')
# else:
#     # Confirm被点击后，不显示任何信息
#     st.empty()

