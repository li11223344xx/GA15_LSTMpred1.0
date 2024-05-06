# **AMPredST**

## **Table of contents:**

- [About the web application](#about-the-webapp)
- [Structure of the repository](#structure-of-the-repository)
- [Credits](#credits)
- [Further details](#details)
- [Contact](#contact)

## **About the project**

[Antimicrobial peptides](https://en.wikipedia.org/wiki/Antimicrobial_peptides) (AMPs) are small bioactive drugs, commonly with fewer than 50 amino acids, which have appeared as promising compounds to control infectious disease caused by multi-drug resistant bacteria or superbugs. These superbugs are not treatable with the available drugs because of the development of some mechanisms to avoid the action of these compounds, which is known as antimicrobial resistance (AMR). According to the World Health Organization, AMR is one of the [top ten global public health threats facing humanity in this century](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance), so it is important to search for AMPs that combat these superbugs and prevent AMR.

**AMPredST** is a web application that allows users to predict the antimicrobial activity and general properties of AMPs using a machine learning-based classifier. The appication is based on a [previous project](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp) that analyzed the best molecular descriptors and machine learning model to predict the antimicrobial activity of AMPs. The best model was `ExtraTreesClassifier` with max_depth of 50 and n_estimators of 200 as hyperparameters, and `Amino acid Composition` as the molecular descriptors.

<a href="https://ampredst.streamlit.app/" title="AMPredST"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"></a><br>

![ampredst-gif](./ampredst.gif)

## **Structure of the repository**

The main files and directories of this repository are:

|File|Description|
|:-:|---|
|[ExtraTreesClassifier_maxdepth50_nestimators200.zip](RandomForest_maxdepth10_nestimators200.zip)|Compressed file of the best classifier|
|[streamlit_app.py](streamlit_app.py)|Script for the streamlit web application|
|[requirements.txt](requirements.txt)|File with names of the packages required for the streamlit web application|
|[style.css](style.css)|css file to customize specific feature of the web application|

## **Credits**

- This project was inspired by the [notebook](https://github.com/dataprofessor/peptide-ml) and [video](https://www.youtube.com/watch?v=0NrFIGLwW0Q&feature=youtu.be) from [Dataprofessor](https://github.com/dataprofessor) about this topic.
- The [datasets](https://biocom-ampdiscover.cicese.mx/dataset), some ideas, and references to compare the performance of the best model were obtained from this [scientific article](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251).

## **Further details**
More details about the exploratory data analysis, data preparation, and model selection are available in this [GitHub repository](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp).


## **Contact**
[![](https://img.shields.io/twitter/follow/sayalaruano?style=social)](https://twitter.com/sayalaruano)

If you have comments or suggestions about this project, you can [open an issue](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp/issues/new) in this repository, or email me at sebasar1245@gamil.com.

# **AMPredST**

## **目录：**

- [关于网络应用](#about-the-webapp)
- [仓库结构](#structure-of-the-repository)
- [鸣谢](#credits)
- [更多详情](#details)
- [联系方式](#contact)

## **关于项目**

[抗菌肽](https://en.wikipedia.org/wiki/Antimicrobial_peptides)（AMPs）是一种小型生物活性药物，通常含有不到50个氨基酸，它们已经成为控制多重耐药细菌或超级细菌引起的传染性疾病的有前途的化合物。这些超级细菌由于发展了一些机制来避免这些化合物的作用，被称为抗菌耐药性（AMR）。根据世界卫生组织的说法，AMR是本世纪面临的[十大全球公共卫生威胁之一](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance)，因此寻找能够对抗这些超级细菌并防止AMR的AMPs是非常重要的。

**AMPredST**是一个网络应用，允许用户使用基于机器学习的分类器预测AMPs的抗菌活性和一般性质。该应用基于[之前的项目](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)，分析了预测AMPs抗菌活性的最佳分子描述符和机器学习模型。最佳模型是`ExtraTreesClassifier`，其超参数为最大深度50和估算器数量200，分子描述符为`Amino acid Composition`。

<a href="https://ampredst.streamlit.app/" title="AMPredST"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"></a><br>

![ampredst-gif](./ampredst.gif)

## **仓库结构**

本仓库的主要文件和目录包括：

|文件|描述|
|:-:|---|
|[ExtraTreesClassifier_maxdepth50_nestimators200.zip](RandomForest_maxdepth10_nestimators200.zip)|最佳分类器的压缩文件|
|[streamlit_app.py](streamlit_app.py)|Streamlit网络应用脚本|
|[requirements.txt](requirements.txt)|网络应用所需包名的文件|
|[style.css](style.css)|用于自定义网络应用特定功能的CSS文件|

## **鸣谢**

- 本项目受到[Dataprofessor](https://github.com/dataprofessor)关于此主题的[笔记本](https://github.com/dataprofessor/peptide-ml)和[视频](https://www.youtube.com/watch?v=0NrFIGLwW0Q&feature=youtu.be)的启发。
- [数据集](https://biocom-ampdiscover.cicese.mx/dataset)、一些想法以及比较最佳模型性能的参考资料均来源于这篇[科学文章](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00251)。

## **更多详情**
关于探索性数据分析、数据准备和模型选择的更多细节可在此[GitHub仓库](https://github.com/sayalaruano/CapstoneProject-MLZoomCamp)中找到。

## **联系方式**
[![](https://img.shields.io/twitter/follow/sayalaruano?style=social)](https://twitter.com/sayalaruano)

如果您对此项目有任何评论或建议，可以在此仓库中[提出问题](https://github.com/sayalaruano/ML_AMPs_prediction_streamlitapp/issues/new)，或通过电子邮件与我联系：sebasar1245@gamil.com。