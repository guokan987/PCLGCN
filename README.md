# PCLGCN

This is a pytorch code of PCLGCN(IEEE Transcation of intelligent transportation systems) under review, thus don't refer or use it!!!
# datasets: Chinese Cities speed data and POI data in JiNan, XiAn, ShenZhen and ChengDu 

# Instructions for use：

Fristly, you should run the data genenration program to generate the speed data

Secondly, you should run the train.py and train_h.py, and in two python file, you will find some models name, P+model represents the GCN_POI methods which don't utilize POI Correlation module(validation model), and MM+model represents the PCLGCN (this is our model)

首先，你需要通过运行generate开头的文件来生成数据（生成后的数据比较庞大）

其次，你需要运行train.py and train_h.py文件

最后，运行的时候你需要选择模型，一般来说，一个原始的图卷积预测模型例如HGCN，它对应了两个模型:一个是PHGCN，它对应HGCN_POI；另一个MMHGCN，它对应PCLHCN.


The code and datasets website:https://pan.baidu.com/s/1kie4I0PHZICu8LCqcbHr2g  key：wcrk 
数据集和代码地址：链接：https://pan.baidu.com/s/1kie4I0PHZICu8LCqcbHr2g  提取码：wcrk 

# Have a nice day!
