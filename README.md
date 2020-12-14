# Fault Localization of Loop Network Devices based on MEC Platform

According to the training data provided, the algorithm model is trained, and the fault test data are analyzed by using the model, so as to quickly predict the fault trend or locate the fault root cause and reproduce the fault occurrence process.

## Directory

```
.
.
├── README.md    #
│
├── requirements.txt    # 
│
├── documents    # will be update
│    ├── 基于MEC平台的网络设备故障定因项目介绍v2.docx    # 文档基础说明第二版
│    ├── 基于MEC平台的网络设备故障定因项目介绍.pdf    # pdf版的说明文档
│    ├── 基于MEC平台的网络设备故障定因赛题说明.pdf    # 赛题说明文档
│    ├── eclat告警规则和关系网络分析.docx    # eclat算法说明
│    ├── IP RAN派单规则.docx    # 派单规则文档 
│    └── ip_ran综合网管告警监控指引手册-2017-11-30.doc    # 指引手册
│
├── dataSet    # save the data 
│    ├──A_pattern    # regulization data for dataset A
│    │    └──pattern.txt    
│    │
|    └──B_pattern    # regulization data for dataset A
│         └──pattern.txt    
│
├── scripts    # code   
│    └── main.py    # main code
│
├── output                  # output save folder
│    ├──processed    # will be update
|    │    ├── A-score.csv    
|    │    └── B-score.csv    
|    │ 
│    └──result    # will be update
|         ├── A-result.csv    #
|         └── B-result.csv    # 
│
├── model    # model save 
    ├── A-rule.csv    # will be update
    └── B-rule.csv    # will be update

```



## How to use

1. cmd line 

   ```
   python main.py
   ```

   

2. output 

   ```
model/    # model 
   output/process/    # node score 
output/result/    # result
   ```
   
   

## requirements

based on python3.6.9

