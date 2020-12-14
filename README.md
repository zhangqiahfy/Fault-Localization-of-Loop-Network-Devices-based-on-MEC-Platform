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
│    ├── 基于MEC平台的网络设备故障定因赛题说明.pdf    # question description
│    ├── GuoChuang-ChapterIX.pdf    # Presentation pdf
│    └── GuoChuang-ChapterIX.ppt    # Presentation slide
│
├── dataSet    # save the data 
│    ├──A_pattern    # regulization data for dataset A
│    │    └──pattern.txt    
│    │
|    └──B_pattern    # regulization data for dataset B
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
└── model    # model save 
    ├── A-rule.csv    # will be update
    └── B-rule.csv    # will be update

```



## How to use

1. in cmd line,

   ```
   python main.py
   ```

   

2. output data will be saved as the below folder

   

   ```
model/    # model 
   output/process/    # node score 
   output/result/    # result
   ```
   
   

## requirements

based on python3.6.9

