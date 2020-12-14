# %load "alarm_AB_0831 - 副本.py"
import os
import pandas as pd
import time
import datetime
# import moxing as mox
from dateutil.parser import parse
import numpy as np
from pygrok import Grok
import matplotlib.pyplot as plt
import re

def match_list(txtData, pattern, dirPath):
    """
    用于数据集的正则化，此处利用正则标准库pygrok，对一条告警数据进行匹配，修改为一条字典数据，多条数据存放在list中
    :param txtData:读取的文本数据
    :param pattern: 正则数据的字典
    :param dirPath: 正则库存放的文件夹
    :return: 返回一个list形式，list中的每个元素为dict
    """

    logList = []
    grok = Grok(pattern, custom_patterns_dir=dirPath)
    for line in txtData:
        logList.append(grok.match(line))
    return list(filter(None, logList))


def load_A_data(aDataPath, file_names):
    """
    读取A数据集的数据，并且对数据进行正则化解析，最后返回标准dataframe数据集

    :param aDataPath: A数据集的位置
    :param file_names:  A数据集内部的数据名称，每个名称包含具体的文件夹路径
    :return: 返回处理后的A数据集
    """

    dirPath = "A_pattern"

    # 定义正则文档 A

    pattern = "%{TimeStamp:time} %{HostName:position} %{Test:test}%{dd:dd}%{ModuleName:module}/%" \
              "{Severity:level}/%{Brief:Brief}%{i:i}\\[%{Count:Count}\\]:%{Description:description}"
    pattern2 = "%{TimeStamp:time} %{HostName:position} %{ModuleName:module}/%{Severity:level}/%" \
               "{Brief:Brief}:%{Description:description}"

    # 循环读取A数据集数据
    # 进行正则化解析
    log_data = []
    for file_name in file_names:  # 这里主要根据解析不同的样本集进行修改
        # 循环读取,并处理为list,
        with open(aDataPath + file_name) as f:
            file_data = f.readlines()
            log_data.extend(match_list(file_data, pattern, dirPath))
            log_data.extend(match_list(file_data, pattern2, dirPath))

    # 把数据更改为dataframe
    # 把数据中的时间列，冲字符串格式改为时间戳格式
    # 删除某些无意义列
    dataAll = pd.DataFrame(log_data)  # 转为dataframe
    dataAll["time"] = dataAll["time"].apply(lambda x: x.split('+')[0])  # 删除时间字符串中 +08:00 数据
    dataAll["time"] = dataAll["time"].apply(lambda x: parse(x))  # 转为时间戳
    dataAll['time'] = pd.to_datetime(dataAll['time'], format='%Y-%m-%d %H:%M:%S')  # 修改时间格式
    dataAll = dataAll.drop(["test", "Count", "dd", "i"], axis=1)  # 删除无意义列

    dataAll["level"] = dataAll["level"].map(lambda x: int(x))  # 把level列从字符串改为int类型

    return dataAll


####下面是B数据集################
def read_as_line(dataPath, name):
    """
    读取数据，多行转为一行
    B数据集中，一条告警存在多行情况，修改为一行
    :param dataPath: 数据地址
    :param name: 数据名字
    :return: 数据集字符串
    """

    with open(dataPath + name) as f:
        f1 = f.read()
        f1 = f1.replace("\n", "")
        f1 = f1.replace("\r", "")
        f1 = f1.replace("An ", "\nAn ")
        f1 = f1.replace("A ", "\nA ")
        f1 = f1.split("\n")
    return f1


def load_B_data(bDataPath, files_with_id, files_without_id):
    """
    B数据集的读取，读取B数据集的数据，并且正则化解析，最后返回全部数据
    B数据集中数据分为两个大类，一类中含有id这个字段，一类中没有id，因此使用两种正则化pattern
    每个大类下面，又分为两个小类别
    :param bDataPath: B数据集的地址
    :param files_with_id: 含有id字段的数据
    :param files_without_id: 不含有id字段的数据
    :return: 标准的dataframe数据
    """

    dirPath = "B_pattern"  # 正则化数据库地址

    # 读取含有id字段数据
    # 并转化为csv 
    log_type_one = []
    log_type_two = []

    # 定义正则规则
    # 由于告警字段内容的不统一，可能出现某些告警没有通过正则化而丢失的情况。这里根据人为判断，告警再分为两个小类。
    # pattern没有cleared字段，pattern2含有cleared字段
    pattern = "%{type:type} %{code:code} %{ID:ID} %{code:code1} %{level:levelLabel} %{level_:level} " \
              "%{cleared:occurred} %{time_:time} %{sent:sent} %{position:position}\%%{module:module}\%%" \
              "{description:description}"
    pattern2 = "%{type:type} %{code:code} %{ID:ID} %{code:code1} %{level:levelLabel} %{level_:level} " \
               "%{cleared:occurred} %{time_:time}, %{cleared:cleared} %{time_:time2} %{sent:sent} %" \
               "{position:position}\%%{module:module}\%%{description:description}"

    # 循环读取, 每条告警处理为字典类型数据，多个字典拼成list格式
    # list转化为dataframe
    for file_name in files_with_id:
        file_data = read_as_line(bDataPath, file_name)
        log_type_one.extend(match_list(file_data, pattern, dirPath))
        log_type_two.extend(match_list(file_data, pattern2, dirPath))
    with_id_one = pd.DataFrame(log_type_one)
    with_id_two = pd.DataFrame(log_type_two)

    # 读取没有id字段的数据
    # 并转化为csv

    log_type_one = []
    log_type_two = []

    # 定义正则规则
    # 由于告警字段内容的不统一，可能出现某些告警没有通过正则化而丢失的情况。这里根据人为判断，告警再分为两个小类。
    # pattern没有cleared字段，pattern2含有cleared字段
    pattern = "%{type:type} %{code:code} %{level:levelLabel} %{level_:level} %{cleared:occurred} %" \
              "{time:time} %{sent:sent} %{position:position} \%%{module:module}\%%{description:description}"
    pattern2 = "%{type:type} %{code:code} %{level:levelLabel} %{level_:level} %{cleared:occurred} %" \
               "{time:time} %{cleared:cleared} %{time:time2} %{sent:sent} %{position:position} \%%" \
               "{module:module}\%%{description:description}"

    # 循环读取, 每条告警处理为字典类型数据，多个字典拼成list格式
    # list转化为dataframe
    for file_name in files_without_id:
        file_data = read_as_line(bDataPath, file_name)
        log_type_one.extend(match_list(file_data, pattern, dirPath))
        log_type_two.extend(match_list(file_data, pattern2, dirPath))
    without_id_one = pd.DataFrame(log_type_one)
    without_id_two = pd.DataFrame(log_type_two)

    # 所有数据集合为总数据集
    # 时间数据处理为时间戳
    # 月份超过五月份，则判断为2019年的数据

    dataAll = pd.concat([with_id_one, with_id_two, without_id_one, without_id_two], axis=0).reset_index(
        drop=True)
    dataAll["time"] = dataAll["time"].apply(lambda x: parse(x))
    dataAll["time"] = dataAll["time"].map(lambda x: x if x < pd.to_datetime('20200501') else x - pd.Timedelta(days=366))
    dataAll = dataAll.sort_values(by=['time'], axis=0, ascending=True).reset_index(drop=True)
    dataAll["level"] = dataAll["level"].map(lambda x: int(x))

    # dataAll = dataAll[dataAll["occurred"] != "cleared at"]  # 删除所有被
    # dataAll = dataAll.drop(["type","levelLabel","sent","occurred"],axis=1)

    return dataAll


# train_data-02-15 14:52:51 ~~~2020-04-13 09:37:08：
# test_data的时间跨度是2019-02-26 02:48:58 ~~~2019-04-02 17:40:25：
def threshold_alarm(train_data, test_data):
    """
    针对数据集进行告警的剔除，使用多种方式剔除告警
    错位计算时间间隔：
        a)	错位寻找下一次告警，并计算两次告警之间的时间。
        b)	删除间隔时间小于中位数的数据。
        c)	中位数由训练集与测试集共同数据计算的。
    发生次数与平均周期：
        a)	计算发生总次数，计算最初与最后的发生时间差。
        b)	时间差除以总次数得到平均周期。
        c)	判断条件一：次数大于均值加上三倍的标准（异常值判断）。
        d)	判断条件二：平均周期小于中位数。
        e)	删除同时满足上述两个条件的数据。

    :param train_data: 训练数据集
    :param test_data: 测试数据集
    :return: 剔除后的数据集
    """

    # 按时间排序
    train_data = train_data.sort_values(by='time', ascending=True)
    test_data = test_data.sort_values(by='time', ascending=True)
    # 删除train中的重复数据
    # 拼接唯一告警
    train_data = train_data.drop_duplicates(['time', 'position', 'module', 'level', 'Brief'], "first")
    train_data['position_Brief'] = train_data['position'] + '&' + train_data['Brief']
    # 删除test中的重复数据
    # 拼接唯一告警
    test_data = test_data.drop_duplicates(['time', 'position', 'module', 'level', 'Brief'], "first")
    test_data['position_Brief'] = test_data['position'] + '&' + test_data['Brief']

    # 统计告警元数据
    # 计算相同两个告警之间的实际时间间隔
    # 如果实际时间间隔国小，不可能是主要告警的
    # 默认主告警不会频繁的发生
    # 下面统计实际时间间隔的中位数。
    train_data['next_time'] = train_data.groupby(by=['position_Brief'])['time'].shift(-1)
    train_data['time_gaps'] = train_data.apply(lambda x: (x['next_time'] - x['time']).total_seconds(), axis=1)
    train_data = train_data.reset_index(drop=True)
    train_time_graps_median = train_data[~train_data['time_gaps'].isna()]['time_gaps'].median()

    test_data['next_time'] = test_data.groupby(by=['position_Brief'])['time'].shift(-1)
    test_data['time_gaps'] = test_data.apply(lambda x: (x['next_time'] - x['time']).total_seconds(), axis=1)
    test_data = test_data.reset_index()
    test_time_graps_median = test_data[~test_data['time_gaps'].isna()]['time_gaps'].median()

    # 计算训练集与测试集实际时间间隔的中位数的平均值值
    # 定义新的frequent_flag字段，用于判断告警是否剔除的字段
    # 当间隔小于平均中位数时，frequent_flag为1
    # 当大于中位数的时候，frequent_flag为0
    time_graps_threshold = (test_time_graps_median + train_time_graps_median) / 2
    train_data['frequent_flag'] = 0
    train_data.loc[train_data['time_gaps'] < time_graps_threshold, 'frequent_flag'] = 1
    test_data['frequent_flag'] = 0
    test_data.loc[test_data['time_gaps'] < time_graps_threshold, 'frequent_flag'] = 1

    # 统计总间隔，总发生次数，与平均发生周期
    # 总间隔是这类告警第一次发生的时间，和最后一次发生的时间的差
    # 次数指的是在期间内发生的总次数
    # 计算平均周期，用总间隔除以总次数
    train_position_Brief = train_data.groupby(['position_Brief'])['time'].agg(
        {'count': 'count', 'max': 'max', 'min': 'min'})
    train_position_Brief = train_position_Brief.sort_values(by='count', ascending=False)
    train_position_Brief = train_position_Brief.reset_index()
    train_position_Brief['time_gaps'] = train_position_Brief.apply(lambda x: (x['max'] - x['min']).total_seconds(),
                                                                   axis=1)
    train_position_Brief['time_period'] = train_position_Brief['time_gaps'] / train_position_Brief['count']
    test_position_Brief = test_data.groupby(['position_Brief'])['time'].agg(
        {'count': 'count', 'max': 'max', 'min': 'min'})
    test_position_Brief = test_position_Brief.sort_values(by='count', ascending=False)
    test_position_Brief = test_position_Brief.reset_index()
    test_position_Brief['time_gaps'] = test_position_Brief.apply(lambda x: (x['max'] - x['min']).total_seconds(),
                                                                 axis=1)
    test_position_Brief['time_period'] = test_position_Brief['time_gaps'] / test_position_Brief['count']

    # 删除告警
    # 考虑平均时间间隔较小偏小，但是总次数偏大的告警
    # 判断条件一：次数大于均值加上三倍的标准（异常值判断）。
    # 判断条件二：平均周期小于中位数。
    # 同时满足两个条件的被删除
    position_Brief = pd.concat([train_position_Brief, test_position_Brief])
    position_Brief = position_Brief.reset_index(drop=True)
    position_Brief_count = position_Brief.groupby(['position_Brief'])['count'].agg({'count': 'sum'})
    position_Brief_time_gaps = position_Brief.groupby(['position_Brief'])['time_gaps'].agg({'time_gaps': 'sum'})
    position_Brief_time_period = position_Brief.groupby(['position_Brief'])['time_period'].agg({'time_period': 'mean'})
    position_Brief_count = position_Brief_count.reset_index()
    position_Brief_time_gaps = position_Brief_time_gaps.reset_index()
    position_Brief_time_period = position_Brief_time_period.reset_index()

    position_Brief_summary = pd.merge(position_Brief_count, position_Brief_time_gaps, on='position_Brief', how='outer')
    position_Brief_summary = pd.merge(position_Brief_summary, position_Brief_time_period, on='position_Brief',
                                      how='outer')
    time_period_median = position_Brief_summary['time_period'].median()
    count_threshold = position_Brief_summary['count'].mean() + 3 * position_Brief_summary['count'].std()
    # 发生次数超过异常值（均值+3标准差）并且周期小于中位数
    position_Brief_loss = position_Brief_summary[(position_Brief_summary['count'] > count_threshold) & (
            position_Brief_summary['time_period'] < time_period_median)]
    train_data_pro = train_data[~train_data['position_Brief'].isin(position_Brief_loss.position_Brief)]

    # 根据frequent_flag剔除
    # 删除frequent_flag为1的数据
    train_data_pro = train_data_pro[train_data_pro['frequent_flag'] == 0]
    test_data_pro = test_data[~test_data['position_Brief'].isin(position_Brief_loss.position_Brief)]
    test_data_pro = test_data_pro[test_data_pro['frequent_flag'] == 0]
    train_data_pro.drop(columns=['position_Brief', 'next_time', 'time_gaps', 'frequent_flag'], inplace=True)
    test_data_pro.drop(columns=['position_Brief', 'next_time', 'time_gaps', 'frequent_flag'], inplace=True)

    return train_data_pro, test_data_pro


def delete_alarm(df, hparam, lparam):
    """

    :hparam 出现次数超过 hparam 的告警，全部删除
    :lparam 出现次数少于 lparam 的告警，全部删除
    :param df:
    :return:
    """
    counts = df["net_title"].value_counts()
    for i in range(len(counts)):
        if counts[i] > hparam or counts[i] < lparam:
            df = df[-df['net_title'].isin([counts.index[i]])]
    df = df.reset_index(drop=True)

    return df


def time_window(alarm, timeLabel):
    """
    时间窗口为60分钟，步长为15分钟（或其他）
    之前告警数据是按照时间顺序排列的
    现在从头开始，第一步循环：将0-60分钟的告警数据提取出来，并加上一个新字段id(窗口序号),值为1；
    第二步循环：将15-75分的告警数据提取出来，粘在第一步提取的数据之后，窗口序号字段值为2；\
    之后依次类推，直到所有告警数据被提取出来并形成新的告警数据

    """
    title_new_collect = []
    threeMins = datetime.timedelta(minutes=60)  # 时间窗口的大小
    oneMins = datetime.timedelta(minutes=45)  # 滑动
    twoMins = datetime.timedelta(minutes=15)  # 间隔
    time_stop = alarm.loc[0, timeLabel] + threeMins
    time_start = alarm.loc[0, timeLabel] + oneMins
    i = 0
    while (i < len(alarm[timeLabel])):
        title_new_collect_part = []
        j = 0
        if time_stop < alarm.loc[i, timeLabel]:
            time_stop = alarm.loc[i, timeLabel] + threeMins
            time_start = time_stop - twoMins
        while (alarm.loc[i, timeLabel] <= time_stop):

            if alarm.loc[i, timeLabel] > time_start:
                j = j + 1

            title_new_collect_part.append(alarm.loc[i, 'title_new'])
            i = i + 1
            if i >= len(alarm[timeLabel]):
                break
        title_new_collect_part = list(set(title_new_collect_part))
        if title_new_collect_part != []:
            title_new_collect.append(title_new_collect_part)
        i = i - j
        time_stop = time_stop + oneMins
        time_start = time_start + oneMins

    return title_new_collect


def eclat(prefix, items, minsup, freq_items, L):
    """
    产生频繁项集方法，采用自调用方式，循环产生所有频繁项
    :param prefix:
    :param items:
    :param minsup: 最小支持度
    :param freq_items: 频繁项
    :param L: 所有项的list集合
    :return: freq_items, L
    """
    while items:

        i, itids = items.pop()
        isupp = len(itids)

        if isupp >= minsup and len(prefix) < 2:  # 只生成项数2和1的频繁项，节省运算时间， 理论上对结果没有影响

            freq_items[frozenset(sorted(prefix + [i]))] = isupp
            L.append(frozenset(sorted(prefix + [i])))
            suffix = []
            for j, ojtids in items:
                jtids = itids & ojtids
                if len(jtids) >= minsup:
                    suffix.append((j, jtids))
            freq_items, L = eclat(prefix + [i], sorted(suffix, key=lambda item: len(item[1]), reverse=True), minsup,
                                  freq_items, L)
    return freq_items, L


def eclat_zc(data_set, minsup):
    """
    Eclat倒排方法，
    将数据倒排，采用调用eclat方法，产生频繁项
    :param data_set:
    :param min_support:
    :return:
    """

    data = {}
    trans_num = 0
    for trans in data_set:
        trans_num += 1
        for item in trans:
            if item not in data:
                data[item] = set()
            data[item].add(trans_num)

    freq_items = {}
    L = []
    freq_items, L = eclat([], sorted(data.items(), key=lambda item: len(item[1]), reverse=True), minsup, freq_items, L)
    return freq_items, L



def get_Subset(fromList, toList):
    """
    生成集合的所有子集

    :param fromList: 输入的list
    :param toList: 输出的子集合
    :return:
    """
    for i in range(len(fromList)):
        t = [fromList[i]]
        tt = frozenset(set(fromList) - set(t))
        if not tt in toList:
            toList.append(tt)
            tt = list(tt)
            if len(tt) > 1:
                toList = get_Subset(tt, toList)
    return toList


def cal_ConfKulcIr(freqSet, H, supportData, ruleList, ruleListequal, minConf=0.5):
    """
    计算一些指标
    提升度lift：
        lift = p(a & b) / p(a)*p(b)
    由于数据中，无用数据过多，提升值过大，无意义
    计算KULC + IR
    kulc为两个提升度的平均值
        KULC = 0.5* p(a & b) / p(a) + 0.5* p(a & b) / p(b)
    不平衡因子IR 可以描述数据中 a 跟 b 的平衡性，如果过大，则a，b数量差距过大
        ir = |p(a) - p(b)|/（p(a) + p(b)-p(a & b))
    知道KULC与IR之后可根据实际情况，进行分析关联性

    :param freqSet:
    :param H:
    :param supportData:
    :param ruleList: 主从关系
    :param ruleListequal: 共生关系
    :param minConf:
    :return:
    """

    for conseq in H:
        # 只记录单个项之间的关系
        if len(freqSet - conseq) == 1 and len(conseq) == 1:

            conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算置信度
            kulc = (supportData[freqSet] / supportData[freqSet - conseq] + supportData[freqSet] / supportData[
                conseq]) * 0.5
            ir = abs(supportData[conseq] - supportData[freqSet - conseq]) / (
                    supportData[conseq] + supportData[freqSet - conseq] - supportData[freqSet])
            # conseq, '-->', freqSet - conseq,
            conf2 = supportData[freqSet] / supportData[conseq]
            if conf == 1 and conf2 == 1:
                ruleListequal.append([list(freqSet - conseq)[0], list(conseq)[0], conf, kulc, ir])
            elif conf >= minConf and conf >= conf2:
                ruleList.append([list(freqSet - conseq)[0], list(conseq)[0], conf, kulc, ir])


# 生成规则
def gen_rule(L, supportData, minConf=0.5):
    """
    生成规则
    根据频繁项及事务集生成规则
    :param L:
    :param supportData:
    :param minConf:
    :return:
    """
    bigRuleList = []
    bigRuleListequal = []
    for freqset in L:

        if len(freqset) > 1:  # 只对集合长度是2的list进行处理
            Hl = list(freqset)
            all_subset = []
            get_Subset(Hl, all_subset)
            cal_ConfKulcIr(freqset, all_subset, supportData, bigRuleList, bigRuleListequal, minConf)  # 计算置信度，KULC，IR数值

    return bigRuleList, bigRuleListequal


def sub_rule(df1, df2, df):

    """
    关联规则关系的数据集字段的分割，并命名
    :param df1:
    :param df2:
    :param df:
    :return:
    """
    ruleMainName = df1["main"].str.split('&&', expand=True)
    ruleMainName.columns = ["main_position", "main_module", "main_Brief"]
    ruleSubName = df2["sub"].str.split('&&', expand=True)
    ruleSubName.columns = ["sub_position", "sub_module", "sub_Brief"]
    ruleDf = pd.concat([ruleMainName, ruleSubName, df[["置信度", "KULC", "不平衡因子IR"]]], axis=1)  # 拼接
    ruleDf = ruleDf.sort_values(by=["main_position", "main_module", "main_Brief", "置信度", "KULC"],
                                ascending=[False, False, False, False, True]).reset_index(drop=True)
    return ruleDf


def deal_Rule(rules, value_num_title):

    """
    生成的告警中，告警信息为数值化映射后的数字，需要更改为原始字符
    ruleA为主告警名称
    ruleB为从告警
    :param rules: rule原始数据
    :param value_num_title: 数值化映射关系表
    :param flag:
    :return:
    """
    rulePart = pd.DataFrame(rules)
    rulePart.columns = ["main", "sub", "置信度", "KULC", "不平衡因子IR"]

    rule_main = []
    for rule in rulePart["main"]:
        rule_main.append(value_num_title[1][value_num_title[0][rule]])
    rule_main = pd.DataFrame(rule_main).astype(str)

    rule_sub = []
    for rule2 in rulePart["sub"]:
        rule_sub.append(value_num_title[1][value_num_title[0][rule2]])
    rule_sub = pd.DataFrame(rule_sub).astype(str)
    rule_main.columns = ["main"]
    rule_sub.columns = ["sub"]

    rule_all = sub_rule(rule_main, rule_sub, rulePart)

    return rule_all


def deal_raw_data(alarm):
    """
    处理数据
    告警字段进行数值化映射
    :param alarm:
    :return:
    """
    alarm["net_title"] = alarm["position"].str.cat(alarm["module"], sep="&&").str.cat(alarm["Brief"],
                                                                                      sep="&&")  # 拼接

    value_num_title = pd.factorize(alarm["net_title"])  # 给label编号
    num_title_df = pd.DataFrame(value_num_title[0])  # 转化为dataframe
    num_title_df.columns = ["title_new"]  # 命名
    alarm = pd.concat([alarm, num_title_df], axis=1)  # 拼接
    print("赋值告警信息新标号成功!")

    event_time = pd.to_datetime(alarm["time"], format='%Y-%m-%d %H:%M:%S')
    alarm = alarm.drop(["time"], axis=1)
    alarm = pd.concat([alarm, event_time], axis=1)
    print("转化时间信息成功!")
    ## 按时间排序
    alarm_sorted = alarm.sort_values(by="time", ascending=True).reset_index(drop=True)

    print("重新排序成功!")

    return alarm_sorted, value_num_title


def get_rule(data,timeLabel):
    startall = time.clock()  # 记录程序开始时间
    start = time.clock()  # 单项运行时间的记录

    alarm, value_num_title = deal_raw_data(data)  # 处理原始数据
    alarm.to_csv("alarmAll.csv", header=True, index=False, encoding="utf-8", sep=',')
    # mox.file.copy('alarmAll.csv', os.path.join(Context.get_output_path(), 'alarmAll.csv'))
    end = time.clock()
    print("处理数据用时: %.1f s" % (end - start))

    start = time.clock()
    title_new_collect = time_window(alarm, timeLabel)  # 获取时间窗
    end = time.clock()
    print("处理时间窗用时: %.1f s" % len(title_new_collect))
    print("处理时间窗用时: %.1f s" % (end - start))

    start = time.clock()
    # minsup = 0.5
    # 尽可能收集更多的数据
    supportData, fil = eclat_zc(title_new_collect, 0.5)  # 收集频繁项
    end = time.clock()

    print("获得频繁项集用时: %.1f s" % (end - start))
    print("频繁项集共：%d 项!" % (len(supportData)))

    # 最小置信度用0.5
    # 收集关联规则
    start = time.clock()
    RuleList, RuleListequal = gen_rule(fil, supportData, minConf=0.5)
    end = time.clock()
    print("获取关联规则用时: %.1f s" % (end - start))

    print("一共产生 %d 条关联信息!" % (len(RuleList) + len(RuleListequal)))
    print("%d rule,%d equal" % (len(RuleList), len(RuleListequal)))
    # 处理原始的关联规则
    # 如果存在共生，则计算，
    if len(RuleList) != 0:
        rule = deal_Rule(RuleList, value_num_title)
    else:
        rule = pd.DataFrame([])

    # 如果存在共生，则计算，
    if len(RuleListequal) != 0:
        rule_equal = deal_Rule(RuleListequal, value_num_title)
    else:
        rule = pd.DataFrame([])


    endall = time.clock()
    print("运行程序总共用时 % .1f s" % (endall - startall))

    return rule, rule_equal


def get_score(dataAll, ruleAll, flag_name):
    """
    计算节点得分
        a)	给每个节点计算输入与输出得分，并计算输入输出的和
        b)	设计得分公式：
                score = sum*out_score/in_score
        c)	根节点一定是输出大于属于的数据，并且根节点的输入输出的和一定比较大
    选择根设备
        a)	根据得分，选取评分第一的节点，把position与brief分离出来。
        b)	回到原始数据中，根据position与brief找到最早的一条告警作为根因告警。
        c)	选择position作为根因设备，对应的description作为日志。对应time作为发生时间。
    下级设备（被传递设备）：
        a)	选择根节点向后选取十分钟，作为时间窗口。
        b)	在时间窗口内的所有数据拿出作为待处理被影响节点列表--A。
        c)	根据关联规则表，找到根节点作为主节点的所有规则表，选择规则表中的从节点形成节点列表—B。
        d)	如果在A中出现的设备，但是不在B中，则排除此设备。如果在A中出现的设备同时在B中也出现，则保留此设备作为被传播设备保留。
        e)	选择输出被传播设备，与被传播设备对应的日志作为输出结果。

    :param dataAll:
    :param ruleAll:
    :param flag_name:
    :return:
    """
    ruleAll['main_position_Brief'] = ruleAll['main_position'] + '&&' + ruleAll['main_Brief']
    ruleAll['sub_position_Brief'] = ruleAll['sub_position'] + '&&' + ruleAll['sub_Brief']

    # 生成用于画图的元组
    ruleTruple = list(zip(ruleAll["main_position"].str.cat(ruleAll["main_Brief"], sep="&&"),
                          ruleAll["sub_position"].str.cat(ruleAll["sub_Brief"], sep="&&")))
    # print(' ruleTruple', ruleTruple)
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from(ruleTruple)


    # 分别计算输入输出得分
    # 计算输入与输出分数的和
    degreeInScore = pd.DataFrame([G.in_degree()]).T
    degreeInScore = degreeInScore.reset_index()
    degreeInScore.columns = ['node', 'in_score']
    degreeOutScore = pd.DataFrame([G.out_degree()]).T
    degreeOutScore = degreeOutScore.reset_index()
    degreeOutScore.columns = ['node', 'out_score']

    resultScore = pd.merge(degreeInScore, degreeOutScore, on='node', how='left')
    resultScore["sum"] = resultScore["in_score"] + resultScore["out_score"]

    resultScore.to_csv("resultScore.csv", header=True, index=False, encoding="utf-8", sep=',')
    # mox.file.copy('resultScore.csv', os.path.join(Context.get_output_path(), 'resultScore.csv'))
    resultScore['score'] = resultScore['out_score'] * resultScore['sum'] / resultScore['in_score']
    resultScore = resultScore.sort_values(by="score", ascending=False).reset_index(drop=True)
    dataAll['position_Brief'] = dataAll['position'] + '&&' + dataAll['Brief']

    # 取出得分最高，且得分超过一定数值的节点
    score_max =  resultScore.loc[(resultScore['out_score'] >= 2) & (resultScore['in_score'] >0)]['score'].max()
    
    node = resultScore.loc[(resultScore['score'] == score_max) & (resultScore['out_score'] >= 2), 'node']
    print('node', len(node))

    result_data = pd.DataFrame([{"字段":"flag_name", "数据结果":flag_name}])
    for i in node:
        # print(i)
        result = {}
        df_main = dataAll[dataAll['position_Brief'] == i].reset_index()
        rule_before = i
        # 故障相关设备名
        Device1 = df_main[df_main['time'] == df_main.sort_values(by="time")["time"][0]]['position'][0]
        print('Device1', Device1)
        result["Device1"] = Device1
        Log1 = df_main[df_main['time'] == df_main.sort_values(by="time")["time"][0]]['description'][0]
        print('Log1', Log1)
        result["Log1"] = Log1
        F_time = df_main.sort_values(by="time")["time"][0]
        sub_rule = ruleAll[(ruleAll['main_position_Brief'] == rule_before) & (ruleAll['sub_position'] != Device1)]
        sub_position_Brief = ruleAll[ruleAll['main_position_Brief'] == rule_before]['sub_position_Brief']
        df_sub = dataAll[(F_time <= dataAll['time']) & (dataAll['time'] < F_time + datetime.timedelta(seconds=600)) & (
            dataAll['position_Brief'].isin(sub_position_Brief))]
        df_sub = df_sub[df_sub['position']!= Device1]
        # 传递下级设备名称（数量<3）
        Spread2device1 = list(df_sub.loc[df_sub['position'] != Device1, 'position'].unique())
        print('Spread2device1', Spread2device1)
        lenth = len(Spread2device1) if len(Spread2device1) < 3 else 3
        for i in range(lenth):
            result["Spread2device" + str(i + 1)] = Spread2device1[i]

        # 传递下级日志（数量<5）
        Spread2log1 = list(df_sub['description'].values)
        print('Spread2log1', Spread2log1)
        lenth = len(Spread2log1) if len(Spread2log1)<5 else 5
        for i in range(lenth):
            result["Spread2log" + str(i + 1)] = Spread2log1[i]
            
        # 传递给下一个设备的传递参数
        sub_rule0 = sub_rule[sub_rule['sub_position_Brief'].isin(df_sub['position_Brief'])]  # 相关的规则库输出
        sub_rule0 = sub_rule0.drop_duplicates(['置信度', 'main_position_Brief', 'sub_position_Brief'], "first")
        # print('sub_rule',sub_rule)

        SpreadParaForD1 = list(sub_rule0['置信度'].values)
        print('SpreadParaForD1', SpreadParaForD1)
        
        for i in range(lenth):
            result["SpreadParaForD" + str(i + 1)] = SpreadParaForD1[i]
            
        result = pd.DataFrame([result]).T.reset_index().rename(columns={"index": "字段", 0: "数据结果"})
        result_data = pd.concat([result_data,result],axis=0)

    return result_data


def get_result(dataAll, flag_name):
    """
    过滤小于6的告警
    产生关联规则
    根据关联规则计算得分，并输出结果
    :param dataAll:
    :param flag_name:
    :return:
    """
    dataAll["level"] = dataAll["level"].map(lambda x: int(x))
    dataPart = dataAll[dataAll["level"] <5].reset_index(drop=True)

    time_label = "time"
    rule, rule_equal = get_rule(dataPart, time_label)
    ruleAll = rule #pd.concat([rule, rule_equal], axis=0).reset_index(drop=True)

    # 获取分数，使用networkx
    ruleAll.to_csv("ruleAll.csv", header=True, index=False, encoding="utf-8", sep=',')
    # mox.file.copy('ruleAll.csv', os.path.join(Context.get_output_path(), 'ruleAll.csv'))
    result = get_score(dataAll, ruleAll, flag_name)
    return result

# 统计一种高级发生的次数，以及时间差的均值，
def create_brief(train_data, test_data):
    """
    为B数据集添加Brief字段：
        a)	用此字段与position构成唯一告警。
        b)	A数据集天然含有brief字段，用于描述告警具体信息。
        c)	B数据集需要根据告警描述字段（description）构造brief字段（此处参考A中brief与description的关系构造）：
            i.	删除 [clear] 之后的数据，clear表示告警清除，这里直接进行分割，剔除clear字段之后所有数据。
            ii.	删除小括号，中括号，大括号之中的内容：
                1.	括号内部数据表示一些具体内容，直接进行删除。
            iii.	删除数字：
                1.	数字很多代表ip地址等，很多告警属于同一种，直接进行删除。
            iv.	删除标点符号：
                1.	数据中很多标点，空格等，虽然两条告警描述信息一致，但是由于标点空格等符号不同，导致会判断为两条告警。
            v.	最终效果：
                1.	一条告警，最终只保留字母信息。

    :return:
    """
    train_data['Brief'] = train_data['description']
    train_data['Brief'] = train_data['Brief'].map(lambda x: x.split('[clear]')[0] if '[clear]' in x else x)
    train_data['Brief'] = train_data['Brief'].map(lambda x: re.sub('\[.*?\]|\{.*?\}|\(.*?\) |\d*|\"|\/', '', x))
    train_data['Brief'] = train_data['Brief'].map(lambda x: re.sub('[^a-zA-Z]','',x))
    test_data['Brief'] = test_data['description']
    test_data['Brief'] = test_data['Brief'].map(lambda x: x.split('[clear]')[0] if '[clear]' in x else x)
    test_data['Brief'] = test_data['Brief'].map(lambda x: re.sub('\[.*?\]|\{.*?\}|\(.*?\) |\d*|\"|\/', '', x))
    test_data['Brief'] = test_data['Brief'].map(lambda x: re.sub('[^a-zA-Z]','',x))
    train_data = train_data.drop_duplicates(['time', 'position', 'module', 'level', 'Brief'], "first")

    test_data = test_data.drop_duplicates(['time', 'position', 'module', 'level', 'Brief'], "first")

    return train_data, test_data

if __name__ == "__main__":
    "=============================================A数据集============================================="
    aDataPath = './A/'
    train_file_names = [
        'training data/CSG-1/log.log',
        'training data/CSG-1/log0.log',
        'training data/CSG-2/log.log',
        'training data/CSG-2/log0.log',
        'training data/CSG-3/log.log',
        'training data/CSG-3/log0.log',
        'training data/CSG-4/log.log',
        'training data/CSG-5/log.log',
        'training data/CSG-5/log0.log',
        'training data/CSG-6/log.log',
        'training data/CSG-6/log0.log',
        'training data/CSG-7/log.log',
        'training data/CSG-7/log0.log',
        'training data/CSG-8/log.log',
        'training data/CSG-8/log0.log',
        'training data/CSG-9/log.log',
        'training data/CSG-9/log0.log',
    ]
    train_aDataAll = load_A_data(aDataPath, train_file_names)  # 读取A训练集

    test_file_names = [
        'test data/ASG-2_10.24.91.27/2019-03-27.log',
        'test data/ASG-2_10.24.91.27/2019-03-28.log',
        'test data/ASG-2_10.24.91.27/2019-03-29.log',
        'test data/CSG-1_10.25.95.128/log.log',
        'test data/CSG-2_10.25.95.129/log.log',
        'test data/CSG-3_10.25.95.131/log.log',
        'test data/CSG-4_10.25.95.135/log.log',
        'test data/CSG-5_10.25.95.130/log.log',
        'test data/CSG-6_10.25.95.132/log.log',
        'test data/CSG-7_10.25.95.133/log.log',
        'test data/CSG-8_10.25.95.134/log.log',
    ]
    test_aDataAll = load_A_data(aDataPath, test_file_names)  # 读取A测试集
    train_aDataAll_pro, test_aDataAll_pro = threshold_alarm(train_aDataAll, test_aDataAll)  # 过滤告警
    aResult = get_result(test_aDataAll_pro, "A")
    aResult = aResult.add_suffix("-A")
    "=============================================B数据集============================================="
    bDataPath = './B/'
    # 训练样本
    train_with_id = [
        '/training data/CSG-3_10.28.202.122.log',
        '/training data/CSG-5_10.28.202.118.log',
        '/training data/CSG-11_10.28.202.125.log',
        '/training data/CSG-14_10.28.202.124.log'
    ]
    train_without_id = ['/training data/ASG-1_10.24.168.25.log',
                        '/training data/ASG-2_10.24.168.24.log',
                        '/training data/CSG-1_10.28.202.116.log',
                        '/training data/CSG-2_10.28.202.117.log',
                        '/training data/CSG-3_10.28.202.122.log',
                        '/training data/CSG-4_10.28.202.115.log',
                        '/training data/CSG-5_10.28.202.118.log',
                        '/training data/CSG-6_10.28.202.114.log',
                        '/training data/CSG-7_10.28.202.113.log',
                        '/training data/CSG-8_10.28.202.112.log',
                        '/training data/CSG-9_10.28.202.111.log',
                        '/training data/CSG-10_10.28.202.110.log',
                        '/training data/CSG-11_10.28.202.125.log',
                        '/training data/CSG-12_10.28.202.109.log',
                        '/training data/CSG-13_10.28.202.126.log']

    train_bDataAll = load_B_data(bDataPath, train_with_id, train_without_id)  # 读取B训练集

    test_with_id = ["/test data/ASG-1.log", "/test data/ASG-2.log"]
    test_without_id = ["/test data/CSG-1.log", "/test data/CSG-2.log", "/test data/CSG-3.log",
                       "/test data/CSG-4.log", "/test data/CSG-5.log", "/test data/CSG-6.log",
                       "/test data/CSG-7.log"]
    test_bDataAll = load_B_data(bDataPath, test_with_id, test_without_id)   # 读取B测试集
    # B数据集的处理
    train_bDataAll.drop(columns=['cleared', 'code', 'levelLabel', 'occurred', 'sent', 'time2', 'type'], inplace=True)
    test_bDataAll.drop(columns=['ID', 'cleared', 'code', 'code1', 'levelLabel', 'occurred', 'sent', 'time2', 'type'],
                       inplace=True)

    # 因为B数据集没有Brief，把description进行提炼，去除数字，剔除[]里面的内容等。
    train_bDataAll, test_bDataAll = create_brief(train_bDataAll, test_bDataAll)

    train_bDataAll_pro, test_bDataAll_pro = threshold_alarm(train_bDataAll, test_bDataAll)  # 过滤告警
    bResult = get_result(test_bDataAll_pro, "B")
    bResult = bResult.add_suffix("-B")
    result = pd.concat([aResult, bResult], axis=1)
    result.to_csv("result.csv",index=0)