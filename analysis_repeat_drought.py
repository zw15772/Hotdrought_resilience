
from global_init import *
# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intra_CV_anaysis import Check_data
from meta_info import *
from pprint import pprint
mpl.use('TkAgg')

# this_root='/Users/liyang/Projects_data/Hotdrought_Resilience/'
# data_root = join(this_root,'data')
# results_root = join(this_root,'results')
# temp_root = join(this_root,'temp')

result_root_this_script = join(results_root, 'analysis_repeat_drought')

class Pick_drought_events_year:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_year', result_root_this_script, mode=2)

        pass

    def run(self):
        # 1 pick events in annual scale
        # self.tif_to_spatial_dict()
        # self.pick_year_events()
        # 2 pick repeated events
        # self.pick_repeated_annual_events()
        self.tif_to_spatial_dict_cru_temp()
        self.find_hot_years()

        pass

    def pick_year_events(self):
        # threshold = -1.5
        outdir = join(self.this_class_arr,'picked_events')
        T.mk_dir(outdir)
        # fdir = data_root + 'SPEI12/per_pix/'
        spi_12_dir = join(data_root,'SPI/per_pix/spi12')
        SPI_spatial_dict = T.load_npy_dir(spi_12_dir)
        events_dic = {}
        params_list = []
        spatial_dic = {}
        for pix in tqdm(SPI_spatial_dict):
            # spatial_dic[pix] = 1
            # continue
            vals = SPI_spatial_dict[pix]
            threshold = -2
            params = (vals,threshold)
            events_list = self.kernel_find_drought_period(params)
            if len(events_list) == 0:
                continue
            drought_year_list = []
            for drought_range in events_list:
                # drought_val = T.pick_vals_from_1darray(vals,drought_range)
                # plt.plot(drought_range,drought_val,lw=4)
                # plt.scatter(drought_range,drought_val,s=10,c='r',zorder=99)
                min_index = T.pick_min_indx_from_1darray(vals,drought_range)
                min_index_gs = min_index % 12 + 1
                drought_year = min_index // 12
                drought_year_list.append(drought_year)
            events_dic[pix] = drought_year_list
            spatial_dic[pix] = len(drought_year_list)
            # print(pix,drought_year_list)
            # plt.show()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # T.save_npy(events_dic,outdir+'drought_year_events_2sd')
        outf = join(outdir,'drought_years')
        T.save_npy(events_dic,outf)
        pass

    def pick_repeated_annual_events(self):
        events_f = join(self.this_class_arr,'picked_events/drought_years.npy')
        outdir = join(self.this_class_arr,'pick_repeated_annual_events')
        outf = join(outdir,'repeated_events_annual.npy')
        T.mk_dir(outdir)
        events_dic = T.load_npy(events_f)
        spatial_dic = {}
        for pix in tqdm(events_dic):
            events = events_dic[pix]
            if len(events) <= 1:
                continue
            # spatial_dic[pix] = len(events)
            events = list(set(events))
            events.sort()
            repeated_events = self.kernel_pick_repeat_events(events,39)

            if len(repeated_events) == 0:
                continue
            # print(repeated_events)
            # spatial_dic[pix] = len(repeated_events)
            spatial_dic[pix] = repeated_events
        T.save_npy(spatial_dic,outf)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()

    def kernel_pick_repeat_events(self,events,year_num):
        # year_num = 16
        # events = [2,3,5,8,10]
        # events = [3,4,5,6]
        events_mark = []
        for i in range(year_num):
            if i in events:
                events_mark.append(1)
            else:
                events_mark.append(0)
        window = 4
        # print(events)
        events_list = []
        for i in range(len(events_mark)):
            select = events_mark[i:i+window]
            if select[0] == 0:
                continue
            if select[-1] == 0 and select.count(1) == 3:
                continue
            build_list = list(range(i,i+window))
            select_index = []
            for j in build_list:
                if j in events:
                    select_index.append(j)
            if not select_index[-1] - select_index[0] >= 2:
                continue
            # 前两年不能有干旱事件
            if select_index[0] - 1 in events:
                continue
            if select_index[0] - 2 in events:
                continue
            # 后两年不能有干旱事件
            if select_index[-1] + 1 in events:
                continue
            if select_index[-1] + 2 in events:
                continue
            if len(select_index) == 4:
                continue
            if select_index[0] - 2 < 0:
                continue
            if select_index[-1] + 2 >= year_num:
                continue
            events_list.append(select_index)
        # print(events_list)
        # exit()
        return events_list


    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        threshold = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:# SPEI
                drought_month.append(i)
            else:
                drought_month.append(-99)
        # plt.plot(drought_month)
        # plt.show()
        events = []
        event_i = []
        for ii in drought_month:
            if ii > -99:
                event_i.append(ii)
            else:
                if len(event_i) > 0:
                    events.append(event_i)
                    event_i = []
                else:
                    event_i = []

        flag = 0
        events_list = []
        # 不取两个端点
        for i in events:
            # 去除两端pdsi值小于-0.5
            if 0 in i or len(pdsi) - 1 in i:
                continue
            new_i = []
            for jj in i:
                new_i.append(jj)
            # print(new_i)
            # exit()
            flag += 1
            vals = []
            for j in new_i:
                try:
                    vals.append(pdsi[j])
                except:
                    print(j)
                    print('error')
                    print(new_i)
                    exit()
            # print(vals)

            # if 0 in new_i:
            # SPEI
            min_val = min(vals)
            if min_val < -99999:
                continue
            if min_val < threshold:
                level = 4
            else:
                level = 0

            events_list.append(new_i)
        return events_list


    def tif_to_spatial_dict_cru_temp(self):
        spi_tif_dir = join(data_root, 'CRU_temp/tif')
        outdir = join(data_root, 'CRU_temp/per_pix')
        T.mk_dir(outdir, True)
        Pre_Process().data_transform(spi_tif_dir, outdir)

    def find_hot_years(self):
        cru_temp_per_pix_dir = join(data_root, 'CRU_temp/per_pix')
        outdir = join(self.this_class_arr,'hot_years')
        T.mk_dir(outdir,force=True)
        temp_spatial_dict = T.load_npy_dir(cru_temp_per_pix_dir)
        year_range = global_year_range_list
        year_range = np.array(year_range,dtype=int)

        hot_year_spatial_dict = {}

        for pix in tqdm(temp_spatial_dict):
            vals = temp_spatial_dict[pix]
            vals[vals<-999] = np.nan
            vals[vals>99999] = np.nan
            if T.is_all_nan(vals):
                continue
            vals_reshape = np.reshape(vals,(-1,12))
            vals_annual_mean = np.nanmean(vals_reshape,axis=1)
            vals_annual_mean_detrend = T.detrend_vals(vals_annual_mean)
            T_quantile = np.percentile(vals_annual_mean_detrend, 90)
            hot_index_True_False = vals_annual_mean_detrend > T_quantile
            hot_years = year_range[hot_index_True_False]
            hot_year_spatial_dict[pix] = hot_years
        outf = join(outdir,'hot_years.npy')
        T.save_npy(hot_year_spatial_dict,outf)

    def tif_to_spatial_dict(self):
        spi_tif_dir = join(data_root,'SPI/tif/1982-2020/spi12')
        outdir = join(data_root,'SPI/per_pix/spi12')
        T.mk_dir(outdir,True)
        Pre_Process().data_transform(spi_tif_dir,outdir)

class Pick_multi_year_drought_events_year:

    def __init__(self):
        self.outdir = r'F:\Hotdrought_Resilience\results\analysis_multi_year_drought\\'
        T.mk_dir(self.outdir, True)

        pass
    def run(self):
        self.pick_multiyear_drought_events_year()
        self.add_temp_during_drought()
        # self.add_NDVI_min_mean_during_drought()
        self.add_total_NDVI_during_and_post_drought()
        #
        # #
        self.add_hot_drought()
        # self.generation_drought_character_df()


    def pick_multiyear_drought_events_year(self):
        # 载入数据
        spi_12_dir = join(data_root, 'SPI/per_pix/spi12')
        SPI_dict = T.load_npy_dir(spi_12_dir)
        years = np.arange(1982, 2021)

        df_droughts = self.detect_multiyear_droughts(
            SPI_dict=SPI_dict,
            years=years,
            drought_threshold=-2,
            min_duration=2,

        )
        outdir=self.outdir+'Dataframe\\'
        T.mk_dir(outdir)
        outpath = outdir + 'multiyear_droughts.df'

        T.save_df(df_droughts, outpath)
        T.df_to_excel(df_droughts, outpath)

    def detect_multiyear_droughts(self, SPI_dict, years, drought_threshold, min_duration,):
        """
        识别每个像元的 multi-year drought 事件并提取属性

        Parameters
        ----------
        SPI_dict : dict
            {pix: np.array of SPI values, shape = (n_years, 12)}
        years : list or np.array
            年份列表，对应 SPI 的第0维
        drought_threshold : float
            定义干旱阈值（SPI < threshold）
        min_duration : int
            定义 multi-year drought 最少持续年数
        recovery_gap : int
            干旱结束后，若 recovery_gap 年内再次干旱，则视为未恢复（排除该事件）
        """

        result_records = []



        for pix, spi_2d in tqdm(SPI_dict.items(), desc="Detecting multiyear droughts"):
            vals=SPI_dict[pix]
            vals[vals<-999] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            vals_reshape = np.reshape(vals,(-1,12))
            spi_2d = np.array(vals_reshape, dtype=float)  # shape: (n_years, 12)
            if spi_2d.ndim != 2 or spi_2d.shape[0] != len(years):
                continue


            # === Step 1: 年度 SPI ===
            spi_annual = np.nanmin(spi_2d, axis=1)

            # === Step 2: 干旱标记 ===
            drought_mask = spi_annual < drought_threshold

            # === Step 3: 连续干旱段识别 ===
            start, end = np.nan, np.nan
            events = []
            for i, is_drought in enumerate(drought_mask):
                if is_drought:
                    if np.isnan(start):
                        start = i
                    end = i
                else:
                    if not np.isnan(start):
                        events.append((int(start), int(end)))
                        start, end = np.nan, np.nan
            if not np.isnan(start):
                events.append((int(start), int(end)))

            # === Step 4: 筛选多年代干旱 ===
            multiyear_events = []
            for (s, e) in events:
                duration = e - s + 1
                if duration >= min_duration:
                    multiyear_events.append((s, e))

            # === Step 5: 记录事件 ===
            for (s, e) in multiyear_events:
                sub_spi = spi_2d[s:e + 1, :]
                min_idx = np.nanargmin(sub_spi)
                min_val = np.nanmin(sub_spi)
                year_idx, month_idx = np.unravel_index(min_idx, sub_spi.shape)
                min_year = years[s + year_idx]
                min_month = month_idx + 1

                drought_years = [int(y) for y in years[s:e + 1]]
                # === Step 6: 干旱严重度 ===

                sub_spi = spi_annual[s:e + 1]
                print(sub_spi)
                if len(sub_spi) > 0:
                    severity = np.nansum(np.abs(sub_spi))
                else:
                    severity = np.nan

                # === Step 6.5: 干旱后4年的 SPI 平均值 ===
                post_idx = [e + j for j in range(1, 5) if (e + j) < len(spi_annual)]
                if len(post_idx) > 0:
                    post_mean_spi = np.nanmean(spi_annual[post_idx])
                else:
                    post_mean_spi = np.nan

                record = {
                    "pix": pix,
                    "drought_years": drought_years,
                    "SPI_min": float(min_val),
                    "SPI_min_year": int(min_year),
                    "SPI_min_month": int(min_month),
                    "duration": len(drought_years),
                    "Drought_severity": float(severity),
                    "Post4yr_mean_SPI": float(post_mean_spi),
                }
                result_records.append(record)


            # === 输出 ===
        df_out = pd.DataFrame(result_records)
        T.save_df(df_out, self.outdir + 'clean_events.df')
        return df_out



    def add_temp_during_drought(self):
        dff=join(self.outdir,'Dataframe/multiyear_droughts.df')

        temp_dic_path=join(data_root,r'CRU_temp\annual_growth_season_temp_detrend_zscore_10degree')


        # === 1. 读取数据 ===
        df = T.load_df(dff)
        temp_dic = T.load_npy_dir(temp_dic_path)

        # === 2. 存放结果 ===
        max_temp_list = []
        max_temp_year_list = []

        mean_temp_list = []

        # === 2. 初始化结果列表 ===


        # === 3. 假设数据从1982年开始 ===
        base_year = 1982

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_years = row['drought_years']

            # 转换字符串 '[2002, 2003, 2004]' 为列表
            if isinstance(drought_years, str):
                try:
                    drought_years = eval(drought_years)
                except:
                    drought_years = []

            # 没有温度数据
            if pix not in temp_dic:
                max_temp_list.append(np.nan)
                max_temp_year_list.append(np.nan)
                mean_temp_list.append(np.nan)
                continue

            temp_arr = np.array(temp_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(temp_arr))

            # 筛选干旱年份对应的索引
            drought_indices = [np.where(all_years == y)[0][0] for y in drought_years if y in all_years]

            if len(drought_indices) == 0:
                max_temp_list.append(np.nan)
                max_temp_year_list.append(np.nan)
                mean_temp_list.append(np.nan)
                continue

            drought_temps = temp_arr[drought_indices]

            # === 计算指标 ===
            max_temp = np.nanmax(drought_temps)
            max_idx = np.nanargmax(drought_temps)
            max_temp_year = drought_years[max_idx]
            mean_temp = np.nanmean(drought_temps)

            # === 添加结果 ===
            max_temp_list.append(max_temp)
            max_temp_year_list.append(max_temp_year)
            mean_temp_list.append(mean_temp)


        # === 4. 添加到 DataFrame ===
        df['max_temp'] = max_temp_list
        df['max_temp_year'] = max_temp_year_list

        df['mean_temp'] = mean_temp_list


        outf=dff
        T.save_df(df,outf)
        T.df_to_excel(df,outf,random=True)


    def add_NDVI_min_mean_during_drought(self):

        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')

        temp_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend_relative_change')

        # === 1. 读取数据 ===
        df = T.load_df(dff)
        temp_dic = T.load_npy_dir(temp_dic_path)

        # === 2. 存放结果 ===
        min_NDVI_list = []
        min_NDVI_year_list = []

        mean_NDVI_list = []

        # === 2. 初始化结果列表 ===

        # === 3. 假设数据从1982年开始 ===
        base_year = 1982

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_years = row['drought_years']

            # 转换字符串 '[2002, 2003, 2004]' 为列表
            if isinstance(drought_years, str):
                try:
                    drought_years = eval(drought_years)
                except:
                    drought_years = []

            # 没有NDVI数据
            if pix not in temp_dic:
                min_NDVI_list.append(np.nan)
                min_NDVI_year_list.append(np.nan)
                mean_NDVI_list.append(np.nan)
                continue

            temp_arr = np.array(temp_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(temp_arr))

            # 筛选干旱年份对应的索引
            drought_indices = [np.where(all_years == y)[0][0] for y in drought_years if y in all_years]

            if len(drought_indices) == 0:
                min_NDVI_list.append(np.nan)
                min_NDVI_year_list.append(np.nan)
                mean_NDVI_list.append(np.nan)
                continue

            drought_temps = temp_arr[drought_indices]

            # === 计算指标 ===
            min_NDVI = np.nanmin(drought_temps)
            min_NDVI_year=drought_years[np.nanargmin(drought_temps)]


            mean_NDVI = np.nanmean(drought_temps)

            # === 添加结果 ===
            min_NDVI_list.append(min_NDVI)
            min_NDVI_year_list.append(min_NDVI_year)

            mean_NDVI_list.append(mean_NDVI)

        # === 4. 添加到 DataFrame ===
        df['min_NDVI'] = min_NDVI_list
        df['min_NDVI_year'] = min_NDVI_year_list

        df['mean_NDVI'] = mean_NDVI_list


        outf = dff
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)

    pass

    def add_total_NDVI_during_and_post_drought(self):

        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')
        temp_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend')

        # === 1. 读取数据 ===
        df = T.load_df(dff)
        ndvi_dic = T.load_npy_dir(temp_dic_path)

        # === 2. 初始化结果 ===

        rt_list = []
        post1_list, post2_list, post3_list, post4_list = [], [], [], []
        drought_len_list = []

        base_year = 1982

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_years = row['drought_years']
            print(pix, drought_years)

            # --- 转换 drought_years ---
            if isinstance(drought_years, str):
                try:
                    drought_years = eval(drought_years)
                except:
                    drought_years = []
            elif not isinstance(drought_years, list):
                drought_years = []

            if len(drought_years) == 0:
                rt_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            # --- NDVI 缺失 ---
            if pix not in ndvi_dic:
                rt_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            ndvi_arr = np.array(ndvi_dic[pix], dtype=float)
            NDVI_average = np.nanmean(ndvi_arr)
            all_years = np.arange(base_year, base_year + len(ndvi_arr))

            # --- 提取干旱期 ---
            drought_indices = [np.where(all_years == y)[0][0] for y in drought_years if y in all_years]
            if len(drought_indices) == 0:
                rt_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            ndvi_drought = ndvi_arr[drought_indices]
            total_drought = np.nanmean(ndvi_drought)
            drought_len = len(drought_indices)
            drought_end_year = max(drought_years)

            # --- 提取干旱后 1–4 年 ---
            post_vals = []
            for offset in [1, 2, 3, 4]:
                year_target = drought_end_year + offset
                if year_target in all_years:
                    idx = np.where(all_years == year_target)[0][0]
                    post_vals.append(ndvi_arr[idx])
                else:
                    post_vals.append(np.nan)

            # === 累积 ===
            post1 = post_vals[0]
            post2 = np.nanmean(post_vals[:2])  # 1 + 2
            post3 = np.nanmean(post_vals[:3])  # 1 + 2 + 3
            post4 = np.nanmean(post_vals[:4])  # 1 + 2 + 3 + 4


            def safe_ratio(a, b):
                if np.isnan(a) or np.isnan(b) or b == 0:
                    return np.nan
                return a / b

            post1_ratio = safe_ratio(post1, NDVI_average)
            post2_ratio = safe_ratio(post2, NDVI_average)
            post3_ratio = safe_ratio(post3, NDVI_average)
            post4_ratio = safe_ratio(post4, NDVI_average)
            rt_ratio=safe_ratio(total_drought, NDVI_average)


            # === 存储结果 ===
            rt_list.append((rt_ratio))
            post1_list.append(post1_ratio)
            post2_list.append(post2_ratio)
            post3_list.append(post3_ratio)
            post4_list.append(post4_ratio)
            drought_len_list.append(drought_len)

        # === 3. 写入 DataFrame ===
        df["NDVI_rt"] = rt_list
        df["NDVI_post1_rs"] = post1_list
        df["NDVI_post2_rs"] = post2_list
        df["NDVI_post3_rs"] = post3_list
        df["NDVI_post4_rs"] = post4_list


        # === 4. 保存 ===
        outf=join(self.outdir,'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)

    def safe_divide(self,a, b):
        """安全除法：当分母为0或NaN时返回np.nan"""
        if np.isnan(a) or np.isnan(b) or abs(b) < 1e-8:
            return np.nan
        return abs(a / b)

    def add_post_NDVI_to_df(self):
        """
        向干旱事件表中添加 NDVI 恢复期（post-drought）数据：
            - post1_NDVI: 干旱结束后1年的NDVI
            - post2_NDVI: 干旱结束后2年的NDVI
            - post12_NDVI_mean: post1和post2的平均
        """

        # === 1. 读取数据 ===
        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')

        NDVI_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend_relative_change')

        df = T.load_df(dff)
        ndvi_dic = T.load_npy_dir(NDVI_dic_path)

        # === 2. 初始化结果列表 ===
        post1_mean_list,post12_mean_list,post123_mean_list, post1234_mean_list = [], [], [],[]

        base_year = 1982  # NDVI起始年份

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_years = row['drought_years']

            # 把字符串转换成list
            if isinstance(drought_years, str):
                try:
                    drought_years = eval(drought_years)
                except:
                    drought_years = []

            # 没有NDVI数据的像元
            if pix not in ndvi_dic:
                post1_mean_list.append(np.nan)

                post12_mean_list.append(np.nan)
                post123_mean_list.append(np.nan)
                post1234_mean_list.append(np.nan)
                continue

            ndvi_arr = np.array(ndvi_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(ndvi_arr))

            if len(drought_years) == 0:
                post1_mean_list.append(np.nan)

                post12_mean_list.append(np.nan)
                post123_mean_list.append(np.nan)
                post1234_mean_list.append(np.nan)
                continue

            # === 干旱结束年份 ===
            drought_end_year = max(drought_years)

            # === post1 和 post2 年份 ===
            post1_year = drought_end_year + 1
            post2_year = drought_end_year + 2
            post3_year=drought_end_year+3
            post4_year=drought_end_year+4

            # 检查年份是否在范围内
            if post1_year not in all_years:
                post1_ndvi = np.nan
            else:
                post1_ndvi = ndvi_arr[np.where(all_years == post1_year)[0][0]]

            if post2_year not in all_years:
                post2_ndvi = np.nan
            else:
                post2_ndvi = ndvi_arr[np.where(all_years == post2_year)[0][0]]

            if post3_year not in all_years:
                post3_ndvi = np.nan
            else:
                post3_ndvi = ndvi_arr[np.where(all_years == post3_year)[0][0]]

            if post4_year not in all_years:
                post4_ndvi = np.nan
            else:
                post4_ndvi = ndvi_arr[np.where(all_years == post4_year)[0][0]]

            # === 平均值 ===
            post1_mean = np.nanmean([post1_ndvi])
            post12_mean = np.nanmean([post1_ndvi, post2_ndvi])
            post123_mean = np.nanmean([post1_ndvi, post2_ndvi, post3_ndvi])
            post1234_mean = np.nanmean([post1_ndvi, post2_ndvi, post3_ndvi, post4_ndvi])


            post12_mean_list.append(post12_mean)
            post1_mean_list.append(post1_mean)
            post123_mean_list.append(post123_mean)
            post1234_mean_list.append(post1234_mean)
        # === 3. 添加到表中 ===
        df['post1_NDVI'] = post1_mean_list

        df['post12_NDVI'] = post12_mean_list
        df['post123_NDVI'] = post123_mean_list
        df['post1234_NDVI'] = post1234_mean_list

        # === 4. 保存结果 ===
        outf=join(self.outdir,'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)

    def add_hot_drought(self):
        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')

        df = T.load_df(dff)


        df['drought_type'] = np.where(df['max_temp'] > 1, 'Hot', 'Normal')

        outf = join(self.outdir, 'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)


    def generation_drought_character_df(self):

        import ast
        dff=join(self.outdir,'Dataframe/multiyear_droughts.df')


        df = T.load_df(dff)

        all_records = []

        # === Step 1. group by pix ===
        grouped = df.groupby('pix')

        for pix, group in tqdm(grouped, total=len(grouped)):
            durations = []
            hot_count, normal_count = 0, 0

            for _, row in group.iterrows():
                # --- drought years ---
                drought_years_raw = row.get("drought_years", [])
                if isinstance(drought_years_raw, str):
                    try:
                        drought_years = ast.literal_eval(drought_years_raw)
                    except:
                        drought_years = []
                elif isinstance(drought_years_raw, list):
                    drought_years = drought_years_raw
                else:
                    drought_years = []

                dur = len(drought_years)
                if dur > 0:
                    durations.append(dur)

                # --- drought type ---
                dtype = str(row.get("drought_type", "")).lower()
                if "hot" in dtype:
                    hot_count += 1
                elif "normal" in dtype:
                    normal_count += 1

            # --- aggregate ---
            if len(durations) == 0:
                mean_dur = np.nan
                max_dur = np.nan
            else:
                mean_dur = np.mean(durations)
                max_dur = np.max(durations)

            all_records.append({
                "pix": pix,
                "multi_drought_count": len(group),
                "hot_drought_count": hot_count,
                "normal_drought_count": normal_count,
                "mean_duration": mean_dur,
                "max_duration": max_dur
            })

        df_out = pd.DataFrame(all_records)
        outpath = join(self.outdir, 'Dataframe/drought_characterastic.df')
        T.save_df(df_out, outpath)
        T.df_to_excel(df_out, outpath.replace(".df", ".xlsx"))

class check_Data():
    ## pix 35 lat and lon -104
    def __init__(self):
        pass
    def run(self):
        self.check_dic_time_series()
        pass

    def check_dic_time_series(self):
        fdir_SPI=data_root+rf'\SPI\dic\spi12\\'
        dic_spi=T.load_npy_dir(fdir_SPI)

        NDVI_fdir=data_root+rf'\NDVI4g\annual_growth_season_NDVI_detrend_relative_change\\'
        NDVI_dic=T.load_npy_dir(NDVI_fdir)

        clean_events=np.load(results_root+rf'\\analysis_multi_year_drought\\Dataframe\\clean_events.npy',allow_pickle=True).tolist()

        for pix in dic_spi.keys():
            if pix not in NDVI_dic:
                continue


            lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
            # if abs(lon + 104) > 0.05:
            #     continue
            # if abs(lat - 35) > 0.05:
            #     continue
            # if abs(lon + 105) > 0.05 or abs(lat - 43) > 0.05:
            #     continue
            spi = np.array(dic_spi[pix], dtype=float).ravel()

            # reshape 并计算年均
            spi_reshape = spi.reshape(39, -1)
            spi_annual_average = np.nanmin(spi_reshape, axis=1)
            NDVI_annual=np.array(NDVI_dic[pix], dtype=float).ravel()
            years=np.arange(1982,2021)

            fig, ax1 = plt.subplots(figsize=(9, 5))

            # === 左轴：SPI ===
            ax1.plot(years ,spi_annual_average, '-', color='royalblue', label='SPI (annual)')
            ax1.axhline(-1.5, color='red', linestyle='--', linewidth=1.2, label='SPI = -1')
            ax1.set_ylabel("SPI", color='royalblue', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='royalblue')
            ax1.set_ylim(-3, 3)

            # === 右轴：NDVI ===
            ax2 = ax1.twinx()
            ax2.plot(years, NDVI_annual, '-', color='forestgreen', label='NDVI')
            ax2.set_ylabel("NDVI", color='forestgreen', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='forestgreen')

            # === 标注干净 multi-drought 年份 ===
            pix_events = [e for e in clean_events if e["pix"] == pix]

            # 初始化 drought_years，防止未定义
            drought_years = []

            if pix_events:
                drought_years = sorted({y for e in pix_events for y in e["drought_years"]})
                print(f"{pix} 的干旱年份: {drought_years}")
            else:
                print(f"{pix} 没有检测到事件")

            # 找出这些年份在序列中的索引
            if len(drought_years) > 0:
                idxs = [np.where(years == y)[0][0] for y in drought_years if y in years]

                ax1.scatter(years[idxs], spi_annual_average[idxs],
                            s=120, facecolors='none', edgecolors='orange',
                            linewidths=2, label='NDVI (clean multi-drought years)')

            # === 图例与标题 ===
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=False)

            ax1.set_xlabel("Year")
            title = f"SPI–NDVI Time Series"
            if pix is not None:
                title += f" | Pixel {pix}"
            plt.title(title, fontsize=13)
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()

class Dataframe:

    def __init__(self):
        self.result_root = results_root + r'\\analysis_multi_year_drought\\Dataframe\\'
        self.dff = join(self.result_root, f'multiyear_droughts.df')
        # self.dff = join(self.result_root, 'drought_characterastic.df')
        pass

    def run(self):
        # self.gen_events_df()
        df = self.__load_df()
        # df = self.add_first_drought_NDVI_anomaly(df)
        # df = self.add_second_drought_NDVI_anomaly(df)

        df=self.add_aridity_to_df(df)
        df = self.add_MODIS_LUCC_to_df(df)
        df = self.add_koppen_to_df(df)
        df=self.add_landcover_data_to_df(df)
        df=self.add_landcover_classfication_to_df(df)

        # #
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

        # self.check_df(df)

        pass


    def gen_events_df(self):
        if isfile(self.dff):
            print('file exsited')
            pause()

            print('file exsited')
            pause()

            print('file exsited')
            pause()
        start_year = global_start_year
        outdir = join(self.this_class_arr,'Dataframe')
        T.mkdir(outdir)
        events_f = join(Pick_drought_events_year().this_class_arr,'pick_repeated_annual_events/repeated_events_annual.npy')
        events_dic = T.load_npy(events_f)
        hot_year_f = join(Pick_drought_events_year().this_class_arr,'hot_years/hot_years.npy')
        hot_year_dic = T.load_npy(hot_year_f)
        pix_list = []
        event_list = []
        first_event_list = []
        second_event_list = []
        post_list = []
        spell_list = []
        spell_len_list = []
        repeat_mode_list = []
        is_hot_drought_list = []
        for pix in tqdm(events_dic,desc='add_events_to_df'):
            events = events_dic[pix]
            hot_years = hot_year_dic[pix]
            # print('hot_years',hot_years)
            # exit()
            if len(events) == 0:
                continue
            for event in events:
                pix_list.append(pix)
                event_list.append(np.array(event)+start_year)
                if len(event) == 2:
                    first_event = tuple([event[0]])
                    second_event = tuple([event[1]])
                    spell = tuple(range(first_event[0]+1,second_event[0]))
                    if event[1] - event[0] == 2:
                        repeat_mode = 'DSD'
                    elif event[1] - event[0] == 3:
                        repeat_mode = 'DSSD'
                    else:
                        raise ValueError
                elif len(event) == 3:
                    e1 = event[0]
                    e2 = event[1]
                    e3 = event[2]
                    if e2 - e1 == 1:
                        first_event = (e1,e2)
                        second_event = tuple([e3])
                        repeat_mode = 'DDSD'
                    else:
                        first_event = tuple([e1])
                        second_event = (e2, e3)
                        repeat_mode = 'DSDD'
                    spell = tuple(range(first_event[-1] + 1, second_event[0]))
                else:
                    raise UserWarning

                hot = False
                # print('first_event',np.array(first_event)+start_year)
                for year in (np.array(first_event)+start_year):
                    if year in hot_years:
                        hot = True
                        break
                if hot == True:
                    is_hot_drought_list.append('Hot-drought')
                else:
                    is_hot_drought_list.append('Normal-drought')

                first_event_list.append(np.array(first_event)+start_year)
                second_event_list.append(np.array(second_event) + start_year)
                spell_list.append(np.array(spell)+start_year)
                spell_len_list.append(len(spell))
                repeat_mode_list.append(repeat_mode)
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_year'] = event_list
        df['first_drought'] = first_event_list
        df['first_drought_type'] = is_hot_drought_list
        df['spell'] = spell_list
        df['spell_length'] = spell_len_list
        df['second_drought'] = second_event_list
        df['repeat_mode'] = repeat_mode_list
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)


    def check_df(self,df):
        df = df.dropna(how='any')
        T.print_head_n(df)
        DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        plt.show()
        pass

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df

    def add_MODIS_LUCC_to_df(self, df):
        f=data_root+'/Basedata/MODIS_LUCC_resample_05.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = f.split('.')[0]
        print(f_name)
        val_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['MODIS_LUCC'] = val_list
        return df

    def add_koppen_to_df(self, df):
        f = data_root + '/Basedata/Koeppen_reclassification.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
        f_name = f.split('.')[0]
        print(f_name)
        val_list = []

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)
        df['Koppen'] = val_list
        return df


    def add_landcover_data_to_df(self, df):

        f = data_root + rf'/Basedata/glc2000_v1_1_05_deg_unify.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        f_name = f.split('.')[0]
        print(f_name)

        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            val_list.append(vals)

        df['landcover_GLC'] = val_list
        return df
    def add_landcover_classfication_to_df(self, df):

        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            landcover=row['landcover_GLC']
            if landcover==0 or landcover==4:
                val_list.append('Evergreen')
            elif landcover==2 or landcover==3 or landcover==5:
                val_list.append('Deciduous')
            elif landcover==6:
                val_list.append('Mixed')
            elif landcover==11 or landcover==12:
                val_list.append('Shrub')
            elif landcover==13 or landcover==14:
                val_list.append('Grass')
            elif landcover==16 or landcover==17 or landcover==18:
                val_list.append('Cropland')
            elif landcover==19 :
                val_list.append('Bare')
            else:
                val_list.append(-999)
        df['landcover_classfication']=val_list

        return df


    def add_aridity_to_df(self,df):  ## here is original aridity index not classification

        f=data_root+rf'/Basedata/aridity_index.tif'

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)

        val_dic = DIC_and_TIF().spatial_arr_to_dic(array)

        # val_array = np.load(fdir + f)

        # val_dic = DIC_and_TIF().spatial_arr_to_dic(val_array)
        f_name='Aridity'
        print(f_name)
        val_list=[]
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix=row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            val=val_dic[pix]
            if val<-99:
                val_list.append(np.nan)
                continue
            val_list.append(val)
        df[f'{f_name}']=val_list

        return df

    def drop_field_df(self, df):
        for col in df.columns:
            print(col)
        # exit()
        df = df.drop(columns=[


                              'max_temp_month',





                              ])
        return df

class PLOT_multi_year_drought_characteristic():
    def __init__(self):

        self.result_root=r'F:\Hotdrought_Resilience\results\\analysis_multi_year_drought\\'
        self.dff=join(self.result_root,'Dataframe/drought_characterastic.df')
        self.outdir=join(self.result_root,'PLOT_drought_characteristic')
        T.mk_dir(self.outdir,True)

        print(self.result_root)
        pass
    def run(self):
        self.plot_spatial_map_multidrought_count()
        self.plot_spatial_map_multidrought_duration()
        self.plot_hot_drought_vs_cold_drought_ratio()



        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['MODIS_LUCC'] != 12]
        df = df[df['Koppen'] != 5]
        df = df[df['landcover_classfication'] != 'Bare']

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def plot_spatial_map_multidrought_count(self):

        df = T.load_df(self.dff)
        print(len(df))
        df = self.df_clean(df)
        print(len(df))
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            val = row['multi_drought_count']
            spatial_dic[pix] = val



        array = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir = join(self.outdir, 'tiff')
        T.mk_dir(outdir)

        outf = join(outdir, 'multi_drought_count.tif')
        print(outf)

        DIC_and_TIF().arr_to_tif(array, outf)

        pass

    def plot_hot_drought_vs_cold_drought_ratio(self):

        df = T.load_df(self.dff)
        print(len(df))
        df = self.df_clean(df)
        print(len(df))
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            count_hot = row['hot_drought_count']
            count_cold = row['normal_drought_count']
            ration=count_hot/(count_hot+count_cold)
            spatial_dic[pix] = ration

        array = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir = join(self.outdir, 'tiff')
        T.mk_dir(outdir)

        outf = join(outdir, 'hot_drought_ratio.tif')
        print(outf)

        DIC_and_TIF().arr_to_tif(array, outf)


        pass

    def plot_spatial_map_multidrought_duration(self):
        df = T.load_df(self.dff)
        print(len(df))
        df = self.df_clean(df)
        print(len(df))
        spatial_dic = {}
        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            val = row['max_duration']
            spatial_dic[pix] = val

        array = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir = join(self.outdir, 'tiff')
        T.mk_dir(outdir)

        outf = join(outdir, 'max_duration.tif')
        print(outf)

        DIC_and_TIF().arr_to_tif(array, outf)

        pass

        pass


class PLOT_multi_year_drought_vegetation():
    def __init__(self):

        self.result_root=r'F:\Hotdrought_Resilience\results\\analysis_multi_year_drought\\'
        self.dff=join(self.result_root,'Dataframe/multiyear_droughts.df')
        self.outdir=join(self.result_root,'PLOT_vegetation_response')
        T.mk_dir(self.outdir,True)

        print(self.result_root)
        pass
    def run(self):
        # self.plot_bar_1()
        # self.plot_NDVI_during_drought()
        # self.plot_NDVI_post_drought_vals()
        # self.plot_NDVI_post_drought_resilience()
        # self.plot_NDVI_hot_normal_during_drought_NDVI()
        # self.plot_hot_minus_normal_during_drought_NDVI()
        self.heatmap_post_drought()

        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        df=df[df['MODIS_LUCC'] != 12]
        df = df[df['Koppen'] != 5]
        df = df[df['Koppen'] > -9999]
        df = df[df['landcover_classfication'] != 'Bare']

        df = df[df['landcover_classfication'] != 'Cropland']

        return df

    def plot_bar(self):
        dff=join(self.result_root,'Dataframe/arr/Dataframe//dataframe.df')
        # dff=r'F:\Hotdrought_Resilience\results\analysis_repeat_drought\Dataframe\arr\Dataframe\dataframe.df'

        df = T.load_df(dff)
        df = self.df_clean(df)
        # df=df[df['first_drought_type']=='Normal-drought']
        # df = df[df['first_drought_type'] == 'Hot-drought']

        T.print_head_n(df)



        aridity_col = 'Aridity'

        aridity_bin = np.linspace(0, 2.5, 11)
        df_groupe1, bin_list_str = T.df_bin(df, aridity_col, aridity_bin)

        val_list_first = []

        val_list_second = []

        bin_centers = []
        name_list = []

        for name, group in df_groupe1:
            bin_left = name[0].left
            bin_right = name[0].right
            bin_centers.append((bin_left + bin_right) / 2)
            name_list.append(f"{bin_left:.2f}-{bin_right:.2f}")
            val_first = np.nanmean(group[f'first_drought_NDVI_relative_change'])
            val_second = np.nanmean(group[f'second_drought_NDVI_relative_change'])
            # val=np.nanmean(group['rs_4years'])
            val_list_first.append(val_first)
            val_list_second.append(val_second)

        x = np.arange(len(name_list))
        width = 0.35  # 一般宽度设置小一点看起来更舒服

        plt.figure(figsize=(7, 4))
        plt.bar(x - width / 2, val_list_first, width=width, color='steelblue', edgecolor='k', alpha=0.8,
                label='First drought')
        plt.bar(x + width / 2, val_list_second, width=width, color='indianred', edgecolor='k', alpha=0.8,
                label='Second drought')

        plt.xticks(x, name_list, rotation=45, ha='right', fontsize=12)
        plt.xlabel('Aridity Index', fontsize=12)
        plt.ylabel('Mean GS NDVI', fontsize=12)
        plt.yticks(fontsize=12)


        plt.ylim(-10, 2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        pass
    pass


    def plot_bar_1(self):

        dff = self.dff
        df = T.load_df(dff)
        df = self.df_clean(df)


        print(len(df))
        df = df[df['Post4yr_mean_SPI'] > -1]
        df = df[df['Aridity'] <0.65]
        # df = df[df['max_temp']>1]

        # === Step 1. 分组 duration ===
        def duration_group(x):
            if x == 2:
                return 2
            elif x == 3:
                return 3
            else:
                return 4  # 4 表示 ≥4

        df['duration_group'] = df['duration'].apply(duration_group)

        # Step 2: 计算每个 duration 下各阶段 unrecovered ratio
        threshold = .95
        summary = []

        for d in [2, 3, 4]:
            sub = df[df['duration_group'] == d]
            if len(sub) == 0:
                continue

            # 干旱年 (RT)
            rt_unrecover = np.sum(sub['NDVI_rt'] < threshold) / len(sub)

            # 干旱后 1–4 年 (RS1–RS4)
            rs1_unrecover = np.sum(sub['NDVI_post1_rs'] < threshold) / len(sub)
            rs2_unrecover = np.sum(sub['NDVI_post2_rs'] < threshold) / len(sub)
            rs3_unrecover = np.sum(sub['NDVI_post3_rs'] < threshold) / len(sub)
            rs4_unrecover = np.sum(sub['NDVI_post4_rs'] < threshold) / len(sub)


            summary.append({
                'duration_group': d,
                'rt': rt_unrecover,
                'rs1': rs1_unrecover,
                'rs2': rs2_unrecover,
                'rs3': rs3_unrecover,
                'rs4': rs4_unrecover,
                'n_events': len(sub)
            })

        df_sum = pd.DataFrame(summary)
        print(df_sum)
        # --- 绘图 ---

        labels = ['RT', 'RS1', 'RS2', 'RS3', 'RS4']
        x = np.arange(len(labels))
        width = 0.18

        plt.figure(figsize=(9, 5))

        for idx, d in enumerate([2, 3, 4, ]):
            sub = df_sum[df_sum['duration_group'] == d]
            plt.bar(x + (idx - 1.5) * width,
                    [sub['rt'].values[0], sub['rs1'].values[0], sub['rs2'].values[0], sub['rs3'].values[0],
                     sub['rs4'].values[0]],
                    width, label=f'Dur={d if d < 5 else ">4"}')

        plt.xticks(x, labels)
        plt.ylabel('Unrecovered ratio (NDVI < 0.95)')
        plt.ylim(0, 1.3)
        plt.legend(title='Duration')
        plt.title('Unrecovered ratio across drought durations and recovery years')
        plt.show()

    def plot_NDVI_during_drought(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        spatial_dic_mean={}
        spatial_dic_min={}
        spatial_dic_total={}


        df_group=T.df_groupby(df,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list_min=df_pix['min_NDVI'].tolist()
            val_list_mean=df_pix['mean_NDVI'].tolist()
            val_list_total=df_pix['NDVI_total_drought'].tolist()

            val_mean=np.nanmean(val_list_mean)
            val_min=np.nanmin(val_list_min)
            val_list_total=np.nanmean(val_list_total)
            spatial_dic_mean[pix]=val_mean
            spatial_dic_min[pix]=val_min
            spatial_dic_total[pix]=val_list_total

        array_mean=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_mean)
        array_min=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_min)
        array_total=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_total)
        outdir=join(self.outdir,'tiff')
        T.mk_dir(outdir)

        outf=join(outdir,'mean_NDVI.tif')
        DIC_and_TIF().arr_to_tif(array_mean, outf)
        print(outf)

        outf=join(outdir,'min_NDVI.tif')
        DIC_and_TIF().arr_to_tif(array_min, outf)
        print(outf)

        outf=join(outdir,'NDVI_total_drought.tif')
        DIC_and_TIF().arr_to_tif(array_total, outf)
        print(outf)

        pass



    def plot_NDVI_post_drought_vals(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        spatial_dic_post1={}
        spatial_dic_post2={}
        spatial_dic_post3={}
        spatial_dic_post4={}



        df_group=T.df_groupby(df,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list_post1=df_pix['NDVI_post1_total'].tolist()
            val_list_post2=df_pix['NDVI_post2_total'].tolist()
            val_list_post3=df_pix['NDVI_post3_total'].tolist()
            val_list_post4=df_pix['NDVI_post4_total'].tolist()

            val_post1_mean=np.nanmean(val_list_post1)

            val_post2_mean=np.nanmean(val_list_post2)
            val_post3_mean = np.nanmean(val_list_post3)
            val_post4_mean=np.nanmean(val_list_post4)

            spatial_dic_post1[pix] = val_post1_mean
            spatial_dic_post2[pix]=val_post2_mean
            spatial_dic_post3[pix] = val_post3_mean
            spatial_dic_post4[pix]=val_post4_mean

        array_post1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post1)
        array_post2 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post2)
        array_post3 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post3)
        array_post4 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post4)

        outdir = join(self.outdir, 'tiff')
        T.mk_dir(outdir)

        outf = join(outdir, 'NDVI_post1_total.tif')
        DIC_and_TIF().arr_to_tif(array_post1, outf)
        print(outf)

        outf = join(outdir, 'NDVI_post2_total.tif')
        DIC_and_TIF().arr_to_tif(array_post2, outf)
        print(outf)

        outf = join(outdir, 'NDVI_post3_total.tif')
        DIC_and_TIF().arr_to_tif(array_post3, outf)
        print(outf)

        outf = join(outdir, 'NDVI_post4_total.tif')
        DIC_and_TIF().arr_to_tif(array_post4, outf)
        print(outf)

        pass


    def plot_NDVI_post_drought_resilience(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        df=df[df['Post4yr_mean_SPI']>-2]
        # df=df[df['drought_type']=='Hot']
        # df=df[df['drought_type']=='Normal']
        spatial_dic_post1={}
        spatial_dic_post2={}
        spatial_dic_post3={}
        spatial_dic_post4={}
        spatial_dic_during_drought={}



        df_group=T.df_groupby(df,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list_during_drought=df_pix['NDVI_rt'].tolist()
            val_list_post1=df_pix['NDVI_post1_rs'].tolist()
            val_list_post2=df_pix['NDVI_post2_rs'].tolist()
            val_list_post3=df_pix['NDVI_post3_rs'].tolist()
            val_list_post4=df_pix['NDVI_post4_rs'].tolist()

            val_list_during_drought_mean=np.nanmean(val_list_during_drought)



            val_post1_mean=np.nanmean(val_list_post1)

            val_post2_mean=np.nanmean(val_list_post2)
            val_post3_mean = np.nanmean(val_list_post3)
            val_post4_mean=np.nanmean(val_list_post4)

            spatial_dic_post1[pix] = val_post1_mean
            spatial_dic_post2[pix]=val_post2_mean
            spatial_dic_post3[pix] = val_post3_mean
            spatial_dic_post4[pix]=val_post4_mean
            spatial_dic_during_drought[pix]=val_list_during_drought_mean

        array_post1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post1)
        array_post2 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post2)
        array_post3 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post3)
        array_post4 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post4)
        array_during_drought = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_during_drought)

        outdir = join(self.outdir, 'tiff','ALL')
        T.mk_dir(outdir)

        outf = join(outdir, 'NDVI_post1_rs.tif')
        DIC_and_TIF().arr_to_tif(array_post1, outf)
        print(outf)

        outf = join(outdir, 'NDVI_post2_rs.tif')
        DIC_and_TIF().arr_to_tif(array_post2, outf)
        print(outf)

        outf = join(outdir, 'NDVI_post3_rs.tif')
        DIC_and_TIF().arr_to_tif(array_post3, outf)
        print(outf)

        outf = join(outdir, 'NDVI_post4_rs.tif')
        DIC_and_TIF().arr_to_tif(array_post4, outf)
        print(outf)

        outf = join(outdir, 'NDVI_rt.tif')
        DIC_and_TIF().arr_to_tif(array_during_drought, outf)
        print(outf)

        pass




    def plot_NDVI_hot_normal_during_drought_NDVI(self):
        df=T.load_df(self.dff)
        print(len(df))
        #
        # T.print_head_n(df);exit()
        print(len(df))
        df=self.df_clean(df)
        spatial_dic={}
        df=df[df['drought_type']=='Hot']


        df_group=T.df_groupby(df,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list=df_pix['NDVI_total_drought'].tolist()
            val_mean=np.nanmean(val_list)
            spatial_dic[pix]=val_mean

        array = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir = join(self.outdir, 'tiff','hot_drought')
        T.mk_dir(outdir,True)

        outf = join(outdir, 'NDVI_total_drought.tif')
        print(outf)

        DIC_and_TIF().arr_to_tif(array, outf)

    def heatmap_post_drought(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff = rf'F:\Hotdrought_Resilience\results\analysis_multi_year_drought\Dataframe\\multiyear_droughts.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        # print(len(df))
        df = df[df['Post4yr_mean_SPI'] > -1]
        # df=df[df['duration']==3]
        df = self.df_clean(df)
        # print(len(df));exit()
        df = df[df['Aridity'] <= 0.65]

        # plt.show();exit()

        T.print_head_n(df)
        x_var = 'Drought_severity'
        y_var = 'mean_temp'
        z_var = 'NDVI_post2_rs'

        plt.hist(df[x_var])
        plt.show()
        plt.hist(df[y_var])
        plt.show()

        bin_x = np.linspace(5, 10, 7, )

        bin_y = np.linspace(-1,2, 7)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(8, 8))

        matrix_dict, x_ticks_list, y_ticks_list = self.df_bin_2d(df, val_col_name=z_var,
                                                                 col_name_x=x_var,
                                                                 col_name_y=y_var, bin_x=bin_x, bin_y=bin_y, round_x=4,
                                                                 round_y=4)
        # pprint(matrix_dict);exit()

        my_cmap = T.cmap_blend(color_list=['#000000', 'r', 'b'])
        my_cmap = 'Spectral'
        self.plot_df_bin_2d_matrix(matrix_dict, 0, 1, x_ticks_list, y_ticks_list, cmap=my_cmap,
                                   is_only_return_matrix=False)
        plt.colorbar()
        plt.xticks(rotation=45)
        plt.tight_layout()
        pprint(matrix_dict)
        # plt.show()

        matrix_dict_count, x_ticks_list, y_ticks_list = self.df_bin_2d_count(df, val_col_name=z_var,
                                                                             col_name_x=x_var,
                                                                             col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        pprint(matrix_dict_count)
        scatter_size_dict = {
            (1, 20): 5,
            (20, 50): 20,
            (50, 100): 50,
            (100, 200): 75,
            (200, 400): 100,
            (400, 800): 200,
            (800, np.inf): 250
        }
        matrix_dict_count_normalized = {}
        # Normalize counts for circle size
        for key in matrix_dict_count:
            num = matrix_dict_count[key]
            for key2 in scatter_size_dict:
                if num >= key2[0] and num < key2[1]:
                    matrix_dict_count_normalized[key] = scatter_size_dict[key2]
                    break
        pprint(matrix_dict_count_normalized)
        reverse_x = list(range(len(bin_y) - 1))[::-1]
        reverse_x_dict = {}
        for i in range(len(bin_y) - 1):
            reverse_x_dict[i] = reverse_x[i]
        # print(reverse_x_dict);exit()
        for x, y in matrix_dict_count_normalized:
            plt.scatter(y, reverse_x_dict[x], s=matrix_dict_count_normalized[(x, y)], c='gray', edgecolors='none',
                        alpha=.5)
        for x, y in matrix_dict_count_normalized:
            plt.scatter(y, reverse_x_dict[x], s=matrix_dict_count_normalized[(x, y)], c='none', edgecolors='gray',
                        alpha=1)

        plt.xlabel('Drought severity')
        plt.ylabel('Temp zscore')

        plt.show()
        # plt.savefig(outf)
        # plt.close()

    def df_bin_2d(self, df, val_col_name, col_name_x, col_name_y, bin_x, bin_y, round_x=2, round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                # print(len(df_group_x_i));exit()

                ## calculate ration and vals<1
                vals = [i for i in vals if i < 1]
                # print(vals);exit()

                rt_mean = len(vals) / len(df) * 100
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict, x_ticks_list, y_ticks_list

    def df_bin_2d_count(self, df, val_col_name, col_name_x, col_name_y, bin_x, bin_y, round_x=2, round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict, x_ticks_list, y_ticks_list

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'], observed=True)
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str

    def plot_df_bin_2d_matrix(self, matrix_dict, vmin, vmax, x_ticks_list, y_ticks_list, cmap='RdBu',
                              is_only_return_matrix=False):
        print(x_ticks_list)
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])
        # plt.colorbar()
        # plt.show()

    def plot_hot_minus_normal_during_drought_NDVI(self):
        fdir=join(self.outdir, 'tiff','normal_drought')
        sdir=join(self.outdir, 'tiff','hot_drought')
        outdir=join(self.outdir, 'tiff','hot_minus_normal_drought')
        T.mk_dir(outdir,True)

        for f in os.listdir(fdir):
            if f.endswith('.tif'):
                fpath=join(fdir,f)
                array_normal, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
                array_normal[array_normal<-999]=np.nan
                array_normal[array_normal>999]=np.nan
                spath=join(sdir,f)
                array_hot, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(spath)
                array_hot[array_hot<-999]=np.nan
                array_hot[array_hot>999]=np.nan

                array=array_hot-array_normal
                array[array<-999]=np.nan
                array[array>999]=np.nan
                array[array==0]=np.nan
                outf=join(outdir,f)
                ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, array)




def main():
    # Pick_drought_events_year().run()
    # Pick_multi_year_drought_events_year().run()
    # Dataframe().run()
    PLOT_multi_year_drought_vegetation().run()
    # PLOT_multi_year_drought_characteristic().run()
    # check_Data().run()
    pass

if __name__ == '__main__':
    main()
