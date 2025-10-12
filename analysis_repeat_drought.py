
from global_init import *
# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        # self.pick_multiyear_drought_events_year()
        # self.add_temp_during_drought()
        # self.add_NDVI_during_drought()
        self.add_post_NDVI_to_df()


    def pick_multiyear_drought_events_year(self):
        # 载入数据
        spi_12_dir = join(data_root, 'SPI/per_pix/spi12')
        SPI_dict = T.load_npy_dir(spi_12_dir)
        years = np.arange(1982, 2021)

        df_droughts = self.detect_multiyear_droughts(
            SPI_dict=SPI_dict,
            years=years,
            drought_threshold=-1.0,
            min_duration=2,
            recovery_gap=2
        )
        outdir=self.outdir+'Dataframe\\'
        T.mk_dir(outdir)
        outpath = outdir + 'multiyear_droughts.df'

        T.save_df(df_droughts, outpath)
        T.df_to_excel(df_droughts, outpath)

    def detect_multiyear_droughts(self, SPI_dict, years, drought_threshold=-2, min_duration=2, recovery_gap=2):
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



            # === Step 1: 年均 SPI12 ===
            spi_annual = np.nanmean(spi_2d, axis=1)

            # === Step 2: 干旱年标记 ===
            drought_mask = spi_annual < drought_threshold

            # === Step 3: 连续干旱段识别 ===
            start, end = None, None
            events = []
            for i, is_drought in enumerate(drought_mask):
                if is_drought:
                    if start is None:
                        start = i
                    end = i
                else:
                    if start is not None:
                        events.append((start, end))
                        start, end = None, None
            if start is not None:
                events.append((start, end))

            # === Step 4: 筛选多年代事件 ===
            for (s, e) in events:
                duration = e - s + 1
                if duration < min_duration:
                    continue

                # 检查恢复期是否干扰
                next_window = drought_mask[e + 1: e + 1 + recovery_gap]
                if np.any(next_window):  # 未来2–3年内再次干旱 → 排除
                    continue

                # === Step 5: 获取最小SPI及时间 ===
                sub_spi = spi_2d[s:e + 1, :]
                min_idx = np.nanargmin(sub_spi)
                min_val = np.nanmin(sub_spi)
                year_idx, month_idx = np.unravel_index(min_idx, sub_spi.shape)
                min_year = years[s + year_idx]
                min_month = month_idx + 1

                # === Step 6: 记录结果 ===
                drought_years = [int(y) for y in years[s:e + 1]]
                record = {
                    "pix": pix,
                    "drought_years": drought_years,
                    "SPI_min": float(min_val),
                    "SPI_min_year": int(min_year),
                    "SPI_min_month": int(min_month)
                }
                result_records.append(record)

        df_out = pd.DataFrame(result_records)
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


    def add_NDVI_during_drought(self):

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

            # 没有温度数据
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
            max_idx = np.nanargmax(drought_temps)
            max_NDVI_year = drought_years[max_idx]
            mean_NDVI = np.nanmean(drought_temps)

            # === 添加结果 ===
            min_NDVI_list.append(min_NDVI)
            min_NDVI_year_list.append(max_NDVI_year)
            mean_NDVI_list.append(mean_NDVI)

        # === 4. 添加到 DataFrame ===
        df['min_NDVI'] = min_NDVI_list
        df['min_NDVI_year'] = min_NDVI_year_list

        df['mean_NDVI'] = mean_NDVI_list


        outf = dff
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)

    pass


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
        post1_list, post2_list, post12_mean_list = [], [], []

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
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post12_mean_list.append(np.nan)
                continue

            ndvi_arr = np.array(ndvi_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(ndvi_arr))

            if len(drought_years) == 0:
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post12_mean_list.append(np.nan)
                continue

            # === 干旱结束年份 ===
            drought_end_year = max(drought_years)

            # === post1 和 post2 年份 ===
            post1_year = drought_end_year + 1
            post2_year = drought_end_year + 2

            # 检查年份是否在范围内
            if post1_year not in all_years:
                post1_ndvi = np.nan
            else:
                post1_ndvi = ndvi_arr[np.where(all_years == post1_year)[0][0]]

            if post2_year not in all_years:
                post2_ndvi = np.nan
            else:
                post2_ndvi = ndvi_arr[np.where(all_years == post2_year)[0][0]]

            # === 平均值 ===
            if np.isnan(post1_ndvi) and np.isnan(post2_ndvi):
                post12_mean = np.nan
            else:
                post12_mean = np.nanmean([post1_ndvi, post2_ndvi])

            post1_list.append(post1_ndvi)
            post2_list.append(post2_ndvi)
            post12_mean_list.append(post12_mean)

        # === 3. 添加到表中 ===
        df['post1_NDVI'] = post1_list
        df['post2_NDVI'] = post2_list
        df['post12_NDVI_mean'] = post12_mean_list

        # === 4. 保存结果 ===
        outf=join(self.outdir,'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)




class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, f'Dataframe/dataframe.df')
        pass

    def run(self):
        # self.gen_events_df()
        df = self.__load_df()
        # df = self.add_first_drought_NDVI_anomaly(df)
        # df = self.add_second_drought_NDVI_anomaly(df)
        # df = self.add_post_drought_n_years(df)
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

    def add_first_drought_NDVI_anomaly(self,df):
        fdir = join(data_root,r'NDVI4g\annual_growth_season_NDVI_relative_change')
        spatial_dict = T.load_npy_dir(fdir)
        result_dict = {}
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add NDVI anomaly'):
            pix = row['pix']
            if not pix in spatial_dict:
                continue
            vals = spatial_dict[pix]
            vals_dict = T.dict_zip(global_year_range_list,vals)
            first_drought_year_list = row['first_drought']
            drought_year_vals = []
            for year in first_drought_year_list:
                val = vals_dict[year]
                drought_year_vals.append(val)
            drought_year_vals_mean = np.nanmean(drought_year_vals)
            # result_dict[i] = {}
            result_dict[i] = drought_year_vals_mean
        df['first_drought_NDVI_relative_change'] = result_dict
        return df

    def add_second_drought_NDVI_anomaly(self,df):
        fdir = join(data_root, r'NDVI4g\annual_growth_season_NDVI_relative_change')
        spatial_dict = T.load_npy_dir(fdir)
        result_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df), desc='add NDVI anomaly'):
            pix = row['pix']
            if not pix in spatial_dict:
                continue
            vals = spatial_dict[pix]
            vals_dict = T.dict_zip(global_year_range_list, vals)
            second_drought_year_list = row['second_drought']
            drought_year_vals = []
            for year in second_drought_year_list:
                val = vals_dict[year]
                drought_year_vals.append(val)
            drought_year_vals_mean = np.nanmean(drought_year_vals)
            # result_dict[i] = {}
            result_dict[i] = drought_year_vals_mean
        df['second_drought_NDVI_relative_change'] = result_dict
        return df

    def add_post_drought_n_years(self,df):
        for n in [1,2,3,4]:
            fdir = join(data_root, r'NDVI4g\annual_growth_season_NDVI_relative_change')
            spatial_dict = T.load_npy_dir(fdir)
            result_dict = {}
            for i, row in tqdm(df.iterrows(), total=len(df), desc='add NDVI anomaly'):
                pix = row['pix']
                if not pix in spatial_dict:
                    continue
                vals = spatial_dict[pix]
                vals_dict = T.dict_zip(global_year_range_list, vals)
                second_drought_year_list = row['second_drought']
                the_last_year = second_drought_year_list[-1]
                post_drought_vals = []
                for year in range(the_last_year+1,the_last_year+n+1):
                    if not year in vals_dict:
                        post_drought_vals = []
                        break
                    val = vals_dict[year]
                    post_drought_vals.append(val)
                if len(post_drought_vals) == 0:
                    continue
                post_drought_vals_mean = np.nanmean(post_drought_vals)
                # result_dict[i] = {}
                result_dict[i] = post_drought_vals_mean
            df[f'post_drought_{n}_years_relative_change'] = result_dict
        return df

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

class PLOT_repeat_drought():
    def __init__(self):

        self.result_root=r'F:\Hotdrought_Resilience\results\analysis_repeat_drought\\'
        self.dff=join(self.result_root,'Dataframe/arr/Dataframe//dataframe.df')
        self.outdir=join(self.result_root,'PLOT')
        T.mk_dir(self.outdir,True)

        print(self.result_root)
        pass
    def run(self):
        # self.plot_bar_1()
        self.plot_Rt_Rs_spatial_map()
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
        dff=join(self.result_root,'Dataframe/arr/Dataframe//dataframe.df')
        # dff=r'F:\Hotdrought_Resilience\results\analysis_repeat_drought\Dataframe\arr\Dataframe\dataframe.df'

        df = T.load_df(dff)
        df = self.df_clean(df)
        # df=df[df['first_drought_type']=='Normal-drought']
        df = df[df['first_drought_type'] == 'Hot-drought']

        T.print_head_n(df)
        df=df[df['Aridity']>.65]
        val_first=np.nanmean(df[f'first_drought_NDVI_relative_change'])
        val_second = np.nanmean(df[f'second_drought_NDVI_relative_change'])

        plt.bar([1],[val_first],width=0.3,color='steelblue',edgecolor='k',alpha=0.8,label='First drought')
        plt.bar([2],[val_second],width=0.3,color='indianred',edgecolor='k',alpha=0.8,label='Second drought')
        plt.xticks([1,2],['First drought','Second drought'],rotation=45,ha='right',fontsize=12)
        plt.xlabel('Aridity Index', fontsize=12)
        plt.ylabel('Mean GS NDVI', fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(-10, 2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


        pass
    def plot_Rt_Rs_spatial_map(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        spatial_dic={}

        df_hot=df[df['first_drought_type'] == 'Hot-drought']
        df_group=T.df_groupby(df_hot,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list=df_pix['first_drought_NDVI_relative_change'].tolist()
            val_mean=np.nanmean(val_list)
            spatial_dic[pix]=val_mean

        array=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir=join(self.outdir,'hot_drought')
        T.mk_dir(outdir)

        outf=join(outdir,'first_drought_NDVI_relative_change.tif')
        print(outf)

        DIC_and_TIF().arr_to_tif(array,outf)


def main():
    # Pick_drought_events_year().run()
    Pick_multi_year_drought_events_year().run()
    # Dataframe().run()
    # PLOT_repeat_drought().run()
    pass

if __name__ == '__main__':
    main()
