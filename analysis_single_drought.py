
from global_init import *
# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta_info import *
from pprint import pprint
mpl.use('TkAgg')

result_root_this_script = join(results_root, 'analysis')

class Pick_drought_events:

    def __init__(self):
        self.outdir=rf'F:\Hotdrought_Resilience\results\analysis_single_year_drought\\'
        T.mkdir(self.outdir,force=True)

        self.threshold = -1.1

    def run(self):

        self.detect_single_year_droughts()
        ## add attribute

        # self.check_dic_time_series()


        pass

    def detect_single_year_droughts(self):
        """
        Detect single-year drought events (duration = 1 year),
        ensuring no extreme drought (SPI < extreme_threshold) in ±4 years.
        """
        fdir = data_root + r'\SPI\per_pix\spi12\\'
        outdir = results_root + r'\analysis_single_year_drought\\Dataframe\\'
        T.mk_dir(outdir, force=True)

        SPI_dict = T.load_npy_dir(fdir)
        years = np.arange(1982, 2021)

        extreme_threshold = -1.1
        recovery_gap = 4
        min_duration=2

        result_records = []


        for pix, vals in tqdm(SPI_dict.items(), desc="Detecting single-year droughts"):


            vals[vals < -999] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue

            spi_2d = np.reshape(vals, (-1, 12))
            if spi_2d.shape[0] != len(years):
                continue

            # === Step 1: 年度最小 SPI ===
            spi_annual = np.nanmin(spi_2d, axis=1)
            drought_mask = spi_annual < extreme_threshold

            # === Step 2: 连续干旱段识别 ===
            events = []
            start, end = np.nan, np.nan
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

            # === Step 3: 标记 multi-year events ===
            multiyear_years = set()
            for (s, e) in events:
                duration = e - s + 1
                if duration >= min_duration:
                    for k in range(s, e + 1):
                        multiyear_years.add(k)

            # === Step 4: 检查 single-year events ===
            for i, is_drought in enumerate(drought_mask):
                if not is_drought:
                    continue
                if i in multiyear_years:
                    continue
                if i - recovery_gap < 0 or i + recovery_gap >= len(years):
                    continue
                if any(idx in multiyear_years for idx in range(i - recovery_gap, i + recovery_gap + 1) if idx != i):
                    continue

                # === Step X: 检查 pre/post 4 年是否有 single/multi 干旱 ===
                pre_window = list(range(i - recovery_gap, i))
                post_window = list(range(i + 1, i + recovery_gap + 1))

                pre4_single = any(
                    (idx in np.where(drought_mask)[0]) and (idx not in multiyear_years) for idx in pre_window if
                    0 <= idx < len(years))
                post4_single = any(
                    (idx in np.where(drought_mask)[0]) and (idx not in multiyear_years) for idx in post_window if
                    0 <= idx < len(years))
                pre4_multi = any(idx in multiyear_years for idx in pre_window if 0 <= idx < len(years))
                post4_multi = any(idx in multiyear_years for idx in post_window if 0 <= idx < len(years))


                # === Step 5 : 计算 severity ===
                monthly_spi = spi_2d[i, :]
                drought_months = monthly_spi[monthly_spi < 0]
                if len(drought_months) == 0:
                    continue
                severity = np.nansum(np.abs(drought_months))

                # === Step 6: 记录 ===
                sub_spi = spi_2d[i, :]
                min_val = np.nanmin(sub_spi)
                min_month = np.nanargmin(sub_spi) + 1
                record = {
                    "pix": pix,
                    "drought_year": int(years[i]),
                    "SPI_min": float(min_val),
                    "SPI_min_month": int(min_month),
                    "Drought_severity": float(severity),
                    "pre4_single": pre4_single,
                    "post4_single": post4_single,
                    "pre4_multi": pre4_multi,
                    "post4_multi": post4_multi,
                }
                result_records.append(record)

        pprint(result_records)

        T.save_npy(result_records, join(outdir, 'single_year_droughts.npy'))

        # === 保存输出 ===
        df_out = pd.DataFrame(result_records)
        T.save_df(df_out, join(outdir, 'single_year_droughts.df'))
        T.df_to_excel(df_out, join(outdir, 'single_year_droughts'))
        print(f"Detected {len(df_out)} single-year droughts.")
        return df_out



    def check_dic_time_series(self):
        fdir_SPI = data_root + rf'\SPI\dic\spi12\\'
        dic_spi = T.load_npy_dir(fdir_SPI)

        NDVI_fdir = data_root + rf'\NDVI4g\annual_growth_season_NDVI_detrend_relative_change\\'
        NDVI_dic = T.load_npy_dir(NDVI_fdir)
        df=T.load_df(results_root+rf'\\analysis_single_year_drought\Dataframe\\single_year_droughts.df')


        for pix in dic_spi.keys():
            if pix not in NDVI_dic:
                continue

            lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)
            # if abs(lon + 104) > 0.05:
            #     continue
            # if abs(lat - 35) > 0.05:
            #     continue
            # if abs(lon + 104) > 0.05 or abs(lat - 43) > 0.05:
            #     continue
            spi = np.array(dic_spi[pix], dtype=float).ravel()

            # reshape 并计算年均
            spi_reshape = spi.reshape(39, -1)
            spi_annual = np.nanmin(spi_reshape, axis=1)
            NDVI_annual = np.array(NDVI_dic[pix], dtype=float).ravel()
            years = np.arange(1982, 2021)

            # === 查找该像素的干旱事件 ===
            pix_events = df[df["pix"] == pix]
            T.print_head_n(pix_events)


            # === 绘图 ===
            fig, ax1 = plt.subplots(figsize=(9, 5))

            # SPI 曲线
            ax1.plot(years, spi_annual, '-o', color='royalblue', label='SPI (annual)')
            ax1.axhline(-1.5, color='red', linestyle='--', linewidth=1.2, label='SPI = -1.5')
            ax1.set_ylabel("SPI", color='royalblue', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='royalblue')
            ax1.set_ylim(-3, 3)

            # 标出干旱事件
            for _, e in pix_events.iterrows():

                drought_years = e["drought_year"]
                plt.scatter(drought_years, spi_annual[drought_years - 1982], marker='o', color='red', s=100)



            # NDVI 曲线
            ax2 = ax1.twinx()
            ax2.plot(years, NDVI_annual, '-', color='forestgreen', label='NDVI')
            ax2.set_ylabel("NDVI", color='forestgreen', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='forestgreen')

            # === 图例与标题 ===
            fig.suptitle(f'Pixel {pix}, Lon={lon:.2f}, Lat={lat:.2f}', fontsize=13)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')

            plt.tight_layout()
            plt.show()

            # 可设置只显示一个像素方便检查



class Dataframe:

    def __init__(self):
        self.result=results_root+rf'\\analysis_single_year_drought\Dataframe\\'

        self.dff = join(self.result,'single_year_droughts.df')
        pass

    def run(self):
        # self.copy_df() ## only one time

        df = self.__df_init()
        # df = self.add_SOS_EOS(df)
        # df = self.filter_drought_events_via_SOS_EOS(df)
        df=self.add_temp_to_df(df)
        df=self.add_hot_drought(df)
        # self.check_df(df)

        df=self.add_total_NDVI_during_and_post_drought(df)
        df=self.add_aridity_to_df(df)
        df=self.add_MODIS_LUCC_to_df(df)
        df=self.add_koppen_to_df(df)

        df=self.add_landcover_data_to_df(df)
        df=self.add_landcover_classfication_to_df(df)
        # df=self.drop_field_df(df)
        # df=self.add_tif_to_df(df)


        #
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)
        pass

    def copy_df(self):
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
            pause()
        outdir = join(self.result,'dataframe')
        T.mk_dir(outdir)
        fpath = join(Pick_drought_events().this_class_arr,'drought_dataframe/drought_dataframe.df')
        df = T.load_df(fpath)
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

        pass

    def add_SOS_EOS(self,df):
        SOS_EOS_dir = join(data_root,'CRU_temp/extract_SOS_EOS_index')

        SOS_tif = join(SOS_EOS_dir,'Start_month_10_degree.tif')
        EOS_tif = join(SOS_EOS_dir,'End_month_10_degree.tif')
        SOS_dict = DIC_and_TIF().spatial_tif_to_dic(SOS_tif)
        EOS_dict = DIC_and_TIF().spatial_tif_to_dic(EOS_tif)
        sos_list = []
        eos_list = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='add SOS EOS'):
            pix = row['pix']
            sos = SOS_dict[pix]
            eos = EOS_dict[pix]
            sos_list.append(sos)
            eos_list.append(eos)
        df['sos'] = sos_list
        df['eos'] = eos_list
        df=df.dropna(subset=['sos','eos'])
        return df

    def filter_drought_events_via_SOS_EOS(self,df):
        picked_index_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            drought_mon = row['drought_mon']
            sos = row['sos']
            eos = row['eos']
            sos = int(sos)
            eos = int(eos)
            # print(sos==eos)
            # print(sos,eos)

            if sos == eos:
                if drought_mon != sos:
                    continue
            elif sos < eos:
                if drought_mon < sos or drought_mon > eos:
                    continue
            elif sos > eos:
                if drought_mon > sos or drought_mon < eos:
                    continue
            else:
                raise ValueError(sos,eos,sos==eos)
            picked_index_list.append(i)

        df = df.loc[picked_index_list]

        return df

    def add_temp_to_df(self, df):

        Temp_fdir = join(data_root, rf'CRU_temp\annual_growth_season_temp_detrend_zscore_10degree')
        Temp_dict = T.load_npy_dir(Temp_fdir)
        Temp_year_list = list(range(1982, 2021))

        result_dict = {}

        for i, row in tqdm(df.iterrows(), total=len(df), desc='add Temp'):
            pix = row['pix']
            # sos = row['sos']
            # eos = row['eos']

            # if np.isnan(sos) or np.isnan(eos):
            #     continue

            year = row['drought_year']
            if not pix in Temp_dict:
                continue
            temp = Temp_dict[pix]
            temp = list(temp)
            if T.is_all_nan(temp):
                continue
            if not len(Temp_year_list) == len(temp):
                print(pix)
                print(len(temp), len(Temp_year_list))
                print('----')
                continue
            Temp_year_dict = T.dict_zip(Temp_year_list, temp)
            # pprint(NDVI_reshape_dict);exit()

            Temp_mean = Temp_year_dict[year]
            result_dict[i] = Temp_mean
        df['Temp'] = result_dict
        df = df.dropna(subset=['Temp'])
        df = df.reset_index(drop=True)
        return df

    def add_hot_drought(self,df):


        df['drought_type'] = np.where(df['Temp'] > 1, 'Hot', 'Normal')

        return df

    def check_df(self,df):
        global_land_tif = join(this_root,'conf/land.tif')
        DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        plt.show()
        pass




    def add_GS_NDVI(self,df):
        NDVI_fdir = join(data_root,'NDVI4g/annual_growth_season_NDVI_detrend_relative_change/')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            # sos = row['sos']
            # eos = row['eos']
            drought_month = row['drought_mon']
            #
            # if np.isnan(sos) or np.isnan(eos):
            #     continue
            # sos = int(sos)
            # eos = int(eos)
            year = row['drought_year']
            if not pix in NDVI_dict:
                continue
            NDVI = NDVI_dict[pix]
            NDVI = list(NDVI)
            if T.is_all_nan(NDVI):
                continue
            if not len(NDVI_year_list) == len(NDVI):
                print(pix)
                print(len(NDVI),len(NDVI_year_list))
                print('----')
                continue
            NDVI_year_dict = T.dict_zip(NDVI_year_list,NDVI)
            # pprint(NDVI_reshape_dict);exit()

            NDVI_drought_year_GS_mean = NDVI_year_dict[year]
            result_dict[i] = NDVI_drought_year_GS_mean
        df['GS_NDVI_relative_change'] = result_dict
        df = df.dropna(subset=['GS_NDVI_relative_change'])
        df = df.reset_index(drop=True)
        return df

    def add_NDVI_min_mean_during_drought(self,df):



        temp_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend_relative_change')

        # === 1. 读取数据 ===

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
            min_NDVI_year = drought_years[np.nanargmin(drought_temps)]

            mean_NDVI = np.nanmean(drought_temps)

            # === 添加结果 ===
            min_NDVI_list.append(min_NDVI)
            min_NDVI_year_list.append(min_NDVI_year)

            mean_NDVI_list.append(mean_NDVI)

        # === 4. 添加到 DataFrame ===
        df['min_NDVI'] = min_NDVI_list
        df['min_NDVI_year'] = min_NDVI_year_list

        df['mean_NDVI'] = mean_NDVI_list

        return df

    def add_GS_values_post_n(self,df):
        post_n_years = 4
        NDVI_fdir = join(data_root,'NDVI4g/annual_growth_season_NDVI_detrend_relative_change/')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']


            if np.isnan(sos) or np.isnan(eos):
                continue
            sos = int(sos)
            eos = int(eos)
            year = row['drought_year']
            if not pix in NDVI_dict:
                continue
            NDVI = NDVI_dict[pix]
            NDVI = list(NDVI)
            if T.is_all_nan(NDVI):
                continue
            if not len(NDVI_year_list) == len(NDVI):
                print(pix)
                print(len(NDVI),len(NDVI_year_list))
                print('----')
                continue
            NDVI_year_dict = T.dict_zip(NDVI_year_list,NDVI)
            # pprint(NDVI_reshape_dict);exit()
            post_n_year_list = []
            for n in range(post_n_years):
                post_n_year_list.append(year+n+1)
            post_n_year_values = []
            for year in post_n_year_list:
                if not year in NDVI_year_dict:
                    post_n_year_values = []
                    break
                NDVI_drought_year_GS = NDVI_year_dict[year]
                post_n_year_values.append(NDVI_drought_year_GS)
            if len(post_n_year_values) == 0:
                continue
            NDVI_drought_year_GS_mean = np.nanmean(post_n_year_values)
            result_dict[i] = NDVI_drought_year_GS_mean
        df[f'GS_NDVI_post_{post_n_years}_relative_change'] = result_dict
        df = df.dropna(subset=[f'GS_NDVI_post_{post_n_years}_relative_change'])
        df = df.reset_index(drop=True)
        return df

    def add_total_NDVI_during_and_post_drought(self,df):


        temp_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend')

        # === 1. 读取数据 ===

        ndvi_dic = T.load_npy_dir(temp_dic_path)


        # === 2. 初始化结果 ===
        total_drought_list = []
        post1_list, post2_list, post3_list, post4_list = [], [], [], []
        drought_len_list = []

        base_year = 1982

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_year = row['drought_year']
            if drought_year==[]:
                continue


            print(pix, drought_year)

            # --- 转换 drought_years ---
            if drought_year is np.nan or (isinstance(drought_year, float) and np.isnan(drought_year)):
                total_drought_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

                # --- NDVI 缺失 ---
            if pix not in ndvi_dic:
                total_drought_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

                # --- NDVI 时间序列 ---
            ndvi_arr = np.array(ndvi_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(ndvi_arr))

            # --- 如果该干旱年不在NDVI时间范围内 ---
            if drought_year not in all_years:
                total_drought_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            # --- 提取干旱年 NDVI ---
            drought_idx = np.where(all_years == drought_year)[0][0]

            total_drought=ndvi_arr[drought_idx]

            # --- 提取干旱后 1–4 年 NDVI ---
            post_vals = []
            for offset in [1, 2, 3, 4]:
                year_target = drought_year + offset
                if year_target in all_years:
                    idx = np.where(all_years == year_target)[0][0]
                    post_vals.append(ndvi_arr[idx])
                else:
                    post_vals.append(np.nan)

            # === 累积 NDVI ===
            post1 = (post_vals[0])
            post2 = np.nansum((post_vals[:2]))
            post3 = np.nansum((post_vals[:3]))
            post4 = np.nansum((post_vals[:4]))

            def safe_ratio(a, b):
                if np.isnan(a) or np.isnan(b) or b == 0:
                    return np.nan
                return a / b

            post1_ratio = safe_ratio(post1, total_drought)
            post2_ratio = safe_ratio(post2, total_drought)
            post3_ratio = safe_ratio(post3, total_drought)
            post4_ratio = safe_ratio(post4, total_drought)

            total_drought_list.append(total_drought)
            post1_list.append(post1_ratio)
            post2_list.append(post2_ratio)
            post3_list.append(post3_ratio)
            post4_list.append(post4_ratio)


        # === 3. 写入 DataFrame ===
        df["NDVI_total_drought"] = total_drought_list
        df["NDVI_post1_total"] = post1_list
        df["NDVI_post2_total"] = post2_list
        df["NDVI_post3_total"] = post3_list
        df["NDVI_post4_total"] = post4_list

        return df


    def add_tif_to_df(self,df,):
        fdir=results_root+rf'\Plot_result\hot\\'
        for f in os.listdir(fdir):
            print(f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fdir+f)
            array = np.array(array, dtype=float)
            val_dic = DIC_and_TIF().spatial_arr_to_dic(array)
            f_name=f.split('.')[0]
            print(f_name)
            val_list=[]


            for i, row in tqdm(df.iterrows(), total=len(df)):
                pix = row['pix']
                if not pix in val_dic:
                    val_list.append(np.nan)
                    continue
                vals = val_dic[pix]
                val_list.append(vals)
            df[f'{f_name}_hot_pixel_average'] = val_list

        return df

        pass

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


                              'GS_NDVI_post_2_hot_pixel_average',

                              'GS_NDVI_post_1_hot_pixel_average',
            'GS_NDVI_post_3_hot_pixel_average',
            'GS_NDVI_post_4_hot_pixel_average',



                              ])
        return df





    def __df_init(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        return df


class PLOT_single_drought_vegetation():
    def __init__(self):

        self.result_root=r'F:\Hotdrought_Resilience\results\\analysis_single_year_drought\\'
        self.dff=join(self.result_root,'Dataframe/single_year_droughts.df')
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
        self.heatmap_during_drought()

        pass

    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        # df=df[df['MODIS_LUCC'] != 12]
        df = df[df['Koppen'] != 5]
        df = df[df['landcover_classfication'] != 'Bare']

        # df = df[df['landcover_classfication'] != 'Cropland']

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

    def heatmap_during_drought(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff = rf'F:\Hotdrought_Resilience\results\analysis_single_year_drought\Dataframe\\single_year_droughts.df'
        df = T.load_df(dff)
        df = self.df_clean(df)
        print(len(df))

        # plt.show();exit()

        T.print_head_n(df)
        x_var = 'Drought_severity'
        y_var = 'Temp'
        z_var = 'NDVI_total_drought'

        plt.hist(df[x_var])
        plt.show()
        plt.hist(df[y_var])
        plt.show()

        bin_x = np.linspace(0, 20, 11, )

        bin_y = np.linspace(-2, 2, 11)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(8, 8))

        matrix_dict, x_ticks_list, y_ticks_list = T.df_bin_2d(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y, round_x=4,
                                                              round_y=4)
        # pprint(matrix_dict);exit()

        my_cmap = T.cmap_blend(color_list=['#000000', 'r', 'b'])
        my_cmap = 'RdBu'
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

    def df_bin_2d_count(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
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
        return matrix_dict,x_ticks_list,y_ticks_list

    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'],observed=True)
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
    def plot_df_bin_2d_matrix(self,matrix_dict,vmin,vmax,x_ticks_list,y_ticks_list,cmap='RdBu',
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
        plt.imshow(matrix,cmap=cmap,vmin=vmin,vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])
        # plt.colorbar()
        # plt.show()

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



    def plot_NDVI_post_drought_resilience(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        df=df[df['drought_type']=='Hot']
        # df=df[df['drought_type']=='Normal']
        spatial_dic_post1={}
        spatial_dic_post2={}
        spatial_dic_post3={}
        spatial_dic_post4={}
        spatial_dic_during_drought={}



        df_group=T.df_groupby(df,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list_post1=df_pix['NDVI_post1_total'].tolist()

            val_list_post2=df_pix['NDVI_post2_total'].tolist()
            val_list_post3=df_pix['NDVI_post3_total'].tolist()
            val_list_post4=df_pix['NDVI_post4_total'].tolist()
            val_list_during_drought=df_pix['NDVI_total_drought'].tolist()
            print(val_list_during_drought)



            val_post1_mean=np.nanmean(val_list_post1)

            val_post2_mean=np.nanmean(val_list_post2)
            val_post3_mean = np.nanmean(val_list_post3)
            val_post4_mean=np.nanmean(val_list_post4)
            val_list_during_drought_mean=np.nanmean(val_list_during_drought)

            spatial_dic_post1[pix] = val_post1_mean-val_list_during_drought_mean
            spatial_dic_post2[pix]=val_post2_mean-val_list_during_drought_mean
            spatial_dic_post3[pix] = val_post3_mean-val_list_during_drought_mean
            spatial_dic_post4[pix]=val_post4_mean-val_list_during_drought_mean
            spatial_dic_during_drought[pix]=val_list_during_drought_mean

        array_post1 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post1)
        array_post2 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post2)
        array_post3 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post3)
        array_post4 = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_post4)
        array_during_drought = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_during_drought)

        outdir = join(self.outdir, 'tiff','Hot')
        T.mk_dir(outdir, force=True)

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

        outf = join(outdir, 'NDVI_total_drought.tif')
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

    # Pick_drought_events().run()
    # Dataframe().run()
    PLOT_single_drought_vegetation().run()




if __name__ == '__main__':
    main()
    pass