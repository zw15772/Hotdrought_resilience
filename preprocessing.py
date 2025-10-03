from global_init import *

# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import xarray as xr
import matplotlib
matplotlib.use('TkAgg')

class GIMMS_NDVI:
    def __init__(self):
        self.datadir = join(data_root, 'NDVI4g')
        pass
    def run(self):
        # self.resample()
        # self.monthly_compose()
        # self.scaling()
        self.per_pix()
        self.check_data()

        pass

    def resample(self):
        fdir = join(self.datadir, 'tif_raw')
        outdir = join(self.datadir, 'bi_weekly_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, 0.5)

    def monthly_compose(self):
        fdir = join(self.datadir,'bi_weekly_05')
        outdir = join(self.datadir,'monthly_tif')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')
        pass
    def scaling(self):
        ## data*0.0001 and
        fdir=join(self.datadir,'monthly_tif')
        outdir=join(self.datadir,'scaling_tif')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = array*0.0001
            array[array<0] = np.nan
            array[array>1] = np.nan
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)

        pass

    def per_pix(self):
        fdir = join(self.datadir,'scaling_tif')
        outdir = join(self.datadir,'per_pix')
        T.mk_dir(outdir)
        Pre_Process().data_transform(fdir,outdir)

    def check_data(self):
        fdir = join(self.datadir,'per_pix')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_num = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            if T.is_all_nan(vals):
                continue
            num = len(vals)
            # plt.plot(vals)
            # plt.show()
            spatial_dict_num[pix] = num
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_num)
        plt.imshow(arr)
        plt.show()

    def extract_all_gs_NDVI_based_temp(self):
        """
        批处理: 提取所有像素的 LAI 生长季 (多年序列)
        输出:
            result_dic : {pix: ndarray 或 None}, shape = (n_years, season_len)
        """
        LAI_dic_fdir =join(data_root,'NDVI4g','per_pix')
        growing_season_fdir = join(data_root,'CRU_temp','extract_growing_season','extract_growing_season_10degree')
        outdir = join(data_root,'NDVI4g','extract_growing_season','extract_growing_season_10degree')
        T.mk_dir(outdir, force=True)

        start_dic = {}
        end_dic = {}
        len_dic = {}

        for f in T.listdir(LAI_dic_fdir):

            result_dic = {}
            LAI_dic = T.load_npy(join(LAI_dic_fdir,f))
            gs_dic = T.load_npy(join(growing_season_fdir,f))

            for pix in tqdm(LAI_dic, desc=f"Extracting GS LAI from {f}"):
                LAI_val = np.array(LAI_dic[pix], dtype=float)

                if pix not in gs_dic:
                    result_dic[pix] = None
                    continue

                start = gs_dic[pix]["start"]
                end = gs_dic[pix]["end"]
                start_dic[pix] = start
                end_dic[pix] = end

                if start is None or end is None:
                    result_dic[pix] = None
                    continue

                n_years = len(LAI_val) // 12
                gs_vals = []

                for y in range(n_years):
                    year_vals = LAI_val[y * 12:(y + 1) * 12]  # 当年12个月

                    if start <= end:
                        # 生长季在同一年内
                        gs_vals.append(year_vals[start:end + 1])
                    else:
                        # 跨年 → 当年[start:12] + 下一年[:end+1]
                        if y < n_years - 1:  # 不是最后一年
                            next_year_vals = LAI_val[(y + 1) * 12:(y + 2) * 12]
                            gs_vals.append(np.concatenate([year_vals[start:], next_year_vals[:end + 1]]))
                        else:
                            # 最后一年无法取下一年 → 丢弃 or 标记 None
                            gs_vals.append(None)

                result_dic[pix] = gs_vals
                len_dic[pix] = len(gs_vals)

            # 保存生长季 LAI
            outf = join(outdir,f)
            np.save(outf, result_dic)

        # 可选：保存 start/end 空间分布检查
        array_start = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(start_dic)
        array_end = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(end_dic)
        array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)

        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_start, outdir + 'start.tif')

        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outdir + 'len.tif')

        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_end, outdir + 'end.tif')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)

        plt.imshow(array_start, cmap="jet")
        plt.colorbar(label="Month Index (0=Jan)")

        plt.subplot(1, 2, 2)

        plt.imshow(array_end, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()


    def annual_growth_season_NDVI(self):
        """
        计算每个像素逐年的生长季 LAI 平均值
        - 北半球: 保留22年 (2003–2024)
        - 南半球: 如果是跨年生长季 → 只保留21年 (2004–2024)
                  如果是全年生长季 → 保留22年
        """
        fdir = join(data_root,'NDVI4g','extract_growing_season','extract_growing_season_10degree')
        outdir = join(data_root,'NDVI4g','annual_growing_season_NDVI')
        T.mk_dir(outdir, force=True)
        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not '.npy' in f:
                continue
            dic = T.load_npy(join(fdir,f))
            result_dic = {}

            for pix in dic:

                r,c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                # # print(lon,lat)
                # if not lon == 149.5:
                #     continue
                # if not lat== -36.5:
                #     continue
                gs_array = dic[pix]  # shape = (n_years, season_len) 或 None


                if gs_array is None:
                    result_dic[pix] = None
                    continue

                if gs_array[-1] is None:
                    gs_array = gs_array[:-1]

                n_years = len(gs_array)

                if n_years < 38:
                    result_dic[pix] = None
                    continue


                # print(gs_array)
                # gs_array = np.array(gs_array,dtype=float)
                # print(gs_array.shape)
                # plt.imshow(gs_array,cmap='jet')
                # plt.colorbar()
                # plt.show()
                # exit()

                # 逐年平均
                annual_mean = np.nanmean(gs_array, axis=1)  # (n_years,)
                print(len(annual_mean))
                # plt.plot(annual_mean)
                # plt.show()



                result_dic[pix] = annual_mean
                len_dic[pix] = len(annual_mean)

            outf = join(outdir,f)
            np.save(outf, result_dic)
        array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()




    pass

class SPI:
    def __init__(self):
        self.datadir = join(data_root,'SPI','tif','1982-2020',)
        pass
    def run(self):
        self.per_pix()


        pass
    def per_pix(self):
        fdir = join(self.datadir, 'spi09')
        outdir =  join(data_root,'SPI','dic','spi09')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir, outdir)
class temperature:
    def __init__(self):
        self.datadir = join(data_root,'CRU_tmax',)
        pass
    def run(self):
        # self.nc_to_tif()
        # self.filiter()

        # self.per_pix()
        # self.long_term_mean()
        self.extract_SOS_EOS_index()
        pass


    def nc_to_tif(self):
        params_list = []

        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            # var_name = f.split('_')[0]
            var_name = 'tmx'
            f_path = join(fdir,f)
            # T.nc_to_tif(f_path,var_name,outdir)
            ncin = Dataset(f_path, 'r')
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
            time_list = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units
            basetime_unit = basetime_str.split('since')[0]
            basetime_unit = basetime_unit.strip()
            basetime = basetime_str.strip(f'{basetime_unit} since ')
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
            data = ncin.variables[var_name]
            for time_i in range(len(time_list)):
                if basetime_unit == 'days':
                    date = basetime + datetime.timedelta(days=int(time_list[time_i]))
                elif basetime_unit == 'years':
                    date1 = basetime.strftime('%Y-%m-%d')
                    base_year = basetime.year
                    date2 = f'{int(base_year + time_list[time_i])}-01-01'
                    delta_days = Tools().count_days_of_two_dates(date1, date2)
                    date = basetime + datetime.timedelta(days=delta_days)
                elif basetime_unit == 'month' or basetime_unit == 'months':
                    date1 = basetime.strftime('%Y-%m-%d')
                    base_year = basetime.year
                    base_month = basetime.month
                    date2 = f'{int(base_year + time_list[time_i] // 12)}-{int(base_month + time_list[time_i] % 12)}-01'
                    delta_days = Tools().count_days_of_two_dates(date1, date2)
                    date = basetime + datetime.timedelta(days=delta_days)
                elif basetime_unit == 'seconds':
                    date = basetime + datetime.timedelta(seconds=int(time_list[time_i]))
                elif basetime_unit == 'hours':
                    date = basetime + datetime.timedelta(hours=int(time_list[time_i]))
                else:
                    raise Exception('basetime unit not supported')
                time_str = time_list[time_i]
                mon = date.month
                year = date.year
                day = date.day
                outf_name = f'{year}{mon:02d}{day:02d}.tif'
                outpath = join(outdir, outf_name)
                # if isfile(outpath):
                #     continue
                arr = data[time_i]
                arr = np.array(arr)[::-1]

                longitude_start = -180
                latitude_start = 90
                pixelWidth = lon[1] - lon[0]
                pixelHeight = lat[0] - lat[1]
                # print(pixelHeight)
                ToRaster().array2raster(outpath, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                # exit()
    def filiter(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'filter_tif')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array<-999] = np.nan
            array[array>999] = np.nan
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)
        pass

    def per_pix(self):
        fdir = join(self.datadir, 'filter_tif')
        outdir =  join(self.datadir,'per_pix',)
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir, outdir)



    def long_term_mean(self):
        fdir=join(data_root,'CRU_temp','per_pix')

        outdir=join(data_root,'CRU_temp','long_term_mean')
        T.mk_dir(outdir,True)

        for f in T.listdir(fdir):
            result_dic={}

            dic=T.load_npy(join(fdir,f))
            for pix in tqdm(dic):
                val=dic[pix]
                val=np.array(val,dtype=float)
                if np.isnan(np.nanmean(val)):
                    result_dic[pix]=np.nan
                    continue
                else:
                    ## reshape to 24*12
                    val=val.reshape(-1,12)
                    # plt.imshow(val)
                    # plt.show()
                    val_mean=np.nanmean(val,axis=0)
                    # plt.plot(val)
                    # plt.show()
                    result_dic[pix]=val_mean

            np.save(join(outdir,f),result_dic)

    def extract_SOS_EOS_index(self):
        """
        根据  的月均温度，提取每个像素的生长季
        条件：温度 > 10 °C
        支持跨年生长季（例如南半球 11 月 - 次年 3 月）
        """

        fdir=join(data_root,'CRU_temp','long_term_mean')


        outdir=join(data_root,'CRU_temp','extract_growing_season_5degree')
        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):
            dic = T.load_npy(join(fdir,f))
            result_dic = {}

            for pix in tqdm(dic):
                val = np.array(dic[pix], dtype=float)

                # 找到温度 > 10°C 的月份索引
                index = np.where(val > 5)[0]

                if len(index) == 0:
                    # 没有超过 10°C 的月份 → 无生长季
                    result_dic[pix] = {"start": None, "end": None, "season": None}
                    continue

                # 检查连续性
                diffs = np.diff(index)

                if np.any(diffs > 1):
                    # 有断点，可能跨年
                    if index[0] == 0 and index[-1] == 11:
                        # 处理跨年情况：把首尾两段合并
                        split_points = np.where(diffs > 1)[0]
                        groups = []
                        start_idx = 0
                        for sp in split_points:
                            groups.append(index[start_idx:sp + 1])
                            start_idx = sp + 1
                        groups.append(index[start_idx:])

                        # 合并首尾段
                        if groups[0][0] == 0 and groups[-1][-1] == 11:
                            merged = np.concatenate((groups[-1], groups[0]))
                            groups = [merged] + groups[1:-1]

                        # 选择最长的连续段作为 growing season
                        longest = max(groups, key=len)
                        first_index, last_index = longest[0], longest[-1]
                    else:
                        # 普通断点情况（取第一个到最后一个）
                        first_index, last_index = index[0], index[-1]
                else:
                    # 全年连续，没有跨年
                    first_index, last_index = index[0], index[-1]

                # 提取生长季温度曲线
                if first_index <= last_index:
                    growing_season = val[first_index:last_index + 1]
                else:
                    # 跨年 → 拼接两段
                    growing_season = np.concatenate((val[first_index:], val[:last_index + 1]))

                result_dic[pix] = {
                    "start": int(first_index),
                    "end": int(last_index),
                    "season": growing_season
                }

            # 保存结果
            outf = join(outdir,f)
            T.save_npy(result_dic,outf)

    def extract_all_gs_temp_based_temp(self):
        """
        批处理: 提取所有像素的 LAI 生长季 (多年序列)
        输出:
            result_dic : {pix: ndarray 或 None}, shape = (n_years, season_len)
        """
        LAI_dic_fdir = join(data_root, 'NDVI4g', 'per_pix')
        growing_season_fdir = join(data_root, 'CRU_temp', 'extract_growing_season', 'extract_growing_season_10degree')
        outdir = join(data_root, 'NDVI4g', 'extract_growing_season', 'extract_growing_season_10degree')
        T.mk_dir(outdir, force=True)

        start_dic = {}
        end_dic = {}
        len_dic = {}

        for f in T.listdir(LAI_dic_fdir):

            result_dic = {}
            LAI_dic = T.load_npy(join(LAI_dic_fdir, f))
            gs_dic = T.load_npy(join(growing_season_fdir, f))

            for pix in tqdm(LAI_dic, desc=f"Extracting GS LAI from {f}"):
                LAI_val = np.array(LAI_dic[pix], dtype=float)

                if pix not in gs_dic:
                    result_dic[pix] = None
                    continue

                start = gs_dic[pix]["start"]
                end = gs_dic[pix]["end"]
                start_dic[pix] = start
                end_dic[pix] = end

                if start is None or end is None:
                    result_dic[pix] = None
                    continue

                n_years = len(LAI_val) // 12
                gs_vals = []

                for y in range(n_years):
                    year_vals = LAI_val[y * 12:(y + 1) * 12]  # 当年12个月

                    if start <= end:
                        # 生长季在同一年内
                        gs_vals.append(year_vals[start:end + 1])
                    else:
                        # 跨年 → 当年[start:12] + 下一年[:end+1]
                        if y < n_years - 1:  # 不是最后一年
                            next_year_vals = LAI_val[(y + 1) * 12:(y + 2) * 12]
                            gs_vals.append(np.concatenate([year_vals[start:], next_year_vals[:end + 1]]))
                        else:
                            # 最后一年无法取下一年 → 丢弃 or 标记 None
                            gs_vals.append(None)

                result_dic[pix] = gs_vals
                len_dic[pix] = len(gs_vals)

            # 保存生长季 LAI
            outf = join(outdir, f)
            np.save(outf, result_dic)

        # 可选：保存 start/end 空间分布检查
        array_start = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(start_dic)
        array_end = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(end_dic)
        array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)

        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_start, outdir + 'start.tif')

        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outdir + 'len.tif')

        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_end, outdir + 'end.tif')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)

        plt.imshow(array_start, cmap="jet")
        plt.colorbar(label="Month Index (0=Jan)")

        plt.subplot(1, 2, 2)

        plt.imshow(array_end, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()



    def annual_growth_season_temp(self):
        """
        计算每个像素逐年的生长季 LAI 平均值
        - 北半球: 保留22年 (2003–2024)
        - 南半球: 如果是跨年生长季 → 只保留21年 (2004–2024)
                  如果是全年生长季 → 保留22年
        """
        fdir = join(data_root,'NDVI4g','extract_growing_season','extract_growing_season_10degree')
        outdir = join(data_root,'NDVI4g','annual_growing_season_NDVI')
        T.mk_dir(outdir, force=True)
        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not '.npy' in f:
                continue
            dic = T.load_npy(join(fdir,f))
            result_dic = {}

            for pix in dic:

                r,c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                # # print(lon,lat)
                # if not lon == 149.5:
                #     continue
                # if not lat== -36.5:
                #     continue
                gs_array = dic[pix]  # shape = (n_years, season_len) 或 None


                if gs_array is None:
                    result_dic[pix] = None
                    continue

                if gs_array[-1] is None:
                    gs_array = gs_array[:-1]

                n_years = len(gs_array)

                if n_years < 38:
                    result_dic[pix] = None
                    continue


                # print(gs_array)
                # gs_array = np.array(gs_array,dtype=float)
                # print(gs_array.shape)
                # plt.imshow(gs_array,cmap='jet')
                # plt.colorbar()
                # plt.show()
                # exit()

                # 逐年平均
                annual_mean = np.nanmean(gs_array, axis=1)  # (n_years,)
                print(len(annual_mean))
                # plt.plot(annual_mean)
                # plt.show()



                result_dic[pix] = annual_mean
                len_dic[pix] = len(annual_mean)

            outf = join(outdir,f)
            np.save(outf, result_dic)
        array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()













class extract_growing_season_not_used:  ## not use in this project
    def __init__(self):

        pass



    def run(self):
        # self.resample()
        # self.SOS_EOS_doy()
        self.SOS_EOS_mon()
        # self.extract_phenology_monthly_variables()
        pass

    @Decorator.shutup_gdal
    def resample(self):
        fdir = join(data_root,'MODIS_phenology/tif')
        outdir = join(data_root,'MODIS_phenology/tif_05')
        for folder in tqdm(T.listdir(fdir)):
            for f in T.listdir(join(fdir,folder)):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir,folder,f)
                outdir_i = join(outdir,folder)
                T.mk_dir(outdir_i,force=True)
                outf = join(outdir_i,f)
                ToRaster().resample_reproj(fpath,outf,0.5)

    def SOS_EOS_doy(self):
        outdir = join(data_root,'MODIS_phenology/SOS_EOS_doy')
        T.mk_dir(outdir,force=True)
        base_date = datetime.datetime(1970,1,1)
        fdir = join(data_root,'MODIS_phenology/tif_05')
        num_cycle_fdir = join(fdir,'NumCycles')
        array_void = 0
        for f in T.listdir(num_cycle_fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(num_cycle_fdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array_void = np.zeros_like(array)
            break
        array_void = array_void.astype('float')

        flag = 0
        for f in T.listdir(num_cycle_fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(num_cycle_fdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array = array.astype('float')
            array[array>30000] = np.nan
            array_void += array
            flag += 1
        array_mean = array_void / flag
        array_mean[array_mean!=1] = np.nan
        spatial_dict = DIC_and_TIF().spatial_arr_to_dic(array_mean)

        year_range = list(range(2001,2021))
        sos_dir = join(fdir,'Greenup_1')
        # eos_dir = join(fdir,'Senescence_1')
        eos_dir = join(fdir,'Dormancy_1')
        gs_range_dict = {}
        for year in tqdm(year_range):
            base_date_i = datetime.datetime(year,1,1)
            sos_fpath = join(sos_dir,f'{year}_01_01.tif')
            eos_fpath = join(eos_dir,f'{year}_01_01.tif')
            sos_array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(sos_fpath)
            eos_array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(eos_fpath)
            sos_array = sos_array.astype('float')
            eos_array = eos_array.astype('float')
            sos_array[sos_array > 30000] = np.nan
            eos_array[eos_array > 30000] = np.nan
            sos_dict = DIC_and_TIF().spatial_arr_to_dic(sos_array)
            eos_dict = DIC_and_TIF().spatial_arr_to_dic(eos_array)
            spatial_dict_1 = {}
            for pix in sos_dict:
                sos = sos_dict[pix]
                eos = eos_dict[pix]
                if np.isnan(sos) or np.isnan(eos):
                    continue
                sos = int(sos)
                eos = int(eos)
                sos_date = base_date + datetime.timedelta(days=sos)
                eos_date = base_date + datetime.timedelta(days=eos)
                sos_year = sos_date.year
                eos_year = eos_date.year
                # if not sos_year == year:
                #     continue
                # if not eos_year == year:
                #     continue
                sos_doy = sos_date - base_date_i
                eos_doy = eos_date - base_date_i
                sos_doy = sos_doy.days
                eos_doy = eos_doy.days
                # if not eos_doy > sos_doy:
                #     continue
                # sos_mon = self.__doy_to_month(sos_doy)
                # eos_mon = self.__doy_to_month(eos_doy)
                if not pix in gs_range_dict:
                    gs_range_dict[pix] = {}
                gs_range_dict[pix][year] = np.array([sos_doy,eos_doy])
        df = T.dic_to_df(gs_range_dict,'pix')
        outf = join(outdir,'SOS_EOS_doy.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def SOS_EOS_mon(self):
        fdir = join(data_root,'MODIS_phenology/SOS_EOS_doy')
        outdir = join(data_root,'MODIS_phenology/SOS_EOS_mon')
        T.mk_dir(outdir,force=True)
        df = T.load_df(join(fdir,'SOS_EOS_doy.df'))
        sos_spatial_dict = {}
        eos_spatial_dict = {}
        year_list = range(2001,2021)

        for i,row in df.iterrows():
            pix = row['pix']
            sos_list = []
            eos_list = []
            for year in year_list:
                sos_eos = row[year]
                if type(sos_eos) == float:
                    continue
                sos,eos = row[year]
                sos_list.append(sos)
                eos_list.append(eos)
            sos_mean = np.nanmean(sos_list)
            eos_mean = np.nanmean(eos_list)
            sos_spatial_dict[pix] = sos_mean
            eos_spatial_dict[pix] = eos_mean
        sos_mon_spatial_dict = {}
        eos_mon_spatial_dict = {}
        for pix  in sos_spatial_dict:
            sos = sos_spatial_dict[pix]
            eos = eos_spatial_dict[pix]
            if sos < 0:
                sos_doy = 365 + int(sos)
            else:
                sos_doy = int(sos)
            if eos > 365:
                eos_doy = int(eos) - 365
            else:
                eos_doy = int(eos)
            sos_mon = self.__doy_to_month(sos_doy)
            eos_mon = self.__doy_to_month(eos_doy)
            sos_mon_spatial_dict[pix] = sos_mon
            eos_mon_spatial_dict[pix] = eos_mon
        sos_outf = join(outdir,'sos_mon.tif')
        eos_outf = join(outdir,'eos_mon.tif')
        DIC_and_TIF().pix_dic_to_tif(sos_mon_spatial_dict,sos_outf)
        DIC_and_TIF().pix_dic_to_tif(eos_mon_spatial_dict,eos_outf)



    def __doy_to_month(self,doy):
        '''
        :param doy: day of year
        :return: month
        '''
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month

    def extract_phenology_monthly_variables(self):
        fdir = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/SPI/dic/spi09/'

        outdir = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/SPI/extract_phenology_monthly//spi09//'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/4GST//4GST_global.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}
        for pix in phenology_dic:
            val=phenology_dic[pix]['Onsets']
            if isinstance(val, np.ndarray):
                print("skip array:", val)
                new_spatial_dic[pix] = np.nan
                continue

            new_spatial_dic[pix]=val

        spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
        plt.show()
        exit()
        for f in T.listdir(fdir):

            outf = outdir + f
            #
            # if os.path.isfile(outf):
            #     continue
            # print(outf)
            spatial_dict = dict(np.load(fdir + f, allow_pickle=True, encoding='latin1').item())
            dic_DOY = {15: 0,
                       30: 0,
                       45: 1,
                       60: 1,
                       75: 2,
                       90: 2,
                       105: 3,
                       120: 3,
                       135: 4,
                       150: 4,
                       165: 5,
                       180: 5,
                       195: 6,
                       210: 6,
                       225: 7,
                       240: 7,
                       255: 8,
                       270: 8,
                       285: 9,
                       300: 9,
                       315: 10,
                       330: 10,
                       345: 11,
                       360: 11, }

            result_dic = {}
            for pix in tqdm(spatial_dict):
                if not pix in phenology_dic:
                    continue

                r, c = pix

                SeasType = phenology_dic[pix]['SeasType']
                if SeasType == 2:

                    SOS = phenology_dic[pix]['Onsets']
                    try:
                        SOS = float(SOS)

                    except:
                        continue

                    SOS = int(SOS)
                    SOS_biweekly = dic_DOY[SOS]

                    EOS = phenology_dic[pix]['Offsets']
                    EOS = int(EOS)
                    EOS_biweekly = dic_DOY[EOS]

                    time_series = spatial_dict[pix]

                    time_series = np.array(time_series)
                    time_series_flatten = time_series.flatten()
                    time_series_flatten_extraction = time_series_flatten[(EOS_biweekly + 1):-(12 - EOS_biweekly - 1)]
                    time_series_flatten_extraction_reshape = time_series_flatten_extraction.reshape(-1, 12)
                    non_growing_season_list = []
                    growing_season_list = []
                    for vals in time_series_flatten_extraction_reshape:
                        if T.is_all_nan(vals):
                            continue
                        ## non-growing season +growing season is 365

                        non_growing_season = vals[0:SOS_biweekly]
                        growing_season = vals[SOS_biweekly:]
                        # print(growing_season)
                        non_growing_season_list.append(non_growing_season)
                        growing_season_list.append(growing_season)
                    # print(len(growing_season_list))
                    non_growing_season_list = np.array(non_growing_season_list)
                    growing_season_list = np.array(growing_season_list)


                elif SeasType == 3:
                    # SeasClass=phenology_dic[pix]['SeasClss']
                    # ## whole year is growing season
                    # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix)
                    # print(lat,lon)
                    # print(SeasType)
                    # print(SeasClass)
                    time_series = spatial_dict[pix]
                    time_series = np.array(time_series)
                    time_series_flatten = time_series.flatten()
                    time_series_flatten_extraction = time_series_flatten[12:]
                    time_series_flatten_extraction_reshape = time_series_flatten_extraction.reshape(-1, 12)
                    non_growing_season_list = []
                    growing_season_list = time_series_flatten_extraction_reshape

                else:
                    SeasClss = phenology_dic[pix]['SeasClss']
                    print(SeasType, SeasClss)
                    continue

                result_dic[pix] = {'SeasType': SeasType,
                                   'non_growing_season': non_growing_season_list,
                                   'growing_season': growing_season_list,
                                   'ecosystem_year': time_series_flatten_extraction_reshape}

            np.save(outf, result_dic)


def main():

    # GIMMS_NDVI().run()
    # SPI().run()
    # temperature().run()
    extract_growing_season().run()

if __name__ == '__main__':
    main()
    pass