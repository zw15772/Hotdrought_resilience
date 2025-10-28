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
import os
from tqdm import tqdm
import xarray
from os.path import join
from scipy.stats import gamma, norm
# import climate_indices
# from climate_indices import compute
# from climate_indices import indices
import matplotlib
matplotlib.use('TkAgg')

class Download_TerraClimate():
    def __init__(self):
        self.datadir = r'F:\Hotdrought_Resilience\data\\'


    def run (self):
        # self.download_all()
        # self.nc_to_tif_time_series_fast()
        # self.resample()
        # self.average()

        self.per_pix()


    def download_all(self):
        param_list = []
        # product_list = ['srad','pdsi','vpd','ppt','tmax','tmin']
        # product_list = ['aet']
        # product_list = ['vpd']
        product_list = ['tmax']
        for product in product_list:
            for y in range(1982, 2021):
                param_list.append([product, str(y)])
                params = [product, str(y)]
                # self.download(params)
        MULTIPROCESS(self.download, param_list).run(process=8, process_or_thread='t')

    def download(self, params):
        product, y = params
        outdir = join(self.datadir, product, 'nc')
        T.mk_dir(outdir, force=True)
        url = 'https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc'.format(product, y)
        print(url)
        while 1:
            try:
                outf = join(outdir, '{}_{}.nc'.format(product, y))
                if os.path.isfile(outf):
                    return None
                req = requests.request('GET', url)
                content = req.content
                fw = open(outf, 'wb')
                fw.write(content)
                return None
            except Exception as e:
                print(url, 'error sleep 5s')
                time.sleep(5)

    def nc_to_tif_time_series_fast(self):

        fdir = join(self.datadir, 'GLEAM','SMs','nc')
        outdir = join(self.datadir, 'GLEAM', 'SMs','tif')
        Tools().mk_dir(outdir, force=True)
        for f in tqdm(os.listdir(fdir)):


            outdir_name = f.split('.')[0]
            # print(outdir_name)

            yearlist = list(range(1982, 2021))
            fpath = join(fdir, f)
            nc_in = xarray.open_dataset(fpath)
            print(nc_in)
            time_bnds = nc_in['time']
            for t in range(len(time_bnds)):
                date = time_bnds[t]['time']
                date = pd.to_datetime(date.values)
                date_str = date.strftime('%Y%m%d')
                date_str = date_str.split()[0]
                outf = join(outdir, f'{date_str}.tif')
                array = nc_in['SMs'][t]

                array = np.array(array)
                # array[array < 0] = np.nan

                longitude_start, latitude_start, pixelWidth, pixelHeight = -180, 90, 0.1, -0.1
                ToRaster().array2raster(outf, longitude_start, latitude_start,
                                        pixelWidth, pixelHeight, array, ndv=-999999)
                # exit()

            # nc_to_tif_template(fdir+f,var_name='lai',outdir=outdir,yearlist=yearlist)
            try:
                self.nc_to_tif_template(fdir + f, var_name='SMs', outdir=outdir, yearlist=yearlist)
            except Exception as e:
                print(e)
                continue
    def nc_to_tif_template(self, fname, var_name, outdir, yearlist):
        try:
            ncin = Dataset(fname, 'r')
            print(ncin.variables.keys())
            # exit()
            time = ncin.variables['time'][:]

        except:
            raise UserWarning('File not supported: ' + fname)
        # lon,lat = np.nan,np.nan
        try:
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
        except:
            try:
                lat = ncin.variables['latitude'][:]
                lon = ncin.variables['longitude'][:]
            except:
                try:
                    lat = ncin.variables['lat_FULL'][:]
                    lon = ncin.variables['lon_FULL'][:]
                except:
                    raise UserWarning('lat or lon not found')
        shape = np.shape(lat)
        try:
            time = ncin.variables['time_counter'][:]
            basetime_str = ncin.variables['time_counter'].units
        except:
            time = ncin.variables['time'][:]
            basetime_str = ncin.variables['time'].units

        basetime_unit = basetime_str.split('since')[0]
        basetime_unit = basetime_unit.strip()
        print(basetime_unit)
        print(basetime_str)
        if basetime_unit == 'days':
            timedelta_unit = 'days'
        elif basetime_unit == 'years':
            timedelta_unit = 'years'
        elif basetime_unit == 'month':
            timedelta_unit = 'month'
        elif basetime_unit == 'months':
            timedelta_unit = 'month'
        elif basetime_unit == 'seconds':
            timedelta_unit = 'seconds'
        elif basetime_unit == 'hours':
            timedelta_unit = 'hours'
        else:
            raise Exception('basetime unit not supported')
        basetime = basetime_str.strip(f'{timedelta_unit} since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
                    except:
                        try:
                            basetime = datetime.datetime.strptime(basetime, '%Y-%m')
                        except:
                            try:
                                basetime_ = basetime.split('T')[0]
                                # print(basetime_)
                                basetime = datetime.datetime.strptime(basetime_, '%Y-%m-%d')
                                # print(basetime)
                            except:

                                raise UserWarning('basetime format not supported')
        data = ncin.variables[var_name]
        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in tqdm(range(len(time))):
            if basetime_unit == 'days':
                date = basetime + datetime.timedelta(days=int(time[time_i]))
            elif basetime_unit == 'years':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                date2 = f'{int(base_year + time[time_i])}-01-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'month' or basetime_unit == 'months':
                date1 = basetime.strftime('%Y-%m-%d')
                base_year = basetime.year
                base_month = basetime.month
                date2 = f'{int(base_year + time[time_i] // 12)}-{int(base_month + time[time_i] % 12)}-01'
                delta_days = Tools().count_days_of_two_dates(date1, date2)
                date = basetime + datetime.timedelta(days=delta_days)
            elif basetime_unit == 'seconds':
                date = basetime + datetime.timedelta(seconds=int(time[time_i]))
            elif basetime_unit == 'hours':
                date = basetime + datetime.timedelta(hours=int(time[time_i]))
            else:
                raise Exception('basetime unit not supported')
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year not in yearlist:
                continue
            day = date.day
            outf_name = f'{year}{mon:02d}{day:02d}.tif'
            outpath = join(outdir, outf_name)
            if isfile(outpath):
                continue
            arr = data[time_i]
            arr = np.array(arr)
            lon_list = []
            lat_list = []
            value_list = []
            for i in range(len(arr)):
                for j in range(len(arr[i])):
                    lon_i = xx[i][j]
                    if lon_i > 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)

    def resample(self):
        fdir = join(self.datadir, rf'GPP_CEDAR\LT_CFE-Hybrid_NT\TIFF\\')
        outdir = join(self.datadir, rf'GPP_CEDAR\LT_CFE-Hybrid_NT\\resample_01')
        T.mk_dir(outdir, True)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, 0.1)

    def average(self):
        fdir_tmax = join(self.datadir, rf'terraclimate\tmax\resample_01\\')
        fdir_min = join(self.datadir, rf'terraclimate\tmin\resample_01\\')
        outdir = join(self.datadir, rf'terraclimate\tmax_min\average\\')
        T.mk_dir(outdir, True)
        for f in tqdm(T.listdir(fdir_tmax)):
            if not f.endswith('.tif'):
                continue
            fpath_tmax = join(fdir_tmax, f)
            print(fpath_tmax)
            fpath_min = join(fdir_min, f)
            # print(fpath_min);exit()
            outpath = join(outdir, f)
            array_tmax, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_tmax)
            array_min, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath_min)
            array = (array_tmax + array_min) / 2
            DIC_and_TIF(pixelsize=0.1).arr_to_tif(array, outpath)


    def per_pix(self):
        fdir = join(self.datadir, rf'terraclimate\tmax_min\\average\\')
        outdir =  join(self.datadir,rf'terraclimate\tmax_min\\per_pix')
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir, outdir)


class GIMMS_NDVI:
    def __init__(self):
        self.datadir = r'F:\Hotdrought_Resilience\data\\'
        pass
    def run(self):
        # self.resample()
        # self.monthly_compose_LAI4g()
        # self.scaling()
        self.per_pix()
        # self.check_data()
        # self.extract_all_gs_NDVI_based_temp()
        # self.annual_growth_season_NDVI()
        # self.annual_growth_season_NDVI_anomaly()
        # self.annual_growth_season_NDVI_detrend()


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

    def monthly_compose_LAI4g(self):
        fdir = join(self.datadir,'LAI4g','scaling_tif')
        outdir = join(self.datadir,'LAI4g','monthly_tif')
        T.mk_dir(outdir)
        self.monthly_compose(fdir,outdir,method='max')
        pass

    def monthly_compose(self, indir, outdir, date_fmt='yyyymmdd', method='mean'):
        '''
        :param method: "mean", "max" or "sum"
        :param date_fmt: 'yyyymmdd' or 'doy'
        :return:
        '''
        Tools().mkdir(outdir)
        year_list = []
        month_list = []
        for f in Tools().listdir(indir):
            y, m, d = self.get_year_month_day(f, date_fmt=date_fmt)
            year_list.append(y)
            month_list.append(m)
        year_list = Tools().drop_repeat_val_from_list(year_list)
        month_list = Tools().drop_repeat_val_from_list(month_list)
        compose_path_dic = {}
        for y in year_list:
            for m in month_list:
                date = (y, m)
                compose_path_dic[date] = []
        for f in Tools().listdir(indir):
            y, m, d = self.get_year_month_day(f, date_fmt=date_fmt)
            date = (y, m)
            compose_path_dic[date].append(join(indir, f))
        for date in compose_path_dic:
            flist = compose_path_dic[date]
            y, m = date
            print(f'{y}{m:02d}')
            outfname = f'{y}{m:02d}.tif'
            outpath = join(outdir, outfname)
            if os.path.isfile(outpath):
                continue
            self.compose_tif_list(flist, outpath, method=method)

    def get_year_month_day(self, fname, date_fmt='yyyymmdd'):
        try:
            if date_fmt == 'yyyymmdd':
                fname_split = fname.split('.')
                if not len(fname_split) == 2:
                    raise
                date = fname_split[0]
                if not len(date) == 8:
                    raise
                date_int = int(date)
                y = date[:4]
                m = date[4:6]
                d = date[6:]
                y = int(y)
                m = int(m)
                d = int(d)
                date_obj = datetime.datetime(y, m, d)  # check date availability
                return y, m, d
            elif date_fmt == 'doy':
                fname_split = fname.split('.')
                if not len(fname_split) == 2:
                    raise
                date = fname_split[0]
                if not len(date) == 7:
                    raise
                y = date[:4]
                doy = date[4:]
                doy = int(doy)
                date_base = datetime.datetime(int(y), 1, 1)
                time_delta = datetime.timedelta(doy - 1)
                date_obj = date_base + time_delta
                y = date_obj.year
                m = date_obj.month
                d = date_obj.day
                return y, m, d
        except:
            if date_fmt == 'yyyymmdd':
                raise UserWarning(
                    f'------\nfname must be yyyymmdd.tif e.g. 19820101.tif\nplease check your fname "{fname}"')
            elif date_fmt == 'doy':
                raise UserWarning(
                    f'------\nfname must be yyyyddd.tif e.g. 1982001.tif\nplease check your fname "{fname}"')

    def compose_tif_list(self, flist, outf, less_than=-9999, method='mean'):
        # less_than -9999, mask as np.nan
        if len(flist) == 0:
            return
        tif_template = flist[0]
        void_dic = DIC_and_TIF(tif_template=tif_template).void_spatial_dic()
        for f in tqdm(flist, desc='transforming...'):
            if not f.endswith('.tif'):
                continue
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
            for r in range(len(array)):
                for c in range(len(array[0])):
                    pix = (r, c)
                    val = array[r][c]
                    void_dic[pix].append(val)
        spatial_dic = {}
        for pix in tqdm(void_dic, desc='calculating mean...'):
            vals = void_dic[pix]
            vals = np.array(vals, dtype=float)
            vals[vals < less_than] = np.nan
            if method == 'mean':
                compose_val = np.nanmean(vals)
            elif method == 'max':
                compose_val = np.nanmax(vals)
            elif method == 'sum':
                compose_val = np.nansum(vals)
            else:
                raise UserWarning(f'{method} is invalid, should be "mean" "max" or "sum"')
            spatial_dic[pix] = compose_val
        DIC_and_TIF(tif_template=tif_template).pix_dic_to_tif(spatial_dic, outf)


    def scaling(self):
        ## data*0.0001 and
        fdir=r'F:\Hotdrought_Resilience\data\LAI4g\resample_01\\'
        outdir=r'F:\Hotdrought_Resilience\data\LAI4g\scaling_tif\\'
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)

            # array = array*0.0001   NDVI
            array = array * 0.01
            array[array<0] = np.nan
            array[array>7] = np.nan
            ToRaster().array2raster(outpath, originX, originY, pixelWidth, pixelHeight, array)

        pass

    def per_pix(self):
        fdir = join(self.datadir,'LAI4g','monthly_tif')
        outdir = join(self.datadir,'LAI4g','per_pix')
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
        growing_season_fdir = join(data_root, 'CRU_temp', 'extract_SOS_EOS_index', 'extract_growing_season_10degree')
        outdir = join(data_root,'NDVI4g','extract_growing_season','extract_growing_season_10degree')
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
                    continue

                start = gs_dic[pix]["start"]
                end = gs_dic[pix]["end"]
                start_dic[pix] = start
                end_dic[pix] = end

                if start is None or end is None:
                    continue

                n_years = len(LAI_val) // 12
                gs_vals = []
                if n_years <38:
                    print(n_years)
                    continue


                for y in range(n_years):
                    year_vals = LAI_val[y * 12:(y + 1) * 12]
                    if isinstance(year_vals, np.ndarray) and np.all(np.isnan(year_vals)):
                        continue
                    else:

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
                                gs_vals.append(np.nan)

                result_dic[pix] = np.array(gs_vals, dtype=object)
                len_dic[pix] = len(gs_vals)

            # 保存生长季 LAI
            outf = join(outdir, f)
            np.save(outf, result_dic)


        # # 可选：保存 start/end 空间分布检查
        # array_start = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(start_dic)
        # array_end = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(end_dic)
        # array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        #
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_start, outdir + 'start.tif')
        #
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outdir + 'len.tif')
        #
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_end, outdir + 'end.tif')
        #
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        #
        # plt.imshow(array_start, cmap="jet")
        # plt.colorbar(label="Month Index (0=Jan)")
        #
        # plt.subplot(1, 2, 2)
        #
        # plt.imshow(array_end, cmap="jet")
        # plt.colorbar(label="Month Index (11=Dec)")
        # plt.show()


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
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                gs_array = dic[pix]

                # 若该像素为空或为 None
                if gs_array is None:
                    continue

                # 若是列表或object数组（包含 array([...]) + np.nan）
                if isinstance(gs_array, (list, np.ndarray)):
                    # 转换为object数组（确保兼容np.nan）
                    gs_array = np.array(gs_array, dtype=object)
                    if len(gs_array)<38:
                        continue



                    if len(gs_array) == 0:
                        continue

                    # 检查最后一个是否是nan（南半球补齐）
                    if isinstance(gs_array[-1], float) and np.isnan(gs_array[-1]):
                        gs_array = gs_array[:-1]

                    # 再次转换成2D数组（每年一个array）
                    try:
                        stacked = np.vstack(gs_array)
                    except Exception as e:
                        print(f"Warning: failed to stack pix {pix}, reason: {e}")
                        result_dic[pix] = np.nan
                        continue


                    try:
                        annual_mean = np.nanmean(stacked, axis=1)
                    except ZeroDivisionError:
                        print("All NaN at pix:", pix)
                        continue

                    lon, lat = DIC_and_TIF().pix_to_lon_lat(pix)


                    if lat < 0:
                        if len(annual_mean) == 38:
                            annual_mean = np.append(annual_mean, np.nan)


                    if lat >= 0 and len(annual_mean) != 39:
                        continue

                    # plt.plot(annual_mean)
                    # print(annual_mean)
                    # print(len(annual_mean))
                    # plt.show()


                    result_dic[pix] = annual_mean
                    len_dic[pix] = len(annual_mean)



            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)
        array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        outtif=join(outdir,'annual_growing_season_NDVI_len.tif')
        DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outtif)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()




    pass

    def annual_growth_season_NDVI_anomaly(self):
        """
        计算每个像素逐年的生长季 LAI 平均值
        - 北半球: 保留22年 (2003–2024)
        - 南半球: 如果是跨年生长季 → 只保留21年 (2004–2024)
                  如果是全年生长季 → 保留22年
        """
        fdir = join(data_root,'NDVI4g','annual_growing_season_NDVI')
        outdir = join(data_root,'NDVI4g','annual_growth_season_NDVI_detrend_relative_change')
        T.mk_dir(outdir, force=True)
        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                vals = dic[pix]
                vals=np.array(vals, dtype=object)


                if len(vals) == 0:
                    continue

                if np.isnan(np.nanmean(vals)):
                    continue
                if len(vals) <38:
                    continue
                # print(type(vals), vals.dtype)
                vals=list(vals)
                detrend_vals=T.detrend_vals(vals)
                # plt.plot(vals)
                # plt.plot(detrend_vals)
                # plt.title(pix)
                # plt.legend(['vals','detrend_vals'])
                # plt.show()
                average_val=np.nanmean(detrend_vals)
                # print(average_val)
                anomaly=(detrend_vals-average_val)/average_val*100
                if np.isnan(average_val):
                    continue
                # plt.plot(detrend_vals)
                # plt.plot(anomaly)
                # plt.plot(vals)
                # plt.title(pix)
                # plt.legend(['detrend','anomaly','vals'])
                # plt.show()

                result_dic[pix] = anomaly
                len_dic[pix] = len(anomaly)


            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)
        # array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        # outtif=join(outdir,'annual_growing_season_NDVI_len.tif')
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outtif)
        # plt.imshow(array_len, cmap="jet")
        # plt.colorbar(label="Month Index (11=Dec)")
        # plt.show()


    def annual_growth_season_NDVI_detrend(self):
        """

        """
        fdir = join(data_root,'NDVI4g','annual_growing_season_NDVI')
        outdir = join(data_root,'NDVI4g','annual_growth_season_NDVI_detrend')
        T.mk_dir(outdir, force=True)
        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                vals = dic[pix]
                vals=np.array(vals, dtype=object)


                if len(vals) == 0:
                    continue

                if np.isnan(np.nanmean(vals)):
                    continue
                if len(vals) <38:
                    continue
                # print(type(vals), vals.dtype)
                vals=list(vals)
                detrend_vals=T.detrend_vals(vals)
                # plt.plot(vals)
                # plt.plot(detrend_vals)
                # plt.title(pix)
                # plt.legend(['vals','detrend_vals'])
                # plt.show()
                #

                result_dic[pix] = detrend_vals
                len_dic[pix] = len(detrend_vals)


            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)
        # array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        # outtif=join(outdir,'annual_growing_season_NDVI_len.tif')
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outtif)
        # plt.imshow(array_len, cmap="jet")
        # plt.colorbar(label="Month Index (11=Dec)")
        # plt.show()

class extract_growing_season_temp():  ## use this
    def __init__(self):
        self.datadir = join(data_root,'terraclimate',)
        pass
    def run(self):
        # self.nc_to_tif()
        # self.filiter()

        # self.per_pix()
        # self.long_term_mean()
        # self.extract_SOS_EOS_index()
        # self.extract_all_gs_temp_based_temp()

        # self.annual_growth_season_temp()
        # self.check_data()
        self.annual_growth_season_temp_detrend_zscore()
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
        fdir=join(data_root,rf'terraclimate\tmax_min\per_pix',)

        outdir=join(data_root,rf'terraclimate\tmax_min','long_term_mean')
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

        fdir=join(data_root,rf'terraclimate\tmax_min\long_term_mean')


        outdir=join(data_root,rf'terraclimate\tmax_min\\','extract_SOS_EOS_index')
        T.mk_dir(outdir, force=True)

        for f in T.listdir(fdir):
            dic = T.load_npy(join(fdir,f))
            result_dic = {}

            for pix in tqdm(dic):
                val = np.array(dic[pix], dtype=float)

                # 找到温度 > 10°C 的月份索引
                index = np.where(val > 10)[0]

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
        LAI_dic_fdir = join(data_root, rf'terraclimate\PDSI\per_pix', )
        growing_season_fdir = join(data_root, rf'terraclimate\tmax_min', 'extract_SOS_EOS_index_10degree',)
        outdir = join(data_root, 'terraclimate\PDSI', 'extract_growing_season_10degree', )

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
                    continue

                start = gs_dic[pix]["start"]
                end = gs_dic[pix]["end"]
                start_dic[pix] = start
                end_dic[pix] = end

                if start is None or end is None:
                    continue

                n_years = len(LAI_val) // 12
                gs_vals = []
                if n_years <38:
                    print(n_years)
                    continue


                for y in range(n_years):
                    year_vals = LAI_val[y * 12:(y + 1) * 12]
                    if isinstance(year_vals, np.ndarray) and np.all(np.isnan(year_vals)):
                        continue
                    else:

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
                                gs_vals.append(np.nan)

                result_dic[pix] = np.array(gs_vals, dtype=object)
                len_dic[pix] = len(gs_vals)

            # 保存生长季 LAI
            outf = join(outdir, f)
            np.save(outf, result_dic)



        # 可选：保存 start/end 空间分布检查
        array_start = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(start_dic)
        array_end = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(end_dic)
        array_len = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(len_dic)
        outtifdir = join(data_root, f'terraclimate\PDSI'  ,'tifs')
        T.mk_dir(outtifdir, force=True)

        DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_start, join(outtifdir, 'start.tif'))

        DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_len, join(outtifdir, 'len.tif'))

        DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_end, join(outtifdir, 'end.tif'))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)

        plt.imshow(array_start, cmap="jet")
        plt.colorbar(label="Month Index (0=Jan)")

        plt.subplot(1, 2, 2)

        plt.imshow(array_end, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()

    def annual_growth_season_temp(self):
            # """
            # 计算每个像素逐年的生长季温度平均值
            # - 北半球: 1982-2020 39 year
            # - 南半球: 若为跨年生长季 → 38年 + 1个 nan (补齐为39年)
            #           若为全年生长季 → 保留39年
            # """""
        fdir = join(self.datadir, 'PDSI','extract_growing_season_10degree')
        outdir = join(self.datadir,'PDSI', 'annual_growth_season_10degree')
        T.mk_dir(outdir, force=True)

        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                gs_array = dic[pix]

                # 若该像素为空或为 None
                if gs_array is None:
                    continue

                # 若是列表或object数组（包含 array([...]) + np.nan）
                if isinstance(gs_array, (list, np.ndarray)):
                    # 转换为object数组（确保兼容np.nan）
                    gs_array = np.array(gs_array, dtype=object)


                    if len(gs_array) == 0:
                        continue

                    # 检查最后一个是否是nan（南半球补齐）
                    if isinstance(gs_array[-1], float) and np.isnan(gs_array[-1]):
                        gs_array = gs_array[:-1]

                    # 再次转换成2D数组（每年一个array）
                    try:
                        stacked = np.vstack(gs_array)
                    except Exception as e:
                        print(f"Warning: failed to stack pix {pix}, reason: {e}")
                        continue

                    # 年平均 (对每一行求平均)
                    annual_mean = np.nanmin(stacked, axis=1)
                    # print(len(annual_mean))
                    # plt.plot(annual_mean)
                    # plt.show()

                    result_dic[pix] = annual_mean
                    len_dic[pix] = len(annual_mean)

            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)

        # 可视化每个像素的有效年数
        array_len = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Available Years")
        plt.title("Valid Year Counts per Pixel")
        plt.show()


    def check_data(self):
        fdir = join(self.datadir,'PDSI','extract_growing_season_10degree')
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_num = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            vals=np.array(vals, dtype=object)
            print(vals)

            if np.all(np.isnan(vals)):
                continue
            if np.isnan(vals).any():
                continue
            num = len(vals)
            # plt.plot(vals)
            # plt.show()
            spatial_dict_num[pix] = num
        arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_num)
        plt.imshow(arr)
        plt.show()


    def annual_growth_season_temp_detrend_zscore(self):
        """
        计算每个像素逐年的生长季 LAI 平均值
        - 北半球: 保留22年 (2003–2024)
        - 南半球: 如果是跨年生长季 → 只保留21年 (2004–2024)
                  如果是全年生长季 → 保留22年
        """
        fdir = join(data_root,'CRU_temp','annual_growth_season_temp_10degree')
        outdir = join(data_root,'CRU_temp','annual_growth_season_temp_detrend_zscore_10degree')
        T.mk_dir(outdir, force=True)
        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                vals = dic[pix]
                vals=np.array(vals, dtype=object)
                # print(vals)

                if len(vals) == 0:
                    continue
                if np.isnan(np.nanmean(vals)):
                    continue
                if len(vals) <38:
                    continue
                # print(type(vals), vals.dtype)
                vals=list(vals)
                detrend_vals=T.detrend_vals(vals)
                average_val=np.nanmean(detrend_vals)
                std_val=np.nanstd(detrend_vals)
                if std_val==0:
                    continue
                anomaly=(detrend_vals-average_val)/std_val
                if np.isnan(average_val):
                    continue
                # plt.plot(anomaly)
                # plt.plot(detrend_vals)
                # plt.plot(vals)
                #
                # plt.title(pix)
                # plt.legend(['zscore','detrend','original'])
                # plt.show()

                result_dic[pix] = anomaly
                len_dic[pix] = len(anomaly)


            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)
        array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        # outtif=join(outdir,'annual_growing_season_temp_len.tif')
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outtif)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()

class PDSI:
    def __init__(self):
        self.datadir = join(data_root,'PDSI')
        pass

    def run(self):
        self.extract_all_gs_PDSI_based_temp()
        self.extract_all_gs_PDSI_based_temp()


    def extract_all_gs_PDSI_based_temp(self):
        """
        批处理: 提取所有像素的 LAI 生长季 (多年序列)
        输出:
            result_dic : {pix: ndarray 或 None}, shape = (n_years, season_len)
        """
        LAI_dic_fdir = join(data_root, rf'terraclimate\PDSI\per_pix', )
        growing_season_fdir = join(data_root, rf'terraclimate\tmax_min', 'extract_SOS_EOS_index_10degree',)
        outdir = join(data_root, 'terraclimate\PDSI', 'extract_growing_season_10degree', )

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
                    continue

                start = gs_dic[pix]["start"]
                end = gs_dic[pix]["end"]
                start_dic[pix] = start
                end_dic[pix] = end

                if start is None or end is None:
                    continue

                n_years = len(LAI_val) // 12
                gs_vals = []
                if n_years <38:
                    print(n_years)
                    continue


                for y in range(n_years):
                    year_vals = LAI_val[y * 12:(y + 1) * 12]
                    if isinstance(year_vals, np.ndarray) and np.all(np.isnan(year_vals)):
                        continue
                    else:

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
                                gs_vals.append(np.nan)

                result_dic[pix] = np.array(gs_vals, dtype=object)
                len_dic[pix] = len(gs_vals)

            # 保存生长季 LAI
            outf = join(outdir, f)
            np.save(outf, result_dic)



        # 可选：保存 start/end 空间分布检查
        array_start = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(start_dic)
        array_end = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(end_dic)
        array_len = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(len_dic)
        outtifdir = join(data_root, f'terraclimate\PDSI'  ,'tifs')
        T.mk_dir(outtifdir, force=True)

        DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_start, join(outtifdir, 'start.tif'))

        DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_len, join(outtifdir, 'len.tif'))

        DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_end, join(outtifdir, 'end.tif'))

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)

        plt.imshow(array_start, cmap="jet")
        plt.colorbar(label="Month Index (0=Jan)")

        plt.subplot(1, 2, 2)

        plt.imshow(array_end, cmap="jet")
        plt.colorbar(label="Month Index (11=Dec)")
        plt.show()

    def annual_growth_season_PDSI(self):
            # """
            # 计算每个像素逐年的生长季温度平均值
            # - 北半球: 1982-2020 39 year
            # - 南半球: 若为跨年生长季 → 38年 + 1个 nan (补齐为39年)
            #           若为全年生长季 → 保留39年
            # """""
        fdir = join(self.datadir, 'PDSI','extract_growing_season_10degree')
        outdir = join(self.datadir,'PDSI', 'annual_growth_season_10degree')
        T.mk_dir(outdir, force=True)

        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                gs_array = dic[pix]

                # 若该像素为空或为 None
                if gs_array is None:
                    continue

                # 若是列表或object数组（包含 array([...]) + np.nan）
                if isinstance(gs_array, (list, np.ndarray)):
                    # 转换为object数组（确保兼容np.nan）
                    gs_array = np.array(gs_array, dtype=object)


                    if len(gs_array) == 0:
                        continue

                    # 检查最后一个是否是nan（南半球补齐）
                    if isinstance(gs_array[-1], float) and np.isnan(gs_array[-1]):
                        gs_array = gs_array[:-1]

                    # 再次转换成2D数组（每年一个array）
                    try:
                        stacked = np.vstack(gs_array)
                    except Exception as e:
                        print(f"Warning: failed to stack pix {pix}, reason: {e}")
                        continue

                    # 年平均 (对每一行求平均)
                    annual_mean = np.nanmin(stacked, axis=1)
                    # print(len(annual_mean))
                    # plt.plot(annual_mean)
                    # plt.show()

                    result_dic[pix] = annual_mean
                    len_dic[pix] = len(annual_mean)

            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)

        # 可视化每个像素的有效年数
        array_len = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Available Years")
        plt.title("Valid Year Counts per Pixel")
        plt.show()



class GPP:
    def __init__(self):
        self.datadir = join(data_root,'GPP_CEDAR')
        pass

    def run(self):
        # self.extract_all_gs_GPP_based_temp()
        # self.annual_growth_season_GPP()
        self.annual_growth_season_GPP_detrend()
        pass

    def extract_all_gs_GPP_based_temp(self):
        """
        批处理: 提取所有像素的 LAI 生长季 (多年序列)
        输出:
            result_dic : {pix: ndarray 或 None}, shape = (n_years, season_len)
        """
        LAI_dic_fdir = join(self.datadir, rf'LT_Baseline_NT\per_pix', )
        growing_season_fdir = join(data_root, rf'terraclimate\tmax_min', 'extract_SOS_EOS_index_10degree', )
        outdir = join(self.datadir, rf'LT_Baseline_NT\extract_growing_season_10degree')

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
                    continue

                start = gs_dic[pix]["start"]
                end = gs_dic[pix]["end"]
                start_dic[pix] = start
                end_dic[pix] = end

                if start is None or end is None:
                    continue

                n_years = len(LAI_val) // 12
                gs_vals = []
                if n_years < 38:
                    print(n_years)
                    continue

                for y in range(n_years):
                    year_vals = LAI_val[y * 12:(y + 1) * 12]
                    if isinstance(year_vals, np.ndarray) and np.all(np.isnan(year_vals)):
                        continue
                    else:

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
                                gs_vals.append(np.nan)

                result_dic[pix] = np.array(gs_vals, dtype=object)
                len_dic[pix] = len(gs_vals)

            # 保存生长季 LAI
            outf = join(outdir, f)
            np.save(outf, result_dic)

        # 可选：保存 start/end 空间分布检查
        # array_start = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(start_dic)
        # array_end = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(end_dic)
        # array_len = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(len_dic)
        # outtifdir = join(data_root, f'terraclimate\PDSI', 'tifs')
        # T.mk_dir(outtifdir, force=True)
        #
        # DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_start, join(outtifdir, 'start.tif'))
        #
        # DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_len, join(outtifdir, 'len.tif'))
        #
        # DIC_and_TIF(pixelsize=0.1).arr_to_tif(array_end, join(outtifdir, 'end.tif'))
        #
        # plt.figure(figsize=(12, 5))
        # plt.subplot(1, 2, 1)
        #
        # plt.imshow(array_start, cmap="jet")
        # plt.colorbar(label="Month Index (0=Jan)")
        #
        # plt.subplot(1, 2, 2)
        #
        # plt.imshow(array_end, cmap="jet")
        # plt.colorbar(label="Month Index (11=Dec)")
        # plt.show()

    def annual_growth_season_GPP(self):
        # """
        # 计算每个像素逐年的生长季温度平均值
        # - 北半球: 1982-2020 39 year
        # - 南半球: 若为跨年生长季 → 38年 + 1个 nan (补齐为39年)
        #           若为全年生长季 → 保留39年
        # """""
        fdir = join(self.datadir, rf'LT_Baseline_NT', 'extract_growing_season_10degree')
        outdir = join(self.datadir, 'LT_Baseline_NT', 'annual_growth_season_10degree')
        T.mk_dir(outdir, force=True)

        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                gs_array = dic[pix]

                # 若该像素为空或为 None
                if gs_array is None:
                    continue

                # 若是列表或object数组（包含 array([...]) + np.nan）
                if isinstance(gs_array, (list, np.ndarray)):
                    # 转换为object数组（确保兼容np.nan）
                    gs_array = np.array(gs_array, dtype=object)

                    if len(gs_array) == 0:
                        continue

                    # 检查最后一个是否是nan（南半球补齐）
                    if isinstance(gs_array[-1], float) and np.isnan(gs_array[-1]):
                        gs_array = gs_array[:-1]

                    # 再次转换成2D数组（每年一个array）
                    try:
                        stacked = np.vstack(gs_array)
                    except Exception as e:
                        print(f"Warning: failed to stack pix {pix}, reason: {e}")
                        continue

                    # 年平均 (对每一行求平均)
                    annual_mean = np.nanmin(stacked, axis=1)
                    # print(len(annual_mean))
                    # plt.plot(annual_mean)
                    # plt.show()

                    result_dic[pix] = annual_mean
                    len_dic[pix] = len(annual_mean)

            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)

        # 可视化每个像素的有效年数
        array_len = DIC_and_TIF(pixelsize=0.1).pix_dic_to_spatial_arr(len_dic)
        plt.imshow(array_len, cmap="jet")
        plt.colorbar(label="Available Years")
        plt.title("Valid Year Counts per Pixel")
        plt.show()


    def annual_growth_season_GPP_detrend(self):
        """

        """
        fdir = join(self.datadir, 'LT_CFE-Hybrid_NT', 'annual_growth_season_10degree')
        outdir = join(self.datadir, 'LT_CFE-Hybrid_NT', 'annual_growth_season_10degree_detrend')
        T.mk_dir(outdir, force=True)
        len_dic = {}

        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.npy'):
                continue

            dic = T.load_npy(join(fdir, f))
            result_dic = {}

            for pix in dic:
                r, c = pix
                # lon,lat=DIC_and_TIF().pix_to_lon_lat(pix) # # print(lon,lat) #
                # if not lon == 149.5:
                #     continue #
                # if not lat== -36.5: #
                #     continue
                vals = dic[pix]
                vals=np.array(vals, dtype=object)


                if len(vals) == 0:
                    continue

                if np.isnan(np.nanmean(vals)):
                    continue
                if len(vals) <38:
                    continue
                # print(type(vals), vals.dtype)
                vals=list(vals)
                detrend_vals=T.detrend_vals(vals)
                # plt.plot(vals)
                # plt.plot(detrend_vals)
                # plt.title(pix)
                # plt.legend(['vals','detrend_vals'])
                # plt.show()


                result_dic[pix] = detrend_vals
                len_dic[pix] = len(detrend_vals)


            # 保存每个文件结果
            outf = join(outdir, f)
            np.save(outf, result_dic)
        # array_len = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(len_dic)
        # outtif=join(outdir,'annual_growing_season_NDVI_len.tif')
        # DIC_and_TIF(pixelsize=0.5).arr_to_tif(array_len, outtif)
        # plt.imshow(array_len, cmap="jet")
        # plt.colorbar(label="Month Index (11=Dec)")
        # plt.show()
class pick_Drought:
    def __init__(self):
        self.datadir = join(data_root,)
        self.outdir = join(data_root,'terraclimate','PDSI','pick_drought\\')
        T.mk_dir(self.outdir, force=True)
        pass
    def run(self):
        # self.pick_multiyear_drought_events_year()
        self.generate_expected_GPP()
        pass

    def pick_multiyear_drought_events_year(self):
        # 载入数据
        PDSI_dir = join(self.datadir, 'terraclimate', 'PDSI', 'annual_growth_season_10degree')
        PDSI_dict = T.load_npy_dir(PDSI_dir)
        years = np.arange(1982, 2021)

        df_droughts = self.detect_multiyear_droughts(
            PDSI_dict=PDSI_dict,
            years=years,
            drought_threshold=-3,
            min_duration=2,

        )
        outdir = self.outdir + 'multiyear_drought\\'
        T.mk_dir(outdir)
        outpath = outdir + 'multiyear_droughts.npy'

        T.save_npy(df_droughts, outpath)


    def detect_multiyear_droughts(self, PDSI_dict, years, drought_threshold, min_duration, ):
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

        result_records = {}

        for pix in tqdm(PDSI_dict, desc="Detecting multiyear droughts"):

            vals = PDSI_dict[pix]
            vals[vals < -999] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue

            # === Step 1: 年度 SPI ===
            spi_annual = vals

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
                sub_spi = spi_annual[s:e + 1]
                min_idx = np.nanargmin(sub_spi)
                min_val = np.nanmin(sub_spi)

                min_year = years[s + min_idx]


                drought_years = [int(y) for y in years[s:e + 1]]
                # === Step 6: 干旱严重度 ===

                sub_spi = spi_annual[s:e + 1]
                # print(sub_spi)
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

                    "drought_years": drought_years,
                    "PDSI_min": float(min_val),
                    "PDSI_min_year": int(min_year),
                    "duration": len(drought_years),
                    "Drought_severity": float(severity),
                    "Post4yr_mean_PDSI": float(post_mean_spi),
                }
                result_records[pix] = record
            # pprint(result_records)


        return result_records


    def generate_expected_GPP(self):  #### here generate expected GPP
        GPP_dir=join(self.datadir,rf'GPP_CEDAR\LT_Baseline_NT\annual_growth_season_10degree_detrend')
        pick_drought_f=join(self.datadir,rf'terraclimate\PDSI\pick_drought\multiyear_drought\multiyear_droughts.npy')
        dic_GPP=T.load_npy_dir(GPP_dir)
        dic_drought=T.load_npy(pick_drought_f)
        out_dic= {}
        for pix, GPP in tqdm(dic_GPP.items(), desc='Generating expected GPP'):
            if pix not in dic_drought:
                continue

            drought_years = dic_drought[pix].get('drought_years', [])
            if len(GPP) == 0 or np.all(np.isnan(GPP)):
                continue

            years = np.arange(1982, 1982 + len(GPP))

            # === Step 1: 找出非干旱年份 ===
            mask_non_drought = np.array([y not in drought_years for y in years])

            non_drought_GPP = np.array(GPP)[mask_non_drought]

            mean_non_drought = np.nanmean(non_drought_GPP)

            # === Step 2: 用平均值替代干旱年 ===
            GPP_expected = GPP.copy()
            for i, y in enumerate(years):
                if y in drought_years:
                    GPP_expected[i] = mean_non_drought

            out_dic[pix] = GPP_expected

            # === Step 3: 保存结果 ===
        outdir=join(self.datadir,rf'GPP_CEDAR\LT_Baseline_NT\\expected_GPP')
        T.mk_dir(outdir,force=True)
        outf = join(outdir, 'expected_GPP.npy')
        np.save(outf, out_dic)












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



class Calculating_SPI:



    def __init__(self):
        self.fdir = r'F:\Hotdrought_Resilience\data\terraclimate\ppt\tif\\'
        self.outdir = r'F:\Hotdrought_Resilience\data\terraclimate\ppt\spi_dic\\'
        self.outroot = r'F:\Hotdrought_Resilience\data\terraclimate\ppt\chunk_dic_new\\'
        T.mk_dir(self.outroot)

    def run(self):
        # self.tif_to_spatial_dict_distributed()
        # self.concatenate_spatial_dict()
        self.compute_spi_12()
        # self.check_data()
        pass

    def tif_to_spatial_dict_distributed(self):
        '''
        written by Yang
        '''
        precip_dir = self.fdir
        # Pre_Process().data_transform_with_date_list()
        fname_list = []
        flag = 0
        bulk_number_i = 0
        files_in_each_bulk = 60
        files_in_total = len(T.listdir(precip_dir))
        bulk_number = files_in_total // files_in_each_bulk + 1
        for f in T.listdir(precip_dir):
            if not f.endswith('.tif'):
                continue
            fname_list.append(f)
            flag += 1
            if flag == files_in_each_bulk:
                print(fname_list)
                start_date = fname_list[0].split('.')[0]
                end_date = fname_list[-1].split('.')[0]
                outdir_i = join(self.outdir, f"{start_date}-{end_date}")
                if isdir(outdir_i):
                    flag = 0
                    fname_list = []
                    bulk_number_i += 1
                    continue
                print(outdir_i)
                print('------------------')
                T.mk_dir(outdir_i,force=True)
                Pre_Process().data_transform_with_date_list(precip_dir, outdir_i, fname_list, n=100000)
                flag = 0
                fname_list = []
                bulk_number_i += 1
            if bulk_number_i == bulk_number:
                fname_list.append(f)
        print(fname_list)
        start_date = fname_list[0].split('.')[0]
        end_date = fname_list[-1].split('.')[0]
        outdir_i = join(self.outdir, f"{start_date}-{end_date}")
        if isdir(outdir_i):
            return
        T.mk_dir(outdir_i,force=True)
        Pre_Process().data_transform_with_date_list(precip_dir, outdir_i, fname_list, n=100000)

    def concatenate_spatial_dict(self):
        fdir = r'F:\Hotdrought_Resilience\data\terraclimate\ppt\spatial_dict\\'
        outdir = r'F:\Hotdrought_Resilience\data\terraclimate\ppt\concatenate_spatial_dict\\'
        T.mk_dir(outdir,force=True)
        flist = []
        for date_range in T.listdir(fdir):
            for f in T.listdir(join(fdir, date_range)):
                flist.append(f)
            break
        for f in tqdm(flist):
            spatial_dict = {}
            spatial_dict_array = {}
            for date_range in T.listdir(fdir):
                fpath = join(fdir,date_range,f)
                spatial_dict_i = T.load_npy(fpath)
                for pix in spatial_dict_i:
                    spatial_dict[pix] = []
                break
            for date_range in T.listdir(fdir):
                fpath = join(fdir, date_range, f)
                spatial_dict_i = T.load_npy(fpath)
                for pix in spatial_dict_i:
                    vals = spatial_dict_i[pix]
                    spatial_dict[pix].append(vals)
            for pix in spatial_dict:
                vals_list = spatial_dict[pix]
                vals_cat = np.concatenate(vals_list)
                spatial_dict_array[pix] = vals_cat
            outf = join(outdir,f)
            T.save_npy(spatial_dict_array,outf)
        pass

    def compute_spi_12(self):
        """计算一个像素的 SPI-12 序列"""
        # running in wheat.snrenet.arizona.edu
        fdir= '/data/home/wenzhang/Hotdrought_resilience/PPT/per_pix'
        outdir='/data/home/wenzhang/Hotdrought_resilience/PPT/spi_dic'
        scale = 12
        T.mk_dir(outdir,force=True)
        distrib = indices.Distribution('gamma')
        Periodicity = compute.Periodicity(12)
        params_list = []
        for f in T.listdir(fdir):
            params = [fdir,f,scale,distrib,Periodicity,outdir]
            params_list.append(params)
            # self.kernel_compute_spi_12(params)
        MULTIPROCESS(self.kernel_compute_spi_12,params_list).run(process=16)

    def kernel_compute_spi_12(self,params):
        fdir, f, scale, distrib, Periodicity, outdir = params
        fpath = join(fdir, f)
        spatial_dict_i = T.load_npy(fpath)
        spi_dict_i = {}
        for pix in spatial_dict_i:
            vals = spatial_dict_i[pix]
            vals = np.array(vals)
            vals[vals<-999] = np.nan
            if T.is_all_nan(vals):
                continue
            spi = climate_indices.indices.spi(
                values=vals,
                scale=scale,
                distribution=distrib,
                periodicity=Periodicity,
                data_start_year=1958,
                calibration_year_initial=1958,
                calibration_year_final=2000,
            )
            spi_dict_i[pix] = spi
        outf = join(outdir, f)
        T.save_npy(spi_dict_i, outf)
        pass

    def check_data(self):
        fdir=r'F:\Hotdrought_Resilience\data\terraclimate\ppt\spi_dic\\'
        for f in T.listdir(fdir):
            if not 'per_pix_dic_100.npy' in f:
                continue
            spatial_dict = T.load_npy(join(fdir,f))
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                if T.is_all_nan(vals):
                    continue
                plt.plot(vals)
                plt.show()
                break




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
    # Download_TerraClimate().run()
    # extract_growing_season_temp().run()

    # GIMMS_NDVI().run()  ## process vegetion index GPP and LAI

    # PDSI().run()
    # GPP().run()
    pick_Drought().run()
    # Calculating_SPI().run()

    pass


if __name__ == '__main__':
    main()
    pass