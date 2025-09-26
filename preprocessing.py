from global_init import *

# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint


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
        # self.per_pix()
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
        self.datadir = join(data_root,'CRU_temp',)
        pass
    def run(self):
        # self.nc_to_tif()
        # self.filiter()

        self.per_pix()
        pass


    def nc_to_tif(self):
        params_list = []

        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            # var_name = f.split('_')[0]
            var_name = 'tmp'
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

class extract_growing_season:
    def __init__(self):
        pass
    def run(self):
        # self.gen_gs_index()
        self.extract_phenology_monthly_variables()
        pass

    def gen_gs_index(self):
        f_phenology = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/4GST//4GST_global.npy'
        phenology_dic = T.load_npy(f_phenology)
        for pix in phenology_dic:
            dict_i = phenology_dic[pix]
            SeasType = dict_i['SeasType']
            sos = dict_i['Onsets']
            eos = dict_i['Offsets']
            pprint(dict_i)
            exit()

        pass

    def extract_phenology_monthly_variables(self):
        fdir = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/SPI/dic/spi09/'

        outdir = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/SPI/extract_phenology_monthly//spi09//'

        Tools().mk_dir(outdir, force=True)
        f_phenology = rf'/Volumes/SSD1T/Hotdrought_Resilience/data/4GST//4GST_global.npy'
        phenology_dic = T.load_npy(f_phenology)
        new_spatial_dic = {}
        for pix in phenology_dic:
            # val=phenology_dic[pix]['Onsets']
            val=phenology_dic[pix]['SeasType']
            if isinstance(val, np.ndarray):
                print("skip array:", val)
                new_spatial_dic[pix] = np.nan
                continue

            new_spatial_dic[pix]=val

        spatial_array=DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(new_spatial_dic)
        plt.imshow(spatial_array,interpolation='nearest',cmap='jet')
        plt.colorbar()
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