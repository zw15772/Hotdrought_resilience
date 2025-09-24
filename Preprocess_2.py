# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from climate_indices import compute
from climate_indices import indices
import climate_indices
from meta_info import *
from multiprocessing import Pool, shared_memory
from multiprocessing import Process, Value, Array

class ERA5_SM:

    def __init__(self):
        self.datadir = join(data_root, 'ERA5/SM/')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.resample_1_deg()
        # self.sum_all_layers()
        # self.per_pix()
        # self.per_pix_1_deg()
        # self.anomaly()
        # self.anomaly_detrend()
        # self.anomaly_detrend_GS()
        # self.anomaly_GS()
        # self.origin_val_GS()
        # self.climatology_juping()
        self.climatology_mean()
        # self.check_data()
        pass

    def nc_to_tif(self):
        layer_list = ['layer2','layer3','layer4']
        for layer in layer_list:
            fdir = join(self.datadir, 'nc', layer)
            params_list = []
            for year in T.listdir(fdir):
                f = join(fdir, year,'data.nc')
                outdir = join(self.datadir, 'tif',layer, year)
                T.mk_dir(outdir,force=True)
                params = (f,'swvl'+layer.replace('layer',''),outdir)
                params_list.append(params)

                # T.nc_to_tif(f, 'swvl1', outdir)
            MULTIPROCESS(self.kernel_nc_to_tif,params_list).run(process=4)

        pass

    def kernel_nc_to_tif(self,params):
        f, var_i, outdir = params
        T.nc_to_tif(f, var_i, outdir)

    def resample(self):
        layer_list = ['layer2','layer3','layer4']
        for layer in layer_list:
            fdir = join(self.datadir,'tif',layer)
            outdir = join(self.datadir,'tif_025',layer)
            T.mk_dir(outdir,force=True)
            for year in tqdm(T.listdir(fdir)):
                fdir_i = join(fdir,year)
                for f in T.listdir(fdir_i):
                    fpath = join(fdir_i,f)
                    outpath = join(outdir,f)
                    ToRaster().resample_reproj(fpath,outpath,res=0.25)

    def resample_1_deg(self):
        fdir = join(self.datadir, 'sum_all_layers')
        outdir = join(self.datadir, 'sum_all_layers_1_deg')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, res=1)
        pass

    def per_pix(self):
        fdir = join(self.datadir,'sum_all_layers')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(global_start_year,global_end_year+1):
            for m in range(1,13):
                f = '{}{:02d}01.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def per_pix_1_deg(self):
        fdir = join(self.datadir,'sum_all_layers_1_deg')
        outdir = join(self.datadir,'per_pix_1_deg',global_year_range)
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(global_start_year,global_end_year+1):
            for m in range(1,13):
                f = '{}{:02d}01.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix/',global_year_range)
        outdir = join(self.datadir,'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)


    def sum_all_layers(self):
        # outdir = join(self.datadir,'sum_all_layers')
        outdir = join(self.datadir,'sum_2-3_layers')
        T.mk_dir(outdir,force=True)
        # layer_list = ['layer1','layer2','layer3','layer4']
        layer_list = ['layer2','layer3']
        layer_depth_dict = {
            'layer1':0.1,
            'layer2':0.3,
            'layer3':0.6,
            'layer4':1.0
        }
        fdir = join(self.datadir,'tif_025')
        flist = []
        for layer in layer_list:
            fdir_i = join(fdir,layer)
            for f in T.listdir(fdir_i):
                flist.append(f)
            break

        for f in tqdm(flist):
            sum_arr = 0
            for layer in layer_list:
                fpath = join(fdir,layer,f)
                arr = ToRaster().raster2array(fpath)[0]
                arr[arr<0] = np.nan
                total_water = arr * layer_depth_dict[layer] # m3/m2
                total_water = total_water * 1000 # mm
                sum_arr += total_water
            outf = join(outdir,f)
            longitude_start = -180
            latitude_start = 90
            pixelWidth = 0.25
            pixelHeight = -0.25
            ToRaster().array2raster(outf, longitude_start, latitude_start, pixelWidth, pixelHeight, sum_arr)

    def anomaly_detrend(self):
        fdir = join(self.datadir, 'anomaly',global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend',global_year_range)
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir, f)
            T.save_npy(spatial_dict_detrend, outf)

    def anomaly_detrend_GS(self):
        fdir = join(self.datadir, 'anomaly_detrend', global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend_GS', global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)
        pass

    def origin_val_GS(self):
        fdir = join(self.datadir, 'per_pix', global_year_range)
        outdir = join(self.datadir, 'origin_GS', global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_obj_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                if len(consecutive_ranges) == 1:
                    consecutive_ranges = np.array(consecutive_ranges[0])
                    consecutive_ranges = np.reshape(consecutive_ranges, (-1, 12))
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)
        pass

    def anomaly_GS(self):
        fdir = join(self.datadir, 'anomaly', global_year_range)
        outdir = join(self.datadir, 'anomaly_GS', global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_obj_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                if len(consecutive_ranges) == 1:
                    consecutive_ranges = np.array(consecutive_ranges[0])
                    consecutive_ranges = np.reshape(consecutive_ranges, (-1, 12))
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)
        pass

    def climatology_juping(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'climatology_juping',global_year_range)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_juping = {}
            for pix in spatial_dict:
                r,c = pix
                vals = spatial_dict[pix]
                if T.is_all_nan(vals):
                    continue
                juping = Pre_Process().climatology_anomaly(vals)
                spatial_dict_juping[pix] = juping
            T.save_npy(spatial_dict_juping,outpath)

    def climatology_mean(self):
        fdir = join(self.datadir,'per_pix_1_deg',global_year_range)
        outdir = join(self.datadir, 'climatology_mean')
        T.mk_dir(outdir, force=True)
        spatial_dict = T.load_npy_dir(fdir)
        climatology_dict = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            pix_anomaly = []
            climatology_means = []
            for m in range(1, 13):
                one_mon = []
                for i in range(len(vals)):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(vals[i])
                mean = np.nanmean(one_mon)
                std = np.nanstd(one_mon)
                climatology_means.append(mean)
            climatology_dict[pix] = climatology_means
        outf = join(outdir, 'climatology_mean.npy')
        T.save_npy(climatology_dict, outf)

    def check_data(self):
        fdir = join(self.datadir,'anomaly_detrend_GS',global_year_range)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_num = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            num = len(vals)
            spatial_dict_num[pix] = num
        arr = D.pix_dic_to_spatial_arr(spatial_dict_num)
        plt.imshow(arr)
        plt.show()

class ERA5_Tair:

    def __init__(self):
        self.datadir = join(data_root, 'ERA5/Tair/')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.tif_to_perpix_1982_2020()
        # self.resample_1_deg()
        # self.tif_to_perpix_1982_2020_1_deg()
        # self.anomaly_detrend()
        # self.anomaly_detrend_GS()
        # self.annual_mean()
        # self.MAT()
        # self.MAT_F_to_C()
        self.MAT_C_1_deg()
        # self.anomaly_GS()
        # self.origin_val_GS()
        # self.climatology_juping()
        # self.climatology_mean()
        pass

    def nc_to_tif(self):

        fdir = join(self.datadir, 'nc')
        params_list = []
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            year = f.split('.')[0]
            # print(f)
            # exit()
            outdir = join(self.datadir, 'tif')
            T.mk_dir(outdir,force=True)
            params = (fpath,'t2m',outdir)
            params_list.append(params)

            # T.nc_to_tif(f, 'swvl1', outdir)
        MULTIPROCESS(self.kernel_nc_to_tif,params_list).run(process=4)

        pass

    def kernel_nc_to_tif(self,params):
        f, var_i, outdir = params
        T.nc_to_tif(f, var_i, outdir)

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            ToRaster().resample_reproj(fpath,outpath,res=0.25)

    def resample_1_deg(self):
        fdir = join(self.datadir, 'tif')
        outdir = join(self.datadir, 'tif_1_deg')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            ToRaster().resample_reproj(fpath, outpath, res=1)
        pass

    def tif_to_perpix_1982_2020(self):
        fdir = join(self.datadir,'tif_025')
        outdir = join(self.datadir,'per_pix/1982-2020')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2021):
            for m in range(1,13):
                f = '{}{:02d}01.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def tif_to_perpix_1982_2020_1_deg(self):
        fdir = join(self.datadir,'tif_1_deg')
        outdir = join(self.datadir,'per_pix_1_deg/1982-2020')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2021):
            for m in range(1,13):
                f = '{}{:02d}01.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix/1982-2020')
        outdir = join(self.datadir,'anomaly/1982-2020')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def anomaly_detrend(self):
        fdir = join(self.datadir, 'anomaly/1982-2020')
        outdir = join(self.datadir, 'anomaly_detrend/1982-2020')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir, f)
            T.save_npy(spatial_dict_detrend, outf)

    def anomaly_detrend_GS(self):
        fdir = join(self.datadir,'anomaly_detrend',global_year_range)
        outdir = join(self.datadir,'anomaly_detrend_GS',global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict,desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)
        pass

    def annual_mean(self):
        fdir = join(self.datadir,'tif_025')
        outdir = join(self.datadir,'MAT/annual_mean')
        T.mk_dir(outdir,force=True)
        year_list = global_year_list()
        month_list = range(1, 13)
        for year in year_list:
            print(str(year) + '\n')
            flist = []
            for mon in month_list:
                f = f'{year}{mon:02d}01.tif'
                fpath = join(fdir, f)
                flist.append(fpath)
            outf = join(outdir, f'{year}.tif')
            Pre_Process().compose_tif_list(flist, outf, method='mean')

        pass

    def MAT(self):
        fdir = join(self.datadir,'MAT/annual_mean')
        flist = []
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            flist.append(fpath)
        outf = join(self.datadir, 'MAT/MAT.tif')
        Pre_Process().compose_tif_list(flist, outf, method='mean')
        pass

    def MAT_F_to_C(self):
        fpath = join(self.datadir, 'MAT/MAT.tif')
        # arr = ToRaster().raster2array(fpath)[0]
        # arr = arr - 273.15
        spatial_dict = D.spatial_tif_to_dic(fpath)
        C_spatial_dict = {}
        for pix in spatial_dict:
            F_val = spatial_dict[pix]
            C_val = F_val - 273.15
            C_spatial_dict[pix] = C_val
        outf = join(self.datadir, 'MAT/MAT_C.tif')
        D.pix_dic_to_tif(C_spatial_dict,outf)

    def MAT_C_1_deg(self):
        fpath = join(self.datadir, 'MAT/MAT_C.tif')
        outpath = join(self.datadir, 'MAT/MAT_C_1_deg.tif')
        ToRaster().resample_reproj(fpath, outpath, res=1)

    def origin_val_GS(self):
        fdir = join(self.datadir, 'per_pix', global_year_range)
        outdir = join(self.datadir, 'origin_GS', global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_obj_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                # print(gs_mon)
                # exit()
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                if len(consecutive_ranges) == 1:
                    consecutive_ranges = np.array(consecutive_ranges[0])
                    consecutive_ranges = np.reshape(consecutive_ranges, (-1, 12))
                date_dict = dict(zip(list(range(len(date_list))), date_list))
                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)
        pass

    def anomaly_GS(self):
        fdir = join(self.datadir, 'anomaly', global_year_range)
        outdir = join(self.datadir, 'anomaly_GS', global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_obj_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                if len(consecutive_ranges) == 1:
                    consecutive_ranges = np.array(consecutive_ranges[0])
                    consecutive_ranges = np.reshape(consecutive_ranges, (-1, 12))
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)
        pass

    def climatology_juping(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'climatology_juping',global_year_range)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_juping = {}
            for pix in spatial_dict:
                r,c = pix
                vals = spatial_dict[pix]
                if T.is_all_nan(vals):
                    continue
                juping = Pre_Process().climatology_anomaly(vals)
                spatial_dict_juping[pix] = juping
            T.save_npy(spatial_dict_juping,outpath)

    def climatology_mean(self):
        fdir = join(self.datadir,'per_pix_1_deg',global_year_range)
        outdir = join(self.datadir, 'climatology_mean')
        T.mk_dir(outdir, force=True)
        spatial_dict = T.load_npy_dir(fdir)
        climatology_dict = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            pix_anomaly = []
            climatology_means = []
            for m in range(1, 13):
                one_mon = []
                for i in range(len(vals)):
                    mon = i % 12 + 1
                    if mon == m:
                        one_mon.append(vals[i])
                mean = np.nanmean(one_mon)
                std = np.nanstd(one_mon)
                climatology_means.append(mean)
            climatology_dict[pix] = climatology_means
        outf = join(outdir, 'climatology_mean.npy')
        T.save_npy(climatology_dict, outf)

class GLEAM_SM:

    def __init__(self):
        self.datadir = data_root + 'GLEAM/SMsurf/'
        T.mk_dir(self.datadir)
        pass


    def run(self):
        # self.nc_to_tif()
        # self.tif_to_perpix_1982_2015()
        # self.anomaly()
        # self.detrend()
        # self.detrend_GS()
        self.anomaly_detrend_smooth()
        pass


    def nc_to_tif(self):
        f = join(self.datadir,'nc/SMsurf_1980-2020_GLEAM_v3.5a_MO.nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        ncin = Dataset(f, 'r')
        ncin_xarr = xr.open_dataset(f)
        print(ncin.variables)
        # exit()
        lat = ncin['lat']
        lon = ncin['lon']
        pixelWidth = lon[1] - lon[0]
        pixelHeight = lat[1] - lat[0]
        longitude_start = lon[0]
        latitude_start = lat[0]
        time_obj = ncin.variables['time']
        start = datetime.datetime(1900, 1, 1)
        # print(time)
        # for t in time:
        #     print(t)
        # exit()
        flag = 0
        for i in tqdm(range(len(time_obj))):
            # print(i)
            flag += 1
            # print(time[i])
            date = start + datetime.timedelta(days=int(time_obj[i]))
            year = str(date.year)
            # exit()
            month = '%02d' % date.month
            day = '%02d' % date.day
            date_str = year + month
            newRasterfn = join(outdir,date_str + '.tif')
            if os.path.isfile(newRasterfn):
                continue
            # print(date_str)
            # exit()
            # if not date_str[:4] in valid_year:
            #     continue
            # print(date_str)
            # exit()
            # arr = ncin.variables['pet'][i]
            arr = ncin_xarr.variables['SMsurf'][i]
            arr = np.array(arr)
            arr[arr<0] = np.nan
            arr = arr.T
            # plt.imshow(arr)
            # plt.show()
            # print(arr)
            # grid = arr < 99999
            # arr[np.logical_not(grid)] = -999999
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # DIC_and_TIF().arr_to_tif(arr,newRasterfn)
            # plt.imshow(arr,'RdBu')
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()

        pass

    def tif_to_perpix_1982_2015(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'per_pix/1982-2015')
        T.mk_dir(outdir,force=True)
        selected_tif_list = []
        for y in range(1982,2021):
            for m in range(1,13):
                f = '{}{:02d}.tif'.format(y,m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir,outdir,selected_tif_list)

    def anomaly(self):
        fdir = join(self.datadir, 'per_pix/1982-2020')
        outdir = join(self.datadir, 'anomaly/1982-2020')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

    def detrend(self):
        fdir = join(self.datadir, 'anomaly/1982-2020')
        outdir = join(self.datadir, 'detrend/1982-2020')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir, f)
            T.save_npy(spatial_dict_detrend, outf)

    def detrend_GS(self):
        fdir = join(self.datadir, 'per_pix_GS/1982-2020')
        outdir = join(self.datadir, 'per_pix_GS_detrend/1982-2020')
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir, f)
            T.save_npy(spatial_dict_detrend, outf)

    def anomaly_detrend_smooth(self):
        fdir = join(self.datadir, 'detrend', global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend_smooth', global_year_range)
        T.mk_dir(outdir,force=True)

        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_smooth = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                std = np.nanstd(vals)
                if std == 0 or np.isnan(std):
                    continue
                vals_smoothed = SMOOTH().forward_window_smooth(vals,window=3)
                plt.plot(vals)
                plt.plot(vals_smoothed)
                plt.show()

            pass

        pass

class NDVI4g:
    def __init__(self):
        self.datadir = join(data_root,'NDVI4g')
        pass

    def run(self):
        # self.rename()
        # self.resample()
        # self.MVC()
        # self.remove_999()
        # self.tif_to_perpix_1982_2020()
        # self.anomaly()
        # self.climatology_percentage()
        self.climatology_percentage_detrend()
        # self.anomaly_detrend()
        # self.anomaly_detrend_GS()
        pass

    def rename(self):
        fdir = join(self.datadir,'tif')
        for f in tqdm(T.listdir(fdir)):
            date = f.split('.')[-2].split('_')[-1]
            year = date[:4]
            mon = date[4:6]
            day = date[6:]
            if day == '02':
                day = '15'
            newf = f'{year}{mon}{day}.tif'
            new_f_path = join(fdir,newf)
            os.rename(join(fdir,f),new_f_path)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            if os.path.isfile(outf):
                continue
            f_path = join(fdir,f)
            ToRaster().resample_reproj(f_path,outf,0.25)
        pass

    def MVC(self):
        fdir = join(self.datadir,'tif_025')
        outdir = join(self.datadir,'tif_025_MVC')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')

    def tif_to_perpix_1982_2020(self):
        fdir = join(self.datadir, 'tif_025_MVC')
        outdir = join(self.datadir, 'per_pix/1982-2020')
        T.mk_dir(outdir, force=True)
        selected_tif_list = []
        for y in range(1982, 2021):
            for m in range(1, 13):
                f = '{}{:02d}.tif'.format(y, m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list)

    def remove_999(self):
        fdir = join(self.datadir,'tif_025_MVC')
        outdir = join(self.datadir,'tif_025_MVC')
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outpath = join(outdir,f)
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array <= -10000] = np.nan
            array[array > 10000] = np.nan
            ToRaster().array2raster(outpath,originX,originY,pixelWidth,pixelHeight,array)

    def anomaly(self):
        fdir = join(self.datadir, 'per_pix',global_year_range)
        outdir = join(self.datadir, 'anomaly',global_year_range)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)

    def climatology_percentage(self):
        fdir = join(self.datadir, 'per_pix', global_year_range)
        outdir = join(self.datadir, 'climatology_percentage', global_year_range)
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir),desc='climatology_percentage'):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            percentage_dict = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                if T.is_all_nan(vals):
                    continue
                vals = np.array(vals)
                vals_reshape = vals.reshape(-1,12)
                vals_reshape_T = vals_reshape.T
                percentage_matrix = []
                for mon_vals in vals_reshape_T:
                    mean = np.nanmean(mon_vals)
                    percentage = mon_vals / mean
                    percentage_matrix.append(percentage)
                percentage_matrix = np.array(percentage_matrix)
                percentage_matrix = percentage_matrix.T
                percentage_arr = percentage_matrix.flatten()
                percentage_dict[pix] = percentage_arr
            outf = join(outdir,f)
            T.save_npy(percentage_dict,outf)

        pass

    def climatology_percentage_detrend(self):
        fdir = join(self.datadir, 'climatology_percentage', global_year_range)
        outdir = join(self.datadir, 'climatology_percentage_detrend', global_year_range)
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir),desc='climatology_percentage_detrend'):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir,f)
            T.save_npy(spatial_dict_detrend,outf)
        pass

    def anomaly_detrend(self):
        fdir = join(self.datadir, 'anomaly',global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend',global_year_range)
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir, f)
            T.save_npy(spatial_dict_detrend, outf)

    def anomaly_detrend_GS(self):
        fdir = join(self.datadir, 'anomaly_detrend',global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend_GS',global_year_range)
        T.mk_dir(outdir, force=True)
        spatial_dict_GS = {}
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                r,c = pix
                vals = spatial_dict[pix]
                vals_GS = T.monthly_vals_to_annual_val(vals,grow_season=global_northern_hemi_gs)
                spatial_dict_GS[pix] = vals_GS
        outf = join(outdir, 'GS_mean.npy')
        T.save_npy(spatial_dict_GS, outf)

class LAI4g:
    def __init__(self):
        self.datadir = join(data_root,'LAI4g')
        pass

    def run(self):
        # self.rename()
        # self.resample()
        # self.resample_05()
        # self.MVC()
        # self.MVC_05()
        # self.tif_to_perpix_1982_2020()
        # self.tif_to_perpix_1982_2020_05()
        self.anomaly_05()
        pass

    def rename(self):
        fdir = join(self.datadir,'tif')
        for f in tqdm(T.listdir(fdir)):
            date = f.split('.')[-2].split('_')[-1]
            year = date[:4]
            mon = date[4:6]
            day = date[6:]
            if day == '02':
                day = '15'
            newf = f'{year}{mon}{day}.tif'
            new_f_path = join(fdir,newf)
            os.rename(join(fdir,f),new_f_path)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            if os.path.isfile(outf):
                continue
            f_path = join(fdir,f)
            ToRaster().resample_reproj(f_path,outf,0.25)
        pass

    def resample_05(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_05')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            if os.path.isfile(outf):
                continue
            f_path = join(fdir,f)
            ToRaster().resample_reproj(f_path,outf,0.5)
        pass


    def MVC(self):
        fdir = join(self.datadir,'tif_025')
        outdir = join(self.datadir,'tif_025_MVC')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')

    def MVC_05(self):
        fdir = join(self.datadir,'tif_05')
        outdir = join(self.datadir,'tif_05_MVC')
        T.mk_dir(outdir)
        Pre_Process().monthly_compose(fdir,outdir,method='max')

    def tif_to_perpix_1982_2020(self):
        fdir = join(self.datadir, 'tif_025_MVC')
        outdir = join(self.datadir, 'per_pix/1982-2020')
        T.mk_dir(outdir, force=True)
        selected_tif_list = []
        for y in range(1982, 2021):
            for m in range(1, 13):
                f = '{}{:02d}.tif'.format(y, m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list)


    def tif_to_perpix_1982_2020_05(self):
        fdir = join(self.datadir, 'tif_05_MVC')
        outdir = join(self.datadir, 'per_pix_05/1982-2020')
        T.mk_dir(outdir, force=True)
        selected_tif_list = []
        for y in range(1982, 2021):
            for m in range(1, 13):
                f = '{}{:02d}.tif'.format(y, m)
                selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list)

    def anomaly_05(self):
        fdir = join(self.datadir,'per_pix_05/1982-2020')
        outdir = join(self.datadir,'anomaly_05','1982-2020')
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)
        pass

class TerraClimate:

    def __init__(self):
        self.datadir = join(data_root,'TerraClimate')
        # self.product_list = ['pet','ppt']
        # self.product_list = ['tmax']
        self.product_list = ['pet']
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.tif_to_perpix_1982_2020()
        # self.annual_mean()
        # self.MAP()
        self.MAP_1_deg()

        pass

    def nc_to_tif(self):
        params_list = []
        for product in self.product_list:
            fdir = join(self.datadir,product,'nc')
            outdir = join(self.datadir,product,'tif')
            T.mk_dir(outdir)
            for f in tqdm(T.listdir(fdir),desc=product):
                # var_name = f.split('_')[0]
                var_name = product
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
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
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
                    arr = np.array(arr)
                    longitude_start = -180
                    latitude_start = 90
                    pixelWidth = lon[1] - lon[0]
                    pixelHeight = lat[1] - lat[0]
                    ToRaster().array2raster(outpath, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
                    # exit()

    def resample(self):
        for product in self.product_list:
            fdir = join(self.datadir,product,'tif')
            outdir = join(self.datadir,product,'tif_025')
            T.mk_dir(outdir)
            for f in tqdm(T.listdir(fdir),desc=product):
                outf = join(outdir,f)
                if os.path.isfile(outf):
                    continue
                f_path = join(fdir,f)
                ToRaster().resample_reproj(f_path,outf,0.25)

        pass

    def tif_to_perpix_1982_2020(self):
        for product in self.product_list:
            fdir = join(self.datadir,product, 'tif_025')
            outdir = join(self.datadir,product, 'per_pix/1982-2020')
            T.mk_dir(outdir, force=True)
            selected_tif_list = []
            for y in range(1982, 2021):
                for m in range(1, 13):
                    f = '{}{:02d}01.tif'.format(y, m)
                    selected_tif_list.append(f)
            Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list)

    def annual_mean(self):
        fdir = join(self.datadir,'ppt','tif_025')
        outdir = join(self.datadir,'ppt/MAP/annual_mean')
        T.mk_dir(outdir,force=True)
        year_list = global_year_list()
        month_list = range(1,13)
        for year in year_list:
            print(str(year)+'\n')
            flist = []
            for mon in month_list:
                f = f'{year}{mon:02d}01.tif'
                fpath = join(fdir,f)
                flist.append(fpath)
            outf = join(outdir,f'{year}.tif')
            if os.path.isfile(outf):
                continue
            Pre_Process().compose_tif_list(flist,outf,method='sum')
        pass

    def MAP(self):
        fdir = join(self.datadir,'ppt','MAP/annual_mean')
        flist = []
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            flist.append(fpath)
        outf = join(self.datadir,'ppt','MAP/MAP.tif')
        Pre_Process().compose_tif_list(flist, outf, method='mean')
        pass

    def MAP_1_deg(self):
        fpath = join(self.datadir,'ppt','MAP/MAP.tif')
        outpath = join(self.datadir,'ppt','MAP/MAP_1_deg.tif')
        ToRaster().resample_reproj(fpath,outpath,1)
        pass

class MODIS_Phenology:

    def __init__(self):
        self.datadir = join(data_root,'MODIS_Phenology')
        self.product_list = ['Greenup_1','Senescence_1']
        pass

    def run(self):
        # self.reproj()
        self.time_to_doy()
        # self.growin_season()
        pass

    def reproj(self):
        fdir = join(self.datadir, 'unzip')
        outdir = join(self.datadir, 'tif')
        T.mk_dir(outdir)
        for date in tqdm(T.listdir(fdir)):
            fdir_i = join(fdir, date)
            for f in T.listdir(fdir_i):
                product = f.split('.')[1]
                fpath = join(fdir_i, f'{date}.{product}.tif')
                outdir_i = join(outdir, product)
                T.mk_dir(outdir_i)
                outpath = join(outdir_i, date + '.tif')
                SRS = DIC_and_TIF().gen_srs_from_wkt(self.wkt())
                ToRaster().resample_reproj(fpath, outpath, .25, srcSRS=SRS, dstSRS='EPSG:4326')
        pass

    def time_to_doy(self):
        fdir = join(self.datadir, 'tif')
        outdir = join(self.datadir, 'tif_doy')
        T.mk_dir(outdir)
        base_time = datetime.datetime(1970, 1, 1)
        for product in T.listdir(fdir):
            outdir_i = join(outdir, product)
            T.mk_dir(outdir_i)
            fdir_i = join(fdir, product)
            for f in tqdm(T.listdir(fdir_i)):
                fpath = join(fdir_i, f)
                outf = join(outdir_i, f[:4]+ '.tif')
                year = int(f[:4])
                year_base_time = datetime.datetime(year, 1, 1)
                spatial_dict = D.spatial_tif_to_dic(fpath)
                doy_spatial_dict = {}
                for pix in spatial_dict:
                    val = spatial_dict[pix]
                    if val > 32700:
                        continue
                    delta_days = int(val)
                    date = base_time + datetime.timedelta(days=delta_days)
                    doy = date - year_base_time
                    doy_spatial_dict[pix] = doy.days
                arr = D.pix_dic_to_spatial_arr(doy_spatial_dict)
                D.arr_to_tif(arr, outf)
        T.open_path_and_file(outdir)


    def growin_season(self):
        fdir = join(self.datadir, 'tif_doy')
        outdir = join(self.datadir, 'tif_doy', 'Growing_Season')
        T.mk_dir(outdir)
        for f in T.listdir(join(fdir,'Greenup_1')):
            fpath1 = join(fdir,'Greenup_1',f)
            fpath2 = join(fdir,'Senescence_1',f)
            arr1 = D.spatial_tif_to_arr(fpath1)
            arr2 = D.spatial_tif_to_arr(fpath2)
            arr = arr2 - arr1
            outf = join(outdir,f)
            D.arr_to_tif(arr,outf)


    def wkt(self):
        wkt = '''
        PROJCS["Sinusoidal",
    GEOGCS["GCS_Undefined",
        DATUM["Undefined",
            SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Sinusoidal"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",0.0],
    UNIT["Meter",1.0]]'''
        return wkt

class GPP:

    def __init__(self):
        self.datadir = join(data_root,'GPP')
        # self.product = 'LT_Baseline_NT'
        self.product = 'LT_CFE-Hybrid_NT'

        self.datarange = '1982-2020'
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.perpix()
        # self.anomaly()
        # self.anomaly_detrend()
        # self.anomaly_detrend_GS()
        # self.perpix_GS()
        # self.perpix_GS_detrend()
        self.check_per_pix()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc',self.product)
        outdir = join(self.datadir,'tif',self.product,self.datarange)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.nc','.tif'))
            nc = Dataset(fpath)
            arr = nc.variables['GPP_mean'][:][0]
            arr = np.array(arr) * 0.01
            arr[arr<-9] = np.nan
            longitude_start = -180
            latitude_start = 90
            pixelWidth = 0.05
            pixelHeight = -0.05
            ToRaster().array2raster(outf,longitude_start, latitude_start, pixelWidth, pixelHeight, arr)


    def resample(self):
        fdir = join(self.datadir,'tif',self.product,self.datarange)
        outdir = join(self.datadir,'tif025',self.product,self.datarange)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.25)


    def perpix(self):
        fdir = join(self.datadir,'tif025',self.product,self.datarange)
        outdir = join(self.datadir,'per_pix',self.product,self.datarange)
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def anomaly(self):
        fdir = join(self.datadir,'per_pix',self.product,self.datarange)
        outdir = join(self.datadir,'anomaly',self.product,self.datarange)
        T.mk_dir(outdir,force=True)
        Pre_Process().cal_anomaly(fdir,outdir)


    def anomaly_detrend(self):
        fdir = join(self.datadir,'anomaly',self.product,self.datarange)
        outdir = join(self.datadir,'anomaly_detrend',self.product,self.datarange)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir,f)
            T.save_npy(spatial_dict_detrend,outf)

    def anomaly_detrend_GS(self):
        fdir = join(self.datadir, 'anomaly_detrend', self.product, global_year_range)
        outdir = join(self.datadir, 'anomaly_detrend_GS', self.product, global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)

    def perpix_GS(self):
        fdir = join(self.datadir, 'per_pix', self.product, global_year_range)
        outdir = join(self.datadir, 'per_pix_GS', self.product, global_year_range)
        T.mk_dir(outdir, force=True)
        # spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            outpath = join(outdir, f)
            spatial_dict = T.load_npy(fpath)
            annual_spatial_dict = {}
            for pix in tqdm(spatial_dict, desc=f):
                gs_mon = global_get_gs(pix)
                vals = spatial_dict[pix]
                vals_dict = dict(zip(date_list, vals))
                date_list_gs = []
                date_list_index = []
                for i, date in enumerate(date_list):
                    mon = date.month
                    if mon in gs_mon:
                        date_list_gs.append(date)
                        date_list_index.append(i)
                consecutive_ranges = T.group_consecutive_vals(date_list_index)
                date_dict = dict(zip(list(range(len(date_list))), date_list))

                # annual_vals_dict = {}
                annual_gs_list = []
                for idx in consecutive_ranges:
                    date_gs = [date_dict[i] for i in idx]
                    if not len(date_gs) == len(gs_mon):
                        continue
                    year = date_gs[0].year
                    vals_gs = [vals_dict[date] for date in date_gs]
                    vals_gs = np.array(vals_gs)
                    vals_gs[vals_gs < -9999] = np.nan
                    mean = np.nanmean(vals_gs)
                    annual_gs_list.append(mean)
                annual_gs_list = np.array(annual_gs_list)
                if T.is_all_nan(annual_gs_list):
                    continue
                annual_spatial_dict[pix] = annual_gs_list

            T.save_npy(annual_spatial_dict, outpath)

    def perpix_GS_detrend(self):
        fdir = join(self.datadir, 'origin_GS', self.product, global_year_range)
        outdir = join(self.datadir, 'origin_GS_detrend', self.product, global_year_range)
        T.mk_dir(outdir, force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir, f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_detrend = T.detrend_dic(spatial_dict)
            outf = join(outdir, f)
            T.save_npy(spatial_dict_detrend, outf)

    def check_per_pix(self):
        fdir = join(self.datadir, 'anomaly_detrend', self.product, global_year_range)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            # fpath = '/Volumes/NVME4T/hotdrought_CMIP/data/NDVI4g/anomaly_detrend_GS/1982-2020/GS_mean.npy'
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                if T.is_all_nan(spatial_dict[pix]):
                    print(pix)
                    continue
                vals = spatial_dict[pix]
                if np.nanstd(vals) == 0:
                    continue
                plt.plot(vals)
                plt.show()

class GLC2000:
    def __init__(self):
        self.datadir = join(data_root,'GLC2000')
        pass

    def run(self):
        # self.resample()
        # self.unify()
        # self.reclass_lc()
        # self.reclass_tif()
        self.plot_reclass_tif()

        pass

    def resample(self):
        fpath = join(self.datadir,'origin','glc2000_v1_1.tif')
        outdir = join(self.datadir,'resample')
        T.mk_dir(outdir)
        # mojority resample
        outf = join(outdir,'glc2000_025.tif')
        ToRaster().resample_reproj(fpath,outf,0.25)

    def unify(self):
        fpath = join(self.datadir,'resample','glc2000_025.tif')
        outdir = join(self.datadir,'unify')
        T.mk_dir(outdir)
        outf = join(outdir,'glc2000_025.tif')
        D.unify_raster(fpath,outf)
        pass


    def reclass_lc(self):
        outdir = join(self.datadir, 'reclass_lc')
        T.mk_dir(outdir)
        outf = join(outdir,'glc2000_025.npy')
        excel = join(self.datadir,'origin','Global_Legend.xls')
        tif = join(self.datadir,'unify','glc2000_025.tif')
        legend_df = pd.read_excel(excel)
        val_dic = T.df_to_dic(legend_df,'VALUE')
        spatial_dic = DIC_and_TIF().spatial_tif_to_dic(tif)
        lc_dict = {
            'evergreen':1,
            'deciduous':2,
            'shrubs':3,
            'grass':4,
            'crop':5,
        }
        reclass_dic = {}
        reclass_num_dic = {}
        for pix in spatial_dic:
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            val = int(val)
            lc = val_dic[val]['reclass_1']
            # lc = val_dic[val]['reclass_2']
            if type(lc) == float:
                continue
            # if not lc in lc_dict:
            #     continue
            reclass_dic[pix] = lc
            # val = lc_dict[lc]
            # reclass_num_dic[pix] = val
        T.save_npy(reclass_dic,outf)

    def reclass_tif(self):
        fpath = join(self.datadir,'reclass_lc','glc2000_025.npy')
        spatial_dic = T.load_npy(fpath)
        lc_dict = {
            'evergreen': 1,
            'deciduous': 2,
            'mixed': 3,
            'shrubs': 4,
            'grass': 5,
            'crop': 6,
        }
        df = pd.DataFrame.from_dict(lc_dict,orient='index',columns=['value'])
        AI_tif = join(Aridity_index().datadir,'Aridity_index.tif')
        AI_spatial_dic = D.spatial_tif_to_dic(AI_tif)
        dryland_spatial_dict = {}
        for pix in AI_spatial_dic:
            val = AI_spatial_dic[pix]
            if np.isnan(val):
                continue
            if val < 0.65:
                dryland_spatial_dict[pix] = val
        reclass_spatial_dict = {}
        for pix in dryland_spatial_dict:
            if not pix in spatial_dic:
                continue
            lc = spatial_dic[pix]
            if not lc in lc_dict:
                continue
            val = lc_dict[lc]
            reclass_spatial_dict[pix] = val

        outf = join(self.datadir,'reclass_lc','glc2000_025_reclass.tif')
        outf_legend = join(self.datadir,'reclass_lc','glc2000_025_reclass_legend.csv')
        D.pix_dic_to_tif(reclass_spatial_dict,outf)
        df.to_csv(outf_legend)

    def plot_reclass_tif(self):
        fpath = join(self.datadir,'reclass_lc','glc2000_025_reclass.tif')
        color_list = [
            '#066133',
            '#AED8B7',
            '#A362A5',
            '#002387',
            '#A02024',
            '#F7DE75',
        ]
        lc_dict = {
            'evergreen': 1,
            'deciduous': 2,
            'mixed': 3,
            'shrubs': 4,
            'grass': 5,
            'crop': 6,
        }
        # Blue represents high values, and red represents low values.
        cmap = Tools().cmap_blend(color_list)
        Plot().plot_Robinson(fpath,cmap=cmap)
        plt.show()
        pass

class SPI:
    def __init__(self):
        self.datadir = join(data_root,'SPI')
        pass

    def run(self):
        # self.cal_spi()
        # self.pick_SPI_year_range()
        # self.SPI_tif()
        # self.resample()
        self.per_pix()
        # self.every_month()
        # self.check_spi()
        pass

    def cal_spi(self):
        date_range = '1930-2020'
        data_start_year = 1930
        # P_dir = CRU().data_dir + 'pre/per_pix/'
        # P_dic = T.load_npy_dir(P_dir,condition='005')
        P_dic,_ = Load_Data().Precipitation_origin(date_range)
        scale_list = [1,3,6,9,12]
        for scale in scale_list:
            outdir = join(self.datadir,'per_pix',date_range)
            T.mk_dir(outdir,force=True)
            outf = join(outdir,f'spi{scale:02d}')
            # distrib = indices.Distribution('pearson')
            distrib = indices.Distribution('gamma')
            Periodicity = compute.Periodicity(12)
            spatial_dic = {}
            for pix in tqdm(P_dic,desc=f'scale {scale}'):
                r,c = pix
                # if r > 180:
                #     continue
                vals = P_dic[pix]
                vals = np.array(vals)
                vals = T.mask_999999_arr(vals,warning=False)
                if np.isnan(np.nanmean(vals)):
                    continue
                # zscore = Pre_Process().z_score_climatology(vals)
                spi = climate_indices.indices.spi(
                values=vals,
                scale=scale,
                distribution=distrib,
                data_start_year=data_start_year,
                calibration_year_initial=1960,
                calibration_year_final=2000,
                periodicity=Periodicity,
                # fitting_params: Dict = None,
                )
                spatial_dic[pix] = spi
                # plt.plot(spi)
                # plt.show()
            T.save_npy(spatial_dic,outf)

    def pick_SPI_year_range(self):
        fdir = join(self.datadir,'per_pix','1930-2020')
        # year_range = global_VIs_year_range_dict['NDVI3g']
        year_range = global_year_range
        outdir = join(self.datadir,'per_pix_05/per_pix',year_range)
        T.mk_dir(outdir,force=True)
        start_year = 1930
        end_year = 2020
        date_list = []
        for y in range(start_year,end_year + 1):
            for m in range(1,13):
                date = f'{y}-{m:02d}'
                date_list.append(date)
        pick_date_list = []
        pick_year_start = int(year_range.split('-')[0])
        pick_year_end = int(year_range.split('-')[1])
        for y in range(pick_year_start, pick_year_end + 1):
            for m in range(1, 13):
                date = f'{y}-{m:02d}'
                pick_date_list.append(date)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            outf = join(outdir,f)
            dic = T.load_npy(fpath)
            picked_vals_dic = {}
            for pix in tqdm(dic,desc=f):
                vals = dic[pix]
                dic_i = dict(zip(date_list,vals))
                picked_vals = []
                for date in pick_date_list:
                    val = dic_i[date]
                    picked_vals.append(val)
                picked_vals = np.array(picked_vals)
                picked_vals_dic[pix] = picked_vals
            T.save_npy(picked_vals_dic,outf)

    def SPI_tif(self):
        fdir = join(self.datadir,'per_pix_05/per_pix',global_year_range)
        outdir = join(self.datadir,'tif',global_year_range)
        T.mk_dir(outdir,force=True)
        date_list = global_date_list()
        fname_list = []
        for date in date_list:
            year = date.year
            mon = date.month
            fname = f'{year}{mon:02d}.tif'
            fname_list.append(fname)
        for f in tqdm(T.listdir(fdir)):
            spatial_dict = T.load_npy(join(fdir,f))
            outdir_i = join(outdir,f.split('.')[0])
            T.mk_dir(outdir_i)
            DIC_and_TIF().pix_dic_to_tif_every_time_stamp(spatial_dict,outdir_i,fname_list)

    def resample(self):
        fdir = join(self.datadir,'tif',global_year_range)
        outdir = join(self.datadir,'tif_025')
        for scale in T.listdir(fdir):
            folder = join(fdir,scale)
            outdir_i = join(outdir,scale)
            T.mk_dir(outdir_i,force=True)
            for f in tqdm(T.listdir(folder),desc=f'{scale}'):
                fpath = join(folder,f)
                outf = join(outdir_i,f)
                ToRaster().resample_reproj(fpath,outf,0.25)

    def per_pix(self):
        fdir = join(self.datadir,'tif_025')
        outdir = join(self.datadir,'per_pix',global_year_range)
        T.mk_dir(outdir,force=True)
        for scale in T.listdir(fdir):
            folder = join(fdir,scale)
            outdir_i = join(outdir,scale)
            Pre_Process().data_transform(folder,outdir_i)
        pass


    def every_month(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        outdir = join(self.datadir,'every_month',global_year_range)
        params_list = []
        for f in T.listdir(fdir):
            scale = f.split('.')[0]
            outdir_i = join(outdir, scale)
            T.mkdir(outdir_i, force=True)
            param = [fdir,f,outdir_i]
            # self.kernel_every_month(param)
            params_list.append(param)
        MULTIPROCESS(self.kernel_every_month,params_list).run(process=7)

    def kernel_every_month(self,params):
        fdir,f,outdir_i = params
        fpath = join(fdir, f)
        spatial_dict = T.load_npy(fpath)
        month_list = range(1, 13)
        for mon in month_list:
            spatial_dict_mon = {}
            for pix in spatial_dict:
                r, c = pix
                if r > 180:
                    continue
                vals = spatial_dict[pix]
                val_mon = T.monthly_vals_to_annual_val(vals, [mon])
                val_mon[val_mon < -10] = -999999
                num = T.count_num(val_mon, -999999)
                if num > 10:
                    continue
                val_mon[val_mon < -10] = np.nan
                if T.is_all_nan(val_mon):
                    continue
                spatial_dict_mon[pix] = val_mon
            outf = join(outdir_i, f'{mon:02d}')
            T.save_npy(spatial_dict_mon, outf)
        pass

    def check_spi(self):
        fdir = join(self.datadir,'per_pix',global_year_range)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict1 = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                spatial_dict1[pix] = len(vals)
                # spatial_dict1[pix] = np.mean(vals)
            arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict1)
            plt.imshow(arr)
            plt.show()
class Extract_GS:

    def __init__(self):

        pass

    def run(self):

        # self.NDVI()
        # self.LAI4g()
        # self.GPP_LT_Baseline_NT()
        # self.T_air()
        # self.GLEAM_SMSurf()
        # self.ERA5_SM()
        self.check()

    def NDVI(self):
        fdir = '/Users/liyang/Projects_data/Hot_drought2/data/NDVI4g/per_pix_dryland/NDVI4g'
        outdir = '/Users/liyang/Projects_data/Hot_drought2/data/NDVI4g/per_pix_dryland/NDVI4g_gs'
        outf = join(outdir,'gs_mean.npy')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        annual_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            gs_mon = global_get_gs(pix)
            vals = spatial_dict[pix]
            vals_dict = dict(zip(date_list,vals))
            date_list_gs = []
            date_list_index = []
            for i,date in enumerate(date_list):
                mon = date.month
                if mon in gs_mon:
                    date_list_gs.append(date)
                    date_list_index.append(i)
            consecutive_ranges = T.group_consecutive_vals(date_list_index)
            date_dict = dict(zip(list(range(len(date_list))),date_list))

            # annual_vals_dict = {}
            annual_gs_list = []
            for idx in consecutive_ranges:
                date_gs = [date_dict[i] for i in idx]
                if not len(date_gs) == len(gs_mon):
                    continue
                year = date_gs[0].year
                vals_gs = [vals_dict[date] for date in date_gs]
                vals_gs = np.array(vals_gs)
                vals_gs[vals_gs<-9999] = np.nan
                mean = np.nanmean(vals_gs)
                annual_gs_list.append(mean)
            annual_gs_list = np.array(annual_gs_list)
            annual_spatial_dict[pix] = annual_gs_list
        T.save_npy(annual_spatial_dict,outf)

    def LAI4g(self):
        fdir = '/Users/liyang/Projects_data/Hot_drought2/data/LAI4g/per_pix_dryland/LAI4g'
        outdir = '/Users/liyang/Projects_data/Hot_drought2/data/LAI4g/per_pix_dryland/LAI4g_gs'
        outf = join(outdir,'gs_mean.npy')
        T.mk_dir(outdir,force=True)
        spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        annual_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            gs_mon = global_get_gs(pix)
            vals = spatial_dict[pix]
            vals_dict = dict(zip(date_list,vals))
            date_list_gs = []
            date_list_index = []
            for i,date in enumerate(date_list):
                mon = date.month
                if mon in gs_mon:
                    date_list_gs.append(date)
                    date_list_index.append(i)
            consecutive_ranges = T.group_consecutive_vals(date_list_index)
            date_dict = dict(zip(list(range(len(date_list))),date_list))

            # annual_vals_dict = {}
            annual_gs_list = []
            for idx in consecutive_ranges:
                date_gs = [date_dict[i] for i in idx]
                if not len(date_gs) == len(gs_mon):
                    continue
                year = date_gs[0].year
                vals_gs = [vals_dict[date] for date in date_gs]
                vals_gs = np.array(vals_gs)
                vals_gs[vals_gs<-9999] = np.nan
                mean = np.nanmean(vals_gs)
                annual_gs_list.append(mean)
            annual_gs_list = np.array(annual_gs_list)
            annual_spatial_dict[pix] = annual_gs_list
        T.save_npy(annual_spatial_dict,outf)

    def GPP_LT_Baseline_NT(self):
        fdir = '/Users/liyang/Projects_data/Hot_drought2/data/GPP/per_pix/LT_Baseline_NT/1982-2020'
        outdir = '/Users/liyang/Projects_data/Hot_drought2/data/GPP/per_pix/LT_Baseline_NT_GS/1982-2020'
        outf = join(outdir, 'gs_mean.npy')
        T.mk_dir(outdir, force=True)
        spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        annual_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            gs_mon = global_get_gs(pix)
            vals = spatial_dict[pix]
            vals_dict = dict(zip(date_list, vals))
            date_list_gs = []
            date_list_index = []
            for i, date in enumerate(date_list):
                mon = date.month
                if mon in gs_mon:
                    date_list_gs.append(date)
                    date_list_index.append(i)
            consecutive_ranges = T.group_consecutive_vals(date_list_index)
            date_dict = dict(zip(list(range(len(date_list))), date_list))

            # annual_vals_dict = {}
            annual_gs_list = []
            for idx in consecutive_ranges:
                date_gs = [date_dict[i] for i in idx]
                if not len(date_gs) == len(gs_mon):
                    continue
                year = date_gs[0].year
                vals_gs = [vals_dict[date] for date in date_gs]
                vals_gs = np.array(vals_gs)
                vals_gs[vals_gs < -9999] = np.nan
                mean = np.nanmean(vals_gs)
                annual_gs_list.append(mean)
            annual_gs_list = np.array(annual_gs_list)
            if T.is_all_nan(annual_gs_list):
                continue
            annual_spatial_dict[pix] = annual_gs_list
        T.save_npy(annual_spatial_dict, outf)
        pass

    def T_air(self):
        fdir = join(data_root,'ERA5/Tair/per_pix/1982-2020')
        outdir = join(data_root,'ERA5/Tair/origin_GS/1982-2020')
        outf = join(outdir, 'gs_mean.npy')
        T.mk_dir(outdir, force=True)
        spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        annual_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            gs_mon = global_get_gs(pix)
            vals = spatial_dict[pix]
            vals_dict = dict(zip(date_list, vals))
            date_list_gs = []
            date_list_index = []
            for i, date in enumerate(date_list):
                mon = date.month
                if mon in gs_mon:
                    date_list_gs.append(date)
                    date_list_index.append(i)
            consecutive_ranges = T.group_consecutive_vals(date_list_index)
            date_dict = dict(zip(list(range(len(date_list))), date_list))

            # annual_vals_dict = {}
            annual_gs_list = []
            for idx in consecutive_ranges:
                date_gs = [date_dict[i] for i in idx]
                if not len(date_gs) == len(gs_mon):
                    continue
                year = date_gs[0].year
                vals_gs = [vals_dict[date] for date in date_gs]
                vals_gs = np.array(vals_gs)
                vals_gs[vals_gs < -9999] = np.nan
                mean = np.nanmean(vals_gs)
                annual_gs_list.append(mean)
            annual_gs_list = np.array(annual_gs_list)
            if T.is_all_nan(annual_gs_list):
                continue
            annual_spatial_dict[pix] = annual_gs_list
        T.save_npy(annual_spatial_dict, outf)
        pass

    def GLEAM_SMSurf(self):
        fdir = join(data_root,'GLEAM/SMsurf/per_pix/1982-2020')
        outdir = join(data_root,'GLEAM/SMsurf/per_pix_GS/1982-2020')
        outf = join(outdir, 'gs_mean.npy')
        T.mk_dir(outdir, force=True)
        spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        annual_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            gs_mon = global_get_gs(pix)
            vals = spatial_dict[pix]
            vals_dict = dict(zip(date_list, vals))
            date_list_gs = []
            date_list_index = []
            for i, date in enumerate(date_list):
                mon = date.month
                if mon in gs_mon:
                    date_list_gs.append(date)
                    date_list_index.append(i)
            consecutive_ranges = T.group_consecutive_vals(date_list_index)
            date_dict = dict(zip(list(range(len(date_list))), date_list))

            # annual_vals_dict = {}
            annual_gs_list = []
            for idx in consecutive_ranges:
                date_gs = [date_dict[i] for i in idx]
                if not len(date_gs) == len(gs_mon):
                    continue
                year = date_gs[0].year
                vals_gs = [vals_dict[date] for date in date_gs]
                vals_gs = np.array(vals_gs)
                vals_gs[vals_gs < -9999] = np.nan
                mean = np.nanmean(vals_gs)
                annual_gs_list.append(mean)
            annual_gs_list = np.array(annual_gs_list)
            if T.is_all_nan(annual_gs_list):
                continue
            annual_spatial_dict[pix] = annual_gs_list
        T.save_npy(annual_spatial_dict, outf)
        pass

    def ERA5_SM(self):
        fdir = join(data_root,'ERA5/SM/per_pix/1982-2020')
        outdir = join(data_root,'ERA5/SM/per_pix_GS/1982-2020')
        outf = join(outdir, 'gs_mean.npy')
        T.mk_dir(outdir, force=True)
        spatial_dict = T.load_npy_dir(fdir)
        date_list = global_date_list()
        annual_spatial_dict = {}
        for pix in tqdm(spatial_dict):
            gs_mon = global_get_gs(pix)
            vals = spatial_dict[pix]
            vals_dict = dict(zip(date_list, vals))
            date_list_gs = []
            date_list_index = []
            for i, date in enumerate(date_list):
                mon = date.month
                if mon in gs_mon:
                    date_list_gs.append(date)
                    date_list_index.append(i)
            consecutive_ranges = T.group_consecutive_vals(date_list_index)
            date_dict = dict(zip(list(range(len(date_list))), date_list))

            # annual_vals_dict = {}
            annual_gs_list = []
            for idx in consecutive_ranges:
                date_gs = [date_dict[i] for i in idx]
                if not len(date_gs) == len(gs_mon):
                    continue
                year = date_gs[0].year
                vals_gs = [vals_dict[date] for date in date_gs]
                vals_gs = np.array(vals_gs)
                vals_gs[vals_gs < -9999] = np.nan
                mean = np.nanmean(vals_gs)
                annual_gs_list.append(mean)
            annual_gs_list = np.array(annual_gs_list)
            if T.is_all_nan(annual_gs_list):
                continue
            annual_spatial_dict[pix] = annual_gs_list
        T.save_npy(annual_spatial_dict, outf)
        pass

    def check(self):
        fpath = '/Users/liyang/Projects_data/Hot_drought2/data/ERA5/Tair/anomaly_detrend_GS/1982-2020/GS_mean.npy'
        spatial_dict = T.load_npy(fpath)
        spatial_dict_len = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            # print(vals)
            vals_len = len(vals)
            spatial_dict_len[pix] = np.nanmean(vals)
        arr = D.pix_dic_to_spatial_arr(spatial_dict_len)
        plt.imshow(arr)
        plt.show()


class Aridity_index:

    def __init__(self):
        self.datadir = join(data_root,'Aridity_index')
        pass

    def run(self):
        self.cal_AI()
        pass

    def cal_AI(self):
        P_dir = join(TerraClimate().datadir,'ppt/per_pix',global_year_range)
        PET_dir = join(TerraClimate().datadir,'pet/per_pix',global_year_range)
        T.mk_dir(self.datadir)
        outf = join(self.datadir,'Aridity_index.tif')

        AI_spatial_dict = {}
        for f in tqdm(T.listdir(PET_dir)):
            fpath_P = join(P_dir,f)
            fpath_PET = join(PET_dir,f)
            spatial_P = T.load_npy(fpath_P)
            spatial_PET = T.load_npy(fpath_PET)
            for pix in spatial_PET:
                P = spatial_P[pix]
                PET = spatial_PET[pix]
                P = np.array(P)
                PET = np.array(PET)
                if np.std(P) == 0 or np.std(PET) == 0:
                    continue
                P_sum = np.sum(P)
                PET_sum = np.sum(PET)
                AI = P_sum/PET_sum
                if AI < 0:
                    continue
                if AI > 10:
                    continue
                AI_spatial_dict[pix] = AI
        D.pix_dic_to_tif(AI_spatial_dict,outf)

class BNPP:
    def __init__(self):
        self.datadir = join(data_root,'BNPP')
        pass

    def run(self):
        self.resample()
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            outf = join(outdir,f)
            fpath = join(fdir,f)
            ToRaster().resample_reproj(fpath,outf,0.25)

        pass

class Water_table_depth:
    def __init__(self):
        self.datadir = join(data_root,'water_table_depth')

    def run(self):
        # self.nc_to_tif()
        # self.unify()
        self.resample()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mkdir(outdir)
        fpath = join(fdir,'cwdx80.nc')
        outpath = join(outdir,'cwdx80.tif')
        ncin = Dataset(fpath)
        print(ncin.variables)
        lon_list = ncin['lon'][:]
        lat_list = ncin['lat'][:]
        # print(lon_list)
        # print(lat_list)
        # exit()
        arr = ncin['cwdx80'][::][::-1]
        arr = np.array(arr)
        arr[arr<0] = np.nan
        arr[arr>1000] = np.nan
        # plt.imshow(arr)
        # plt.show()
        ToRaster().array2raster(outpath, lon_list[0], lat_list[-1], 0.05, -0.05, arr)

    def unify(self):
        fpath = join(self.datadir,'tif','cwdx80.tif')
        outdir = join(self.datadir,'tif_unify')
        T.mk_dir(outdir)
        outpath = join(self.datadir,'tif_unify','cwdx80.tif')
        D.unify_raster(fpath,outpath)

    def resample(self):
        fpath = join(self.datadir,'tif_unify','cwdx80.tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        outf = join(outdir,'cwdx80.tif')
        ToRaster().resample_reproj(fpath,outf,0.25)

class HWSD:
    def __init__(self):
        self.datadir = join(data_root,'HWSD')
        pass

    def run(self):
        # self.nc_to_tif()
        self.resample()
        pass

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mkdir(outdir)
        fpath = join(fdir,'S_SILT.nc4')
        ncin = Dataset(fpath)
        arr = ncin['S_SILT'][::][::-1]
        arr = np.array(arr)
        arr[arr<0] = np.nan
        newRasterfn = join(outdir,'S_SILT.tif')
        ToRaster().array2raster(newRasterfn, -180, 90, 0.05, -0.05, arr)

    def resample(self):
        fpath = join(self.datadir,'tif','S_SILT.tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        outf = join(outdir,'S_SILT.tif')
        ToRaster().resample_reproj(fpath,outf,0.25)

class SoilGrids:

    def __init__(self):
        self.datadir = join(data_root, 'SoilGrids','SOC')
        pass

    def run(self):
        self.resample()
        self.sum_all_layer()
        pass

    def resample(self):
        fdir = join(self.datadir,'tif_origin')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for folder in tqdm(T.listdir(fdir)):
            print(folder)
            fpath = join(fdir,folder,'global_5000m_84_0.05_unify.tif')
            array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            array[array<=0] = np.nan
            outf = join(outdir,folder+'.tif')
            ToRaster().array2raster(outf,originX,originY,pixelWidth,pixelHeight,array)
            ToRaster().resample_reproj(outf,outf,0.25)

        pass

    def sum_all_layer(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_sum')
        T.mk_dir(outdir)
        outf = join(outdir,'SOC_sum.tif')
        flist = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            flist.append(fpath)
        Pre_Process().compose_tif_list(flist,outf,method='sum')


class Rooting_depth:

    def __init__(self):
        self.datadir = join(data_root, 'Rooting_depth')
        pass

    def run(self):
        # self.nc_to_tif()
        # self.resample()
        # self.unify_raster()
        self.merge_tifs()
        pass


    def nc_to_tif(self):
        fdir = join(self.datadir,'nc')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        for f in os.listdir(fdir):
            if not f.endswith('.nc'):
                continue
            print(f)
            nc = join(fdir,f)
            ncin = Dataset(nc, 'r')
            lat = ncin['lat'][::-1]
            lon = ncin['lon']
            pixelWidth = lon[1] - lon[0]
            pixelHeight = lat[0] - lat[1]
            longitude_start = lon[0]
            latitude_start = lat[-1]

            arr = ncin.variables['root_depth']
            arr = np.array(arr)
            arr[arr <= 0] = -999999
            # arr[arr > 999999] = -999999
            newRasterfn = join(outdir,f.replace('.nc','.tif'))
            ToRaster().array2raster(newRasterfn, longitude_start, latitude_start, pixelWidth, pixelHeight, arr)
            # grid = np.ma.masked_where(grid>1000,grid)
            # plt.imshow(arr,'RdBu',vmin=10,vmax=100)
            # plt.colorbar()
            # plt.show()
            # nc_dic[date_str] = arr
            # exit()
            pass

        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_025')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            outf = join(outdir,f)
            ToRaster().resample_reproj(fpath,outf,0.25)

    def unify_raster(self):
        fdir = join(self.datadir,'tif_025')
        outdir = join(self.datadir,'tif_025_unify')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            outf = join(outdir,f)
            D.unify_raster(fpath,outf)

    def merge_tifs(self):
        fdir = join(self.datadir,'tif_025_unify')
        outdir = join(self.datadir,'tif_025_unify_merge')
        T.mk_dir(outdir)
        flist = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            flist.append(fpath)
        outf = join(outdir,'rooting_depth.tif')
        Pre_Process().compose_tif_list(flist,outf,method='mean')

        pass

class CCI_LC:

    def __init__(self):
        self.datadir = join(data_root, 'CCI_LC')
        pass

    def run(self):
        # self.lccs_class_count()
        self.read_dict()
        pass

    def lccs_class_count(self):
        nc_dir = join(self.datadir,'nc')
        outdir = join(self.datadir,'lccs_class_count')
        for f in T.listdir(nc_dir):
            nc_path = join(nc_dir,f)
            # nc_path = '/Volumes/NVME4T/hotdrought_CMIP/data/CCI_LC/nc/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1996-v2.0.7cds.nc'
            self.array = self.nc_to_array(nc_path)
            self.array = np.array(self.array, dtype=np.int16)

            outf = join(self.datadir,f.replace('.nc','.npy'))
            self.data_transform(outf)
        pass

    def nc_to_array(self,nc_path):
        nc = nc_path
        ncin = Dataset(nc, 'r')
        # print(ncin.variables)
        # for var in ncin.variables:
        #     print(var)
        array = ncin.variables['lccs_class'][0][::]
        return array

    def data_transform(self,outf):
        # 不可并行，内存不足
        row = len(self.array)
        col = len(self.array[0])
        window_height = 0.25 / 180 * row
        window_width = 0.25 / 360 * col
        self.window_height = int(window_height)
        self.window_width = int(window_width)


        # moving window
        spatial_dict = {}
        params_list = []
        for i in tqdm(range(0, row, self.window_height)):
            for j in range(0, col, self.window_width):
                params = (i, j)
                params_list.append(params)
        # split params list into 8 processes
        params_list_split = np.array_split(params_list, 11)
        shm = shared_memory.SharedMemory(create=True, size=self.array.nbytes)
        results = MULTIPROCESS(self.kernel_data_transform, params_list_split).run(process_or_thread='p',process=11)
        for results_i in results:
            print(results_i)

        # T.save_npy(spatial_dict, outf)
    def kernel_data_transform(self, params_list):
        existing_shm = shared_memory.SharedMemory(name=shr_name)
        array = np.ndarray(self.array,dtype=np.int16,buffer=self.existing_shm.buf)
        spatial_dict = {}
        for params in params_list:
            i, j = params
            pix = (i, j)
            array_i = []
            for k in range(i, i + self.window_height):
                for l in range(j, j + self.window_width):
                    array_i.append(array[k][l])
            array_i = np.array(array_i)
            array_i = np.array(array_i, dtype=np.int16)
            dic = {}
            for k in array_i:
                if not k in dic:
                    dic[k] = 1
                else:
                    dic[k] += 1
            dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)
            spatial_dict[pix] = dic
        return spatial_dict



    def group_lc(self):
        group_lc = {
            10:'crop',
            11:'crop',
            12:'crop',
            20:'crop',
            30:'crop',

            40: 'EBF',

            46: 'DBF',
            50: 'DBF',
            54: 'DBF',

            55: 'NEF',
            56: 'NEF',
            60: 'NEF',

            61: 'DNF',
            62: 'DNF',
            66: 'DNF',

            70: 'MIX',

            76: 'shrubland',
            80: 'shrubland',
            81: 'shrubland',

            82: 'grassland',
        }
        return group_lc

    def read_dict(self,fpath):
        spatial_dict = T.load_npy(fpath)
        spatial_dict_group = {}
        lc_type_list = []
        spatial_dict_ratio_all = {}
        pix_dict = {}
        for pix in tqdm(spatial_dict):
            count_dict = spatial_dict[pix]
            spatial_dict_ratio_i = {}
            i,j = pix
            i = i / 90
            j = j / 90
            i = int(i)
            j = int(j)
            pix = (i,j)
            pix_dict[pix] = 1
            for count_dict_i in count_dict:
                lc_type = count_dict_i[0]
                lc_count = count_dict_i[1]
                if not lc_type in lc_type_list:
                    lc_type_list.append(lc_type)
                ratio_i = lc_count / 8100. * 100
                spatial_dict_ratio_i[lc_type] = ratio_i
            # print(spatial_dict_ratio_i)
            spatial_dict_ratio_all[pix] = spatial_dict_ratio_i
        df = T.dic_to_df(spatial_dict_ratio_all,'pix')
        T.print_head_n(df)
        lc_type_list.sort()
        for lc in lc_type_list:
            print(lc)
            spatial_dict_i = T.df_to_spatial_dic(df,lc)
            arr = D.pix_dic_to_spatial_arr(spatial_dict_i)
            outf = join(outdir,f'{lc}.tif')
            D.arr_to_tif(arr,outf)
        T.open_path_and_file(outdir)

def main():
    # ERA5_SM().run()
    # ERA5_Tair().run()
    # GLC2000().run()
    # GLEAM_SM().run()
    # NDVI4g().run()
    # TerraClimate().run()
    # MODIS_Phenology().run()
    LAI4g().run()
    # GPP().run()
    # SPI().run()
    # Extract_GS().run()
    # Aridity_index().run()
    # BNPP().run()
    # Water_table_depth().run()
    # HWSD().run()
    # SoilGrids().run()
    # Rooting_depth().run()
    # CCI_LC().run()

if __name__ == '__main__':
    main()
    pass