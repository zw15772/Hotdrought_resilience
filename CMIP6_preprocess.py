# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from meta_info import *
D_CMIP = DIC_and_TIF(pixelsize=1)

class CMIP6:

    def __init__(self):
        self.datadir = join(data_root, 'CMIP6')
        # self.product_list = ['pr','tas']
        # self.product_list = ['gpp']
        # self.product_list = ['tas']
        pass

    def run(self):
        # self.all_models_summary()
        # self.pick_nc_files()
        # self.nc_to_tif()
        # self.resample()
        # self.unify_tif_reshape()
        # self.clean_tif()
        # self.check_tif_shape()
        # self.models_daterange()
        # self.check_date_consecutiveness()
        # self.per_pix()
        # self.per_pix_detrend()
        # self.per_pix_anomaly()
        self.per_pix_anomaly_2020_2040()
        # self.per_pix_anomaly_based_history_climotology()
        # self.per_pix_juping_based_history_climotology()
        # self.per_pix_juping()
        # self.per_pix_annual()
        # self.per_pix_anomaly_detrend()
        # self.check_anomaly()
        # self.check_per_pix()
        # self.check_anomaly_time_series()
        # self.calculate_SPI()
        # self.tif_ensemble()
        # self.per_pix_ensemble()
        # self.per_pix_ensemble_anomaly()
        # self.per_pix_ensemble_anomaly_detrend()
        # self.per_pix_ensemble_anomaly_dynamic_baseline()
        # self.per_pix_ensemble_anomaly_moving_window_baseline()
        # self.per_pix_ensemble_anomaly_CMIP_historical_baseline()
        # self.per_pix_ensemble_anomaly_based_history_tas()
        # self.per_pix_ensemble_anomaly_based_history_sm()
        # self.per_pix_ensemble_anomaly_based_2020_2060()
        # self.per_pix_ensemble_std_anomaly_based_2020_2060()
        # self.check_individual_model()
        # self.check_ensemble_model()
        # self.modify_FGOALS_f3_L()
        pass

    def all_models_summary(self):
        result_dict = {}
        flag = 0
        # product_list = ['lai']
        product_list = ['vpd']
        for product in product_list:
            fdir = join(self.datadir, product,'nc')
            for f in T.listdir(fdir):
                fname = f.split('.')[0]
                product_i,realm_i,model_i,experiment_i,ensemble_i,_,time_i = fname.split('_')
                dict_i = {'product':product_i, 'realm':realm_i, 'model':model_i, 'experiment':experiment_i, 'ensemble':ensemble_i, 'time':time_i, 'fname':join(fdir,f)}
                flag += 1
                result_dict[flag] = dict_i
            df = T.dic_to_df(result_dict)
            T.df_to_excel(df,join(self.datadir,f'{product}','models_summary'),n=10000)


    def pick_nc_files(self):

        # experiment_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
        # product_list = ['pr','tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['tas']
        # product_list = ['mrsos']
        product_list = ['vpd']
        for product in product_list:
            print(product)
            excel = join(self.datadir,f'{product}', 'models_summary.xlsx')
            df = pd.read_excel(excel)
            # T.print_head_n(df)
            experiment_list = T.get_df_unique_val_list(df, 'experiment')
            fdir = join(self.datadir, product,'picked_nc')
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            df_product = df[df['product']==product]
            for experiment in experiment_list:
                df_i = df_product[df_product['experiment']==experiment]
                outdir_i = join(fdir,experiment)
                if not os.path.exists(outdir_i):
                    os.makedirs(outdir_i)
                model_list = T.get_df_unique_val_list(df_i,'model')
                for model in model_list:
                    df_model = df_i[df_i['model']==model]
                    outdir_model = join(outdir_i,model)
                    if not os.path.exists(outdir_model):
                        os.makedirs(outdir_model)
                    for index,row in df_model.iterrows():
                        fname = row['fname']
                        shutil.copy(fname, outdir_model)


    def kernel_nc_to_tif(self, params):
        try:
            fname, outdir, product = params
            try:
                ncin = Dataset(fname, 'r')
            except:
                return
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
            shape = np.shape(lat)

            time = ncin.variables['time'][:]
            basetime = ncin.variables['time'].units
            basetime = basetime.strip('days since ')
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M:%S')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M:%S.%f')
                    except:
                        basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M')
            data = ncin.variables[product]

            if len(shape) == 2:
                xx,yy = lon,lat
            else:
                xx,yy = np.meshgrid(lon, lat)
            for time_i in range(len(data)):
                # print(time_i)
                date = basetime + datetime.timedelta(days=time[time_i])
                time_str = time[time_i]
                mon = date.month
                year = date.year
                if year > 2100:
                    continue
                outf_name = f'{year}{mon:02d}.tif'
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
                        if lon_i >= 180:
                            lon_i -= 360
                        lat_i = yy[i][j]
                        value_i = arr[i][j]
                        lon_list.append(lon_i)
                        lat_list.append(lat_i)
                        value_list.append(value_i)
                DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list,outpath)
        except:
            fw = open(join(self.datadir,'kernel_nc_to_tif_error.txt'),'a')
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write(f'[{current_time}],')
            fw.write(','.join(params)+'\n')
            fw.close()

    def kernel_nc_to_tif_vpd(self, params):
        fname, outdir, product = params
        try:
            ncin = Dataset(fname, 'r')
        except:
            return
        lat = ncin.variables['lat'][:]
        lon = ncin.variables['lon'][:]
        shape = np.shape(lat)

        time = ncin.variables['time'][:]
        basetime = ncin.variables['time'].units
        basetime = basetime.strip('days since ')
        try:
            basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
        except:
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M:%S.%f')
                except:
                    basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d %H:%M')
        print(ncin.variables.keys())

        data = ncin.variables[product]

        if len(shape) == 2:
            xx, yy = lon, lat
        else:
            xx, yy = np.meshgrid(lon, lat)
        for time_i in range(len(data)):
            # print(time_i)
            date = basetime + datetime.timedelta(days=time[time_i])
            time_str = time[time_i]
            mon = date.month
            year = date.year
            if year > 2100:
                continue
            outf_name = f'{year}{mon:02d}.tif'
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
                    if lon_i >= 180:
                        lon_i -= 360
                    lat_i = yy[i][j]
                    value_i = arr[i][j]
                    lon_list.append(lon_i)
                    lat_list.append(lat_i)
                    value_list.append(value_i)
            DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list, outpath)

    def nc_to_tif(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']
        experiment_list = ['ssp245','ssp585']
        # product_list = ['pr', 'tas']
        # product_list = ['tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['mrsos']
        product_list = ['VPD']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product,'picked_nc')
            outdir = join(self.datadir, product,'tif')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir,experiment)
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir,experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment,model)
                    T.mkdir(outdir_model,force=True)
                    fdir_model = join(fdir_i,model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model,fname)
                        param_i = [fpath, outdir_model,product]
                        param_list.append(param_i)
        MULTIPROCESS(self.kernel_nc_to_tif, param_list).run(process=8)
        # exit()


    def kernel_unify_tif_shape(self,params):
        fpath, outf = params
        # ToRaster().un(fpath, outf)
        DIC_and_TIF().unify_raster(fpath, outf)
        pass

    def unify_tif_reshape(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['tas']
        product_list = ['VPD']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif_resample')
            # fdir = join(self.datadir, product, 'tif_resample_unify_clean')
            fdir = join(self.datadir, product, 'tif_resample_unify')
            # outdir = join(self.datadir, product, 'tif_resample_unify_clean')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                # if not isdir(fdir_i):
                #     continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model, fname)
                        outf = join(outdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year, mon = fname_split[:4], fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        # arr = ToRaster().raster2array(fpath)[0]
                        # fw.write(f'{str(np.shape(arr))},{product},{experiment},{model},{fname}\n')
                        # break
                        if isfile(outf):
                            continue
                        # print(fpath, outf)
                        # exit()
                        param_i = [fpath, outf]
                        # self.kernel_unify_tif_shape(param_i)
                        param_list.append(param_i)
                        # T.open_path_and_file(outdir_model)
        MULTIPROCESS(self.kernel_unify_tif_shape, param_list).run(process=7)
        # exit()


    def check_tif_shape(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['lai']
        # product_list = ['mrsos']
        product_list = ['VPD']
        # product_list = ['pr', 'tas']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fw = open(join(self.datadir, f'{product}','shapes.csv'), 'w')
            fdir = join(self.datadir, product, 'tif_resample_unify_clean')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                for model in tqdm(model_list, desc='model'):
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year, mon = fname_split[:4], fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        arr = ToRaster().raster2array(fpath)[0]
                        fw.write(f'{str(np.shape(arr))},{product},{experiment},{model},{fname}\n')
                        break
                        # param_i = [fpath, outf]
                        # self.kernel_unify_tif_shape(param_i)
                        # param_list.append(param_i)
                        # T.open_path_and_file(outdir_model)
        # MULTIPROCESS(self.kernel_unify_tif_shape, param_list).run(process=6)


    def kernel_resample(self,parms):
        fpath,outf,res = parms
        if isfile(outf):
            return
        # if 'ACCESS-ESM1-5' in fpath:
        #     print(fpath)
        #     exit()
        ToRaster().resample_reproj(fpath, outf, res=res)

    def resample(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        product_list = ['VPD']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif')
            outdir = join(self.datadir, product, 'tif_resample')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model, fname)
                        if not fpath.endswith('.tif'):
                            continue
                        # if 'ACCESS-ESM1-5' in fpath:
                        #     print(fpath)
                        #     exit()
                        outf = join(outdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year, mon = fname_split[:4], fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        params = [fpath,outf,1]
                        param_list.append(params)
                        # ToRaster().resample_reproj(fpath,outf,res=1)
        MULTIPROCESS(self.kernel_resample, param_list).run(process=7)


    def kernel_per_pix(self,params):
        fdir_model, outdir_model, flist_picked = params
        print(fdir_model)
        flist = T.listdir(fdir_model)
        if len(flist) == 0:
            return
        Pre_Process().data_transform_with_date_list(fdir_model,outdir_model,flist_picked)

    def per_pix(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['tas']
        product_list = ['VPD']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif_resample_unify_clean')
            outdir = join(self.datadir, product, 'per_pix')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    flist_picked = []
                    for fname in tqdm(fname_list, desc='fname'):
                        if not fname.endswith('.tif'):
                            continue
                        fpath = join(fdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year,mon = fname_split[:4],fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        flist_picked.append(fname)
                    # print(flist_picked)
                    # exit()
                    outdir_flist = T.listdir(outdir_model)
                    if len(outdir_flist) > 0:
                        continue
                    params = [fdir_model,outdir_model,flist_picked]
                    param_list.append(params)
                    # print(params)
                    # self.kernel_per_pix(params)
        MULTIPROCESS(self.kernel_per_pix, param_list).run(process=7)

    def models_daterange(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        product_list = ['VPD']
        param_list = []
        all_result_dict = {}
        index = 0
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif_resample_unify')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                for model in tqdm(model_list, desc='model'):
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    date_list = []
                    for fname in tqdm(fname_list, desc='fname'):
                        if not fname.endswith('.tif'):
                            continue
                        fpath = join(fdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year,mon = fname_split[:4],fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        date = fname_split
                        date_list.append(date)
                    date_list = sorted(date_list)
                    date_list = tuple(date_list)
                    info_dict = {'date_list':date_list,'product':product,'experiment':experiment,'model':model}
                    all_result_dict[index] = info_dict
                    index += 1
            df = T.dic_to_df(all_result_dict,key_col_str='index',col_order=['product','experiment','model','date_list'])
            T.df_to_excel(df,join(self.datadir,product,'models_daterange'))

    def kernel_per_pix_anomaly_juping(self,params):
        fdir_model, year_list, mon_list, outdir_model = params
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        if isfile(outf):
            return
        try:
            # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            anomaly_dict = {}
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                std = np.std(vals)
                if std == 0:
                    continue
                values_time_series_dict_year = {}
                values_time_series_dict_mon = {}
                year_list_unique = list(set(year_list))
                mon_list_unique = list(set(mon_list))
                for year in year_list_unique:
                    values_time_series_dict_year[year] = {}
                for mon in mon_list_unique:
                    values_time_series_dict_mon[mon] = {}
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    if val == 0:
                        continue
                    # values_time_series_dict_year[year][mon] = val
                    values_time_series_dict_mon[mon][year] = val
                climatology_info_dict = {}
                for mon in range(1, 13):
                    one_mon_val_list = values_time_series_dict_mon[mon].values()
                    one_mon_val_list = list(one_mon_val_list)
                    mean = np.nanmean(one_mon_val_list)
                    std = np.nanstd(one_mon_val_list)
                    climatology_info_dict[mon] = {'mean': mean, 'std': std}
                anomaly = []
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    mean = climatology_info_dict[mon]['mean']
                    std = climatology_info_dict[mon]['std']
                    anomaly_i = val - mean
                    anomaly.append(anomaly_i)
                anomaly = np.array(anomaly)
                anomaly_dict[pix] = anomaly

            T.save_npy(anomaly_dict, outf)
        except:
            fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_anomaly_2020_2040(self,params):
        fdir_model, year_list, mon_list, outdir_model = params
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        if isfile(outf):
            return
        try:
        # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            anomaly_dict = {}
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                std = np.std(vals)
                if std == 0:
                    continue
                values_time_series_dict_year = {}
                values_time_series_dict_mon = {}
                year_list_unique = list(set(year_list))
                mon_list_unique = list(set(mon_list))
                for year in year_list_unique:
                    values_time_series_dict_year[year] = {}
                for mon in mon_list_unique:
                    values_time_series_dict_mon[mon] = {}
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    if val == 0:
                        continue
                    # values_time_series_dict_year[year][mon] = val
                    values_time_series_dict_mon[mon][year] = val
                values_time_series_dict_mon_2020_2040 = {}
                for mon in range(1,13):
                    values_time_series_dict_mon_2020_2040[mon] = {}
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    if year < 2020 or year > 2040:
                        continue
                    values_time_series_dict_mon_2020_2040[mon][year] = val
                climatology_info_dict = {}
                for mon in range(1, 13):
                    one_mon_val_list = values_time_series_dict_mon_2020_2040[mon].values()
                    one_mon_val_list = list(one_mon_val_list)
                    mean = np.nanmean(one_mon_val_list)
                    std = np.nanstd(one_mon_val_list)
                    climatology_info_dict[mon] = {'mean': mean, 'std': std}
                # print(climatology_info_dict)
                # exit()
                anomaly = []
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    mean = climatology_info_dict[mon]['mean']
                    std = climatology_info_dict[mon]['std']
                    anomaly_i = (val - mean) / std
                    anomaly.append(anomaly_i)
                anomaly = np.array(anomaly)
                anomaly_dict[pix] = anomaly

            T.save_npy(anomaly_dict, outf)
        except:
            fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_anomaly(self,params):
        fdir_model, year_list, mon_list, outdir_model = params
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        if isfile(outf):
            return
        try:
            # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            anomaly_dict = {}
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                std = np.std(vals)
                if std == 0:
                    continue
                values_time_series_dict_year = {}
                values_time_series_dict_mon = {}
                year_list_unique = list(set(year_list))
                mon_list_unique = list(set(mon_list))
                for year in year_list_unique:
                    values_time_series_dict_year[year] = {}
                for mon in mon_list_unique:
                    values_time_series_dict_mon[mon] = {}
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    if val == 0:
                        continue
                    # values_time_series_dict_year[year][mon] = val
                    values_time_series_dict_mon[mon][year] = val
                climatology_info_dict = {}
                for mon in range(1, 13):
                    one_mon_val_list = values_time_series_dict_mon[mon].values()
                    one_mon_val_list = list(one_mon_val_list)
                    mean = np.nanmean(one_mon_val_list)
                    std = np.nanstd(one_mon_val_list)
                    climatology_info_dict[mon] = {'mean': mean, 'std': std}
                anomaly = []
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    mean = climatology_info_dict[mon]['mean']
                    std = climatology_info_dict[mon]['std']
                    anomaly_i = (val - mean) / std
                    anomaly.append(anomaly_i)
                anomaly = np.array(anomaly)
                anomaly_dict[pix] = anomaly

            T.save_npy(anomaly_dict, outf)
        except:
            fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_anomaly_based_history_climotology(self,params):
        fdir_model, year_list, mon_list, outdir_model,climatology_spatial_dict = params
        # print(climatology_spatial_dict)
        # exit()
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        # if isfile(outf):
        #     return
        # try:
        # print(1/0)
        dict_i = T.load_npy_dir(fdir_model)
        anomaly_dict = {}
        for pix in tqdm(dict_i):
            vals = dict_i[pix]
            std = np.std(vals)
            if std == 0:
                continue
            year_list_unique = list(set(year_list))
            mon_list_unique = list(set(mon_list))
            if not pix in climatology_spatial_dict:
                continue
            climatology_info_dict = climatology_spatial_dict[pix]

            anomaly = []
            for i in range(len(vals)):
                year = year_list[i]
                mon = mon_list[i]
                val = vals[i]
                mean = climatology_info_dict[mon]['mean']
                std = climatology_info_dict[mon]['std']
                anomaly_i = (val - mean) / std
                anomaly.append(anomaly_i)

            anomaly = np.array(anomaly)
            anomaly_dict[pix] = anomaly

        T.save_npy(anomaly_dict, outf)
        # except:
        #     fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
        #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_juping_based_history_climotology(self,params):
        fdir_model, year_list, mon_list, outdir_model,climatology_spatial_dict = params
        # print(climatology_spatial_dict)
        # exit()
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        # if isfile(outf):
        #     return
        # try:
        # print(1/0)
        dict_i = T.load_npy_dir(fdir_model)
        anomaly_dict = {}
        for pix in tqdm(dict_i):
            vals = dict_i[pix]
            std = np.std(vals)
            if std == 0:
                continue
            year_list_unique = list(set(year_list))
            mon_list_unique = list(set(mon_list))
            if not pix in climatology_spatial_dict:
                continue
            climatology_info_dict = climatology_spatial_dict[pix]
            mon_mean_list = []
            for mon in range(1,13):
                mon_mean = climatology_info_dict[mon]['mean']
                mon_mean_list.append(mon_mean)
            anomaly = []
            for i in range(len(vals)):
                year = year_list[i]
                mon = mon_list[i]
                val = vals[i]
                mean = climatology_info_dict[mon]['mean']
                std = climatology_info_dict[mon]['std']
                anomaly_i = val - mean
                anomaly.append(anomaly_i)

            anomaly = np.array(anomaly)
            anomaly_dict[pix] = anomaly

        T.save_npy(anomaly_dict, outf)
        # except:
        #     fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
        #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def per_pix_detrend(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp126', 'ssp370', 'ssp585']

        # product_list = ['tas']
        product_list = ['gpp']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_detrend')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    dict_i = T.load_npy_dir(fdir_model)
                    dict_i_detrend = T.detrend_dic(dict_i)
                    outf = join(outdir_model, 'detrend.npy')
                    T.save_npy(dict_i_detrend, outf)

    def per_pix_anomaly(self):

        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model]
                    param_list.append(params)
                    self.kernel_per_pix_anomaly(params)
        # exit()
        # MULTIPROCESS(self.kernel_per_pix_anomaly, param_list).run(process=7)
        pass

    def per_pix_anomaly_2020_2040(self):

        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['tas']
        product_list = ['VPD']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly_2020_2040')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model]
                    param_list.append(params)
                    # self.kernel_per_pix_anomaly_2020_2040(params)
        # exit()
        MULTIPROCESS(self.kernel_per_pix_anomaly_2020_2040, param_list).run(process=6)
        pass

    def per_pix_anomaly_based_history_climotology(self):
        history_climatology_dict = T.load_npy(join(data_root, 'CRU/tmp/Climatology_mean_std_1deg/1982-2020.npy'))
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly_history_climotology')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model,history_climatology_dict]
                    param_list.append(params)
                    # self.kernel_per_pix_anomaly_based_history_climotology(params)
                    # exit()
        MULTIPROCESS(self.kernel_per_pix_anomaly_based_history_climotology, param_list).run(process=7)

    def per_pix_juping_based_history_climotology(self):
        history_climatology_dict = T.load_npy(join(data_root, 'CRU/tmp/Climatology_mean_std_1deg/1982-2020.npy'))
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_juping_history_climotology')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model,history_climatology_dict]
                    param_list.append(params)
                    # self.kernel_per_pix_juping_based_history_climotology(params)
        MULTIPROCESS(self.kernel_per_pix_juping_based_history_climotology, param_list).run(process=7)

    def per_pix_juping(self):

        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange,sheet_name=0)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly_juping')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model]
                    param_list.append(params)
                    # self.kernel_per_pix_anomaly_juping(params)
        MULTIPROCESS(self.kernel_per_pix_anomaly_juping, param_list).run(process=7)
        pass

    def per_pix_annual(self):
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['tas']
        # product_list = ['pr']
        product = 'tas'
        param_list = []
        fdir = join(self.datadir, product, 'per_pix_anomaly_juping')
        outdir = join(self.datadir, product, 'per_pix_anomaly_juping_annual')
        for experiment in tqdm(experiment_list, desc='experiment'):
            fdir_i = join(fdir, experiment)
            if not isdir(fdir_i):
                continue
            model_list = T.listdir(fdir_i)
            outdir_experiment = join(outdir, experiment)
            for model in tqdm(model_list, desc='model'):
                outdir_model = join(outdir_experiment, model)
                T.mkdir(outdir_model, force=True)
                fdir_model = join(fdir_i, model)
                f = join(fdir_model, 'anomaly.npy')
                date_f = join(fdir_model, 'date_range.npy')
                date_obj_list = np.load(date_f, allow_pickle=True)

                # print(fdir_model)
                # exit()
                dict_i = T.load_npy(f)
                for pix in dict_i:
                    vals = dict_i[pix]
                    plt.plot(date_obj_list, vals)
                    plt.show()
                    print(vals)
                    print(pix)
                    exit()
                # self.kernel_per_pix_anomaly_juping(params)
        # MULTIPROCESS(self.kernel_per_pix_anomaly_juping, param_list).run(process=7)
        pass


    def per_pix_anomaly_detrend(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp126', 'ssp370', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'per_pix_anomaly')
            outdir = join(self.datadir, product, 'per_pix_anomaly_detrend')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    f = join(fdir_model, 'anomaly.npy')
                    dict_i = T.load_npy(f)
                    dict_i_detrend = T.detrend_dic(dict_i)
                    outf = join(outdir_model, 'anomaly_detrend.npy')
                    T.save_npy(dict_i_detrend, outf)


    def check_per_pix(self):
        fdir = '/Volumes/NVME4T/hotdrought_CMIP/data/CMIP6/mrsos/per_pix/ssp245/CanESM5-1'
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            std = np.nanstd(vals)
            if std == 0:
                continue
            mean = np.nanmean(vals)
            # spatial_dict_mean[pix] = len(vals)
            spatial_dict_mean[pix] = mean
        arr = DIC_and_TIF(pixelsize=1).pix_dic_to_spatial_arr(spatial_dict_mean)
        plt.imshow(arr,cmap='RdBu',interpolation='nearest')
        plt.colorbar()
        plt.show()

    def check_anomaly(self):
        f = '/Volumes/NVME2T/hotcold_drought/data/CMIP6/tas/per_pix_anomaly_history_climotology/ssp245/ACCESS-CM2/anomaly.npy'
        anomaly_dict = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(anomaly_dict):
            spatial_dict[pix] = anomaly_dict[pix][40]
            # vals = anomaly_dict[pix]
            # plt.plot(vals)
            # plt.show()
        arr = DIC_and_TIF(pixelsize=1).pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,cmap='RdBu')
        plt.colorbar()
        plt.show()
        pass


    def check_anomaly_time_series(self):
        f = '/Volumes/NVME2T/hotcold_drought/data/CMIP6/gpp/per_pix_anomaly_detrend/ssp585/CMCC-ESM2/anomaly_detrend.npy'
        anomaly_dict = T.load_npy(f)
        for pix in tqdm(anomaly_dict):
            r,c = pix
            if not r == 40:
                continue
            # if not c == 240:
            #     continue
            vals = anomaly_dict[pix]
            plt.plot(vals)
            plt.title(pix)
            plt.show()
        pass

    def check_date_consecutiveness(self):
        # product = 'gpp'
        # product = 'lai'
        product = 'mrsos'
        dff_daterange = join(self.datadir,product, 'models_daterange.xlsx')
        df_daterange = pd.read_excel(dff_daterange)
        for i,row in df_daterange.iterrows():
            date_list = row['date_list']
            date_list = eval(date_list)
            year_dict = {}
            for ii, date in enumerate(date_list):
                year, mon = date[:4], date[4:]
                year = int(year)
                mon = int(mon)
                if year not in year_dict:
                    year_dict[year] = []
                    year_dict[year].append(mon)
                else:
                    year_dict[year].append(mon)
            for year in year_dict:
                mon_list = year_dict[year]
                print(mon_list)
            exit()

        exit()
        pass


    def kernel_calculate_SPI(self,parmas):

        fdir_model, missing_dates_index, date_range, scale, distrib,Periodicity,outdir_model = parmas
        try:
        # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            spi_dict = {}
            date_range_result = []
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                vals = np.array(vals)
                vals[vals<0] = 0
                ## interpolate missing dates
                if len(missing_dates_index) > 0:
                    for i in missing_dates_index:
                        vals = np.insert(vals, i, np.nan)
                ## interpolate missing dates
                vals = T.interp_nan(vals)
                if len(vals) == 1:
                    continue
                excluded_index = []
                year_selected = -999999
                for i, date_i in enumerate(date_range):
                    year = date_i.year
                    month = date_i.month
                    if month != 1:
                        excluded_index.append(i)
                    if month == 1:
                        year_selected = year
                        break
                if year_selected < 0:
                    raise Exception('year_selected < 0')
                vals_selected = []
                date_range_selected = []
                for i, val in enumerate(vals):
                    if i in excluded_index:
                        continue
                    vals_selected.append(val)
                    date_range_selected.append(date_range[i])
                vals_selected = np.array(vals_selected)
                date_range_selected = np.array(date_range_selected)
                # try:
                #     std = np.std(vals_selected)
                # except:
                #     print(vals_selected)
                #     print(vals)
                #     exit()

                # zscore = Pre_Process().z_score_climatology(vals)
                spi = climate_indices.indices.spi(
                    values=vals_selected,
                    scale=scale,
                    distribution=distrib,
                    data_start_year=year_selected,
                    calibration_year_initial=year_selected,
                    calibration_year_final=year_selected + 30,
                    periodicity=Periodicity,
                    # fitting_params: Dict = None,
                )
                spi_dict[pix] = spi
                date_range_result = date_range_selected
                # anomaly = T.detrend_vals(anomaly)
                # anomaly = Pre_Process().z_score(vals)
                # arr = DIC_and_TIF(pixelsize=1).pix_dic_to_spatial_arr_mean(dict_i)
                # arr = T.mask_999999_arr(arr, warning=False)
                # arr[arr == 0] = np.nan
                # plt.subplot(211)
                # plt.plot(date_range_selected,vals_selected)
                # plt.twinx()
                # plt.plot(date_range_selected,spi,color='r')
                # plt.subplot(212)
                # plt.imshow(arr,aspect='auto')
                # plt.scatter(pix[1],pix[0],color='r')
                # plt.show()
            outf = join(outdir_model, f'SPI{scale}.npy')
            outf_date = join(outdir_model, f'SPI_date_range{scale}.npy')
            T.save_npy(spi_dict, outf)
            T.save_npy(date_range_result, outf_date)
        except Exception as e:
            fw = open(join(self.datadir, 'SPI_error.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write(f'{timestamp},{e},{outdir_model}\n')

        pass

    def calculate_SPI(self):
        ### SPI parameters ###
        scale = 12
        distrib = indices.Distribution('gamma')
        Periodicity = compute.Periodicity(12)
        ### SPI parameters ###
        dff_daterange = join(self.datadir,'models_daterange.xlsx')
        df_daterange = pd.read_excel(dff_daterange)
        # print(df_daterange)
        # exit()
        experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'SPI')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    print(product,experiment,model)
                    date_list = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].tolist()
                    if len(date_list) != 1:
                        continue
                    date_list = date_list[0]
                    date_list = eval(date_list)

                    date_obj_list = []
                    for date in date_list:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        date_obj = datetime.datetime(year, month, 1)
                        date_obj_list.append(date_obj)
                    date_obj_list = sorted(date_obj_list)

                    missing_dates_index = []
                    missing_dates_obj_list = []
                    for i,date_obj in enumerate(date_obj_list):
                        if i + 1 == len(date_obj_list):
                            continue
                        date_obj_1 = date_obj_list[i]
                        date_obj_2 = date_obj_list[i+1]
                        delta = date_obj_2 - date_obj_1
                        delta = delta.days
                        if delta > 31:
                            missing_dates_index.append(i)
                            missing_dates_obj_list.append(date_obj_1)
                    if len(date_obj_list) == 0:
                        continue
                    date_start = date_obj_list[0]
                    date_end = date_obj_list[-1]
                    date_range = pd.date_range(date_start, date_end, freq='MS')

                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    params = [fdir_model, missing_dates_index, date_range, scale,distrib,Periodicity,outdir_model]
                    # if not outdir_model == '/Volumes/NVME2T/hotcold_drought/data/CMIP6/pr/SPI/ssp126/ACCESS-CM2':
                    #     continue
                    param_list.append(params)
                    self.kernel_calculate_SPI(params)
        # MULTIPROCESS(self.kernel_calculate_SPI, param_list).run(process=7)

    def clean_tif(self):
        # product = 'mrsos'
        product = 'VPD'
        # product = 'tas'
        experiment_list = ['ssp245', 'ssp585']
        outdir_father = join(self.datadir,product,'tif_resample_unify_clean')
        for exp in experiment_list:
            fdir = join(self.datadir, product, 'tif_resample_unify', exp)
            outdir = join(outdir_father,exp)
            T.mkdir(outdir,force=True)
            for model in tqdm(T.listdir(fdir),desc=exp):
                outdir_i = join(outdir,model)
                T.mkdir(outdir_i,force=True)
                for f in T.listdir(join(fdir,model)):
                    if not f.endswith('.tif'):
                        continue
                    fpath = join(fdir,model,f)
                    outpath = join(outdir_i,f)
                    arr,originX,originY,pixelWidth,pixelHeight = ToRaster().raster2array(fpath)
                    # arr[arr>=70] = np.nan
                    # arr[arr<=0] = np.nan
                    arr[arr >= 1000] = np.nan
                    arr[arr <= 0] = np.nan
                    ToRaster().array2raster(outpath,originX,originY,pixelWidth,pixelHeight,arr)

    def check_individual_model(self):
        product = 'mrsos'
        # product = 'tas'
        experiment_list = ['ssp245', 'ssp585']
        fdir = join(self.datadir,product,'per_pix')
        for exp in experiment_list:
            fdir_i = join(fdir,exp)
            color_len = len(T.listdir(fdir_i))
            color_list = T.gen_colors(color_len)
            flag = 0
            for model in T.listdir(fdir_i):
                fdir_model = join(fdir_i,model)
                dict_i = T.load_npy_dir(fdir_model)
                all_vals = []
                for pix in tqdm(dict_i):
                    vals = dict_i[pix]
                    if np.nanstd(vals) == 0:
                        continue
                    all_vals.append(vals)
                mean = np.nanmean(all_vals,axis=0)
                plt.plot(mean,label=model,color=color_list[flag])
                flag += 1
            plt.legend()
            plt.show()
        pass

    def check_ensemble_model(self):
        product = 'mrsos'
        # product = 'tas'
        experiment_list = ['ssp245', 'ssp585']
        fdir = join(self.datadir,product,'per_pix_ensemble')
        for exp in experiment_list:
            fdir_i = join(fdir,exp)
            dict_i = T.load_npy_dir(fdir_i)
            all_vals = []
            spatial_dict = {}
            flag = 0
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                if np.nanstd(vals) == 0:
                    continue
                vals = np.array(vals)
                vals[vals<-99] = np.nan
                if True in np.isnan(vals):
                    spatial_dict[pix] = 1
                    continue
                flag += 1
                if flag > 10000:
                    plt.plot(vals)
                    plt.title(pix)
                    plt.show()
                all_vals.append(vals)
            arr = D_CMIP.pix_dic_to_spatial_arr(spatial_dict)
            plt.imshow(arr,cmap='jet',interpolation='nearest')
            plt.show()
            mean = np.nanmean(all_vals,axis=0)
            plt.plot(mean,label=exp)
        plt.legend()
        plt.show()
        pass

    def tif_ensemble(self):
        product = 'mrsos'
        # product = 'tas'
        experiment_list = ['ssp245','ssp585']
        flist = []
        for year in range(2020,2101):
            for mon in range(1,13):
                fname = f'{year}{mon:02d}.tif'
                flist.append(fname)
        outdir = join(self.datadir,product,'tif_resample_unify_ensemble')
        T.mkdir(outdir,force=True)

        for exp in experiment_list:
            outdir_i = join(outdir,exp)
            T.mkdir(outdir_i)
            fdir = join(self.datadir,product,'tif_resample_unify_clean',exp)
            for f in flist:
                print(f,'\n')
                outf = join(outdir_i,f)
                flist_ensemble = []
                for model in T.listdir(fdir):
                    fpath = join(fdir,model,f)
                    if not isfile(fpath):
                        continue
                    flist_ensemble.append(fpath)
                Pre_Process().compose_tif_list(flist_ensemble,outf)

        # fdir = join(self.datadir,'pr','SPI')
        pass

    def modify_FGOALS_f3_L(self):
        experiment_list = ['ssp245', 'ssp585']
        model = 'FGOALS-f3-L'
        product = 'mrsos'
        for exp in experiment_list:
            fdir = join(self.datadir,product,'tif_resample_unify_clean',exp,model)
            for f in tqdm(T.listdir(fdir),desc=exp):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir,f)
                array,originX,originY,pixelWidth,pixelHeight = ToRaster().raster2array(fpath)
                array[array<0] = np.nan
                array = array * 100
                ToRaster().array2raster(fpath,originX,originY,pixelWidth,pixelHeight,array)
        pass

    def per_pix_ensemble(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        product_list = ['tas']
        param_list = []
        for product in product_list:
            fdir = join(self.datadir, product, 'tif_resample_unify_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                Pre_Process().data_transform(fdir_i,outdir_i)

    def per_pix_ensemble_anomaly(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['tas','mrsos']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        product_list = ['mrsos']
        param_list = []
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_anomaly')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                spatial_dict = T.load_npy_dir(fdir_i)
                Pre_Process().cal_anomaly(fdir_i,outdir_i)

    def per_pix_ensemble_anomaly_detrend(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['tas','mrsos']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        product_list = ['mrsos']
        param_list = []
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble_anomaly')
            outdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_detrend')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                spatial_dict = T.load_npy_dir(fdir_i)
                spatial_dict_detrend = T.detrend_dic(spatial_dict)
                outf = join(outdir_i,'anomaly_detrend.npy')
                T.save_npy(spatial_dict_detrend,outf)

    def per_pix_ensemble_anomaly_based_history_tas(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        history_climatology_mean_f = join(data_root,'ERA5/Tair/climatology_mean/climatology_mean.npy')
        history_climatology_mean_dict = T.load_npy(history_climatology_mean_f)

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        product = 'tas'
        fdir = join(self.datadir, product, 'per_pix_ensemble')
        outdir = join(self.datadir, product, 'per_pix_ensemble_juping_based_history')
        for experiment in experiment_list:
            fdir_i = join(fdir, experiment)
            outdir_i = join(outdir, experiment)
            T.mkdir(outdir_i,force=True)
            spatial_dict = T.load_npy_dir(fdir_i)
            for pix in spatial_dict:
                r,c = pix
                if r < 40:
                    continue
                if r > 50:
                    continue
                vals = spatial_dict[pix]
                history_climatology_mean = history_climatology_mean_dict[pix]
                if T.is_all_nan(history_climatology_mean):
                    continue
                history_climatology_mean = list(history_climatology_mean)
                history_climatology_mean = history_climatology_mean * 81
                anomaly = vals - history_climatology_mean

                # plt.scatter(vals,history_climatology_mean)
                # plt.xlabel('vals')
                # plt.ylabel('history_climatology_mean')
                plt.figure()
                plt.plot(vals,'b',label='original')
                plt.plot(history_climatology_mean,'r',label='history_climatology_mean')
                plt.legend()

                plt.twinx()
                plt.plot(anomaly,'g',label='anomaly')
                plt.title(pix)
                plt.legend()
                plt.show()
                # exit()
    def per_pix_ensemble_anomaly_based_history_sm(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        history_climatology_mean_f = join(data_root,'ERA5/SM/climatology_mean/climatology_mean.npy')
        history_climatology_mean_dict = T.load_npy(history_climatology_mean_f)

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        product = 'mrsos'
        fdir = join(self.datadir, product, 'per_pix_ensemble')
        outdir = join(self.datadir, product, 'per_pix_ensemble_juping_based_history')
        for experiment in experiment_list:
            fdir_i = join(fdir, experiment)
            outdir_i = join(outdir, experiment)
            T.mkdir(outdir_i,force=True)
            spatial_dict = T.load_npy_dir(fdir_i)
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                history_climatology_mean = history_climatology_mean_dict[pix]
                if T.is_all_nan(history_climatology_mean):
                    continue
                history_climatology_mean = list(history_climatology_mean)
                print(history_climatology_mean)
                # history_climatology_mean = history_climatology_mean * 81

                # anomaly = vals - history_climatology_mean
                # plt.scatter(vals,history_climatology_mean)
                # plt.xlabel('vals')
                # plt.ylabel('history_climatology_mean')
                # plt.figure()
                # plt.plot(vals)
                plt.plot(history_climatology_mean,'r')
                # plt.twinx()
                # plt.plot(anomaly,'g')
                plt.title(pix)
                plt.show()
                # exit()

    def per_pix_ensemble_anomaly_based_2020_2060(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        base_line_year_num = 40

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        product_list = ['mrsos', 'tas']
        # product_list = ['tas']
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_based_2020_2060')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                spatial_dict = T.load_npy_dir(fdir_i)
                anomaly_spatial_dict = {}
                for pix in tqdm(spatial_dict,desc=f'{product} {experiment}'):
                    r,c = pix
                    if r > 150: # antarctica
                        continue
                    # if r < 20: # north pole
                    #     continue
                    vals = spatial_dict[pix]
                    if np.nanstd(vals) == 0:
                        continue
                    vals = np.array(vals)
                    vals[vals<-999] = np.nan
                    if True in np.isnan(vals):
                        continue
                    base_line_vals = vals[0:12*base_line_year_num]
                    climatology_means = []
                    for m in range(1, 13):
                        one_mon = []
                        for i in range(len(base_line_vals)):
                            mon = i % 12 + 1
                            if mon == m:
                                one_mon.append(base_line_vals[i])
                        mean = np.nanmean(one_mon)
                        std = np.nanstd(one_mon)
                        climatology_means.append(mean)
                    anomaly = vals - climatology_means * 81
                    anomaly_spatial_dict[pix] = anomaly
                outf = join(outdir_i, 'anomaly.npy')
                T.save_npy(anomaly_spatial_dict, outf)

    def per_pix_ensemble_std_anomaly_based_2020_2060(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        base_line_year_num = 40

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        product_list = ['mrsos', 'tas']
        # product_list = ['tas']
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_std_anomaly_based_2020_2060')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                spatial_dict = T.load_npy_dir(fdir_i)
                anomaly_spatial_dict = {}
                for pix in tqdm(spatial_dict,desc=f'{product} {experiment}'):
                    r,c = pix
                    if r > 150: # antarctica
                        continue
                    # if r < 20: # north pole
                    #     continue
                    vals = spatial_dict[pix]
                    if np.nanstd(vals) == 0:
                        continue
                    vals = np.array(vals)
                    vals[vals<-999] = np.nan
                    if True in np.isnan(vals):
                        continue
                    base_line_vals = vals[0:12*base_line_year_num]
                    climatology_means = []
                    for m in range(1, 13):
                        one_mon = []
                        for i in range(len(base_line_vals)):
                            mon = i % 12 + 1
                            if mon == m:
                                one_mon.append(base_line_vals[i])
                        mean = np.nanmean(one_mon)
                        std = np.nanstd(one_mon)
                        climatology_means.append(mean)
                    anomaly = vals - climatology_means * 81
                    std = np.nanstd(vals)
                    std_anomaly = anomaly / std
                    anomaly_spatial_dict[pix] = std_anomaly
                outf = join(outdir_i, 'anomaly.npy')
                T.save_npy(anomaly_spatial_dict, outf)

    def per_pix_ensemble_anomaly_dynamic_baseline(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['tas', 'mrsos'][::-1]
        product_list = ['tas', 'mrsos']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        year_list = list(range(2020,2101))
        year_range_list = [
            [2020, 2040],
            [2041, 2060],
            [2061, 2080],
            [2081, 2100],
        ]

        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_anomaly')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i, force=True)
                spatial_dict = T.load_npy_dir(fdir_i)

                for pix in spatial_dict:
                    r,c = pix
                    if r < 100:
                        continue
                    if c < 100:
                        continue
                    vals = spatial_dict[pix]
                    vals[vals<0] = np.nan
                    if True in np.isnan(vals):
                        continue
                    r,c = pix
                    if r > 148:
                        continue
                    vals_reshape = vals.reshape(-1,12)
                    annual_vals_dict = T.dict_zip(year_list,vals_reshape)
                    dynamic_threshold_anomaly = []
                    for year_range in year_range_list:
                        vals_subsection = []
                        for year in range(year_range[0],year_range[1]+1):
                            vals_subsection.append(annual_vals_dict[year])
                        vals_subsection = np.array(vals_subsection)
                        vals_subsection = vals_subsection.reshape(-1)
                        climatology_anomaly = Pre_Process().climatology_anomaly(vals_subsection)
                        climatology_anomaly = list(climatology_anomaly)
                        dynamic_threshold_anomaly = dynamic_threshold_anomaly + climatology_anomaly
                    # dynamic_threshold_anomaly = np.reshape(dynamic_threshold_anomaly,-1)
                    # print(np.shape((dynamic_threshold_anomaly)))
                    # exit()
                    # print(dynamic_threshold_anomaly)
                    # exit()
                    vals_anomaly = Pre_Process().climatology_anomaly(vals)
                    plt.plot(dynamic_threshold_anomaly,label='dynamic')
                    plt.plot(vals_anomaly,label='long-term anomaly')
                    plt.title(pix)
                    plt.legend()
                    plt.show()
        pass

    def per_pix_ensemble_anomaly_moving_window_baseline(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['pr', 'tas']
        # product_list = ['tas', 'mrsos'][::-1]
        product_list = ['tas', 'mrsos']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        year_list = list(range(2020,2101))
        window_size = 20
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_moving_window_baseline')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i, force=True)
                spatial_dict = T.load_npy_dir(fdir_i)

                anomaly_spatial_dict = {}

                for pix in tqdm(spatial_dict,desc=f'{product} {experiment}'):
                    r,c = pix
                    vals = spatial_dict[pix]
                    vals[vals<0] = np.nan
                    if True in np.isnan(vals):
                        continue
                    r,c = pix
                    if r > 148:
                        continue
                    vals_reshape = vals.reshape(-1,12)
                    # annual_vals_dict = T.dict_zip(year_list,vals_reshape)
                    moving_window_anomaly = []
                    for i in range(len(year_list)):
                        # if i + window_size >= len(year_list):
                        #     break
                        # window_name = f'{i}-{i + window_size}'
                        picked_arr = vals_reshape[i:i + window_size]
                        picked_arr_reshape = picked_arr.reshape(-1)
                        picked_arr_reshape_anomaly = Pre_Process().climatology_anomaly(picked_arr_reshape)
                        year_vals = picked_arr_reshape_anomaly[:12]
                        year_vals = list(year_vals)
                        moving_window_anomaly = moving_window_anomaly + year_vals
                    anomaly_spatial_dict[pix] = moving_window_anomaly
                outf = join(outdir_i, 'anomaly.npy')
                T.save_npy(anomaly_spatial_dict, outf)

    def per_pix_ensemble_anomaly_CMIP_historical_baseline(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        historical_fdir = join(data_root, 'ERA5/Tair/per_pix_1_deg/1982-2020')
        # historical_fdir = join(data_root,'CMIP6_historical/tas/per_pix_ensemble/historical')
        historical_spatial_dict = T.load_npy_dir(historical_fdir)
        # outdir = join(self.datadir,'tif_anomaly_mean','ERA5')
        # outdir = join(self.datadir,'tif_anomaly_mean','ERA5')
        # outdir = join(self.datadir,'tif_anomaly_mean','CMIP6_historical')
        # T.mkdir(outdir,force=True)

        # product_list = ['pr', 'tas']
        # product_list = ['tas', 'mrsos'][::-1]
        # product_list = ['tas', 'mrsos']
        product_list = ['tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        year_list = list(range(2020,2101))
        for product in product_list:

            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_CMIP_historical_baseline')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i, force=True)
                spatial_dict = T.load_npy_dir(fdir_i)

                anomaly_spatial_dict = {}
                for pix in tqdm(spatial_dict,desc=f'{product} {experiment}'):
                    r,c = pix
                    vals = spatial_dict[pix]
                    vals[vals<0] = np.nan
                    if True in np.isnan(vals):
                        continue
                    r,c = pix
                    if r > 148:
                        continue
                    gs = global_get_gs(pix)
                    vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                    std = np.nanstd(vals_gs)
                    historical_vals = historical_spatial_dict[pix]
                    historical_vals_gs = T.monthly_vals_to_annual_val(historical_vals,gs)
                    historical_mean = np.nanmean(historical_vals_gs)
                    anomaly = (vals_gs - historical_mean) / std
                    anomaly_spatial_dict[pix] = anomaly
                outf = join(outdir_i, 'anomaly.npy')
                T.save_npy(anomaly_spatial_dict, outf)


class CMIP6_historical:

    def __init__(self):
        self.datadir = join(data_root, 'CMIP6_historical')
        # self.product_list = ['pr','tas']
        # self.product_list = ['gpp']
        # self.product_list = ['tas']
        pass

    def run(self):
        # self.all_models_summary()
        # self.pick_nc_files()
        # self.nc_to_tif()
        # self.resample()
        # self.unify_tif_reshape()
        # self.check_tif_shape()
        # self.models_daterange()
        # self.check_date_consecutiveness()
        # self.per_pix()
        # self.per_pix_detrend()
        # self.per_pix_anomaly()
        # self.per_pix_anomaly_based_history_climotology()
        # self.per_pix_juping_based_history_climotology()
        # self.per_pix_juping()
        # self.per_pix_annual()
        # self.per_pix_anomaly_detrend()
        # self.check_anomaly()
        # self.check_per_pix()
        # self.check_anomaly_time_series()
        # self.calculate_SPI()
        #
        # self.check_individual_model()
        # self.modify_FGOALS_f3_L()
        # self.clean_tif()
        # self.tif_ensemble()
        # self.per_pix_ensemble()
        # self.check_ensemble_model()

        # self.per_pix_ensemble_anomaly()
        # self.per_pix_ensemble_baseline()
        self.per_pix_ensemble_gs()

        pass

    def all_models_summary(self):
        result_dict = {}
        flag = 0
        # product_list = ['lai']
        product_list = ['mrsos','tas']
        for product in product_list:
            fdir = join(self.datadir, product,'nc')
            for f in T.listdir(fdir):
                fname = f.split('.')[0]
                product_i,realm_i,model_i,experiment_i,ensemble_i,_,time_i = fname.split('_')
                dict_i = {'product':product_i, 'realm':realm_i, 'model':model_i, 'experiment':experiment_i, 'ensemble':ensemble_i, 'time':time_i, 'fname':join(fdir,f)}
                flag += 1
                result_dict[flag] = dict_i
            df = T.dic_to_df(result_dict)
            T.df_to_excel(df,join(self.datadir,f'{product}','models_summary'),n=10000)


    def pick_nc_files(self):

        # experiment_list = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
        # product_list = ['pr','tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['tas']
        product_list = ['mrsos','tas']
        for product in product_list:
            print(product)
            excel = join(self.datadir,f'{product}', 'models_summary.xlsx')
            df = pd.read_excel(excel)
            # T.print_head_n(df)
            experiment_list = T.get_df_unique_val_list(df, 'experiment')
            fdir = join(self.datadir, product,'picked_nc')
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            df_product = df[df['product']==product]
            for experiment in experiment_list:
                df_i = df_product[df_product['experiment']==experiment]
                outdir_i = join(fdir,experiment)
                if not os.path.exists(outdir_i):
                    os.makedirs(outdir_i)
                model_list = T.get_df_unique_val_list(df_i,'model')
                for model in model_list:
                    df_model = df_i[df_i['model']==model]
                    outdir_model = join(outdir_i,model)
                    if not os.path.exists(outdir_model):
                        os.makedirs(outdir_model)
                    for index,row in df_model.iterrows():
                        fname = row['fname']
                        shutil.copy(fname, outdir_model)


    def kernel_nc_to_tif(self, params):
        try:
            fname, outdir, product = params
            try:
                ncin = Dataset(fname, 'r')
            except:
                return
            lat = ncin.variables['lat'][:]
            lon = ncin.variables['lon'][:]
            shape = np.shape(lat)

            time = ncin.variables['time'][:]
            basetime = ncin.variables['time'].units
            basetime = basetime.strip('days since ')
            try:
                basetime = datetime.datetime.strptime(basetime, '%Y-%m-%d')
            except:
                try:
                    basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M:%S')
                except:
                    try:
                        basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M:%S.%f')
                    except:
                        basetime = datetime.datetime.strptime(basetime,'%Y-%m-%d %H:%M')
            data = ncin.variables[product]
            if len(shape) == 2:
                xx,yy = lon,lat
            else:
                xx,yy = np.meshgrid(lon, lat)
            for time_i in range(len(data)):
                # print(time_i)
                date = basetime + datetime.timedelta(days=time[time_i])
                time_str = time[time_i]
                mon = date.month
                year = date.year
                if year > 2100:
                    continue
                if year < 1960:
                    continue
                outf_name = f'{year}{mon:02d}.tif'
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
                        if lon_i >= 180:
                            lon_i -= 360
                        lat_i = yy[i][j]
                        value_i = arr[i][j]
                        lon_list.append(lon_i)
                        lat_list.append(lat_i)
                        value_list.append(value_i)
                DIC_and_TIF().lon_lat_val_to_tif(lon_list, lat_list, value_list,outpath)
        except:
            fw = open(join(self.datadir,'kernel_nc_to_tif_error.txt'),'a')
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write(f'[{current_time}],')
            fw.write(','.join(params)+'\n')
            fw.close()


    def nc_to_tif(self):
        experiment_list = ['historical']
        # product_list = ['pr', 'tas']
        # product_list = ['tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        product_list = ['mrsos','tas']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product,'picked_nc')
            outdir = join(self.datadir, product,'tif')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir,experiment)
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir,experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment,model)
                    T.mkdir(outdir_model,force=True)
                    fdir_model = join(fdir_i,model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model,fname)
                        param_i = [fpath, outdir_model,product]
                        param_list.append(param_i)
                        # self.kernel_nc_to_tif(param_i)
        MULTIPROCESS(self.kernel_nc_to_tif, param_list).run(process=7)
        # exit()


    def kernel_unify_tif_shape(self,params):
        fpath, outf = params
        # ToRaster().un(fpath, outf)
        DIC_and_TIF().unify_raster(fpath, outf)
        pass

    def unify_tif_reshape(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        # product_list = ['pr', 'tas']
        product_list = ['tas','mrsos']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif_resample')
            # fdir = join(self.datadir, product, 'tif_resample_unify')
            outdir = join(self.datadir, product, 'tif_resample_unify')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model, fname)
                        outf = join(outdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year, mon = fname_split[:4], fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        # arr = ToRaster().raster2array(fpath)[0]
                        # fw.write(f'{str(np.shape(arr))},{product},{experiment},{model},{fname}\n')
                        # break
                        # if isfile(outf):
                        #     continue
                        # print(fpath, outf)
                        # exit()
                        param_i = [fpath, outf]
                        # self.kernel_unify_tif_shape(param_i)
                        param_list.append(param_i)
                        # T.open_path_and_file(outdir_model)
        MULTIPROCESS(self.kernel_unify_tif_shape, param_list).run(process=7)
        # exit()


    def check_tif_shape(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['lai']
        product_list = ['mrsos', 'tas']
        # product_list = ['pr', 'tas']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fw = open(join(self.datadir, f'{product}','shapes.csv'), 'w')
            fdir = join(self.datadir, product, 'tif_resample_unify')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                for model in tqdm(model_list, desc='model'):
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year, mon = fname_split[:4], fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        arr = ToRaster().raster2array(fpath)[0]
                        fw.write(f'{str(np.shape(arr))},{product},{experiment},{model},{fname}\n')
                        break
                        # param_i = [fpath, outf]
                        # self.kernel_unify_tif_shape(param_i)
                        # param_list.append(param_i)
                        # T.open_path_and_file(outdir_model)
        # MULTIPROCESS(self.kernel_unify_tif_shape, param_list).run(process=6)


    def kernel_resample(self,parms):
        fpath,outf,res = parms
        if isfile(outf):
            return
        # if 'ACCESS-ESM1-5' in fpath:
        #     print(fpath)
        #     exit()
        ToRaster().resample_reproj(fpath, outf, res=res)

    def resample(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        # product_list = ['pr', 'tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        product_list = ['mrsos', 'tas']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif')
            outdir = join(self.datadir, product, 'tif_resample')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    for fname in tqdm(fname_list, desc='fname'):
                        fpath = join(fdir_model, fname)
                        if not fpath.endswith('.tif'):
                            continue
                        # if 'ACCESS-ESM1-5' in fpath:
                        #     print(fpath)
                        #     exit()
                        outf = join(outdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year, mon = fname_split[:4], fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        params = [fpath,outf,1]
                        param_list.append(params)
                        # ToRaster().resample_reproj(fpath,outf,res=1)
        MULTIPROCESS(self.kernel_resample, param_list).run(process=7)


    def kernel_per_pix(self,params):
        fdir_model, outdir_model, flist_picked = params
        print(fdir_model)
        flist = T.listdir(fdir_model)
        if len(flist) == 0:
            return
        Pre_Process().data_transform_with_date_list(fdir_model,outdir_model,flist_picked)

    def per_pix(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        # product_list = ['tas','mrsos']
        product_list = ['mrsos']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif_resample_unify_clean')
            outdir = join(self.datadir, product, 'per_pix')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    if not model == 'FGOALS-f3-L':
                        continue
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    flist_picked = []
                    for fname in tqdm(fname_list, desc='fname'):
                        if not fname.endswith('.tif'):
                            continue
                        fpath = join(fdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year,mon = fname_split[:4],fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        flist_picked.append(fname)
                    # print(flist_picked)
                    # exit()
                    outdir_flist = T.listdir(outdir_model)
                    if len(outdir_flist) > 0:
                        continue
                    params = [fdir_model,outdir_model,flist_picked]
                    param_list.append(params)
                    # print(params)
                    # self.kernel_per_pix(params)
        MULTIPROCESS(self.kernel_per_pix, param_list).run(process=7)

    def models_daterange(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        # product_list = ['pr', 'tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        product_list = ['mrsos', 'tas']
        param_list = []
        all_result_dict = {}
        index = 0
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'tif_resample_unify')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                for model in tqdm(model_list, desc='model'):
                    fdir_model = join(fdir_i, model)
                    fname_list = T.listdir(fdir_model)
                    date_list = []
                    for fname in tqdm(fname_list, desc='fname'):
                        if not fname.endswith('.tif'):
                            continue
                        fpath = join(fdir_model, fname)
                        fname_split = fname.split('.')[0]
                        year,mon = fname_split[:4],fname_split[4:]
                        year = int(year)
                        if year > 2100:
                            continue
                        date = fname_split
                        date_list.append(date)
                    date_list = sorted(date_list)
                    date_list = tuple(date_list)
                    info_dict = {'date_list':date_list,'product':product,'experiment':experiment,'model':model}
                    all_result_dict[index] = info_dict
                    index += 1
            df = T.dic_to_df(all_result_dict,key_col_str='index',col_order=['product','experiment','model','date_list'])
            T.df_to_excel(df,join(self.datadir,product,'models_daterange'))

    def kernel_per_pix_anomaly_juping(self,params):
        fdir_model, year_list, mon_list, outdir_model = params
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        if isfile(outf):
            return
        try:
            # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            anomaly_dict = {}
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                std = np.std(vals)
                if std == 0:
                    continue
                values_time_series_dict_year = {}
                values_time_series_dict_mon = {}
                year_list_unique = list(set(year_list))
                mon_list_unique = list(set(mon_list))
                for year in year_list_unique:
                    values_time_series_dict_year[year] = {}
                for mon in mon_list_unique:
                    values_time_series_dict_mon[mon] = {}
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    if val == 0:
                        continue
                    # values_time_series_dict_year[year][mon] = val
                    values_time_series_dict_mon[mon][year] = val
                climatology_info_dict = {}
                for mon in range(1, 13):
                    one_mon_val_list = values_time_series_dict_mon[mon].values()
                    one_mon_val_list = list(one_mon_val_list)
                    mean = np.nanmean(one_mon_val_list)
                    std = np.nanstd(one_mon_val_list)
                    climatology_info_dict[mon] = {'mean': mean, 'std': std}
                anomaly = []
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    mean = climatology_info_dict[mon]['mean']
                    std = climatology_info_dict[mon]['std']
                    anomaly_i = val - mean
                    anomaly.append(anomaly_i)
                anomaly = np.array(anomaly)
                anomaly_dict[pix] = anomaly

            T.save_npy(anomaly_dict, outf)
        except:
            fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_anomaly(self,params):
        fdir_model, year_list, mon_list, outdir_model = params
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        if isfile(outf):
            return
        try:
            # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            anomaly_dict = {}
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                std = np.std(vals)
                if std == 0:
                    continue
                values_time_series_dict_year = {}
                values_time_series_dict_mon = {}
                year_list_unique = list(set(year_list))
                mon_list_unique = list(set(mon_list))
                for year in year_list_unique:
                    values_time_series_dict_year[year] = {}
                for mon in mon_list_unique:
                    values_time_series_dict_mon[mon] = {}
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    if val == 0:
                        continue
                    # values_time_series_dict_year[year][mon] = val
                    values_time_series_dict_mon[mon][year] = val
                climatology_info_dict = {}
                for mon in range(1, 13):
                    one_mon_val_list = values_time_series_dict_mon[mon].values()
                    one_mon_val_list = list(one_mon_val_list)
                    mean = np.nanmean(one_mon_val_list)
                    std = np.nanstd(one_mon_val_list)
                    climatology_info_dict[mon] = {'mean': mean, 'std': std}
                anomaly = []
                for i in range(len(vals)):
                    year = year_list[i]
                    mon = mon_list[i]
                    val = vals[i]
                    mean = climatology_info_dict[mon]['mean']
                    std = climatology_info_dict[mon]['std']
                    anomaly_i = (val - mean) / std
                    anomaly.append(anomaly_i)
                anomaly = np.array(anomaly)
                anomaly_dict[pix] = anomaly

            T.save_npy(anomaly_dict, outf)
        except:
            fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_anomaly_based_history_climotology(self,params):
        fdir_model, year_list, mon_list, outdir_model,climatology_spatial_dict = params
        # print(climatology_spatial_dict)
        # exit()
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        # if isfile(outf):
        #     return
        # try:
        # print(1/0)
        dict_i = T.load_npy_dir(fdir_model)
        anomaly_dict = {}
        for pix in tqdm(dict_i):
            vals = dict_i[pix]
            std = np.std(vals)
            if std == 0:
                continue
            year_list_unique = list(set(year_list))
            mon_list_unique = list(set(mon_list))
            if not pix in climatology_spatial_dict:
                continue
            climatology_info_dict = climatology_spatial_dict[pix]

            anomaly = []
            for i in range(len(vals)):
                year = year_list[i]
                mon = mon_list[i]
                val = vals[i]
                mean = climatology_info_dict[mon]['mean']
                std = climatology_info_dict[mon]['std']
                anomaly_i = (val - mean) / std
                anomaly.append(anomaly_i)

            anomaly = np.array(anomaly)
            anomaly_dict[pix] = anomaly

        T.save_npy(anomaly_dict, outf)
        # except:
        #     fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
        #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def kernel_per_pix_juping_based_history_climotology(self,params):
        fdir_model, year_list, mon_list, outdir_model,climatology_spatial_dict = params
        # print(climatology_spatial_dict)
        # exit()
        outf = join(outdir_model, 'anomaly.npy')
        outf_date = join(outdir_model, 'date_range.npy')
        date_range = []
        for year,mon in zip(year_list,mon_list):
            date = datetime.datetime(year, mon, 1)
            date_range.append(date)
        date_range = np.array(date_range)
        np.save(outf_date, date_range)
        # if isfile(outf):
        #     return
        # try:
        # print(1/0)
        dict_i = T.load_npy_dir(fdir_model)
        anomaly_dict = {}
        for pix in tqdm(dict_i):
            vals = dict_i[pix]
            std = np.std(vals)
            if std == 0:
                continue
            year_list_unique = list(set(year_list))
            mon_list_unique = list(set(mon_list))
            if not pix in climatology_spatial_dict:
                continue
            climatology_info_dict = climatology_spatial_dict[pix]
            mon_mean_list = []
            for mon in range(1,13):
                mon_mean = climatology_info_dict[mon]['mean']
                mon_mean_list.append(mon_mean)
            anomaly = []
            for i in range(len(vals)):
                year = year_list[i]
                mon = mon_list[i]
                val = vals[i]
                mean = climatology_info_dict[mon]['mean']
                std = climatology_info_dict[mon]['std']
                anomaly_i = val - mean
                anomaly.append(anomaly_i)

            anomaly = np.array(anomaly)
            anomaly_dict[pix] = anomaly

        T.save_npy(anomaly_dict, outf)
        # except:
        #     fw = open(join(self.datadir, 'kernel_per_pix_anomaly_error_log.csv'), 'a')
        #     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        #     fw.write('[{}],{}\n'.format(timestamp, outdir_model))
        pass

    def per_pix_detrend(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp126', 'ssp370', 'ssp585']

        # product_list = ['tas']
        product_list = ['gpp']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_detrend')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    dict_i = T.load_npy_dir(fdir_model)
                    dict_i_detrend = T.detrend_dic(dict_i)
                    outf = join(outdir_model, 'detrend.npy')
                    T.save_npy(dict_i_detrend, outf)

    def per_pix_anomaly(self):

        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        product_list = ['mrsos']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model]
                    param_list.append(params)
                    self.kernel_per_pix_anomaly(params)
        # exit()
        # MULTIPROCESS(self.kernel_per_pix_anomaly, param_list).run(process=7)
        pass

    def per_pix_anomaly_based_history_climotology(self):
        history_climatology_dict = T.load_npy(join(data_root, 'CRU/tmp/Climatology_mean_std_1deg/1982-2020.npy'))
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly_history_climotology')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model,history_climatology_dict]
                    param_list.append(params)
                    # self.kernel_per_pix_anomaly_based_history_climotology(params)
                    # exit()
        MULTIPROCESS(self.kernel_per_pix_anomaly_based_history_climotology, param_list).run(process=7)

    def per_pix_juping_based_history_climotology(self):
        history_climatology_dict = T.load_npy(join(data_root, 'CRU/tmp/Climatology_mean_std_1deg/1982-2020.npy'))
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_juping_history_climotology')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model,history_climatology_dict]
                    param_list.append(params)
                    # self.kernel_per_pix_juping_based_history_climotology(params)
        MULTIPROCESS(self.kernel_per_pix_juping_based_history_climotology, param_list).run(process=7)

    def per_pix_juping(self):

        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp245', 'ssp585']
        # experiment_list = ['ssp126', 'ssp370', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            dff_daterange = join(self.datadir,product,'models_daterange.xlsx')
            df_daterange = pd.read_excel(dff_daterange,sheet_name=0)
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'per_pix_anomaly_juping')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    daterange = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].values[0]
                    daterange = eval(daterange)
                    year_list = []
                    mon_list = []
                    for date in daterange:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        year_list.append(year)
                        mon_list.append(month)
                    params = [fdir_model, year_list, mon_list, outdir_model]
                    param_list.append(params)
                    # self.kernel_per_pix_anomaly_juping(params)
        MULTIPROCESS(self.kernel_per_pix_anomaly_juping, param_list).run(process=7)
        pass

    def per_pix_annual(self):
        experiment_list = ['ssp245', 'ssp585']

        # product_list = ['tas']
        # product_list = ['pr']
        product = 'tas'
        param_list = []
        fdir = join(self.datadir, product, 'per_pix_anomaly_juping')
        outdir = join(self.datadir, product, 'per_pix_anomaly_juping_annual')
        for experiment in tqdm(experiment_list, desc='experiment'):
            fdir_i = join(fdir, experiment)
            if not isdir(fdir_i):
                continue
            model_list = T.listdir(fdir_i)
            outdir_experiment = join(outdir, experiment)
            for model in tqdm(model_list, desc='model'):
                outdir_model = join(outdir_experiment, model)
                T.mkdir(outdir_model, force=True)
                fdir_model = join(fdir_i, model)
                f = join(fdir_model, 'anomaly.npy')
                date_f = join(fdir_model, 'date_range.npy')
                date_obj_list = np.load(date_f, allow_pickle=True)

                # print(fdir_model)
                # exit()
                dict_i = T.load_npy(f)
                for pix in dict_i:
                    vals = dict_i[pix]
                    plt.plot(date_obj_list, vals)
                    plt.show()
                    print(vals)
                    print(pix)
                    exit()
                # self.kernel_per_pix_anomaly_juping(params)
        # MULTIPROCESS(self.kernel_per_pix_anomaly_juping, param_list).run(process=7)
        pass


    def per_pix_anomaly_detrend(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['ssp126', 'ssp370', 'ssp585']

        product_list = ['tas']
        # product_list = ['gpp']
        # product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'per_pix_anomaly')
            outdir = join(self.datadir, product, 'per_pix_anomaly_detrend')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    f = join(fdir_model, 'anomaly.npy')
                    dict_i = T.load_npy(f)
                    dict_i_detrend = T.detrend_dic(dict_i)
                    outf = join(outdir_model, 'anomaly_detrend.npy')
                    T.save_npy(dict_i_detrend, outf)

    def check_per_pix(self):
        fdir = '/Volumes/NVME4T/hotdrought_CMIP/data/CMIP6/mrsos/per_pix/ssp245/CanESM5-1'
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            std = np.nanstd(vals)
            if std == 0:
                continue
            mean = np.nanmean(vals)
            # spatial_dict_mean[pix] = len(vals)
            spatial_dict_mean[pix] = mean
        arr = DIC_and_TIF(pixelsize=1).pix_dic_to_spatial_arr(spatial_dict_mean)
        plt.imshow(arr,cmap='RdBu',interpolation='nearest')
        plt.colorbar()
        plt.show()

    def check_anomaly(self):
        f = '/Volumes/NVME2T/hotcold_drought/data/CMIP6/tas/per_pix_anomaly_history_climotology/ssp245/ACCESS-CM2/anomaly.npy'
        anomaly_dict = T.load_npy(f)
        spatial_dict = {}
        for pix in tqdm(anomaly_dict):
            spatial_dict[pix] = anomaly_dict[pix][40]
            # vals = anomaly_dict[pix]
            # plt.plot(vals)
            # plt.show()
        arr = DIC_and_TIF(pixelsize=1).pix_dic_to_spatial_arr(spatial_dict)
        plt.imshow(arr,cmap='RdBu')
        plt.colorbar()
        plt.show()
        pass


    def check_anomaly_time_series(self):
        f = '/Volumes/NVME2T/hotcold_drought/data/CMIP6/gpp/per_pix_anomaly_detrend/ssp585/CMCC-ESM2/anomaly_detrend.npy'
        anomaly_dict = T.load_npy(f)
        for pix in tqdm(anomaly_dict):
            r,c = pix
            if not r == 40:
                continue
            # if not c == 240:
            #     continue
            vals = anomaly_dict[pix]
            plt.plot(vals)
            plt.title(pix)
            plt.show()
        pass

    def check_date_consecutiveness(self):
        # product = 'gpp'
        # product = 'lai'
        product = 'mrsos'
        dff_daterange = join(self.datadir,product, 'models_daterange.xlsx')
        df_daterange = pd.read_excel(dff_daterange)
        for i,row in df_daterange.iterrows():
            date_list = row['date_list']
            date_list = eval(date_list)
            year_dict = {}
            for ii, date in enumerate(date_list):
                year, mon = date[:4], date[4:]
                year = int(year)
                mon = int(mon)
                if year not in year_dict:
                    year_dict[year] = []
                    year_dict[year].append(mon)
                else:
                    year_dict[year].append(mon)
            for year in year_dict:
                mon_list = year_dict[year]
                print(mon_list)
            exit()

        exit()
        pass


    def kernel_calculate_SPI(self,parmas):

        fdir_model, missing_dates_index, date_range, scale, distrib,Periodicity,outdir_model = parmas
        try:
        # print(1/0)
            dict_i = T.load_npy_dir(fdir_model)
            spi_dict = {}
            date_range_result = []
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                vals = np.array(vals)
                vals[vals<0] = 0
                ## interpolate missing dates
                if len(missing_dates_index) > 0:
                    for i in missing_dates_index:
                        vals = np.insert(vals, i, np.nan)
                ## interpolate missing dates
                vals = T.interp_nan(vals)
                if len(vals) == 1:
                    continue
                excluded_index = []
                year_selected = -999999
                for i, date_i in enumerate(date_range):
                    year = date_i.year
                    month = date_i.month
                    if month != 1:
                        excluded_index.append(i)
                    if month == 1:
                        year_selected = year
                        break
                if year_selected < 0:
                    raise Exception('year_selected < 0')
                vals_selected = []
                date_range_selected = []
                for i, val in enumerate(vals):
                    if i in excluded_index:
                        continue
                    vals_selected.append(val)
                    date_range_selected.append(date_range[i])
                vals_selected = np.array(vals_selected)
                date_range_selected = np.array(date_range_selected)
                # try:
                #     std = np.std(vals_selected)
                # except:
                #     print(vals_selected)
                #     print(vals)
                #     exit()

                # zscore = Pre_Process().z_score_climatology(vals)
                spi = climate_indices.indices.spi(
                    values=vals_selected,
                    scale=scale,
                    distribution=distrib,
                    data_start_year=year_selected,
                    calibration_year_initial=year_selected,
                    calibration_year_final=year_selected + 30,
                    periodicity=Periodicity,
                    # fitting_params: Dict = None,
                )
                spi_dict[pix] = spi
                date_range_result = date_range_selected
                # anomaly = T.detrend_vals(anomaly)
                # anomaly = Pre_Process().z_score(vals)
                # arr = DIC_and_TIF(pixelsize=1).pix_dic_to_spatial_arr_mean(dict_i)
                # arr = T.mask_999999_arr(arr, warning=False)
                # arr[arr == 0] = np.nan
                # plt.subplot(211)
                # plt.plot(date_range_selected,vals_selected)
                # plt.twinx()
                # plt.plot(date_range_selected,spi,color='r')
                # plt.subplot(212)
                # plt.imshow(arr,aspect='auto')
                # plt.scatter(pix[1],pix[0],color='r')
                # plt.show()
            outf = join(outdir_model, f'SPI{scale}.npy')
            outf_date = join(outdir_model, f'SPI_date_range{scale}.npy')
            T.save_npy(spi_dict, outf)
            T.save_npy(date_range_result, outf_date)
        except Exception as e:
            fw = open(join(self.datadir, 'SPI_error.csv'), 'a')
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            fw.write(f'{timestamp},{e},{outdir_model}\n')

        pass

    def calculate_SPI(self):
        ### SPI parameters ###
        scale = 12
        distrib = indices.Distribution('gamma')
        Periodicity = compute.Periodicity(12)
        ### SPI parameters ###
        dff_daterange = join(self.datadir,'models_daterange.xlsx')
        df_daterange = pd.read_excel(dff_daterange)
        # print(df_daterange)
        # exit()
        experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        product_list = ['pr']
        param_list = []
        for product in tqdm(product_list, desc='product'):
            fdir = join(self.datadir, product, 'per_pix')
            outdir = join(self.datadir, product, 'SPI')
            for experiment in tqdm(experiment_list, desc='experiment'):
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                model_list = T.listdir(fdir_i)
                outdir_experiment = join(outdir, experiment)
                for model in tqdm(model_list, desc='model'):
                    print(product,experiment,model)
                    date_list = df_daterange[(df_daterange['product']==product) & (df_daterange['experiment']==experiment) & (df_daterange['model']==model)]['date_list'].tolist()
                    if len(date_list) != 1:
                        continue
                    date_list = date_list[0]
                    date_list = eval(date_list)

                    date_obj_list = []
                    for date in date_list:
                        year,mon = date[:4],date[4:]
                        year = int(year)
                        month = int(mon)
                        date_obj = datetime.datetime(year, month, 1)
                        date_obj_list.append(date_obj)
                    date_obj_list = sorted(date_obj_list)

                    missing_dates_index = []
                    missing_dates_obj_list = []
                    for i,date_obj in enumerate(date_obj_list):
                        if i + 1 == len(date_obj_list):
                            continue
                        date_obj_1 = date_obj_list[i]
                        date_obj_2 = date_obj_list[i+1]
                        delta = date_obj_2 - date_obj_1
                        delta = delta.days
                        if delta > 31:
                            missing_dates_index.append(i)
                            missing_dates_obj_list.append(date_obj_1)
                    if len(date_obj_list) == 0:
                        continue
                    date_start = date_obj_list[0]
                    date_end = date_obj_list[-1]
                    date_range = pd.date_range(date_start, date_end, freq='MS')

                    outdir_model = join(outdir_experiment, model)
                    T.mkdir(outdir_model, force=True)
                    fdir_model = join(fdir_i, model)
                    # dict_i = T.load_npy_dir(fdir_model)
                    params = [fdir_model, missing_dates_index, date_range, scale,distrib,Periodicity,outdir_model]
                    # if not outdir_model == '/Volumes/NVME2T/hotcold_drought/data/CMIP6/pr/SPI/ssp126/ACCESS-CM2':
                    #     continue
                    param_list.append(params)
                    self.kernel_calculate_SPI(params)
        # MULTIPROCESS(self.kernel_calculate_SPI, param_list).run(process=7)

    def clean_tif(self):
        # product = 'mrsos'
        product = 'tas'
        experiment_list = ['historical']
        outdir_father = join(self.datadir,product,'tif_resample_unify_clean')
        for exp in experiment_list:
            fdir = join(self.datadir, product, 'tif_resample_unify', exp)
            outdir = join(outdir_father,exp)
            T.mkdir(outdir,force=True)
            for model in tqdm(T.listdir(fdir),desc=exp):
                outdir_i = join(outdir,model)
                T.mkdir(outdir_i,force=True)
                for f in T.listdir(join(fdir,model)):
                    if not f.endswith('.tif'):
                        continue
                    fpath = join(fdir,model,f)
                    outpath = join(outdir_i,f)
                    arr,originX,originY,pixelWidth,pixelHeight = ToRaster().raster2array(fpath)
                    # arr[arr>=70] = np.nan
                    # arr[arr<=0] = np.nan
                    arr[arr >= 1000] = np.nan
                    arr[arr <= 0] = np.nan
                    ToRaster().array2raster(outpath,originX,originY,pixelWidth,pixelHeight,arr)

    def check_individual_model(self):
        product = 'mrsos'
        # product = 'tas'
        experiment_list = ['historical']
        fdir = join(self.datadir,product,'per_pix')
        for exp in experiment_list:
            fdir_i = join(fdir,exp)
            color_len = len(T.listdir(fdir_i))
            color_list = T.gen_colors(color_len)
            flag = 0
            for model in T.listdir(fdir_i):
                # if not model == 'FGOALS-f3-L':
                #     continue
                fdir_model = join(fdir_i,model)
                dict_i = T.load_npy_dir(fdir_model)
                all_vals = []
                for pix in tqdm(dict_i):
                    vals = dict_i[pix]
                    if np.nanstd(vals) == 0:
                        continue
                    all_vals.append(vals)
                mean = np.nanmean(all_vals,axis=0)
                plt.plot(mean,label=model,color=color_list[flag])
                if type(mean) != np.float64:
                    print(mean)
                    print(type(mean))
                    plt.text(0,mean[0],model,color='k')
                flag += 1
            plt.legend()
            plt.show()
        pass

    def check_ensemble_model(self):
        product = 'mrsos'
        # product = 'tas'
        experiment_list = ['historical']
        fdir = join(self.datadir,product,'per_pix_ensemble')
        for exp in experiment_list:
            fdir_i = join(fdir,exp)
            dict_i = T.load_npy_dir(fdir_i)
            all_vals = []
            spatial_dict = {}
            flag = 0
            for pix in tqdm(dict_i):
                vals = dict_i[pix]
                if np.nanstd(vals) == 0:
                    continue
                vals = np.array(vals)
                vals[vals<-99] = np.nan
                if True in np.isnan(vals):
                    spatial_dict[pix] = 1
                    continue
                flag += 1
                # if flag > 10000:
                #     plt.plot(vals)
                #     plt.title(pix)
                #     plt.show()
                all_vals.append(vals)
            arr = D_CMIP.pix_dic_to_spatial_arr(spatial_dict)
            plt.imshow(arr,cmap='jet',interpolation='nearest')
            plt.show()
            mean = np.nanmean(all_vals,axis=0)
            plt.plot(mean,label=exp)
        plt.legend()
        plt.show()
        pass

    def tif_ensemble(self):
        product = 'mrsos'
        # product = 'tas'
        experiment_list = ['historical']
        flist = []
        for year in range(1976,2015):
            for mon in range(1,13):
                fname = f'{year}{mon:02d}.tif'
                flist.append(fname)
        outdir = join(self.datadir,product,'tif_resample_unify_clean_ensemble')
        T.mkdir(outdir,force=True)

        for exp in experiment_list:
            outdir_i = join(outdir,exp)
            T.mkdir(outdir_i)
            fdir = join(self.datadir,product,'tif_resample_unify_clean',exp)
            for f in flist:
                print(f,'\n')
                outf = join(outdir_i,f)
                flist_ensemble = []
                for model in T.listdir(fdir):
                    fpath = join(fdir,model,f)
                    if not isfile(fpath):
                        continue
                    flist_ensemble.append(fpath)
                Pre_Process().compose_tif_list(flist_ensemble,outf)

        # fdir = join(self.datadir,'pr','SPI')
        pass

    def modify_FGOALS_f3_L(self):
        experiment_list = ['historical']
        model = 'FGOALS-f3-L'
        product = 'mrsos'
        for exp in experiment_list:
            fdir = join(self.datadir,product,'tif_resample_unify_clean',exp,model)
            for f in tqdm(T.listdir(fdir),desc=exp):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir,f)
                array,originX,originY,pixelWidth,pixelHeight = ToRaster().raster2array(fpath)
                array[array<0] = np.nan
                array = array * 100
                ToRaster().array2raster(fpath,originX,originY,pixelWidth,pixelHeight,array)
        pass

    def per_pix_ensemble(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        product_list = ['tas','mrsos']
        param_list = []
        for product in product_list:
            fdir = join(self.datadir, product, 'tif_resample_unify_clean_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                Pre_Process().data_transform(fdir_i,outdir_i)


    def per_pix_ensemble_baseline(self):

        experiment_list = ['historical']
        product_list = ['tas','mrsos']
        param_list = []
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_baseline')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                spatial_dict = T.load_npy_dir(fdir_i)
                spatial_dict_climatology = {}
                for pix in tqdm(spatial_dict,desc=f'{product} {experiment}'):
                    vals = spatial_dict[pix]
                    vals[vals<0] = np.nan
                    climatology_means = []
                    climatology_std = []
                    for m in range(1, 13):
                        one_mon = []
                        for i in range(len(vals)):
                            mon = i % 12 + 1
                            if mon == m:
                                one_mon.append(vals[i])
                        mean = np.nanmean(one_mon)
                        std = np.nanstd(one_mon)
                        climatology_means.append(mean)
                        climatology_std.append(std)
                    spatial_dict_climatology[pix] = {
                        'mean': climatology_means,
                        'std': climatology_std
                    }
                outf = join(outdir_i, 'climatology_mean.npy')
                T.save_npy(spatial_dict_climatology, outf)


    def per_pix_ensemble_gs(self):
        # experiment_list = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']
        experiment_list = ['historical']

        # product_list = ['pr', 'tas']
        # product_list = ['tas','mrsos']
        product_list = ['tas']
        # product_list = ['pr']
        # product_list = ['gpp']
        # product_list = ['lai']
        # product_list = ['mrsos']
        param_list = []
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_ensemble')
            outdir = join(self.datadir, product, 'per_pix_ensemble_gs')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mkdir(outdir_i,force=True)
                spatial_dict = T.load_npy_dir(fdir_i)
                spatial_dict_gs = {}
                for pix in tqdm(spatial_dict,desc=f'{product} {experiment}'):
                    vals = spatial_dict[pix]
                    gs = global_get_gs_CMIP(pix)
                    vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                    spatial_dict_gs[pix] = vals_gs
                outf = join(outdir_i, 'gs.npy')
                T.save_npy(spatial_dict_gs, outf)


class CMIP_statistic:

    def __init__(self):

        pass

    def run(self):
        self.spatial_trend()
        pass

    def spatial_trend(self):
        f = join(data_root, 'CMIP6/mrsos/per_pix_anomaly/ssp585/MRI-ESM2-0/anomaly.npy')
        spatial_dict = T.load_npy(f)
        spatial_dict_mean = {}
        for pix in tqdm(spatial_dict):
            vals = spatial_dict[pix]
            # vals_mean = np.nanmean(vals)
            a,b,r,p = T.nan_line_fit(list(range(len(vals))),vals)
            spatial_dict_mean[pix] = a
        arr = D_CMIP.pix_dic_to_spatial_arr(spatial_dict_mean)
        plt.imshow(arr,interpolation='nearest',vmin=-0.002,vmax=0.002,cmap='RdBu')
        plt.colorbar()
        plt.show()
        pass


def main():
    CMIP6().run()
    # CMIP6_historical().run()
    # CMIP_statistic().run()

if __name__ == '__main__':
    main()