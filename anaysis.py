
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
        self.result_root=rf'F:\Hotdrought_Resilience\results\analysis\\'
        T.mkdir(self.result_root,force=True)

        self.threshold = -1.5

    def run(self):

        self.pick_normal_drought_events()
        # # self.pick_normal_hot_events()  ## not use here
        self.gen_dataframe()

        pass


    def pick_normal_drought_events(self):
        outdir = join(self.result_root, 'picked_events')
        T.mk_dir(outdir)
        threshold = self.threshold
        scale = 'spi03'

        SPI_fdir = join(data_root,rf'SPI/dic/{scale}')

        SPI_dict = T.load_npy_dir(SPI_fdir)
        events_dic = {}
        params_list = []
        for pix in tqdm(SPI_dict):
            SPI_vals = SPI_dict[pix]
            SPI_vals = np.array(SPI_vals)
            SPI_vals = np.array(SPI_vals)
            SPI_vals[SPI_vals < -999] = np.nan
            params = (SPI_vals, threshold)
            params_list.append(params)
            events_list = self.kernel_find_drought_period(params)
            # print(events_list)
            if len(events_list) == 0:
                continue
            drought_year_list = []
            drought_month_list = []
            min_values_list = []
            for drought_range in events_list:
                min_index = T.pick_min_indx_from_1darray(SPI_vals, drought_range)
                # print(min_index)
                min_values=SPI_vals[min_index]
                min_values_list.append(min_values)
                # print(min_values)
                drought_year = min_index // 12 + global_start_year
                drought_year_list.append(drought_year)
                drought_month = min_index % 12 + 1
                drought_month_list.append(drought_month)
            # drought_year_list = np.array(drought_year_list)
            drought_info_dict = {'drought_year':drought_year_list,'drought_mon':drought_month_list,'drought_val':min_values_list}
            events_dic[pix] = drought_info_dict
        outf = join(outdir, scale)
        T.save_npy(events_dic, outf)


    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        threshold = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:  # SPEI
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

            events_list.append(new_i)
        return events_list


    def pick_normal_hot_events(self):  ## not used
        outdir = join(self.this_class_arr, 'normal_hot_events')
        T.mk_dir(outdir)
        threshold_quantile = 90
        fdir = join(data_root, 'CRU_temp/annual_growth_season_temp_10degree')
        dic_temp = T.load_npy_dir(fdir)
        drought_events_dir = join(self.this_class_arr, 'picked_events')
        for f in T.listdir(drought_events_dir):
            scale = f.split('.')[0]
            fpath = join(drought_events_dir, f)
            drought_events_dict = T.load_npy(fpath)
            hot_dic = {}
            normal_dic = {}
            for pix in tqdm(drought_events_dict,desc=f'{scale}'):
                spi_drought_dict_i = drought_events_dict[pix]
                spi_drought_year = spi_drought_dict_i['drought_year']
                spi_drought_mon = spi_drought_dict_i['drought_mon']
                if not pix in dic_temp:
                    continue
                T_annual_val = dic_temp[pix]
                T_quantile = np.percentile(T_annual_val, threshold_quantile)
                hot_index_True_False = T_annual_val > T_quantile
                hot_years = []
                for i, val in enumerate(hot_index_True_False):
                    if val == True:
                        hot_years.append(i + global_start_year)
                # hot_years = set(hot_years)
                # print(hot_years)
                # exit()
                hot_drought_year = []
                hot_drought_mon = []
                spi_drought_year_spare = []
                spi_drought_mon_spare = []

                for i,dr in enumerate(spi_drought_year):
                    if dr in hot_years:
                        hot_drought_year.append(dr)
                        hot_drought_mon.append(spi_drought_mon[i])
                    else:
                        spi_drought_year_spare.append(dr)
                        spi_drought_mon_spare.append(spi_drought_mon[i])
                hot_dic[pix] = {'drought_year':hot_drought_year,'drought_mon':hot_drought_mon}
                normal_dic[pix] = {'drought_year':spi_drought_year_spare,'drought_mon':spi_drought_mon_spare}
            hot_outf = join(outdir, f'hot-drought_{scale}.npy')
            normal_outf = join(outdir, f'normal-drought_{scale}.npy')
            T.save_npy(hot_dic, hot_outf)
            T.save_npy(normal_dic, normal_outf)


    def gen_dataframe(self):
        outdir = join(self.result_root, 'Dataframe')
        T.mk_dir(outdir)
        drought_events_dir = join(self.result_root, 'picked_events')
        print(drought_events_dir)
        scale = 'spi03'
        pix_list = []
        drought_year_list = []
        drought_mon_list = []
        drought_type_list = []
        drought_scale_list = []
        drought_val_list = []
        drought_year_dict_all = {}

        for f in tqdm(T.listdir(drought_events_dir)):
            fpath = join(drought_events_dir, f)
            var_i = f.split('.')[0]
            drought_type = var_i.split('_')[0]
            scale = var_i.split('.')[0]
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                events_dict = spatial_dict[pix]
                # pprint(events)
                drought_year_list_i = events_dict['drought_year']
                drought_mon_list_i = events_dict['drought_mon']
                drought_severity_list_i = events_dict['drought_val']
                # print(drought_severity_list_i)
                for i in range(len(drought_year_list_i)):
                    pix_list.append(pix)
                    drought_year_list.append(drought_year_list_i[i])
                    drought_mon_list.append(drought_mon_list_i[i])
                    drought_type_list.append(drought_type)
                    drought_scale_list.append(scale)
                    drought_val_list.append(drought_severity_list_i[i])


        # exit()
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_year'] = drought_year_list
        df['drought_mon'] = drought_mon_list
        df['drought_serverity'] = drought_val_list


        df['drought_scale'] = drought_scale_list

        df = df.sort_values(by=['pix','drought_year'])
        df = df.reset_index(drop=True)
        outf = join(outdir,'drought_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)





class Dataframe:

    def __init__(self):
        self.result=results_root+rf'\\analysis\Dataframe\\'

        self.dff = join(self.result,'drought_dataframe.df')
        pass

    def run(self):
        # self.copy_df() ## only one time

        df = self.__df_init()
        # df = self.add_SOS_EOS(df)
        # df = self.filter_drought_events_via_SOS_EOS(df)
        # df=self.add_temp_to_df(df)
        # self.check_df(df)
        # df = self.add_GS_NDVI(df)

        df = self.add_GS_values_post_n(df)
        # df=self.add_aridity_to_df(df)
        # df=self.add_MODIS_LUCC_to_df(df)
        # df=self.add_koppen_to_df(df)
        # # df=self.add_tif_to_df(df)
        # df=self.add_landcover_data_to_df(df)
        # df=self.add_landcover_classfication_to_df(df)
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

    def check_df(self,df):
        global_land_tif = join(this_root,'conf/land.tif')
        DIC_and_TIF().plot_df_spatial_pix(df,global_land_tif)
        plt.show()
        pass

    def add_temp_to_df(self,df):

        Temp_fdir = join(data_root, rf'CRU_temp\annual_growth_season_temp_zscore_10degree')
        Temp_dict = T.load_npy_dir(Temp_fdir)
        Temp_year_list = list(range(1982, 2021))

        result_dict = {}

        for i, row in tqdm(df.iterrows(), total=len(df), desc='add Temp'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']

            if np.isnan(sos) or np.isnan(eos):
                continue

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


    def add_GS_NDVI(self,df):
        NDVI_fdir = join(data_root,'NDVI4g/annual_growth_season_NDVI_relative_change')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']
            drought_month = row['drought_mon']

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

            NDVI_drought_year_GS_mean = NDVI_year_dict[year]
            result_dict[i] = NDVI_drought_year_GS_mean
        df['GS_NDVI_relative_change'] = result_dict
        df = df.dropna(subset=['GS_NDVI_relative_change'])
        df = df.reset_index(drop=True)
        return df

    def add_rt(self,df):
        NDVI_fdir = join(data_root,'NDVI4g/annual_growth_season_NDVI_anomaly')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']
            drought_month = row['drought_mon']

            if np.isnan(sos) or np.isnan(eos):
                continue
            sos = int(sos)
            eos = int(eos)
            year = row['drought_year']
            if not pix in NDVI_dict:
                continue
            NDVI = NDVI_dict[pix]
            NDVI = list(NDVI)
            # long_term_mean=np.nanmean(NDVI)
            # print(long_term_mean)
            if T.is_all_nan(NDVI):
                continue
            if not len(NDVI_year_list) == len(NDVI):
                print(pix)
                print(len(NDVI),len(NDVI_year_list))
                print('----')
                continue
            NDVI_year_dict = T.dict_zip(NDVI_year_list,NDVI)
            # pprint(NDVI_reshape_dict);exit()
            ## get previous year NDVI
            pre_year_1 = year-1
            pre_year_2 = year-2
            pre_year_3 = year-3
            pre_year_4 = year-4
            if pre_year_4 < 1982:
                continue
            NDVI_pre_year = np.nanmean(NDVI_year_dict[pre_year_4]+NDVI_year_dict[pre_year_3]+NDVI_year_dict[pre_year_2]+NDVI_year_dict[pre_year_1])
            print(NDVI_pre_year)

            NDVI_drought_year_GS_mean = NDVI_year_dict[year]/NDVI_pre_year
            result_dict[i] = NDVI_drought_year_GS_mean
        df['rt_4years'] = result_dict
        df = df.dropna(subset=['GS_NDVI'])
        df = df.reset_index(drop=True)
        return df

    def add_GS_values_post_n(self,df):
        post_n_years = 4
        NDVI_fdir = join(data_root,'NDVI4g/annual_growth_season_NDVI_relative_change')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']
            drought_month = row['drought_mon']

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

    def add_rs(self,df):
        post_n_years = 4
        NDVI_fdir = join(data_root,'NDVI4g/annual_growth_season_NDVI_anomaly')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']
            drought_month = row['drought_mon']

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

            pre_year_1 = year - 1
            pre_year_2 = year - 2
            pre_year_3 = year - 3
            pre_year_4 = year - 4
            if pre_year_4 < 1982:
                continue
            NDVI_pre_year = np.nanmean(
                NDVI_year_dict[pre_year_4] + NDVI_year_dict[pre_year_3] + NDVI_year_dict[pre_year_2] + NDVI_year_dict[
                    pre_year_1])
            rs = np.nanmean(post_n_year_values)/NDVI_pre_year
            result_dict[i] = rs
        df[f'rs_4years'] = result_dict
        df = df.dropna(subset=['GS_NDVI'])
        df = df.reset_index(drop=True)
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

def main():

    # Pick_drought_events().run()
    Dataframe().run()




if __name__ == '__main__':
    main()
    pass