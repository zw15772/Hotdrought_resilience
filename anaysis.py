from more_itertools.more import combination_index
from pyasn1_modules.rfc3412 import SNMPv3Message

from global_init import *
# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta_info import *
from pprint import pprint

result_root_this_script = join(results_root, 'analysis')

class Pick_drought_events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events', result_root_this_script, mode=2)
        self.threshold = -2

    def run(self):

        # self.pick_normal_drought_events()
        # self.pick_normal_hot_events()
        # self.drought_timing()
        self.gen_dataframe()
        pass


    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        T.mk_dir(outdir)
        threshold = self.threshold
        scale = 'spi03'
        SPI_fdir = join(data_root,f'/Volumes/SSD1T/Hotdrought_Resilience/data/SPI/dic/{scale}')

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
            if len(events_list) == 0:
                continue
            drought_year_list = []
            for drought_range in events_list:
                min_index = T.pick_min_indx_from_1darray(SPI_vals, drought_range)
                drought_year = min_index // 12 + global_start_year
                drought_year_list.append(drought_year)
            drought_year_list = np.array(drought_year_list)
            events_dic[pix] = drought_year_list
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


    def pick_normal_hot_events(self):
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
                spi_drought_year = drought_events_dict[pix]
                if not pix in dic_temp:
                    continue
                T_annual_val = dic_temp[pix]
                T_quantile = np.percentile(T_annual_val, threshold_quantile)
                hot_index_True_False = T_annual_val > T_quantile
                hot_years = []
                for i, val in enumerate(hot_index_True_False):
                    if val == True:
                        hot_years.append(i + global_start_year)
                hot_years = set(hot_years)
                # print(hot_years)
                # exit()
                hot_drought_year = []
                spi_drought_year_spare = []
                for dr in spi_drought_year:
                    if dr in hot_years:
                        hot_drought_year.append(dr)
                    else:
                        spi_drought_year_spare.append(dr)
                hot_dic[pix] = hot_drought_year
                normal_dic[pix] = spi_drought_year_spare
            hot_outf = join(outdir, f'hot-drought_{scale}.npy')
            normal_outf = join(outdir, f'normal-drought_{scale}.npy')
            T.save_npy(hot_dic, hot_outf)
            T.save_npy(normal_dic, normal_outf)

    def drought_timing(self):
        outdir = join(self.this_class_arr, 'drought_timing')
        T.mk_dir(outdir)
        threshold = self.threshold
        scale = 'spi03'
        SPI_fdir = join(data_root, f'/Volumes/SSD1T/Hotdrought_Resilience/data/SPI/dic/{scale}')
        SPI_dict = T.load_npy_dir(SPI_fdir)
        events_dic = {}
        events_mon_dic = {}
        params_list = []
        for pix in tqdm(SPI_dict):
            vals = SPI_dict[pix]
            vals = np.array(vals)
            params = (vals, threshold)
            params_list.append(params)
            events_list = self.kernel_find_drought_period(params)
            if len(events_list) == 0:
                continue
            drought_year_list = []
            drought_month_list = []
            for drought_range in events_list:
                min_index = T.pick_min_indx_from_1darray(vals, drought_range)
                drought_year = min_index // 12 + global_start_year
                drought_month = min_index % 12 + 1
                drought_year_list.append(drought_year)
                drought_month_list.append(drought_month)
            drought_year_list = np.array(drought_year_list)
            drought_month_list = np.array(drought_month_list)
            events_dic[pix] = drought_year_list
            events_mon_dic[pix] = drought_month_list
        outf_year = join(outdir, f'{scale}_drought_year.npy')
        outf_mon = join(outdir, f'{scale}_drought_mon.npy')
        T.save_npy(events_dic, outf_year)
        T.save_npy(events_mon_dic, outf_mon)


    def gen_dataframe(self):
        outdir = join(self.this_class_arr,'drought_dataframe')
        T.mk_dir(outdir)
        drought_events_dir = join(self.this_class_arr, 'normal_hot_events')
        drought_timing_dir = join(self.this_class_arr,'drought_timing')
        scale = 'spi03'
        pix_list = []
        drought_year_list = []
        drought_type_list = []
        drought_scale_list = []
        drought_year_dict_all = {}
        drought_mon_dict_all = {}
        drought_year_f = join(drought_timing_dir,f'{scale}_drought_year.npy')
        drought_mon_f = join(drought_timing_dir,f'{scale}_drought_mon.npy')
        drought_year_dict = T.load_npy(drought_year_f)
        drought_mon_dict = T.load_npy(drought_mon_f)
        drought_year_dict_all[scale] = drought_year_dict
        drought_mon_dict_all[scale] = drought_mon_dict

        for f in tqdm(T.listdir(drought_events_dir)):
            fpath = join(drought_events_dir, f)
            var_i = f.split('.')[0]
            drought_type = var_i.split('_')[0]
            scale = var_i.split('_')[1]
            spatial_dict = T.load_npy(fpath)
            for pix in spatial_dict:
                events = spatial_dict[pix]
                for e in events:
                    pix_list.append(pix)
                    drought_year_list.append(e)
                    drought_type_list.append(drought_type)
                    drought_scale_list.append(scale)
        # exit()
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_year'] = drought_year_list
        df['drought_type'] = drought_type_list
        df['drought_scale'] = drought_scale_list
        # T.print_head_n(df)
        # exit()
        # add drought timing
        # drought_timing_year_list = []
        drought_mon_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            pix = row['pix']
            year = row['drought_year']
            scale = row['drought_scale']
            if not pix in drought_year_dict_all[scale]:
                drought_mon_list.append(np.nan)
                continue
            drought_year = drought_year_dict_all[scale][pix]
            drought_year = list(drought_year)
            if not year in drought_year:
                drought_mon_list.append(np.nan)
                continue
            drought_mon = drought_mon_dict_all[scale][pix]
            drought_mon = list(drought_mon)
            drought_year_index = drought_year.index(year)
            drought_mon_i = drought_mon[drought_year_index]
            drought_mon_list.append(drought_mon_i)

        df['drought_mon'] = drought_mon_list
        df = df.sort_values(by=['pix','drought_type','drought_year'])
        df = df.reset_index(drop=True)
        outf = join(outdir,'drought_dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)


class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe','dataframe.df')
        pass

    def run(self):
        # self.copy_df() ## only one time


        df = self.__df_init()
        df = self.add_SOS_EOS(df)
        # df = self.add_GS_NDVI(df)


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
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        fpath = join(Pick_drought_events().this_class_arr,'picked_events/spi03.df')
        df = T.load_df(fpath)
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)

        pass

    def add_SOS_EOS(self,df):
        SOS_EOS_dir = join(data_root,'CRU_temp/extract_growing_season')

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

    def add_GS_NDVI(self,df):
        NDVI_fdir = join(data_root,'NDVI4g/per_pix')
        NDVI_dict = T.load_npy_dir(NDVI_fdir)
        NDVI_year_list = list(range(1982,2021))

        result_dict = {}

        for i,row in tqdm(df.iterrows(),total=len(df),desc='add GS NDVI'):
            pix = row['pix']
            sos = row['sos']
            eos = row['eos']
            drought_month = row['drought_month']

            if np.isnan(sos) or np.isnan(eos):
                continue
            sos = int(sos)
            eos = int(eos)
            year = row['drought_year']
            NDVI = NDVI_dict[pix]
            if T.is_all_nan(NDVI):
                continue
            NDVI_reshape = np.reshape(NDVI,(-1,12))
            NDVI_reshape_dict = T.dict_zip(NDVI_year_list,NDVI_reshape)

            if sos < eos:
                if drought_month < sos:
                    continue
                if drought_month > eos:
                    continue
                NDVI_drought_year = NDVI_reshape_dict[year]
                NDVI_drought_year_GS = NDVI_drought_year[sos-1:eos]
            else:
                if drought_month >= sos:
                    if not year+1 in NDVI_reshape_dict:
                        continue
                    NDVI_drought_year1 = NDVI_reshape_dict[year]
                    NDVI_drought_year2 = NDVI_reshape_dict[year+1]
                    NDVI_drought_year_GS1 = NDVI_drought_year1[sos-1:]
                    NDVI_drought_year_GS2 = NDVI_drought_year2[:eos]
                elif drought_month <= eos:
                    if not year-1 in NDVI_reshape_dict:
                        continue
                    NDVI_drought_year1 = NDVI_reshape_dict[year-1]
                    NDVI_drought_year2 = NDVI_reshape_dict[year]
                    NDVI_drought_year_GS1 = NDVI_drought_year1[sos-1:]
                    NDVI_drought_year_GS2 = NDVI_drought_year2[:eos]
                else:
                    continue
                NDVI_drought_year_GS = np.concatenate((NDVI_drought_year_GS1,NDVI_drought_year_GS2))
            NDVI_drought_year_GS_mean = np.nanmean(NDVI_drought_year_GS)
            result_dict[i] = NDVI_drought_year_GS_mean
        df['GS_NDVI'] = result_dict
        df = df.dropna(subset=['GS_NDVI'])
        df = df.reset_index(drop=True)
        return df

    def __df_init(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        return df

def main():

    Pick_drought_events().run()
    # Dataframe().run()

if __name__ == '__main__':
    main()
    pass