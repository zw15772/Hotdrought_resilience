from conda.common.io import print_instrumentation_data

from global_init import *

# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta_info import *

result_root_this_script = join(results_root, 'analysis')

class Pick_drought_events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events', result_root_this_script, mode=2)
        self.threshold = -2

    def run(self):

        # self.pick_normal_hot_events()
        # self.gen_dataframe()
        pass



    def kernel_find_drought_period(self, params):
        vals = params[0]
        vals = np.array(vals)
        threshold = params[1]
        threshold_start = -1
        start_of_drought_list = []
        end_of_drought_list = []
        for i in range(len(vals)):
            if i + 1 == len(vals):
                break
            val_left = vals[i]
            vals_right = vals[i + 1]
            if val_left < threshold_start and vals_right > threshold_start:
                end_of_drought_list.append(i+1)
            if val_left > threshold_start and vals_right < threshold_start:
                start_of_drought_list.append(i)

        drought_events_list = []
        for s in start_of_drought_list:
            for e in end_of_drought_list:
                if e > s:
                    drought_events_list.append((s,e))
                    break

        drought_events_list_extreme = []
        drought_timing_list = []
        for event in drought_events_list:
            s = event[0]
            e = event[1]
            min_index = T.pick_min_indx_from_1darray(vals,list(range(s,e)))
            drought_timing_month = min_index % 12 + 1
            min_val = vals[min_index]
            if min_val < threshold:
                drought_events_list_extreme.append(event)
                drought_timing_list.append(drought_timing_month)
        return drought_events_list_extreme,drought_timing_list



        #     drought_vals = vals[s:e]
        #     plt.plot(list(range(s,e)),drought_vals,zorder=10,lw=4)
        #     plt.scatter(min_index,vals[min_index],c='r',zorder=20,s=100)
        # plt.plot(vals,c='k')
        # plt.show()


        # # test plot events

        # exit()
        # T.color_map_choice()
        # color_list = T.gen_colors(len(drought_events_list),palette='Dark2')
        # color_list_random_index = np.random.choice(range(len(drought_events_list)), len(drought_events_list))
        # color_list_random = []
        # for i in color_list_random_index:
        #     color_list_random.append(color_list[i])

        # for i in range(len(drought_events_list)):
        #     start = drought_events_list[i][0]
        #     end = drought_events_list[i][1]
        #     plt.scatter(start, vals[start], color=color_list_random[i],zorder=10,s=50,alpha=0.5)
        #     plt.scatter(end, vals[end], color=color_list_random[i],zorder=10,s=50,alpha=0.5)
        #
        # plt.plot(vals, c='k',lw=0.5)
        # plt.hlines(threshold_start, 0, len(vals))
        # plt.show()




    def pick_normal_hot_events(self):
        outdir = join(self.this_class_arr, 'normal_hot_events')
        T.mk_dir(outdir)
        threshold_quantile = 90
        # gs_dict = Growing_season().longterm_growing_season()
        ## load growing season temperature detrend raw 39years
        fdir = join(data_root,'CRU_temp/annual_growth_season_temp_10degree')
        dic_temp=T.load_npy_dir(fdir)


        drought_events_dir = join(self.this_class_arr, 'picked_events')
        for f in T.listdir(drought_events_dir):
            if not f == 'spi03.df':
                continue
            scale = f.split('.')[0]
            fpath = join(drought_events_dir, f)
            drought_events_df = T.load_df(fpath)
            drought_events_dict=T.df_groupby(drought_events_df,'pix')
            hot_dic = {}
            normal_dic = {}
            for pix in tqdm(drought_events_dict,desc=f'{scale}'):
                spi_drought_year = drought_events_dict[pix]['drought_year'].tolist()
                # print(spi_drought_year);exit()
                if not pix in dic_temp:
                    continue
                temp_growing_season_raw = dic_temp[pix]
                temp_growing_season_raw=np.array(temp_growing_season_raw)
                temp_growing_season_raw = list(temp_growing_season_raw)
                # print(temp_growing_season_raw.shape)
                # if temp_growing_season_raw == None:
                #     continue
                if np.isnan(temp_growing_season_raw[-1]):
                    temp_growing_season_raw = temp_growing_season_raw[:-1]

                temp_growing_season_raw_detrend=T.detrend_vals(temp_growing_season_raw)
                # print(temp_growing_season_raw_detrend)
                # if not pix in global_gs_dict:
                #     continue
                # gs_mon = global_gs_dict[pix]
                # gs_mon = list(gs_mon)
                temp_growing_season_raw_detrend = temp_growing_season_raw
                T_quantile = np.percentile(temp_growing_season_raw_detrend, threshold_quantile)
                hot_index_True_False = temp_growing_season_raw_detrend > T_quantile
                hot_years = []
                for i, val in enumerate(hot_index_True_False):
                    if val == True:
                        hot_years.append(i + global_start_year)
                hot_years = set(hot_years)

                hot_drought_year = []
                spi_drought_year_spare = []
                for dr in spi_drought_year:
                    if dr in hot_years:
                        hot_drought_year.append(dr)
                    else:
                        spi_drought_year_spare.append(dr)
                hot_dic[pix] = hot_drought_year
                normal_dic[pix] = spi_drought_year_spare
                # print(hot_drought_year)
                # print(spi_drought_year_spare)
                # print(spi_drought_year)
                # plt.plot(list(range(1982,2021)),temp_growing_season_raw_detrend)
                # plt.show()
            hot_outf = join(outdir, f'hot-drought_{scale}.npy')
            normal_outf = join(outdir, f'normal-drought_{scale}.npy')
            T.save_npy(hot_dic, hot_outf)
            T.save_npy(normal_dic, normal_outf)

    def gen_dataframe(self):
        fdir = join(self.this_class_arr,'picked_events')
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        outf = join(outdir,'dataframe.df')
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            print(f)
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            df_list.append(df)
        df_concat = pd.concat(df_list)
        df_concat = df_concat.sort_values(by=['pix','scale'])
        T.save_df(df_concat,outf)
        T.df_to_excel(df_concat,outf)

        pass


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

    # Pick_drought_events().run()
    Dataframe().run()

if __name__ == '__main__':
    main()
    pass