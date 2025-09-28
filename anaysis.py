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
        self.threshold = -0.5

    def run(self):
        self.pick_normal_drought_events()
        # self.add_hot_normal_drought()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        T.mk_dir(outdir)
        threshold = self.threshold

        scale_list = global_selected_spi_list
        SPI_dict_all = {}
        for scale in scale_list:
            print('loading data', scale)
            SPI_data,_ = Load_Data().SPI_scale(scale)
            SPI_dict_all[scale] = SPI_data

        for scale in scale_list:
            SPI_dict = SPI_dict_all[scale]
            params_list = []
            outf = join(outdir, f'{scale}.df')
            pix_list = []
            drought_range_list = []
            drougth_timing_list = []
            drought_year_list = []
            for pix in tqdm(SPI_dict,desc=f'{scale}'):
                vals = SPI_dict[pix]
                vals = np.array(vals)
                if True in np.isnan(vals):
                    continue
                params = (vals, threshold)
                # params_list.append(params)
                drought_events_list, drought_timing_list = self.kernel_find_drought_period(params)
                for i in range(len(drought_events_list)):
                    event = drought_events_list[i]
                    s,e = event
                    mid = int((s+e)/2)
                    drought_year = int(mid/12) + global_start_year
                    timing = drought_timing_list[i]
                    pix_list.append(pix)
                    drought_range_list.append(event)
                    drougth_timing_list.append(timing)
                    drought_year_list.append(drought_year)
            df = pd.DataFrame()
            df['pix'] = pix_list
            df['drought_range'] = drought_range_list
            df['drought_year'] = drought_year_list
            df['drought_month'] = drougth_timing_list
            df['scale'] = scale
            T.save_df(df,outf)
            T.df_to_excel(df,outf)


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

    def add_hot_normal_drought(self):
        fdir = join(self.this_class_arr, 'picked_events')
        Temperature_anomaly_data_dict,_ = Load_Data().ERA_Tair_anomaly_detrend()
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            drought_type_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f):
                pix = row['pix']
                drought_range = row['drought_range']
                e = int(drought_range[1])
                s = int(drought_range[0])
                temperature_anomaly_in_drought = Temperature_anomaly_data_dict[pix][s:e+1]
                # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                mean_temperature = np.nanmean(temperature_anomaly_in_drought)
                if mean_temperature > 1:
                    drought_type = 'hot-drought'
                # elif mean_temperature < 0:
                #     drought_type = 'cold-drought'
                else:
                    drought_type = 'normal-drought'
                drought_type_list.append(drought_type)
            df['drought_type'] = drought_type_list
            outf = join(fdir,f)
            T.save_df(df,outf)
            T.df_to_excel(df,outf)

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
        # self.copy_df()

        df = self.__df_init()
        # df = self.add_SOS_EOS(df)
        df = self.add_GS_NDVI(df)


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
        SOS_EOS_dir = join(data_root,'MODIS_phenology/SOS_EOS_mon')
        SOS_tif = join(SOS_EOS_dir,'sos_mon.tif')
        EOS_tif = join(SOS_EOS_dir,'eos_mon.tif')
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