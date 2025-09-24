# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from meta_info import *

result_root_this_script = join(results_root, 'analysis')

class Pick_drought_events:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_Drought_Events', result_root_this_script, mode=2)
        self.threshold = -2

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

class Pick_drought_events_SM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_SM', result_root_this_script, mode=2)
        # self.threshold = -2

    def run(self):
        self.pick_normal_drought_events()
        self.add_hot_normal_drought()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        threshold_list = [
            -0.5, # mild drought
            -1, # moderate drought
            -1.5, # severe drought
            -2, # extreme drought
            -np.inf, # exceptional drought
        ]
        T.mk_dir(outdir)
        # threshold = self.threshold
        # data_dict,_ = Load_Data().ERA_SM_anomaly_detrend()
        data_dict,_ = Load_Data().GLEAM_SMRoot_anomaly_detrend()

        for threshold_i in range(len(threshold_list)):
            threshold_upper = threshold_list[threshold_i]
            if threshold_i == len(threshold_list) - 1:
                break
            threshold_bottom = threshold_list[threshold_i+1]

            pix_list = []
            drought_range_list = []
            drougth_timing_list = []
            drought_year_list = []
            intensity_list = []
            severity_list = []
            severity_mean_list = []

            for pix in tqdm(data_dict,desc=f'pick_{threshold_upper}'):
                r,c = pix
                if r > 600: # Antarctica
                    continue
                vals = data_dict[pix]
                vals = np.array(vals)
                std = np.nanstd(vals)
                if std == 0 or np.isnan(std):
                    continue
                drought_events_list, drought_timing_list = self.kernel_find_drought_period(vals,threshold_upper,threshold_bottom)
                for i in range(len(drought_events_list)):
                    event = drought_events_list[i]
                    s,e = event
                    mid = int((s+e)/2)
                    drought_year = int(mid/12) + global_start_year
                    timing = drought_timing_list[i]
                    duration = e - s
                    intensity = np.nanmin(vals[s:e])
                    severity = np.nansum(vals[s:e])
                    severity_mean = np.nanmean(vals[s:e])

                    pix_list.append(pix)
                    drought_range_list.append(event)
                    drougth_timing_list.append(timing)
                    drought_year_list.append(drought_year)
                    intensity_list.append(intensity)
                    severity_list.append(severity)
                    severity_mean_list.append(severity_mean)

            df = pd.DataFrame()
            df['pix'] = pix_list
            df['drought_range'] = drought_range_list
            df['drought_year'] = drought_year_list
            df['drought_month'] = drougth_timing_list
            df['threshold'] = threshold_upper
            df['intensity'] = intensity_list
            df['severity'] = severity_list
            df['severity_mean'] = severity_mean_list

            outf = join(outdir, f'{threshold_upper}.df')
            T.save_df(df,outf)
            T.df_to_excel(df,outf)


    def kernel_find_drought_period(self,vals,threshold_upper,threshold_bottom,threshold_start=-.5):
        vals = np.array(vals)
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
            if min_val > threshold_bottom and min_val < threshold_upper:
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
            T_max_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f):
                pix = row['pix']
                drought_range = row['drought_range']
                e = int(drought_range[1])
                s = int(drought_range[0])
                temperature_anomaly_in_drought = Temperature_anomaly_data_dict[pix][s:e+1]
                # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                max_temperature = np.nanmax(temperature_anomaly_in_drought)
                if max_temperature > 1:
                    drought_type = 'hot-drought'
                # elif mean_temperature < 0:
                #     drought_type = 'cold-drought'
                else:
                    drought_type = 'normal-drought'
                T_max_list.append(max_temperature)
                drought_type_list.append(drought_type)
            df['drought_type'] = drought_type_list
            df['T_max'] = T_max_list
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

class Pick_drought_events_SM_multi_thresholds:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_SM_multi_thresholds', result_root_this_script, mode=2)
        # self.threshold = -2

    def run(self):
        # self.pick_normal_drought_events()
        self.add_hot_normal_drought()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        # threshold_list = [
        #     -0.5, # mild drought
        #     -1, # moderate drought
        #     -1.5, # severe drought
        #     -2, # extreme drought
        #     -np.inf, # exceptional drought
        # ]
        threshold_list = np.arange(-0.5,-3.1,-0.1)
        threshold_list = [round(i,1) for i in threshold_list]
        T.mk_dir(outdir)
        # threshold = self.threshold
        # data_dict,_ = Load_Data().ERA_SM_anomaly_detrend()
        data_dict,_ = Load_Data().GLEAM_SMRoot_anomaly_detrend()

        for threshold_i in range(len(threshold_list)):
            threshold_upper = threshold_list[threshold_i]
            if threshold_i == len(threshold_list) - 1:
                break
            threshold_bottom = threshold_list[threshold_i+1]

            pix_list = []
            drought_range_list = []
            drougth_timing_list = []
            drought_year_list = []
            intensity_list = []
            severity_list = []
            severity_mean_list = []

            for pix in tqdm(data_dict,desc=f'pick_{threshold_upper}'):
                r,c = pix
                if r > 600: # Antarctica
                    continue
                vals = data_dict[pix]
                vals = np.array(vals)
                std = np.nanstd(vals)
                if std == 0 or np.isnan(std):
                    continue
                drought_events_list, drought_timing_list = self.kernel_find_drought_period(vals,threshold_upper,threshold_bottom)
                for i in range(len(drought_events_list)):
                    event = drought_events_list[i]
                    s,e = event
                    mid = int((s+e)/2)
                    drought_year = int(mid/12) + global_start_year
                    timing = drought_timing_list[i]
                    duration = e - s
                    intensity = np.nanmin(vals[s:e])
                    severity = np.nansum(vals[s:e])
                    severity_mean = np.nanmean(vals[s:e])

                    pix_list.append(pix)
                    drought_range_list.append(event)
                    drougth_timing_list.append(timing)
                    drought_year_list.append(drought_year)
                    intensity_list.append(intensity)
                    severity_list.append(severity)
                    severity_mean_list.append(severity_mean)

            df = pd.DataFrame()
            df['pix'] = pix_list
            df['drought_range'] = drought_range_list
            df['drought_year'] = drought_year_list
            df['drought_month'] = drougth_timing_list
            df['threshold'] = threshold_upper
            df['intensity'] = intensity_list
            df['severity'] = severity_list
            df['severity_mean'] = severity_mean_list

            outf = join(outdir, f'{threshold_upper}.df')
            T.save_df(df,outf)
            T.df_to_excel(df,outf)


    def kernel_find_drought_period(self,vals,threshold_upper,threshold_bottom,threshold_start=-.5):
        vals = np.array(vals)
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
            if min_val > threshold_bottom and min_val < threshold_upper:
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
            T_level_list = []
            T_max_list = []
            # for T_threshold in Temperature_threshold_list:
            for i,row in tqdm(df.iterrows(),total=len(df),desc=f):
                pix = row['pix']
                drought_range = row['drought_range']
                e = int(drought_range[1])
                s = int(drought_range[0])
                temperature_anomaly_in_drought = Temperature_anomaly_data_dict[pix][s:e+1]
                # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                max_temperature = np.nanmax(temperature_anomaly_in_drought)
                T_level = int(max_temperature*10) / 10.
                T_max_list.append(max_temperature)
                T_level_list.append(T_level)
            df['T_level'] = T_level_list
            df['T_max'] = T_max_list
            T.print_head_n(df)
            # exit()
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


class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)

        # print('add NDVI mask')
        # df = self.add_NDVI_mask(df)
        # if is_clean_df == True:
        #     df = self.clean_df(df)
        print('add GLC2000')
        df = self.add_GLC_landcover_data_to_df(df)
        # print('add Aridity Index')
        df = self.add_AI_to_df(df)

        print('add AI_reclass')
        df = self.AI_reclass(df)
        df = df[df['AI_class'] == 'Arid']
        self.df = df

    def clean_df(self,df):

        # df = df[df['lat']>=30]
        # df = df[df['landcover_GLC'] != 'Crop']
        df = df[df['NDVI_MASK'] == 1]
        # df = df[df['ELI_significance'] == 1]
        return df

    def add_GLC_landcover_data_to_df(self, df):
        f = join(data_root,'GLC2000/reclass_lc/glc2000_025.npy')
        val_dic=T.load_npy(f)
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

    def add_NDVI_mask(self,df):
        # f =rf'C:/Users/pcadmin/Desktop/Data/Base_data/NDVI_mask.tif'
        f = join(data_root, 'NDVI4g/NDVI_mask.tif')
        print(f)

        array, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f)
        array = np.array(array, dtype=float)
        val_dic = D.spatial_arr_to_dic(array)
        f_name = 'NDVI_MASK'
        print(f_name)
        # exit()
        val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df)):

            pix = row['pix']
            if not pix in val_dic:
                val_list.append(np.nan)
                continue
            vals = val_dic[pix]
            if vals < -99:
                val_list.append(np.nan)
                continue
            val_list.append(vals)
        df[f_name] = val_list
        return df

    def add_AI_to_df(self, df):
        f = join(data_root, 'Aridity_index/Aridity_index.tif')
        spatial_dict = D.spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, D)
        return df

    def AI_reclass(self,df):
        AI_class = []
        for i,row in tqdm(df.iterrows(),total=len(df),desc='AI_reclass'):
            AI = row['aridity_index']
            if AI < 0.65:
                AI_class.append('Arid')
            elif AI >= 0.65:
                AI_class.append('Humid')
            elif np.isnan(AI):
                AI_class.append(np.nan)
            else:
                print(AI)
                raise ValueError('AI error')
        df['AI_class'] = AI_class
        return df

    def add_koppen(self,df):
        f = join(data_root, 'koppen/koppen_reclass_dic.npy')
        val_dic = T.load_npy(f)
        df = T.add_spatial_dic_to_df(df, val_dic, 'Koppen')
        return df

    # def add_ELI_significance(self,df):
    #     from Chapter5 import analysis
    #     f = join(Water_energy_limited_area().this_class_tif, 'significant_pix_r/ELI_Temp_significance.tif')
    #     spatial_dict = D.spatial_tif_to_dic(f)
    #     df = T.add_spatial_dic_to_df(df, spatial_dict, 'ELI_significance')
    #
    #     return df



class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe.df')
        pass

    def run(self):
        self.copy_df()
        df = self.__gen_df_init()
        # df = Dataframe_func(df).df
        # df = self.add_variables_during_droughts_GS(df,Load_Data().NDVI4g_climatology_percentage_detrend)
        # df = df.dropna()
        # df = self.add_variables_during_droughts(df,Load_Data().Precipitation_anomaly)
        # df = self.add_variables_after_droughts_GS(df,Load_Data().NDVI4g_climatology_percentage_detrend)
        # df = self.add_SPI_characteristic(df)
        # df = self.add_variables_during_droughts(df,Load_Data().ERA_Tair_anomaly_detrend)
        # df = self.add_MAT_MAP(df)

        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def copy_df(self):
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff = join(Pick_drought_events().this_class_arr,'dataframe/dataframe.df')
        df = T.load_df(dff)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def add_variables_during_droughts(self,df,data_obj):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()

        mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_{var_name}'):
            pix = row['pix']
            drought_range = row['drought_range']
            vals = data_dict[pix]
            vals_during_drought = vals[drought_range[0]:drought_range[1]+1]
            mean = np.nanmean(vals_during_drought)
            mean_list.append(mean)
        df[f'{var_name}'] = mean_list

        return df

    def add_SPI_characteristic(self,df):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        scale_list = global_selected_spi_list
        SPI_dict_all = {}
        for scale in scale_list:
            print('loading data', scale)
            SPI_data, _ = Load_Data().SPI_scale(scale)
            SPI_dict_all[scale] = SPI_data

        result_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_SPI_characteristic'):
            pix = row['pix']
            scale = row['scale']
            drought_range = row['drought_range']
            SPI = SPI_dict_all[scale][pix]
            vals_during_drought = SPI[drought_range[0]:drought_range[1]+1]
            # mean = np.nanmean(vals_during_drought)
            duration = len(vals_during_drought)
            intensity = np.nanmin(vals_during_drought)
            severity = np.nansum(vals_during_drought)
            scale_int = int(scale.replace('spi',''))
            result_dict_i = { # todo: add this func to lytools
                'duration':duration,
                'intensity':intensity,
                'severity':severity,
                'scale_int':scale_int,
            }
            result_list.append(result_dict_i)
        colname = result_list[0].keys()
        for col in colname:
            print('add',col)
            col_list = []
            for i in result_list:
                col_list.append(i[col])
            df[col] = col_list
        return df

    def add_MAT_MAP(self,df):
        MAP_f = join(data_root,'TerraClimate/ppt/MAP/MAP.tif')
        MAT_f =join(data_root,'ERA5/Tair/MAT/MAT_C.tif')
        MAP_dict = D.spatial_tif_to_dic(MAP_f)
        MAT_dict = D.spatial_tif_to_dic(MAT_f)
        df = T.add_spatial_dic_to_df(df,MAP_dict,'MAP')
        df = T.add_spatial_dic_to_df(df,MAT_dict,'MAT')
        return df

    def add_variables_during_droughts_GS(self,df,data_obj):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()

        mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_{var_name}'):
            pix = row['pix']
            GS = global_get_gs(pix)

            drought_range = row['drought_range']
            e,s = drought_range[1],drought_range[0]
            picked_index = []
            for idx in range(s,e+1):
                mon = idx % 12 + 1
                if not mon in GS:
                    continue
                picked_index.append(idx)
            if len(picked_index) == 0:
                mean_list.append(np.nan)
                continue
            if not pix in data_dict:
                mean_list.append(np.nan)
                continue
            vals = data_dict[pix]
            picked_vals = T.pick_vals_from_1darray(vals,picked_index)
            mean = np.nanmean(picked_vals)
            if mean == 0:
                mean_list.append(np.nan)
                continue
            mean_list.append(mean)
        df[f'{var_name}_GS'] = mean_list
        return df


    def add_variables_after_droughts_GS(self,df,data_obj):
        n_list = [1,2,3,4]
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()
        for n in n_list:
            delta_mon = n * 12
            mean_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                pix = row['pix']
                GS = global_get_gs(pix)
                if not pix in data_dict:
                    mean_list.append(np.nan)
                    continue
                vals = data_dict[pix]
                drought_range = row['drought_range']
                end_mon = drought_range[-1]
                post_drought_range = []
                for i in range(delta_mon):
                    post_drought_range.append(end_mon + i + 1)
                picked_index = []
                for idx in post_drought_range:
                    mon = idx % 12 + 1
                    if not mon in GS:
                        continue
                    if idx >= len(vals):
                        picked_index = []
                        break
                    picked_index.append(idx)
                if len(picked_index) == 0:
                    mean_list.append(np.nan)
                    continue
                picked_vals = T.pick_vals_from_1darray(vals, picked_index)
                mean = np.nanmean(picked_vals)
                if mean == 0:
                    mean_list.append(np.nan)
                    continue
                mean_list.append(mean)
            df[f'{var_name}_post_{n}_GS'] = mean_list

        return df

class Dataframe_SM:

    def __init__(self):
        # self.Dataframe_mode = 'SM_detrend_NDVI_with_trend_pre_baseline'
        # self.Dataframe_mode = 'SM_detrend_NDVI_with_trend_longterm_baseline'
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir(f'Dataframe_SM_multi_threshold', result_root_this_script, mode=2)
            # T.mk_class_dir('Dataframe_SM_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Dataframe_SM_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Dataframe_SM', result_root_this_script, mode=2)
        self.df_dir = join(self.this_class_arr, 'dataframe')

        pass

    def run(self):
        # self.copy_df()
        # exit()
        for f in T.listdir(self.df_dir):
            if not f.endswith('.df'):
                continue
            print(f)
            self.dff = join(self.df_dir,f)
            df = self.__gen_df_init()
            cols = df.columns
            for col in cols:
                print(col)
            # exit()
            # df = Dataframe_func(df).df
            # df = self.add_rt_pre_baseline(df,Load_Data().NDVI4g_climatology_percentage)
            # df = self.add_rt_pre_baseline(df,Load_Data().NDVI4g_climatology_percentage_detrend)
            # df = self.add_rt_pre_baseline(df,Load_Data().NDVI4g_climatology_percentage_detrend)
            # df = self.add_rt_pre_baseline(df,Load_Data().LT_Baseline_NT_origin)
            # df = self.add_rt_pre_baseline(df,Load_Data().LT_CFE_Hybrid_NT_origin)
            # df = self.add_variables_during_droughts_GS(df,Load_Data().NDVI4g_climatology_percentage)
            # df = self.add_variables_during_droughts_GS(df,Load_Data().NDVI4g_climatology_percentage_detrend)
            # df = df.dropna()
            # df = self.add_variables_during_droughts_GS(df, Load_Data().ERA_Tair_juping)
            # df = self.add_variables_during_droughts_GS(df, Load_Data1().vpd_anomaly)
            # df = self.add_variables_post_droughts_GS(df,Load_Data().NDVI4g_climatology_percentage)
            # df = self.add_variables_post_droughts_GS(df,Load_Data().NDVI4g_climatology_percentage_detrend)
            # df = self.add_variables_during_droughts_GS(df,Load_Data1().vpd_anomaly)
            # df = self.add_variables_during_droughts_GS(df,Load_Data().LT_Baseline_NT_anomaly_detrend)

            # df = self.add_rs_pre_baseline(df,Load_Data().NDVI4g_climatology_percentage)
            # df = self.add_rs_pre_baseline(df,Load_Data().NDVI4g_climatology_percentage_detrend)
            df = self.add_rs_pre_baseline(df,Load_Data().LT_Baseline_NT_origin)
            df = self.add_rs_pre_baseline(df,Load_Data().LT_CFE_Hybrid_NT_origin)
            # df = self.add_variables_during_droughts(df,Load_Data().ERA_Tair_anomaly_detrend)
            # df = self.add_MAT_MAP(df)
            # df = self.add_BNPP(df)
            # df = self.add_water_table_depth(df)
            # df = self.add_soil_type(df)
            # df = self.add_rooting_depth(df)
            # df = self.add_SOC(df)
            # df = self.add_variables_CV(df,Load_Data().ERA_Tair_origin)
            # df = self.add_variables_CV(df,Load_Data().Precipitation_origin)

            T.save_df(df, self.dff)
            T.df_to_excel(df, self.dff)
            # T.print_head_n(df)
            # exit()
        self.merge_df()
        pass

    def __gen_df_init(self):
        if not os.path.isfile(self.dff):
            df = pd.DataFrame()
            T.save_df(df,self.dff)
            return df
        else:
            df,dff = self.__load_df()
            return df

    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        # T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

    def copy_df(self):
        # drought_df_dir = join(Pick_drought_events_SM().this_class_arr,'picked_events')
        drought_df_dir = join(Pick_drought_events_SM_multi_thresholds().this_class_arr,'picked_events')
        outdir = self.df_dir
        T.mk_dir(outdir)
        for f in T.listdir(drought_df_dir):
            if not f.endswith('.df'):
                continue
            outf = join(outdir,f)
            fpath = join(drought_df_dir,f)
            if isfile(outf):
                print('Warning: this function will overwrite the dataframe')
                print('Warning: this function will overwrite the dataframe')
                print('Warning: this function will overwrite the dataframe')
                pause()
                pause()

            df = T.load_df(fpath)
            T.save_df(df,outf)
            T.df_to_excel(df,outf)
        pass

    def add_variables_during_droughts(self,df,data_obj):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()

        mean_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_{var_name}'):
            pix = row['pix']
            drought_range = row['drought_range']
            vals = data_dict[pix]
            vals_during_drought = vals[drought_range[0]:drought_range[1]+1]
            mean = np.nanmean(vals_during_drought)
            mean_list.append(mean)
        df[f'{var_name}'] = mean_list

        return df

    def add_variables_CV(self,df,data_obj):
        outdir = join(self.this_class_tif,'CV')
        T.mk_dir(outdir)
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()
        outf = join(outdir,f'{var_name}.tif')
        print(outf)
        if not os.path.isfile(outf):
            CV_spatial_dict = {}
            for pix in tqdm(data_dict):
                vals = data_dict[pix]
                if np.nanstd(vals) == 0:
                    continue
                vals_reshape = vals.reshape(-1,12)
                annual_vals = []
                for i in range(len(vals_reshape)):
                    annual_vals.append(np.nanmean(vals_reshape[i]))
                vals_CV = np.nanstd(annual_vals) / np.nanmean(annual_vals)
                CV_spatial_dict[pix] = vals_CV
            D.pix_dic_to_tif(CV_spatial_dict,outf)
        CV_spatial_dict = D.spatial_tif_to_dic(outf)
        df = T.add_spatial_dic_to_df(df,CV_spatial_dict,f'{var_name}_CV')
        return df

    def add_MAT_MAP(self,df):
        MAP_f = join(data_root,'TerraClimate/ppt/MAP/MAP.tif')
        MAT_f =join(data_root,'ERA5/Tair/MAT/MAT_C.tif')
        MAP_dict = D.spatial_tif_to_dic(MAP_f)
        MAT_dict = D.spatial_tif_to_dic(MAT_f)
        df = T.add_spatial_dic_to_df(df,MAP_dict,'MAP')
        df = T.add_spatial_dic_to_df(df,MAT_dict,'MAT')
        return df

    def add_BNPP(self,df):
        fpath = join(data_root,'BNPP/tif_025/BNPP_0-200cm.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'BNPP')
        return df

    def add_SOC(self,df):
        soc_tif = join(data_root,'SoilGrids/SOC/tif_sum/SOC_sum.tif')
        spatial_dict = D.spatial_tif_to_dic(soc_tif)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'SOC')
        return df

    def add_rooting_depth(self,df):
        soc_tif = join(data_root,'Rooting_Depth/tif_025_unify_merge/rooting_depth.tif')
        spatial_dict = D.spatial_tif_to_dic(soc_tif)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'rooting_depth')
        return df


    def add_water_table_depth(self,df):
        fpath = join(data_root,'water_table_depth/tif_025/cwdx80.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'water_table_depth')
        return df

    def add_soil_type(self,df):
        fpath = join(data_root,'HWSD/tif_025/S_SILT.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'S_SILT')
        return df

    def add_variables_during_droughts_GS(self,df,data_obj):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()
        # delta_mon = 12
        during_drought_val_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_{var_name}'):
            pix = row['pix']
            GS = global_get_gs(pix)

            drought_range = row['drought_range']
            e,s = drought_range[1],drought_range[0]
            picked_index = []
            for idx in range(s,e+1):
                mon = idx % 12 + 1
                if not mon in GS:
                    continue
                picked_index.append(idx)
            if len(picked_index) == 0:
                during_drought_val_list.append(np.nan)
                continue
            if not pix in data_dict:
                during_drought_val_list.append(np.nan)
                continue
            vals = data_dict[pix]
            picked_vals = T.pick_vals_from_1darray(vals,picked_index)
            mean_during_drought = np.nanmean(picked_vals)
            if mean_during_drought == 0:
                during_drought_val_list.append(np.nan)
                continue

            during_drought_val_list.append(mean_during_drought)
        df[f'{var_name}_GS'] = during_drought_val_list
        return df

    def add_variables_post_droughts_GS(self,df,data_obj):
        n_list = [1,2,3,4]
        # pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()
        for n in n_list:
            delta_mon = n * 12
            post_drought_val_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                pix = row['pix']
                GS = global_get_gs(pix)
                if not pix in data_dict:
                    post_drought_val_list.append(np.nan)
                    continue
                vals = data_dict[pix]
                drought_range = row['drought_range']
                end_mon = drought_range[-1]
                post_drought_range = []
                for i in range(delta_mon):
                    post_drought_range.append(end_mon + i + 1)
                picked_index_post = []
                for idx in post_drought_range:
                    mon = idx % 12 + 1
                    if not mon in GS:
                        continue
                    if idx >= len(vals):
                        picked_index = []
                        break
                    picked_index_post.append(idx)
                if len(picked_index_post) == 0:
                    post_drought_val_list.append(np.nan)
                    continue
                picked_vals_post = T.pick_vals_from_1darray(vals, picked_index_post)
                mean_post = np.nanmean(picked_vals_post)
                if mean_post == 0:
                    post_drought_val_list.append(np.nan)
                    continue
                post_drought_val_list.append(mean_post)

            df[f'{var_name}_post_{n}_GS'] = post_drought_val_list

        return df

    def add_rt_pre_baseline(self,df,data_obj,pre_year=2):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()
        # delta_mon = 12
        rt_list = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_{var_name}'):
            pix = row['pix']
            GS = global_get_gs(pix)

            drought_range = row['drought_range']
            e,s = drought_range[1],drought_range[0]
            picked_index = []
            for idx in range(s,e+1):
                mon = idx % 12 + 1
                if not mon in GS:
                    continue
                picked_index.append(idx)
            if len(picked_index) == 0:
                rt_list.append(np.nan)
                continue
            if not pix in data_dict:
                rt_list.append(np.nan)
                continue
            vals = data_dict[pix]
            picked_vals = T.pick_vals_from_1darray(vals,picked_index)
            mean_during_drought = np.nanmean(picked_vals)
            if mean_during_drought == 0:
                rt_list.append(np.nan)
                continue

            pre_drought_range = []
            for i in range(pre_year * 12):
                idx = s - i - 1
                if idx < 0:
                    pre_drought_range = []
                    break
                pre_drought_range.append(idx)
            if len(pre_drought_range) == 0:
                rt_list.append(np.nan)
                continue
            pre_drought_range = pre_drought_range[::-1]
            picked_index_pre = []
            for idx in pre_drought_range:
                mon = idx % 12 + 1
                if not mon in GS:
                    continue
                picked_index_pre.append(idx)
            if len(picked_index_pre) == 0:
                rt_list.append(np.nan)
                continue
            picked_vals_pre = T.pick_vals_from_1darray(vals, picked_index_pre)
            mean_pre_drought = np.nanmean(picked_vals_pre)
            rt = mean_during_drought / mean_pre_drought
            rt_list.append(rt)
        df[f'{var_name}_rt_pre_baseline_GS'] = rt_list
        return df


    def add_rs_pre_baseline(self,df,data_obj,pre_year=2):
        n_list = [1,2,3,4]
        # pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict, var_name = data_obj()
        for n in n_list:
            delta_mon = n * 12
            rs_list = []
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f'add_post_{n}_{var_name}'):
                pix = row['pix']
                GS = global_get_gs(pix)
                if not pix in data_dict:
                    rs_list.append(np.nan)
                    continue
                vals = data_dict[pix]
                drought_range = row['drought_range']
                end_mon = drought_range[-1]
                post_drought_range = []
                for i in range(delta_mon):
                    post_drought_range.append(end_mon + i + 1)
                picked_index_post = []
                for idx in post_drought_range:
                    mon = idx % 12 + 1
                    if not mon in GS:
                        continue
                    if idx >= len(vals):
                        picked_index = []
                        break
                    picked_index_post.append(idx)
                if len(picked_index_post) == 0:
                    rs_list.append(np.nan)
                    continue
                picked_vals_post = T.pick_vals_from_1darray(vals, picked_index_post)
                mean_post = np.nanmean(picked_vals_post)
                if mean_post == 0:
                    rs_list.append(np.nan)
                    continue

                start_mon = drought_range[0]
                pre_drought_range = []
                for i in range(pre_year * 12):
                    idx = start_mon - i - 1
                    if idx < 0:
                        pre_drought_range = []
                        break
                    pre_drought_range.append(idx)
                if len(pre_drought_range) == 0:
                    rs_list.append(np.nan)
                    continue
                pre_drought_range = pre_drought_range[::-1]
                picked_index_pre = []
                for idx in pre_drought_range:
                    mon = idx % 12 + 1
                    if not mon in GS:
                        continue
                    picked_index_pre.append(idx)
                if len(picked_index_pre) == 0:
                    rs_list.append(np.nan)
                    continue
                picked_vals_pre = T.pick_vals_from_1darray(vals, picked_index_pre)
                mean_pre_drought = np.nanmean(picked_vals_pre)
                rs = mean_post / mean_pre_drought

                rs_list.append(rs)
            df[f'{var_name}_rs_pre_baseline_{n}_GS'] = rs_list

        return df

    def merge_df(self):
        fdir = self.df_dir
        outdir = join(self.this_class_arr,'dataframe_merge')
        T.mk_dir(outdir)
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            print(f)
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            df_list.append(df)
        df_concat = pd.concat(df_list)
        df_concat = df_concat.sort_values(by=['pix','threshold','drought_range'])
        outf = join(outdir,'dataframe_merge.df')
        T.save_df(df_concat,outf)
        T.df_to_excel(df_concat,outf,random=True)
        pass


class Optimal_temperature:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Optimal_temperature', result_root_this_script, mode=2)

    def run(self):
        step = .5
        # self.cal_opt_temp(step)
        # self.tif_opt_temp()
        self.plot_test_cal_opt_temp(step)
        pass

    def cal_opt_temp(self,step):
        dff = join(Dataframe_SM().this_class_arr,'dataframe/-0.5.df')
        df_global = T.load_df(dff)
        pix_list = T.get_df_unique_val_list(df_global,'pix')

        # step = 1  # Celsius
        # outdir = join(self.this_class_tif,f'optimal_temperature')


        # temp_dic,_ = Load_Data().ERA_Tair_origin()
        # temp_dic,_ = Load_Data().Temperature_max_origin()
        # ndvi_dic,vege_name = Load_Data().NDVI4g_origin()
        # ndvi_dic,vege_name = Load_Data().LT_Baseline_NT_origin()
        T_dir = join(data_root,'TerraClimate/tmax/per_pix/1982-2020')
        # NDVI_dir = join(data_root,'NDVI4g/per_pix/1982-2020')
        # vege_name = 'NDVI4g'
        NDVI_dir = join(data_root,'GPP/per_pix/LT_Baseline_NT/1982-2020')
        vege_name = 'LT_Baseline_NT'
        outdir = join(self.this_class_arr, f'optimal_temperature',f'{vege_name}_step_{step}_celsius')
        T.mk_dir(outdir,force=True)
        # outdir_i = join(outdir,f'{vege_name}_step_{step}_celsius.tif')
        param_list = []
        for f in T.listdir(NDVI_dir):
            param = [NDVI_dir,T_dir,outdir,step,f,pix_list]
            param_list.append(param)
            # self.kernel_cal_opt_temp(param)
        MULTIPROCESS(self.kernel_cal_opt_temp,param_list).run(process=7)

    def tif_opt_temp(self):
        outdir = join(self.this_class_tif,f'optimal_temperature','T_max')
        T.mk_dir(outdir,force=True)
        folder_name = 'LT_Baseline_NT_step_0.5_celsius'
        fdir = join(self.this_class_arr,'optimal_temperature',folder_name)
        spatial_dict = T.load_npy_dir(fdir)
        spatial_dict_new = {}
        for pix in spatial_dict:
            val = spatial_dict[pix]
            val = val + 273.15
            spatial_dict_new[pix] = val
        outf = join(outdir,f'{folder_name}.tif')
        D.pix_dic_to_tif(spatial_dict_new,outf)

    def kernel_cal_opt_temp(self,params):
        NDVI_dir,T_dir,outdir,step,f,pix_list = params
        fpath_NDVI = join(NDVI_dir, f)
        fpath_T = join(T_dir, f)
        ndvi_dic = T.load_npy(fpath_NDVI)
        temp_dic = T.load_npy(fpath_T)
        optimal_temp_dic = {}
        for pix in pix_list:
            if not pix in ndvi_dic:
                continue
            ndvi = ndvi_dic[pix]
            temp = temp_dic[pix]
            temp = np.array(temp)
            # temp = np.array(temp) - 273.15  # Kelvin to Celsius
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df = df[df['ndvi'] > 0]
            df = df.dropna()
            if len(df) == 0:
                continue
            max_t = max(df['temp'])
            min_t = int(min(df['temp']))
            t_bins = np.arange(start=min_t, stop=max_t, step=step)
            df_group, bins_list_str = T.df_bin(df, 'temp', t_bins)
            quantial_90_list = []
            x_list = []
            for name, df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                if len(vals) == 0:
                    continue
                quantile_90 = np.nanquantile(vals, 0.9)
                left = name[0].left
                x_list.append(left)
                quantial_90_list.append(quantile_90)
            # multi regression to find the optimal temperature
            # parabola fitting
            x = np.array(x_list)
            y = np.array(quantial_90_list)
            if len(x) < 3:
                continue
            if len(y) < 3:
                continue
            a, b, c = self.nan_parabola_fit(x, y)
            y_fit = a * x ** 2 + b * x + c
            T_opt = x[np.argmax(y_fit)]
            optimal_temp_dic[pix] = T_opt
        outf = join(outdir, f)
        T.save_npy(optimal_temp_dic, outf)

    def plot_test_cal_opt_temp(self,step):
        dff = join(Dataframe_SM().this_class_arr,'dataframe/-0.5.df')
        df_global = T.load_df(dff)
        df_global = df_global[df_global['AI_class']=='Arid']
        pix_list = T.get_df_unique_val_list(df_global,'pix')

        # step = 1  # Celsius
        outdir = join(self.this_class_arr,f'optimal_temperature')
        outf = join(outdir,f'step_{step}_celsius')
        T.mk_dir(outdir)

        temp_dic,_ = Load_Data().ERA_Tair_origin()
        ndvi_dic,_ = Load_Data().NDVI4g_origin()
        # ndvi_dic,_ = Load_Data().LT_Baseline_NT_origin()

        optimal_temp_dic = {}
        for pix in tqdm(pix_list):
            ndvi = ndvi_dic[pix]
            temp = temp_dic[pix]
            temp = np.array(temp) - 273.15 # Kelvin to Celsius
            df = pd.DataFrame()
            df['ndvi'] = ndvi
            df['temp'] = temp
            df = df[df['ndvi'] > 0.1]
            df = df.dropna()
            max_t = max(df['temp'])
            min_t = int(min(df['temp']))
            t_bins = np.arange(start=min_t,stop=max_t,step=step)
            df_group, bins_list_str = T.df_bin(df,'temp',t_bins)
            # ndvi_list = []
            # box_list = []
            color_list = T.gen_colors(len(df_group))
            color_list = color_list[::-1]
            flag = 0
            quantial_90_list = []
            x_list = []
            for name,df_group_i in df_group:
                vals = df_group_i['ndvi'].tolist()
                quantile_90 = np.nanquantile(vals,0.9)
                left = name[0].left
                x_list.append(left)
                plt.scatter([left]*len(vals),vals,s=20,color=color_list[flag])
                flag += 1
                quantial_90_list.append(quantile_90)
                # box_list.append(vals)
                # mean = np.nanmean(vals)
                # ndvi_list.append(mean)
            # multi regression to find the optimal temperature
            # parabola fitting
            x = np.array(x_list)
            y = np.array(quantial_90_list)
            # a,b,c = np.polyfit(x,y,2)
            a,b,c = self.nan_parabola_fit(x,y)
            # plot abc
            # y = ax^2 + bx + c
            y_fit = a*x**2 + b*x + c
            plt.plot(x,y_fit,'k--',lw=2)
            opt_T = x[np.argmax(y_fit)]
            plt.scatter([opt_T],[np.max(y_fit)],s=200,marker='*',color='r',zorder=99)
            print(len(y_fit))
            print(len(quantial_90_list))
            a_,b_,r_,p_ = T.nan_line_fit(y_fit,quantial_90_list)
            r2 = r_**2
            print(r2)
            # exit()


            plt.plot(x_list,quantial_90_list,c='k',lw=2)
            plt.title(f'a={a:.3f},b={b:.3f},c={c:.3f}')
            print(t_bins)
            # # plt.plot(t_bins[:-1],ndvi_list)
            # plt.boxplot(box_list,positions=t_bins[:-1],showfliers=False)
            plt.show()


            # exit()
        #     t_mean_list = []
        #     ndvi_mean_list = []
        #     for i in range(len(t_bins)):
        #         if i + 1 >= len(t_bins):
        #             continue
        #         df_t = df[df['temp']>t_bins[i]]
        #         df_t = df_t[df_t['temp']<t_bins[i+1]]
        #         t_mean = df_t['temp'].mean()
        #         # t_mean = t_bins[i]
        #         ndvi_mean = df_t['ndvi'].mean()
        #         t_mean_list.append(t_mean)
        #         ndvi_mean_list.append(ndvi_mean)
        #
        #     indx_list = list(range(len(ndvi_mean_list)))
        #     max_indx = T.pick_max_indx_from_1darray(ndvi_mean_list,indx_list)
        #     if max_indx > 999:
        #         optimal_temp = np.nan
        #     else:
        #         optimal_temp = t_mean_list[max_indx]
        #     optimal_temp_dic[pix] = optimal_temp
        # T.save_npy(optimal_temp_dic,outf)

    def nan_parabola_fit(self, val1_list, val2_list):
        if not len(val1_list) == len(val2_list):
            raise UserWarning('val1_list and val2_list must have the same length')
        val1_list_new = []
        val2_list_new = []
        for i in range(len(val1_list)):
            val1 = val1_list[i]
            val2 = val2_list[i]
            if np.isnan(val1):
                continue
            if np.isnan(val2):
                continue
            val1_list_new.append(val1)
            val2_list_new.append(val2)
        a,b,c = np.polyfit(val1_list_new, val2_list_new, 2)

        return a,b,c

class Correlation_analysis_phenology:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Correlation_analysis_phenology', result_root_this_script, mode=2)
        pass

    def run(self):
        self.run_corr()
        # self.plot_corr()
        pass

    def run_corr(self):
        outdir = join(self.this_class_tif, 'NDVI_vs_Temp')
        T.mk_dir(outdir)
        NDVI_data, NDVI_name = Load_Data().NDVI4g_origin()
        Temp_data, Temp_name = Load_Data().ERA_Tair_origin()
        date_list = global_date_obj_list()
        spatial_dict = {}
        for pix in tqdm(NDVI_data):
            r,c = pix
            if r > 240:
                continue
            NDVI = NDVI_data[pix]
            Temp = Temp_data[pix]
            # if np.nanstd(NDVI) == 0:
            #     continue
            if T.is_all_nan(NDVI):
                continue
            if True in np.isnan(NDVI):
                continue
            r,p = T.nan_correlation(NDVI,Temp)
            # if r < 0.7:
            #     continue
            # spatial_dict[pix] = r

            if r < 0.7:
                plt.plot(date_list,NDVI)
                plt.twinx()
                plt.plot(date_list,Temp,c='r')
                plt.title(pix)
                plt.show()
        arr = D.pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,vmin=0.6,vmax=1,cmap='RdBu_r',interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        outf = join(outdir, 'NDVI_vs_Temp_mask_07.tif')
        D.pix_dic_to_tif(spatial_dict,outf)

        pass

    def plot_corr(self):
        # tif_path = join(self.this_class_tif, 'NDVI_vs_Temp', 'NDVI_vs_Temp.tif')
        # tif_path = join(self.this_class_tif, 'NDVI_vs_Temp', 'NDVI_vs_Temp_mask.tif')
        tif_path = join(self.this_class_tif, 'NDVI_vs_Temp', 'NDVI_vs_Temp_mask_07.tif')
        # Plot().plot_ortho(tif_path,vmin=0.6,vmax=1,cmap='RdBu_r')
        Plot().plot_ortho(tif_path,vmin=0.7,vmax=.7,cmap='RdBu_r')
        plt.show()
        pass


def t_VPD():
    fdir = '/Volumes/NVME4T/greening_project_redo/data/original_dataset/VPD_dic'
    spatial_dict = T.load_npy_dir(fdir)
    spatial_dict_mean = {}
    for pix in tqdm(spatial_dict):
        vals = spatial_dict[pix]
        print(len(vals))
        exit()
        vals_mean = np.nanmean(vals)
        spatial_dict_mean[pix] = vals_mean
    arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dict_mean)
    arr[arr==0] = np.nan
    plt.imshow(arr,interpolation='nearest',cmap='jet',vmin=0,vmax=1)
    plt.colorbar()
    plt.show()

    pass

def main():
    # Pick_drought_events().run()
    # Pick_drought_events_SM().run()
    # Pick_drought_events_SM_multi_thresholds().run()
    # Dataframe().run()
    # Dataframe_SM().run()
    # Dataframe_SM_trend().run()
    Optimal_temperature().run()
    # Correlation_analysis_phenology().run()
    # t_VPD()
    # a = -0.999
    # a_str = str(a)
    # a_1 = a_str[0:4]
    # print(a)
    # print(a_str)
    # print(a_1)


    pass

if __name__ == '__main__':
    main()