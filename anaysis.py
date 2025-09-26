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

def main():

    Pick_drought_events().run()

if __name__ == '__main__':
    main()
    pass