
from global_init import *
# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from meta_info import *
from pprint import pprint
mpl.use('TkAgg')

this_root='/Users/liyang/Projects_data/Hotdrought_Resilience/'
data_root = join(this_root,'data')
results_root = join(this_root,'results')
temp_root = join(this_root,'temp')

result_root_this_script = join(results_root, 'analysis_repeat_drought')

class Pick_drought_events_year:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_year', result_root_this_script, mode=2)

        pass

    def run(self):
        # 1 pick events in annual scale
        # self.tif_to_spatial_dict()
        # self.pick_year_events()
        # 2 pick repeated events
        # self.pick_repeated_annual_events()
        self.find_hot_years()

        pass

    def pick_year_events(self):
        # threshold = -1.5
        outdir = join(self.this_class_arr,'picked_events')
        T.mk_dir(outdir)
        # fdir = data_root + 'SPEI12/per_pix/'
        spi_12_dir = join(data_root,'SPI/per_pix/spi12')
        SPI_spatial_dict = T.load_npy_dir(spi_12_dir)
        events_dic = {}
        params_list = []
        spatial_dic = {}
        for pix in tqdm(SPI_spatial_dict):
            # spatial_dic[pix] = 1
            # continue
            vals = SPI_spatial_dict[pix]
            threshold = -2
            params = (vals,threshold)
            events_list = self.kernel_find_drought_period(params)
            if len(events_list) == 0:
                continue
            drought_year_list = []
            for drought_range in events_list:
                # drought_val = T.pick_vals_from_1darray(vals,drought_range)
                # plt.plot(drought_range,drought_val,lw=4)
                # plt.scatter(drought_range,drought_val,s=10,c='r',zorder=99)
                min_index = T.pick_min_indx_from_1darray(vals,drought_range)
                min_index_gs = min_index % 12 + 1
                drought_year = min_index // 12
                drought_year_list.append(drought_year)
            events_dic[pix] = drought_year_list
            spatial_dic[pix] = len(drought_year_list)
            # print(pix,drought_year_list)
            # plt.show()
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # plt.imshow(arr,cmap='jet',interpolation='nearest')
        # plt.colorbar()
        # plt.show()
        # T.save_npy(events_dic,outdir+'drought_year_events_2sd')
        outf = join(outdir,'drought_years')
        T.save_npy(events_dic,outf)
        pass

    def pick_repeated_annual_events(self):
        events_f = join(self.this_class_arr,'picked_events/drought_years.npy')
        outdir = join(self.this_class_arr,'pick_repeated_annual_events')
        outf = join(outdir,'repeated_events_annual.npy')
        T.mk_dir(outdir)
        events_dic = T.load_npy(events_f)
        spatial_dic = {}
        for pix in tqdm(events_dic):
            events = events_dic[pix]
            if len(events) <= 1:
                continue
            # spatial_dic[pix] = len(events)
            events = list(set(events))
            events.sort()
            repeated_events = self.kernel_pick_repeat_events(events,39)

            if len(repeated_events) == 0:
                continue
            # print(repeated_events)
            # spatial_dic[pix] = len(repeated_events)
            spatial_dic[pix] = repeated_events
        T.save_npy(spatial_dic,outf)
        # arr = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        # DIC_and_TIF().plot_back_ground_arr()
        # plt.imshow(arr)
        # plt.colorbar()
        # plt.show()

    def kernel_pick_repeat_events(self,events,year_num):
        # year_num = 16
        # events = [2,3,5,8,10]
        # events = [3,4,5,6]
        events_mark = []
        for i in range(year_num):
            if i in events:
                events_mark.append(1)
            else:
                events_mark.append(0)
        window = 4
        # print(events)
        events_list = []
        for i in range(len(events_mark)):
            select = events_mark[i:i+window]
            if select[0] == 0:
                continue
            if select[-1] == 0 and select.count(1) == 3:
                continue
            build_list = list(range(i,i+window))
            select_index = []
            for j in build_list:
                if j in events:
                    select_index.append(j)
            if not select_index[-1] - select_index[0] >= 2:
                continue
            # 前两年不能有干旱事件
            if select_index[0] - 1 in events:
                continue
            if select_index[0] - 2 in events:
                continue
            # 后两年不能有干旱事件
            if select_index[-1] + 1 in events:
                continue
            if select_index[-1] + 2 in events:
                continue
            if len(select_index) == 4:
                continue
            if select_index[0] - 2 < 0:
                continue
            if select_index[-1] + 2 >= year_num:
                continue
            events_list.append(select_index)
        # print(events_list)
        # exit()
        return events_list


    def kernel_find_drought_period(self, params):
        # 根据不同干旱程度查找干旱时期
        pdsi = params[0]
        threshold = params[1]
        drought_month = []
        for i, val in enumerate(pdsi):
            if val < threshold:# SPEI
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
            # print(new_i)
            # exit()
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
            if min_val < threshold:
                level = 4
            else:
                level = 0

            events_list.append(new_i)
        return events_list


    def tif_to_spatial_dict_cru_temp(self):
        spi_tif_dir = join(data_root, 'CRU_temp/tif')
        outdir = join(data_root, 'CRU_temp/per_pix')
        T.mk_dir(outdir, True)
        Pre_Process().data_transform(spi_tif_dir, outdir)

    def find_hot_years(self):
        cru_temp_per_pix_dir = join(data_root, 'CRU_temp/per_pix')
        outdir = join(self.this_class_arr,'hot_years')
        T.mk_dir(outdir,force=True)
        temp_spatial_dict = T.load_npy_dir(cru_temp_per_pix_dir)
        year_range = global_year_range_list
        year_range = np.array(year_range,dtype=int)

        hot_year_spatial_dict = {}

        for pix in tqdm(temp_spatial_dict):
            vals = temp_spatial_dict[pix]
            vals[vals<-999] = np.nan
            vals[vals>99999] = np.nan
            if T.is_all_nan(vals):
                continue
            vals_reshape = np.reshape(vals,(-1,12))
            vals_annual_mean = np.nanmean(vals_reshape,axis=1)
            vals_annual_mean_detrend = T.detrend_vals(vals_annual_mean)
            T_quantile = np.percentile(vals_annual_mean_detrend, 90)
            hot_index_True_False = vals_annual_mean_detrend > T_quantile
            hot_years = year_range[hot_index_True_False]
            hot_year_spatial_dict[pix] = hot_years
        outf = join(outdir,'hot_years.npy')
        T.save_npy(hot_year_spatial_dict,outf)

    def tif_to_spatial_dict(self):
        spi_tif_dir = join(data_root,'SPI/tif/spi12')
        outdir = join(data_root,'SPI/per_pix/spi12')
        T.mk_dir(outdir,True)
        Pre_Process().data_transform(spi_tif_dir,outdir)

class Dataframe:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Dataframe', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, f'Dataframe/dataframe.df')
        pass

    def run(self):
        # self.gen_events_df()
        df = self.__load_df()


        pass


    def gen_events_df(self):
        if isfile(self.dff):
            print('file exsited')
            pause()

            print('file exsited')
            pause()

            print('file exsited')
            pause()
        start_year = global_start_year
        outdir = join(self.this_class_arr,'Dataframe')
        T.mkdir(outdir)
        events_f = join(Pick_drought_events_year().this_class_arr,'pick_repeated_annual_events/repeated_events_annual.npy')
        events_dic = T.load_npy(events_f)
        hot_year_f = join(Pick_drought_events_year().this_class_arr,'hot_years/hot_years.npy')
        hot_year_dic = T.load_npy(hot_year_f)
        pix_list = []
        event_list = []
        first_event_list = []
        second_event_list = []
        post_list = []
        spell_list = []
        spell_len_list = []
        repeat_mode_list = []
        is_hot_drought_list = []
        for pix in tqdm(events_dic,desc='add_events_to_df'):
            events = events_dic[pix]
            hot_years = hot_year_dic[pix]
            # print('hot_years',hot_years)
            # exit()
            if len(events) == 0:
                continue
            for event in events:
                pix_list.append(pix)
                event_list.append(np.array(event)+start_year)
                if len(event) == 2:
                    first_event = tuple([event[0]])
                    second_event = tuple([event[1]])
                    spell = tuple(range(first_event[0]+1,second_event[0]))
                    if event[1] - event[0] == 2:
                        repeat_mode = 'DSD'
                    elif event[1] - event[0] == 3:
                        repeat_mode = 'DSSD'
                    else:
                        raise ValueError
                elif len(event) == 3:
                    e1 = event[0]
                    e2 = event[1]
                    e3 = event[2]
                    if e2 - e1 == 1:
                        first_event = (e1,e2)
                        second_event = tuple([e3])
                        repeat_mode = 'DDSD'
                    else:
                        first_event = tuple([e1])
                        second_event = (e2, e3)
                        repeat_mode = 'DSDD'
                    spell = tuple(range(first_event[-1] + 1, second_event[0]))
                else:
                    raise UserWarning

                hot = False
                # print('first_event',np.array(first_event)+start_year)
                for year in (np.array(first_event)+start_year):
                    if year in hot_years:
                        hot = True
                        break
                if hot == True:
                    is_hot_drought_list.append('Hot-drought')
                else:
                    is_hot_drought_list.append('Normal-drought')

                first_event_list.append(np.array(first_event)+start_year)
                second_event_list.append(np.array(second_event) + start_year)
                spell_list.append(np.array(spell)+start_year)
                spell_len_list.append(len(spell))
                repeat_mode_list.append(repeat_mode)
        df = pd.DataFrame()
        df['pix'] = pix_list
        df['drought_year'] = event_list
        df['first_drought'] = first_event_list
        df['first_drought_type'] = is_hot_drought_list
        df['spell'] = spell_list
        df['spell_length'] = spell_len_list
        df['second_drought'] = second_event_list
        df['repeat_mode'] = repeat_mode_list
        T.save_df(df,self.dff)
        T.df_to_excel(df,self.dff)



    def __load_df(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        print('len(df):',len(df))
        return df,dff

def main():
    # Pick_drought_events_year().run()
    Dataframe().run()
    pass

if __name__ == '__main__':
    main()
