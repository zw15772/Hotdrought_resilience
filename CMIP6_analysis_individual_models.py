# coding=utf-8

from meta_info import *
result_root_this_script = join(results_root, 'CMIP6_analysis_individual_models')
D_CMIP = DIC_and_TIF(pixelsize=1)

class Pick_drought_events_SM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_SM', result_root_this_script, mode=2)
        # self.threshold = -2

    def run(self):
        self.pick_normal_drought_events()
        # self.add_hot_normal_drought()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        self.datadir = join(data_root, 'CMIP6')
        experiment_list = ['ssp245', 'ssp585']

        product_list = ['mrsos']
        for product in product_list:
            fdir = join(self.datadir, product, 'per_pix_anomaly')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                if not isdir(fdir_i):
                    continue
                outdir_i = join(outdir, experiment)
                T.mk_dir(outdir_i, force=True)
                model_list = T.listdir(fdir_i)
                for model in model_list:
                    fpath = join(fdir_i, model,'anomaly.npy')
                    date_f = join(fdir_i, model,'date_range.npy')

                    spatial_dict = T.load_npy(fpath)
                    date = np.load(date_f,allow_pickle=True)
                    date_start = date[0]
                    start_year = date_start.year
                    threshold_upper = -1.5
                    threshold_bottom = -np.inf

                    pix_list = []
                    drought_range_list = []
                    drougth_timing_list = []
                    drought_year_list = []
                    intensity_list = []
                    severity_list = []
                    severity_mean_list = []

                    for pix in tqdm(spatial_dict,desc=f'{experiment}-{model}'):
                        r,c = pix
                        # if r > 600: # Antarctica
                        #     continue
                        vals = spatial_dict[pix]
                        vals = np.array(vals)
                        std = np.nanstd(vals)
                        if std == 0 or np.isnan(std):
                            continue
                        drought_events_list, drought_timing_list = self.kernel_find_drought_period(vals,threshold_upper,threshold_bottom)
                        for i in range(len(drought_events_list)):
                            event = drought_events_list[i]
                            s,e = event
                            mid = int((s+e)/2)
                            drought_year = int(mid/12) + start_year
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

                    outf = join(outdir_i, f'{threshold_upper}_{model}.df')
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

    def run(self):
        # self.pick_normal_drought_events()
        # self.merge_df()
        self.add_hot_normal_drought()
        # self.add_hot_normal_drought_annual_T()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        self.datadir = join(data_root, 'CMIP6')
        experiment_list = ['ssp245', 'ssp585']
        product = 'mrsos'
        fdir = join(self.datadir, product, 'per_pix_anomaly')
        params_list = []
        for experiment in experiment_list:
            fdir_i = join(fdir, experiment)
            outdir_i = join(outdir, experiment)
            T.mk_dir(outdir_i, force=True)
            threshold_list = np.arange(-0.5,-3.1,-0.1)
            threshold_list = [round(i,1) for i in threshold_list]
            model_list = T.listdir(fdir_i)
            for model in model_list:
                params = (fdir_i,model,threshold_list,outdir_i)
                params_list.append(params)
                # self.kernel_pick_normal_drought_events(params)
        MULTIPROCESS(self.kernel_pick_normal_drought_events,params_list).run(process=7)


    def kernel_pick_normal_drought_events(self,params):
        fdir_i,model,threshold_list,outdir_i = params
        fpath = join(fdir_i, model, 'anomaly.npy')
        date_f = join(fdir_i, model, 'date_range.npy')
        date_list = np.load(date_f, allow_pickle=True)
        data_dict = T.load_npy(fpath)
        outdir_model = join(outdir_i, model)
        T.mk_dir(outdir_model, force=True)

        for threshold_i in range(len(threshold_list)):
            threshold_upper = threshold_list[threshold_i]
            if threshold_i == len(threshold_list) - 1:
                break
            threshold_bottom = threshold_list[threshold_i + 1]

            pix_list = []
            drought_start_list = []
            drought_end_list = []
            drougth_timing_list = []
            drought_year_list = []
            intensity_list = []
            severity_list = []
            severity_mean_list = []

            for pix in data_dict:
                r, c = pix
                if r > 600:  # Antarctica
                    continue
                vals = data_dict[pix]
                vals = np.array(vals)
                std = np.nanstd(vals)
                if std == 0 or np.isnan(std):
                    continue
                drought_events_list, drought_timing_list = self.kernel_find_drought_period(vals, threshold_upper,
                                                                                           threshold_bottom)
                if len(drought_events_list) == 0:
                    continue
                # print(drought_events_list)
                for i in range(len(drought_events_list)):
                    event = drought_events_list[i]
                    s, e = event
                    mid = int((s + e) / 2)
                    # drought_year = int(mid/12) + 2020
                    start_date = date_list[s]
                    end_date = date_list[e]
                    # print(drought_date_range)
                    # exit()
                    drought_date = date_list[mid]
                    drought_year = drought_date.year
                    timing = drought_timing_list[i]
                    duration = e - s
                    intensity = np.nanmin(vals[s:e])
                    severity = np.nansum(vals[s:e])
                    severity_mean = np.nanmean(vals[s:e])

                    pix_list.append(pix)
                    drought_start_list.append(start_date)
                    drought_end_list.append(end_date)
                    drougth_timing_list.append(timing)
                    drought_year_list.append(drought_year)
                    intensity_list.append(intensity)
                    severity_list.append(severity)
                    severity_mean_list.append(severity_mean)

            df = pd.DataFrame()
            df['pix'] = pix_list
            # df['drought_range'] = drought_range_list
            df['drought_start'] = drought_start_list
            df['drought_end'] = drought_end_list
            df['drought_year'] = drought_year_list
            df['drought_month'] = drougth_timing_list
            df['threshold'] = threshold_upper
            df['intensity'] = intensity_list
            df['severity'] = severity_list
            df['severity_mean'] = severity_mean_list

            outf = join(outdir_model, f'{threshold_upper}.df')
            T.save_df(df, outf)
            T.df_to_excel(df, outf)

        pass


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
        experiment_list = ['ssp245', 'ssp585']
        params_list = []
        for experiment in experiment_list:
            fdir = join(self.this_class_arr, 'dataframe_merge',experiment)
            fdir_T = join(data_root,'CMIP6/tas/per_pix_anomaly_2020_2040',experiment)
            # fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_moving_window_baseline',experiment)
            for model in T.listdir(fdir):
                params = (fdir,fdir_T,model)
                params_list.append(params)
                # self.kernel_add_hot_normal_drought(params)
        MULTIPROCESS(self.kernel_add_hot_normal_drought,params_list).run(process=7)

    def kernel_add_hot_normal_drought(self,params):
        fdir,fdir_T,model = params
        # print(params)
        model_dir = join(fdir, model)
        T_model_dir = join(fdir_T, model)
        if not isdir(T_model_dir):
            return
        Temperature_model_f = join(T_model_dir, 'anomaly.npy')
        Temperature_model_date = join(T_model_dir, 'date_range.npy')
        Temperature_anomaly_data_dict = T.load_npy(Temperature_model_f)
        Temperature_anomaly_date = np.load(Temperature_model_date, allow_pickle=True)
        Temperature_anomaly_date = list(Temperature_anomaly_date)
        for f in T.listdir(model_dir):
            if not f.endswith('.df'):
                continue
            fpath = join(model_dir, f)
            df = T.load_df(fpath)
            T_level_list = []
            T_max_list = []
            for i, row in tqdm(df.iterrows(),total=len(df),desc='\t'.join(params)):
                pix = row['pix']
                drought_start = row['drought_start']
                drought_end = row['drought_end']
                try:
                    start_date_index = Temperature_anomaly_date.index(drought_start)
                    end_date_index = Temperature_anomaly_date.index(drought_end)

                    temperature_anomaly_in_drought = Temperature_anomaly_data_dict[pix][start_date_index:end_date_index + 1]
                    # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                    max_temperature = np.nanmax(temperature_anomaly_in_drought)
                    T_level = int(max_temperature * 10) / 10.
                    T_max_list.append(max_temperature)
                    T_level_list.append(T_level)
                except:
                    T_level_list.append(np.nan)
                    T_max_list.append(np.nan)
            df['T_level'] = T_level_list
            df['T_max'] = T_max_list
            outf = join(model_dir, f)
            print(outf)
            T.save_df(df, outf)
            T.df_to_excel(df, outf)


    def add_hot_normal_drought_annual_T(self):
        experiment_list = ['ssp245', 'ssp585']
        for experiment in experiment_list:
            fdir = join(self.this_class_arr, 'picked_events',experiment)
            fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_CMIP_historical_baseline',experiment)
            # fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_moving_window_baseline',experiment)
            Temperature_anomaly_data_dict = T.load_npy_dir(fdir_T)
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
                    annual_temperature = Temperature_anomaly_data_dict[pix]
                    year_index = int(e/12)
                    # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                    max_temperature = annual_temperature[year_index]
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

    def merge_df(self):
        experiment_list = ['ssp245', 'ssp585']
        for exp in experiment_list:
            fdir = join(self.this_class_arr, 'picked_events',exp)
            outdir = join(self.this_class_arr,'dataframe_merge',exp)
            T.mk_dir(outdir,force=True)

            for model in T.listdir(fdir):
                fdir_model = join(fdir,model)
                outdir_model = join(outdir,model)
                T.mk_dir(outdir_model,force=True)
                df_list = []
                for f in T.listdir(fdir_model):
                    if not f.endswith('.df'):
                        continue
                    print(f)
                    fpath = join(fdir_model,f)
                    df = T.load_df(fpath)
                    df_list.append(df)
                df_concat = pd.concat(df_list)
                df_concat = df_concat.sort_values(by=['pix','threshold'])
                outf = join(outdir_model,'dataframe_merge.df')
                T.save_df(df_concat,outf)
                T.df_to_excel(df_concat,outf,random=True)
                pass

class Pick_drought_events_SM_multi_thresholds_ensemble_annual:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_SM_multi_thresholds_ensemble_annual', result_root_this_script, mode=2)
        # self.threshold = -2

    def run(self):
        # self.pick_normal_drought_events()
        # self.add_hot_normal_drought()
        self.add_hot_normal_drought_annual_T()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        self.datadir = join(data_root, 'CMIP6')
        experiment_list = ['ssp245', 'ssp585']
        product = 'mrsos'
        # fdir = join(self.datadir, product, 'per_pix_ensemble_std_anomaly_based_2020_2060')
        fdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_detrend')
        # fdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_moving_window_baseline')
        # fdir = join(self.datadir, product, 'per_pix_ensemble_std_anomaly_based_2020_2060')
        for experiment in experiment_list:
            fdir_i = join(fdir, experiment)
            outdir_i = join(outdir, experiment)
            T.mk_dir(outdir_i, force=True)
            threshold_list = np.arange(-0.5,-3.1,-0.1)
            threshold_list = [round(i,1) for i in threshold_list]
            # threshold = self.threshold
            # data_dict,_ = Load_Data().ERA_SM_anomaly_detrend()
            data_dict = T.load_npy_dir(fdir_i)

            for threshold_i in range(len(threshold_list)):
                threshold_upper = threshold_list[threshold_i]
                if threshold_i == len(threshold_list) - 1:
                    break
                threshold_bottom = threshold_list[threshold_i+1]

                pix_list = []
                drought_range_list = []
                drougth_timing_list = []
                drought_year_list = []

                all_vals = []

                for pix in tqdm(data_dict,desc=f'pick_{threshold_upper}'):
                    r,c = pix
                    if r > 600: # Antarctica
                        continue
                    vals = data_dict[pix]
                    vals = np.array(vals)
                    std = np.nanstd(vals)
                    if std == 0 or np.isnan(std):
                        continue
                    gs = global_get_gs_CMIP(pix)
                    vals_gs = T.monthly_vals_to_annual_val(vals,gs)
                    drought_index_list = self.kernel_find_drought_period_annual(vals_gs,threshold_upper,threshold_bottom)
                    for i in range(len(drought_index_list)):
                        ind = drought_index_list[i]
                        drought_year = ind + 2020
                        pix_list.append(pix)
                        drought_year_list.append(drought_year)
                df = pd.DataFrame()
                df['pix'] = pix_list
                df['drought_year'] = drought_year_list
                df['threshold'] = threshold_upper

                outf = join(outdir_i, f'{threshold_upper}.df')
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

    def kernel_find_drought_period_annual(self,vals,threshold_upper,threshold_bottom,threshold_start=-.5):
        vals = np.array(vals)
        drought_year_list = []

        for i in range(len(vals)):
            val = vals[i]
            if threshold_bottom < val < threshold_upper:
                drought_year_list.append(i)
        return drought_year_list


    def add_hot_normal_drought(self):
        experiment_list = ['ssp245', 'ssp585']
        for experiment in experiment_list:
            fdir = join(self.this_class_arr, 'picked_events',experiment)
            fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_std_anomaly_based_2020_2060',experiment)
            # fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_moving_window_baseline',experiment)
            Temperature_anomaly_data_dict = T.load_npy_dir(fdir_T)
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

    def add_hot_normal_drought_annual_T(self):
        experiment_list = ['ssp245', 'ssp585']
        for experiment in experiment_list:
            fdir = join(self.this_class_arr, 'picked_events',experiment)
            fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_CMIP_historical_baseline',experiment)
            # fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_moving_window_baseline',experiment)
            Temperature_anomaly_data_dict = T.load_npy_dir(fdir_T)
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
                    drought_year = row['drought_year']
                    if not pix in Temperature_anomaly_data_dict:
                        T_max_list.append(np.nan)
                        T_level_list.append(np.nan)
                        continue
                    annual_temperature = Temperature_anomaly_data_dict[pix]
                    year_index = drought_year - 2020
                    # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                    max_temperature = annual_temperature[year_index]
                    T_level = int(max_temperature*10) / 10.
                    T_max_list.append(max_temperature)
                    T_level_list.append(T_level)
                df['T_level'] = T_level_list
                df['T_max'] = T_max_list
                df = df.dropna()
                T.print_head_n(df)
                # exit()
                outf = join(fdir,f)
                T.save_df(df,outf)
                T.df_to_excel(df,outf)

    def gen_dataframe(self):
        experiment_list = ['ssp245', 'ssp585']
        for experiment in experiment_list:
            fdir = join(self.this_class_arr,'picked_events',experiment)
            outdir = join(self.this_class_arr,'dataframe',experiment)
            T.mk_dir(outdir,force=True)
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


class Dataframe_SM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir(f'Dataframe_SM_multi_threshold_annual', result_root_this_script, mode=2)
            # T.mk_class_dir(f'Dataframe_SM_multi_threshold', result_root_this_script, mode=2)
            # T.mk_class_dir('Dataframe_SM_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Dataframe_SM_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Dataframe_SM', result_root_this_script, mode=2)
        self.df_dir = join(self.this_class_arr, 'dataframe')

        pass

    def run(self):
        # self.copy_df()
        params_list = []
        for exp in ['ssp245', 'ssp585']:
            for model in T.listdir(join(self.df_dir,exp)):
                params = (exp,model)
                params_list.append(params)
                # self.kernel_run(params)
        MULTIPROCESS(self.kernel_run,params_list).run(process=7)
        pass


    def kernel_run(self,params):
        exp,model = params
        dff = join(self.df_dir, exp, model, 'dataframe_merge.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        df = Dataframe_func(df).df
        df = df.dropna()

        T.save_df(df, dff)
        T.df_to_excel(df, dff)
        pass

    def copy_df(self):
        experiment_list = ['ssp245', 'ssp585']
        for exp in experiment_list:
            fdir = join(Pick_drought_events_SM_multi_thresholds().this_class_arr,'dataframe_merge',exp)
            outdir = join(self.this_class_arr,'dataframe',exp)
            T.mk_dir(outdir,force=True)
            for model in T.listdir(fdir):
                print(exp,model)
                fdir_model = join(fdir,model)
                outdir_model = join(outdir,model)
                T.mk_dir(outdir_model,force=True)
                for f in T.listdir(fdir_model):
                    fpath = join(fdir_model,f)
                    outpath = join(outdir_model,f)
                    if isfile(outpath):
                        print('Warning: this function will overwrite the dataframe')
                        print('Warning: this function will overwrite the dataframe')
                        print('Warning: this function will overwrite the dataframe')
                        pause()
                        pause()
                    shutil.copy(fpath,outpath)
        pass

    def add_variables_during_droughts(self,df,fpath,var_name):
        pix_list = T.get_df_unique_val_list(df, 'pix')
        data_dict = T.load_npy(fpath)

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


    def add_BNPP(self,df):
        fpath = join(data_root,'BNPP/tif_025/BNPP_0-200cm.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'BNPP')
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
        outdir = join(self.this_class_arr,'dataframe_merge',self.scenario)
        T.mk_dir(outdir,force=True)
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.df'):
                continue
            print(f)
            fpath = join(fdir,f)
            df = T.load_df(fpath)
            df_list.append(df)
        df_concat = pd.concat(df_list)
        df_concat = df_concat.sort_values(by=['pix','threshold'])
        outf = join(outdir,'dataframe_merge.df')
        T.save_df(df_concat,outf)
        T.df_to_excel(df_concat,outf,random=True)
        pass


class Dataframe_func:

    def __init__(self,df,is_clean_df=True):
        print('add lon lat')
        df = self.add_lon_lat(df)

        # print('add NDVI mask')
        # df = self.add_NDVI_mask(df)
        # if is_clean_df == True:
        #     df = self.clean_df(df)
        # print('add GLC2000')
        # df = self.add_GLC_landcover_data_to_df(df)
        # print('add Aridity Index')
        df = self.add_AI_to_df(df)
        df = self.add_MAT_MAP(df)

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
        val_dic = D_CMIP.spatial_arr_to_dic(array)
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
        f = join(data_root, 'Aridity_index/Aridity_index_1deg.tif')
        spatial_dict = D_CMIP.spatial_tif_to_dic(f)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'aridity_index')
        return df

    def add_lon_lat(self,df):
        df = T.add_lon_lat_to_df(df, D_CMIP)
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

    def add_MAT_MAP(self,df):
        MAP_f = join(data_root,'TerraClimate/ppt/MAP/MAP_1_deg.tif')
        MAT_f =join(data_root,'ERA5/Tair/MAT/MAT_C_1_deg.tif')
        MAP_dict = D.spatial_tif_to_dic(MAP_f)
        MAT_dict = D.spatial_tif_to_dic(MAT_f)
        df = T.add_spatial_dic_to_df(df,MAP_dict,'MAP')
        df = T.add_spatial_dic_to_df(df,MAT_dict,'MAT')
        return df


class CMIP6_drought_events_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('CMIP6_drought_events_analysis', result_root_this_script, mode=2)
        self.experiment_list = ['ssp245', 'ssp585']
        pass

    def run(self):
        # self.copy_df()
        # self.add_VPD()
        # self.add_drought_type()
        # self.drought_frequency_ts()
        # self.tif_drought_frequency_spatial()
        # self.tif_drought_frequency_spatial_ensemble()
        # self.tif_drought_frequency_spatial_all_drt()
        # self.plot_drought_frequency_spatial()
        # self.plot_drought_frequency_spatial_ensemble()
        # self.plot_variables_during_drought_ts()
        # self.tif_variables_spatial_trend_during_drought()
        # self.ensemble_tif_variables_spatial_trend_during_drought()
        self.plot_ensemble_tif_variables_spatial_trend_during_drought()
        pass

    def copy_df(self):
        fdir = join(Dataframe_SM().this_class_arr,'dataframe')
        # T.open_path_and_file(fdir)
        # exit()
        outdir = join(self.this_class_arr,'dataframe')
        for exp in self.experiment_list:
            fdir_i = join(fdir,exp)
            # print(fdir_i)
            # exit()
            outdir_i = join(outdir,exp)
            T.mk_dir(outdir_i,force=True)
            for model in T.listdir(fdir_i):
                fdir_model = join(fdir_i,model)
                outdir_model = join(outdir_i,model)
                T.mk_dir(outdir_model,force=True)
                for f in T.listdir(fdir_model):
                    fpath = join(fdir_model,f)
                    outpath = join(outdir_model,f)
                    shutil.copy(fpath,outpath)

    def add_drought_type(self):
        experiment_list = ['ssp245', 'ssp585']
        for exp in experiment_list:
            fdir = join(self.this_class_arr,'dataframe',exp)
            for model in T.listdir(fdir):
                dff = join(fdir,model,'dataframe_merge.df')
                df = T.load_df(dff)
                column_list = df.columns.tolist()
                if not 'T_max' in column_list:
                    os.remove(dff)
                    continue
                drt_list = []
                for i,row in tqdm(df.iterrows(),total=len(df),desc=f'{exp}_{model}'):
                    T_max = row['T_max']
                    # if T_max > 0:
                    if T_max > 2:
                        drought_type = 'hot-drought'
                    else:
                        drought_type = 'normal-drought'
                    drt_list.append(drought_type)
                df['drought_type'] = drt_list
                T.save_df(df,dff)
                T.df_to_excel(df,dff)

    def add_VPD(self):
        experiment_list = ['ssp245', 'ssp585']
        params_list = []
        for experiment in experiment_list:
            fdir = join(self.this_class_arr, 'dataframe',experiment)
            fdir_T = join(data_root,'CMIP6/VPD/per_pix_anomaly_2020_2040',experiment)
            # fdir_T = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly_moving_window_baseline',experiment)
            for model in T.listdir(fdir):
                params = (fdir,fdir_T,model)
                params_list.append(params)
                # self.kernel_add_hot_normal_drought(params)
        MULTIPROCESS(self.kernel_add_VPD,params_list).run(process=7)

    def kernel_add_VPD(self,params):
        fdir,fdir_T,model = params
        # print(params)
        model_dir = join(fdir, model)
        T_model_dir = join(fdir_T, model)
        if not isdir(T_model_dir):
            return
        Temperature_model_f = join(T_model_dir, 'anomaly.npy')
        Temperature_model_date = join(T_model_dir, 'date_range.npy')
        Temperature_anomaly_data_dict = T.load_npy(Temperature_model_f)
        Temperature_anomaly_date = np.load(Temperature_model_date, allow_pickle=True)
        Temperature_anomaly_date = list(Temperature_anomaly_date)
        for f in T.listdir(model_dir):
            if not f.endswith('.df'):
                continue
            fpath = join(model_dir, f)
            df = T.load_df(fpath)
            T_max_list = []
            for i, row in tqdm(df.iterrows(),total=len(df),desc='\t'.join(params)):
                pix = row['pix']
                drought_start = row['drought_start']
                drought_end = row['drought_end']
                try:
                    start_date_index = Temperature_anomaly_date.index(drought_start)
                    end_date_index = Temperature_anomaly_date.index(drought_end)

                    temperature_anomaly_in_drought = Temperature_anomaly_data_dict[pix][start_date_index:end_date_index + 1]
                    # temperature_anomaly = Temperature_anomaly_data_dict[pix]
                    max_temperature = np.nanmean(temperature_anomaly_in_drought)
                    T_max_list.append(max_temperature)
                except:
                    T_max_list.append(np.nan)
            df['VPD_anomaly'] = T_max_list
            outf = join(model_dir, f)
            print(outf)
            T.save_df(df, outf)
            T.df_to_excel(df, outf)

    def drought_frequency_ts(self):
        fdir = join(self.this_class_arr,'dataframe')
        outdir = join(self.this_class_png,'drought_frequency_ts')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        year_list = range(2020,2101)
        for exp in self.experiment_list:
            exp_dir = join(fdir,exp)
            result_dict = {}
            flag = 0
            for model in T.listdir(exp_dir):
                model_dir = join(exp_dir,model)
                for f in T.listdir(model_dir):
                    if not f.endswith('.df'):
                        continue
                    fpath = join(model_dir,f)
                    df = T.load_df(fpath)
                    df = df[df['threshold']<-1.]
                    all_pix_list = T.get_df_unique_val_list(df,'pix')
                    all_pix_len = len(all_pix_list)
                    for drt in drought_type_list:
                        df_drt = df[df['drought_type']==drt]
                        df_year_dict = T.df_groupby(df_drt,'drought_year')
                        x = []
                        y = []
                        for year in year_list:
                            if not year in df_year_dict:
                                x.append(year)
                                y.append(np.nan)
                                continue
                            df_year = df_year_dict[year]
                            spatial_dict = {}
                            pix_list = T.get_df_unique_val_list(df_year,'pix')
                            # len_pix = len(pix_list)
                            len_pix = len(df_year)
                            ratio = len_pix / all_pix_len * 100
                            x.append(year)
                            y.append(ratio)
                        # y = SMOOTH().smooth_convolve(y,window_len=5)
                        flag += 1
                        result_dict[flag] = {
                            'x':x,
                            'y':y,
                            'drought_type':drt,
                            'model':model,
                        }
            df_result = T.dic_to_df(result_dict)
            T.print_head_n(df_result)
            plt.figure(figsize=(10,5))
            for drt in drought_type_list:
                df_drt = df_result[df_result['drought_type']==drt]
                year_list = df_result['x'][0]
                y_list = df_drt['y'].tolist()
                y_list = np.array(y_list)
                y_mean = np.nanmean(y_list,axis=0)
                y_std = np.nanstd(y_list,axis=0)
                plt.plot(year_list,y_mean,label='mean')
                plt.fill_between(year_list,y_mean-y_std,y_mean+y_std,alpha=0.2)
            plt.title(exp)
            outf = join(outdir,f'{exp}.pdf')
            plt.legend()
            plt.savefig(outf)
            plt.close()

        T.open_path_and_file(outdir)

    def tif_drought_frequency_spatial(self):
        fdir = join(self.this_class_arr, 'dataframe')
        outdir = join(self.this_class_tif,'drought_frequency')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        for exp in self.experiment_list:
            exp_dir = join(fdir, exp)
            exp_outdir = join(outdir,exp)
            result_dict = {}
            flag = 0
            for model in T.listdir(exp_dir):
                model_dir = join(exp_dir, model)
                model_outdir = join(exp_outdir, model)
                T.mk_dir(model_outdir,force=True)
                for f in T.listdir(model_dir):
                    if not f.endswith('.df'):
                        continue
                    fpath = join(model_dir, f)
                    df = T.load_df(fpath)
                    # T.print_head_n(df)
                    df = df[df['threshold']<-1.5]
                    for drt in drought_type_list:
                        df_drt = df[df['drought_type'] == drt]
                        pix_list = T.get_df_unique_val_list(df_drt, 'pix')
                        df_groupby_pix = T.df_groupby(df_drt, 'pix')
                        spatial_dict = {}
                        for pix in pix_list:
                            df_pix = df_groupby_pix[pix]
                            len_pix = len(df_pix)
                            spatial_dict[pix] = len_pix
                        outf = join(model_outdir,f'{drt}.tif')
                        print(outf)
                        D_CMIP.pix_dic_to_tif(spatial_dict,outf)
                        # exit()

    def tif_drought_frequency_spatial_ensemble(self):
        fdir = join(self.this_class_tif, 'drought_frequency')
        outdir = join(self.this_class_tif,'drought_frequency_ensemble')
        T.mk_dir(outdir)
        for exp in self.experiment_list:
            exp_dir = join(fdir, exp)
            exp_outdir = join(outdir,exp)
            T.mk_dir(exp_outdir,force=True)
            for drt in global_drought_type_list:
                tif_path_list = []
                for model in T.listdir(exp_dir):
                    tif_path = join(exp_dir,model,f'{drt}.tif')
                    tif_path_list.append(tif_path)
                outf = join(exp_outdir,f'{drt}.tif')
                print(outf)
                Pre_Process().compose_tif_list(tif_path_list,outf)
        pass


    def plot_drought_frequency_spatial(self):
        fdir = join(self.this_class_tif,'drought_frequency')
        outdir = join(self.this_class_png,'drought_frequency')
        T.mk_dir(outdir)
        color_list = [
                         '#844000',
                         '#fc9831',
                         '#fffbd4',
                         '#ffffff',
                         # '#86b9d2',
                         # '#064c6c',
                     ]
        cmap = Tools().cmap_blend(color_list[::-1])
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            plt.figure(figsize=(10,5))
            Plot().plot_Robinson(fpath,res=100000,cmap=cmap,vmin=0,vmax=5)
            plt.title(f)
            outf = join(outdir,f'{f}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)
            # exit()

    def plot_drought_frequency_spatial_ensemble(self):
        fdir = join(self.this_class_tif,'drought_frequency_ensemble')
        outdir = join(self.this_class_png,'drought_frequency_ensemble')
        T.mk_dir(outdir)
        color_list = [
                         '#844000',
                         '#fc9831',
                         '#fffbd4',
                         '#ffffff',
                         # '#86b9d2',
                         # '#064c6c',
                     ]
        cmap = Tools().cmap_blend(color_list[::-1])
        for exp in T.listdir(fdir):
            print(exp)
            exp_dir = join(fdir,exp)
            fpath = join(exp_dir,'hot-drought.tif')
            plt.figure(figsize=(10,5))
            Plot().plot_Robinson(fpath,res=100000,cmap=cmap,vmin=0,vmax=20)
            plt.title(exp)
            # plt.show()
            outf = join(outdir,f'{exp}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def tif_drought_frequency_spatial_all_drt(self):
        fdir = join(self.this_class_arr, 'dataframe')
        outdir = join(self.this_class_tif,'drought_frequency')
        T.mk_dir(outdir)
        for exp in self.experiment_list:
            fdir_i = join(fdir, exp)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                # df = df[df['threshold']<-2]
                pix_list = T.get_df_unique_val_list(df, 'pix')
                df_groupby_pix = T.df_groupby(df, 'pix')
                spatial_dict = {}
                for pix in pix_list:
                    df_pix = df_groupby_pix[pix]
                    len_pix = len(df_pix)
                    spatial_dict[pix] = len_pix
                outf = join(outdir,f'all_drought_type_{exp}.tif')
                print(outf)
                D_CMIP.pix_dic_to_tif(spatial_dict,outf)



    def plot_variables_during_drought_ts(self):
        # col = 'T_max'
        col = 'intensity'
        fdir = join(self.this_class_arr, 'dataframe')
        outdir = join(self.this_class_png, 'variables_during_drought_ts')
        T.mk_dir(outdir)
        drought_type_list = ['hot-drought']
        # drought_type_list = ['normal-drought']
        year_list = range(2020, 2101)
        plt.figure(figsize=(10, 5))
        for exp in self.experiment_list:
            exp_dir = join(fdir, exp)
            result_dict = {}
            flag = 0
            for model in T.listdir(exp_dir):
                model_dir = join(exp_dir, model)
                for f in T.listdir(model_dir):
                    if not f.endswith('.df'):
                        continue
                    fpath = join(model_dir, f)
                    df = T.load_df(fpath)
                    print(df.columns.tolist())
                    # vals = df[col].tolist()
                    # plt.hist(vals,bins=100)
                    # plt.show()
                    # exit()
                    # df = df[df['threshold'] < -1.]
                    all_pix_list = T.get_df_unique_val_list(df, 'pix')
                    all_pix_len = len(all_pix_list)
                    for drt in drought_type_list:
                        df_drt = df[df['drought_type'] == drt]
                        # df_drt = df
                        df_year_dict = T.df_groupby(df_drt, 'drought_year')
                        x = []
                        y = []
                        for year in year_list:
                            if not year in df_year_dict:
                                x.append(year)
                                y.append(np.nan)
                                continue
                            df_year = df_year_dict[year]
                            # T.print_head_n(df_year)
                            # exit()
                            spatial_dict = {}
                            T_vals = df_year[col].tolist()
                            T_mean = np.nanmean(T_vals)
                            x.append(year)
                            y.append(T_mean)
                        # y = SMOOTH().smooth_convolve(y,window_len=5)
                        flag += 1
                        result_dict[flag] = {
                            'x': x,
                            'y': y,
                            'drought_type': drt,
                            'model': model,
                        }
            df_result = T.dic_to_df(result_dict)
            T.print_head_n(df_result)
            for drt in drought_type_list:
                df_drt = df_result[df_result['drought_type'] == drt]
                year_list = df_result['x'][0]
                y_list = df_drt['y'].tolist()
                y_list = np.array(y_list)
                y_mean = np.nanmean(y_list, axis=0)
                y_std = np.nanstd(y_list, axis=0)
                plt.plot(year_list, y_mean, label=exp)
                plt.fill_between(year_list, y_mean - y_std, y_mean + y_std, alpha=0.2)
        plt.legend()
        outf = join(outdir, f'{col}.pdf')
        plt.legend()
        plt.savefig(outf)
        plt.close()

        T.open_path_and_file(outdir)


    def tif_variables_spatial_trend_during_drought(self):
        fdir = join(self.this_class_arr, 'dataframe')
        outdir = join(self.this_class_tif, 'variables_spatial_trend_during_drought')
        # col = 'T_max'
        col = 'intensity'
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        for exp in self.experiment_list:
            exp_dir = join(fdir,exp)
            exp_outdir = join(outdir,col, exp)
            result_dict = {}
            flag = 0
            for model in T.listdir(exp_dir):
                model_dir = join(exp_dir, model)
                model_outdir = join(exp_outdir, model)
                T.mk_dir(model_outdir, force=True)
                for f in T.listdir(model_dir):
                    if not f.endswith('.df'):
                        continue
                    fpath = join(model_dir, f)
                    df = T.load_df(fpath)
                    # T.print_head_n(df)
                    # df = df[df['threshold'] < -1.5]
                    for drt in drought_type_list:
                        df_drt = df[df['drought_type'] == drt]
                        # T.print_head_n(df_drt)
                        # exit()
                        pix_list = T.get_df_unique_val_list(df_drt, 'pix')
                        df_groupby_pix = T.df_groupby(df_drt, 'pix')
                        spatial_dict = {}
                        for pix in pix_list:
                            df_pix = df_groupby_pix[pix]
                            df_pix_sort = df_pix.sort_values('drought_year')
                            drought_year = df_pix_sort['drought_year'].to_list()
                            vals = df_pix_sort[col].to_list()
                            try:
                                a,b,r,p = T.nan_line_fit(drought_year,vals)
                                spatial_dict[pix] = a
                            except:
                                continue
                        outf = join(model_outdir, f'{drt}.tif')
                        print(outf)
                        D_CMIP.pix_dic_to_tif(spatial_dict, outf)

        pass


    def ensemble_tif_variables_spatial_trend_during_drought(self):
        fdir = join(self.this_class_tif, 'variables_spatial_trend_during_drought')
        outdir = join(self.this_class_tif, 'variables_spatial_trend_during_drought_ensemble')
        # col = 'T_max'
        col = 'intensity'
        T.mk_dir(outdir)
        for exp in self.experiment_list:
            exp_dir = join(fdir,col,exp)
            exp_outdir = join(outdir,col,exp)
            T.mk_dir(exp_outdir,force=True)
            for drt in global_drought_type_list:
                tif_path_list = []
                for model in T.listdir(exp_dir):
                    tif_path = join(exp_dir,model,f'{drt}.tif')
                    tif_path_list.append(tif_path)
                    # print(tif_path)
                # print('--')
                # print(tif_path_list)
                # exit()
                outf = join(exp_outdir,f'{drt}.tif')
                print(outf)
                Pre_Process().compose_tif_list(tif_path_list,outf)

    def plot_ensemble_tif_variables_spatial_trend_during_drought(self):
        fdir = join(self.this_class_tif, 'variables_spatial_trend_during_drought_ensemble')
        outdir = join(self.this_class_png, 'variables_spatial_trend_during_drought_ensemble')
        # col = 'T_max'
        col = 'intensity'
        T.mk_dir(outdir)
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        drought_type_list = [ 'hot-drought']
        # cmap = Tools().cmap_blend(color_list[::-1])
        cmap = Tools().cmap_blend(color_list)
        for exp in self.experiment_list:
            exp_dir = join(fdir,col,exp)
            exp_outdir = join(outdir,col,exp)
            T.mk_dir(exp_outdir,force=True)
            for drt in drought_type_list:
                fpath = join(exp_dir,f'{drt}.tif')
                plt.figure(figsize=(10,5))
                # Plot().plot_Robinson(fpath,res=100000,cmap=cmap,vmin=-0.1,vmax=0.1)
                Plot().plot_Robinson(fpath,res=100000,cmap=cmap,vmin=-0.01,vmax=0.01)
                plt.title(f'{col}_{exp}_{drt}')
                print(f'{col}_{exp}_{drt}')
                # plt.show()
                outf = join(exp_outdir,f'{col}_{exp}_{drt}.png')
                plt.savefig(outf,dpi=300)
                plt.close()
        T.open_path_and_file(outdir)


def main():
    # Pick_drought_events_SM().run()
    # Pick_drought_events_SM_multi_thresholds().run()
    # Pick_drought_events_SM_multi_thresholds_ensemble_annual().run()
    # Dataframe_SM().run()

    CMIP6_drought_events_analysis().run()
    # CMIP6_climate_variables_analysis().run()
    # Crit_P_T().run()
    pass


if __name__ == '__main__':

    main()
