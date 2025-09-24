# coding=utf-8

from meta_info import *
result_root_this_script = join(results_root, 'CMIP6_analysis')
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

class Pick_drought_events_SM_ensemble:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_SM_ensemble', result_root_this_script, mode=2)
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
            # fdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_moving_window_baseline')
            fdir = join(self.datadir, product, 'per_pix_ensemble_std_anomaly_based_2020_2060')
            for experiment in experiment_list:
                fdir_i = join(fdir, experiment)
                outdir_i = join(outdir, experiment)
                T.mk_dir(outdir_i, force=True)

                spatial_dict = T.load_npy_dir(fdir_i)
                start_year = 2020
                threshold_upper = -1.5
                threshold_bottom = -np.inf

                pix_list = []
                drought_range_list = []
                drougth_timing_list = []
                drought_year_list = []
                intensity_list = []
                severity_list = []
                severity_mean_list = []

                for pix in tqdm(spatial_dict,desc=f'{experiment}'):
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

    def add_hot_normal_drought(self):
        fdir = join(self.this_class_arr, 'picked_events')
        fdir_temp = join(data_root,'CMIP6/tas/per_pix_ensemble_anomaly')
        experiment_list = ['ssp245', 'ssp585']
        for exp in experiment_list:
            fdir_i = join(fdir,exp)
            fdir_temp_i = join(fdir_temp,exp)
            Temperature_anomaly_data_dict = T.load_npy_dir(fdir_temp_i)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                fpath = join(fdir_i,f)
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
                outf = join(fdir_i,f)
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

class Pick_drought_events_SM_multi_thresholds_ensemble:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_drought_events_SM_multi_thresholds_ensemble', result_root_this_script, mode=2)
        # self.threshold = -2

    def run(self):
        # self.pick_normal_drought_events()
        # self.add_hot_normal_drought()
        # self.add_hot_normal_drought_annual_T()
        # self.gen_dataframe()
        pass

    def pick_normal_drought_events(self):
        outdir = join(self.this_class_arr, 'picked_events')
        self.datadir = join(data_root, 'CMIP6')
        experiment_list = ['ssp245', 'ssp585']
        product = 'mrsos'
        # fdir = join(self.datadir, product, 'per_pix_ensemble_std_anomaly_based_2020_2060')
        # fdir = join(self.datadir, product, 'per_pix_ensemble_anomaly_moving_window_baseline')
        fdir = join(self.datadir, product, 'per_pix_ensemble_std_anomaly_based_2020_2060')
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
                        drought_year = int(mid/12) + 2020
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
        # self.scenario = 'ssp245'
        self.scenario = 'ssp585'
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir(f'Dataframe_SM_multi_threshold_annual', result_root_this_script, mode=2)
            # T.mk_class_dir(f'Dataframe_SM_multi_threshold', result_root_this_script, mode=2)
            # T.mk_class_dir('Dataframe_SM_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Dataframe_SM_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Dataframe_SM', result_root_this_script, mode=2)
        self.df_dir = join(self.this_class_arr, 'dataframe',self.scenario)

        pass

    def run(self):
        self.copy_df()
        # exit()
        for f in T.listdir(self.df_dir):
            if not f.endswith('.df'):
                continue
            print(f)
            self.dff = join(self.df_dir,f)
            df = self.__gen_df_init()
            df = Dataframe_func(df).df
            df = df.dropna()
            # temp_anomaly_f = join(data_root,f'CMIP6/tas/per_pix_ensemble_anomaly_moving_window_baseline',self.scenario,'anomaly.npy')
            # temp_anomaly_f = join(data_root,f'CMIP6/tas/per_pix_ensemble_std_anomaly_based_2020_2060',self.scenario,'anomaly.npy')
            # df = self.add_variables_during_droughts(df,temp_anomaly_f,'Tair_anomaly_GS')

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
        drought_df_dir = join(Pick_drought_events_SM_multi_thresholds_ensemble_annual().this_class_arr,'picked_events',self.scenario)
        outdir = self.df_dir
        T.mk_dir(outdir,force=True)
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
        self.copy_df()
        self.add_drought_type()
        self.drought_frequency_ts()
        self.tif_drought_frequency_spatial()
        self.tif_drought_frequency_spatial_all_drt()
        self.plot_drought_frequency_spatial()
        self.plot_T_during_drought_ts()
        # self.tif_every_10_years_drought_events()
        # self.plot_every_10_years_drought_events()
        pass

    def copy_df(self):
        fdir = join(Dataframe_SM().this_class_arr,'dataframe_merge')
        # T.open_path_and_file(fdir)
        # exit()
        outdir = join(self.this_class_arr,'dataframe')
        for exp in self.experiment_list:
            fdir_i = join(fdir,exp)
            # print(fdir_i)
            # exit()
            outdir_i = join(outdir,exp)
            T.mk_dir(outdir_i,force=True)
            for f in T.listdir(fdir_i):
                fpath = join(fdir_i,f)
                print(fpath)
                outpath = join(outdir_i,f)
                shutil.copy(fpath,outpath)

    def add_drought_type(self):
        experiment_list = ['ssp245', 'ssp585']
        for exp in experiment_list:
            dff = join(self.this_class_arr,'dataframe',exp,'dataframe_merge.df')
            df = T.load_df(dff)
            drt_list = []
            for i,row in tqdm(df.iterrows(),total=len(df),desc=exp):
                T_max = row['T_max']
                # if T_max > 0:
                if T_max > 1:
                    drought_type = 'hot-drought'
                else:
                    drought_type = 'normal-drought'
                drt_list.append(drought_type)
            df['drought_type'] = drt_list
            T.save_df(df,dff)
            T.df_to_excel(df,dff)


    def drought_frequency_ts(self):
        fdir = join(self.this_class_arr,'dataframe')
        outdir = join(self.this_class_png,'drought_frequency_ts')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        for exp in self.experiment_list:
            plt.figure()
            fdir_i = join(fdir,exp)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                fpath = join(fdir_i,f)
                df = T.load_df(fpath)
                # df = df[df['threshold']<-2]
                all_pix_list = T.get_df_unique_val_list(df,'pix')
                all_pix_len = len(all_pix_list)
                for drt in drought_type_list:
                    df_drt = df[df['drought_type']==drt]
                    year_list = T.get_df_unique_val_list(df_drt,'drought_year')
                    df_year_dict = T.df_groupby(df_drt,'drought_year')
                    x = []
                    y = []
                    for year in year_list:
                        df_year = df_year_dict[year]
                        spatial_dict = {}
                        pix_list = T.get_df_unique_val_list(df_year,'pix')
                        # len_pix = len(pix_list)
                        len_pix = len(df_year)
                        ratio = len_pix / all_pix_len * 100
                        x.append(year)
                        y.append(ratio)
                    y = SMOOTH().smooth_convolve(y,window_len=5)
                    plt.plot(x,y,label=f'{drt}-{exp}')
            plt.legend()
            plt.ylim(0,30)
            outf = join(outdir,f'{exp}.pdf')
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def tif_drought_frequency_spatial(self):
        fdir = join(self.this_class_arr, 'dataframe')
        outdir = join(self.this_class_tif,'drought_frequency')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        for exp in self.experiment_list:
            fdir_i = join(fdir, exp)
            for f in T.listdir(fdir_i):
                if not f.endswith('.df'):
                    continue
                fpath = join(fdir_i, f)
                df = T.load_df(fpath)
                T.print_head_n(df)
                # df = df[df['threshold']<-2]
                for drt in drought_type_list:
                    df_drt = df[df['drought_type'] == drt]
                    pix_list = T.get_df_unique_val_list(df_drt, 'pix')
                    df_groupby_pix = T.df_groupby(df_drt, 'pix')
                    spatial_dict = {}
                    for pix in pix_list:
                        df_pix = df_groupby_pix[pix]
                        len_pix = len(df_pix)
                        spatial_dict[pix] = len_pix
                    outf = join(outdir,f'{drt}_{exp}.tif')
                    print(outf)
                    D_CMIP.pix_dic_to_tif(spatial_dict,outf)


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
            Plot().plot_Robinson(fpath,res=100000,cmap=cmap,vmin=0,vmax=20)
            plt.title(f)
            outf = join(outdir,f'{f}.png')
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


    def tif_every_10_years_drought_events(self):
        fdir = join(self.this_class_arr,'dataframe')
        outdir = join(self.this_class_tif,'every_10_years_drought_events')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        year_list_group = list(range(2020,2100,10))
        year_list_group.append(2100)
        year_list_group_list = []
        for i in range(len(year_list_group)):
            if i + 1 == len(year_list_group):
                break
            year_list_group_list.append([year_list_group[i],year_list_group[i+1]])

        for year_list in year_list_group_list:
            for exp in self.experiment_list:
                fdir_i = join(fdir,exp)
                for f in T.listdir(fdir_i):
                    if not f.endswith('.df'):
                        continue
                    fpath = join(fdir_i,f)
                    df = T.load_df(fpath)
                    for drt in drought_type_list:
                        df_drt = df[df['drought_type']==drt]
                        df_year = df_drt[df_drt['drought_year']>=year_list[0]]
                        df_year = df_drt[df_drt['drought_year']<year_list[1]]
                        spatial_dict = {}
                        pix_list = T.get_df_unique_val_list(df_year,'pix')
                        for pix in pix_list:
                            df_pix = df_year[df_year['pix']==pix]
                            drought_events_num = len(df_pix)
                            spatial_dict[pix] = drought_events_num
                        outf = join(outdir,f'{drt}_{exp}_{year_list[0]}-{year_list[-1]}.tif')
                        print(outf)
                        D_CMIP.pix_dic_to_tif(spatial_dict,outf)

    def plot_every_10_years_drought_events(self):
        fdir = join(self.this_class_tif,'every_10_years_drought_events')
        outdir = join(self.this_class_png,'every_10_years_drought_events')
        T.mk_dir(outdir)
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ]
        cmap = Tools().cmap_blend(color_list[::-1])
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            Plot().plot_Robinson(fpath,res=100000,cmap=cmap,vmin=1,vmax=10)
            print(f)
            plt.title(f)
            outf = join(outdir,f'{f}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def plot_T_during_drought_ts(self):
        fdir = join(self.this_class_arr,'dataframe')
        outdir = join(self.this_class_png,'T_during_drought_ts')
        T.mk_dir(outdir)
        experiment_list = ['ssp245', 'ssp585']
        for exp in experiment_list:
            dff = join(fdir,exp,'dataframe_merge.df')
            df = T.load_df(dff)
            year_list = T.get_df_unique_val_list(df,'drought_year')
            plt.figure()
            for drt in global_drought_type_list:
                df_drt = df[df['drought_type']==drt]
                x = []
                y = []
                err = []
                for year in year_list:
                    df_year = df_drt[df_drt['drought_year']==year]
                    vals = df_year['T_max'].tolist()
                    mean = np.nanmean(vals)
                    std = np.nanstd(vals)
                    x.append(year)
                    y.append(mean)
                    err.append(std)
                x = np.array(x)
                y = np.array(y)
                err = np.array(err)
                plt.plot(x,y,label=drt)
                plt.fill_between(x,y-err,y+err,alpha=0.5)
            outf = join(outdir,f'{exp}.pdf')
            plt.legend()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)
        # plt.show()


        pass

class CMIP6_climate_variables_analysis:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('CMIP6_climate_variables_analysis', result_root_this_script, mode=2)
        self.experiment_list = ['ssp245', 'ssp585']
        self.product_list = ['mrsos','tas']
        pass

    def run(self):
        self.tif_every_10_years()
        pass

    def tif_every_10_years(self):
        outdir = join(self.this_class_tif,'every_10_years')
        T.mk_dir(outdir)
        drought_type_list = global_drought_type_list
        year_list_group = list(range(2020,2100,10))
        year_list_group.append(2100)
        year_list_group_list = []
        for i in range(len(year_list_group)):
            if i + 1 == len(year_list_group):
                break
            year_list_group_list.append([year_list_group[i],year_list_group[i+1]])

        for product in self.product_list:
            fdir = join(data_root,'CMIP6',product,'per_pix_ensemble_anomaly')
            for year_list in year_list_group_list:
                for exp in self.experiment_list:
                    fdir_i = join(fdir,exp)
                    T.open_path_and_file(fdir_i)
                    # exit()
                    for f in T.listdir(fdir_i):
                        if not f.endswith('.df'):
                            continue
                        fpath = join(fdir_i,f)
                        df = T.load_df(fpath)
                        for drt in drought_type_list:
                            df_drt = df[df['drought_type']==drt]
                            df_year = df_drt[df_drt['drought_year']>=year_list[0]]
                            df_year = df_drt[df_drt['drought_year']<year_list[1]]
                            spatial_dict = {}
                            pix_list = T.get_df_unique_val_list(df_year,'pix')
                            for pix in pix_list:
                                df_pix = df_year[df_year['pix']==pix]
                                drought_events_num = len(df_pix)
                                spatial_dict[pix] = drought_events_num
                            outf = join(outdir,f'{drt}_{exp}_{year_list[0]}-{year_list[-1]}.tif')
                            print(outf)
                            D_CMIP.pix_dic_to_tif(spatial_dict,outf)

class Crit_P_T:
    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Crit_P_T', result_root_this_script, mode=2)

        #-------------------------------
        self.scenario = 'ssp245'
        # self.scenario = 'ssp585'

        #-------------------------------
        # self.col_name = 'NDVI4g_climatology_percentage_GS_reg'
        # self.threshold = 0.95
        self.col_name = 'NDVI4g_climatology_percentage_post_2_GS_reg'
        self.threshold = 0.98
        #-------------------------------

        self.dff = join(self.this_class_arr,'dataframe',f'{self.scenario}.df')
        pass

    def run(self):
        # self.copy_df()
        self.predict_rs()
        self.statistic_rs()

        pass

    def copy_df(self):
        fpath = join(Dataframe_SM().this_class_arr,'dataframe_merge',self.scenario,'dataframe_merge.df')
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir,force=True)
        outf = join(outdir,f'{self.scenario}.df')
        shutil.copy(fpath,outf)
        df = T.load_df(outf)
        T.df_to_excel(df,outf)

    def predict_rs(self):
        outdir = join(self.this_class_arr,'predict_rs',self.scenario)
        T.mk_dir(outdir,force=True)
        import statistic
        df = T.load_df(self.dff)
        T.print_head_n(df)
        col_name = self.col_name
        # exit()
        history_model_f = join(statistic.Critical_P_and_T().this_class_arr, f'MAT_MAP_multiregression_anomaly/{col_name}.npy')
        history_model = T.load_npy(history_model_f)
        df_list = []
        for key in tqdm(history_model):
            MAT_left = history_model[key]['MAT_left']
            MAT_right = history_model[key]['MAT_right']
            MAP_left = history_model[key]['MAP_left']
            MAP_right = history_model[key]['MAP_right']
            df_MAT = df[(df['MAT']>=MAT_left)&(df['MAT']<=MAT_right)]
            df_MAP = df_MAT[(df_MAT['MAP']>=MAP_left)&(df_MAT['MAP']<=MAP_right)]
            if len(df_MAP) == 0:
                continue
            model = history_model[key]['model']
            SM_vals = df_MAP['intensity'].tolist()
            T_vals = df_MAP['Tair_juping_GS'].tolist()
            predict_val = model.predict(np.array([SM_vals,T_vals]).T)
            df_MAP[col_name] = predict_val
            df_list.append(df_MAP)
        df_merge = pd.concat(df_list)
        outf = join(outdir,f'{col_name}.df')
        T.save_df(df_merge,outf)
        T.df_to_excel(df_merge,outf)

    def statistic_rs(self):
        outdir = join(self.this_class_tif,'statistic_rs')
        outdir_png = join(self.this_class_png,'statistic_rs')
        T.mk_dir(outdir_png)
        T.mk_dir(outdir)
        col_name = self.col_name
        dff = join(self.this_class_arr,'predict_rs',self.scenario,f'{col_name}.df')
        df = T.load_df(dff)
        T.print_head_n(df)

        df_group_pix = T.df_groupby(df,'pix')
        spatial_dict = {}
        for pix in tqdm(df_group_pix):
            df_pix = df_group_pix[pix]
            df_pix_threshold = df_pix[df_pix[col_name]<=self.threshold]
            ratio = len(df_pix_threshold) / len(df_pix) * 100
            spatial_dict[pix] = ratio
        arr = D_CMIP.pix_dic_to_spatial_arr(spatial_dict)
        outf = join(outdir,f'{col_name}-{self.scenario}.tif')
        outf_png = join(outdir_png,f'{col_name}-{self.scenario}.png')
        D_CMIP.arr_to_tif(arr,outf)
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#86b9d2',
            '#064c6c',
        ][::-1]
        # Blue represents high values, and red represents low values.
        cmap = Tools().cmap_blend(color_list)
        Plot().plot_Robinson(outf,res=100000,cmap=cmap,vmin=0,vmax=100)
        plt.savefig(outf_png,dpi=300)
        plt.close()


def main():
    # Pick_drought_events_SM().run()
    # Pick_drought_events_SM_ensemble().run()
    # Pick_drought_events_SM_multi_thresholds_ensemble().run()
    # Pick_drought_events_SM_multi_thresholds_ensemble_annual().run()
    # Dataframe_SM().run()

    CMIP6_drought_events_analysis().run()
    # CMIP6_climate_variables_analysis().run()
    # Crit_P_T().run()
    pass


if __name__ == '__main__':

    main()
