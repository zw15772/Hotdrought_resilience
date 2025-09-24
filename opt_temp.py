# coding=utf-8

from lytools import *
T = Tools()
D = DIC_and_TIF(pixelsize=0.25)


class Optimal_temperature:

    def __init__(self):

        pass

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
