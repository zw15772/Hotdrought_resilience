# coding=utf-8
import shap
from meta_info import *

result_root_this_script = join(results_root, 'statistic')
import xgboost as xgb

class Hot_normal_drought:

    def __init__(self):
        import analysis
        # self.Dataframe_mode = analysis.Dataframe_SM().Dataframe_mode
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir(f'Hot_normal_drought', result_root_this_script, mode=2)
            # T.mk_class_dir(f'Hot_normal_drought/{self.Dataframe_mode}', result_root_this_script, mode=2)
            # T.mk_class_dir('Hot_normal_drought_double_detrend', result_root_this_script, mode=2)
            # T.mk_class_dir('Hot_normal_drought', result_root_this_script, mode=2)
            # T.mk_class_dir('Hot_normal_drought_double_detrend', result_root_this_script, mode=2)
        self.dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        self.df_dir = analysis.Dataframe_SM().df_dir
        pass

    def run(self):
        # self.tif_hot_normal_drought_rt()
        # self.tif_hot_normal_drought_rt_merge()
        # self.tif_hot_normal_drought_rt_merge_extreme()
        self.tif_hot_normal_drought_rt_merge_extreme_hot_normal()
        # self.plot_tif_hot_normal_drought_rt_merge()
        # self.tif_hot_normal_drought_rs()
        # self.tif_hot_normal_drought_rs_merge()
        # self.plot_tif_hot_normal_drought_rs_merge()
        # self.plot_tif_hot_normal_drought_rs_merge_test()
        # self.statistic_pft_hot_normal_merge()
        # self.plot_tif_hot_normal_drought_rs()
        # self.matrix_T_SPI()
        # self.temporal_change()
        # self.temporal_change_percentage_bar()
        # self.tif_spatial_trend()
        # self.tif_spatial_trend_normal_hot()
        # self.plot_sptatial_trend()
        # self.statistic_pft_hot_normal_merge_trend()
        # self.rs_trend_vs_NDVI_trend_scatter()
        # self.hot_normal_delta()
        # self.plot_hot_normal_delta()
        # self.statistic_pft_hot_normal_delta()
        # self.statistic_AI_hot_normal_delta()
        # self.statistic_BNPP_hot_normal_delta()
        # self.statistic_not_rs_rt_area()
        pass

    def tif_hot_normal_drought_rt(self):
        outdir = join(self.this_class_tif, 'tif_hot_normal_rt')
        T.mk_dir(outdir, force=1)
        drought_type_list = global_drought_type_list
        col_name = 'LT_CFE_Hybrid_NT_origin_rs_pre_baseline_2_GS'
        # col_name = 'LT_CFE_Hybrid_NT_origin_rt_pre_baseline_GS'
        for drt in drought_type_list:
            for f in T.listdir(self.df_dir):
                if not f.endswith('.df'):
                    continue
                level = f.replace('.df','')
                dff = join(self.df_dir,f)
                # outf = join(outdir, f'{f}_hot.tif')
                outf = join(outdir, f'{col_name}_{level}_{drt}.tif')
                df = T.load_df(dff)
                df = df[df['AI_class'] == 'Arid']
                T.print_head_n(df)
                # exit()
                # df = df[df['scale']=='spi03']
                # df = df[df['drought_type']=='hot-drought']
                # df = df[df['drought_type']==drt]
                df_pix_dict = T.df_groupby(df,'pix')
                spatial_dict = {}
                print(df.columns)
                for pix in tqdm(df_pix_dict):
                    df_i = df_pix_dict[pix]
                    NDVI_anomaly_detrend = df_i[col_name].values
                    # NDVI_anomaly_detrend = df_i['NDVI-anomaly_detrend_post_1'].values
                    mean = np.nanmean(NDVI_anomaly_detrend)
                    spatial_dict[pix] = mean
                D.pix_dic_to_tif(spatial_dict, outf)

    def tif_hot_normal_drought_rt_merge(self):
        outdir = join(self.this_class_tif, 'tif_hot_normal_rt_merge')
        T.mk_dir(outdir, force=1)
        dff = self.dff_merge
        # col_name = 'LT_CFE_Hybrid_NT_origin_rs_pre_baseline_2_GS'
        col_name = 'LT_CFE_Hybrid_NT_origin_rt_pre_baseline_GS'
        drought_type_list = global_drought_type_list
        # for drt in drought_type_list:
        # outf = join(outdir, f'hot.tif')
        outf = join(outdir, f'{col_name}.tif')
        df = T.load_df(dff)
        df = df[df['AI_class'] == 'Arid']
        T.print_head_n(df)
        # exit()
        # df = df[df['scale']=='spi03']
        # df = df[df['drought_type']=='hot-drought']
        # df = df[df['drought_type']==drt]
        df_pix_dict = T.df_groupby(df,'pix')
        spatial_dict = {}
        print(df.columns)
        for pix in tqdm(df_pix_dict):
            df_i = df_pix_dict[pix]
            NDVI_anomaly_detrend = df_i[col_name].values
            # NDVI_anomaly_detrend = df_i['NDVI-anomaly_detrend_post_1'].values
            mean = np.nanmean(NDVI_anomaly_detrend)
            spatial_dict[pix] = mean
        D.pix_dic_to_tif(spatial_dict, outf)

    def tif_hot_normal_drought_rt_merge_extreme(self):
        outdir = join(self.this_class_tif, 'tif_hot_normal_drought_rt_merge_extreme')
        T.mk_dir(outdir, force=1)
        dff = self.dff_merge
        col_name = 'LT_CFE_Hybrid_NT_origin_rs_pre_baseline_2_GS'
        # col_name = 'LT_CFE_Hybrid_NT_origin_rt_pre_baseline_GS'
        drought_type_list = global_drought_type_list
        # for drt in drought_type_list:
        # outf = join(outdir, f'hot.tif')
        outf = join(outdir, f'{col_name}.tif')
        df = T.load_df(dff)
        # T.print_head_n(df)
        # exit()
        df = df[df['AI_class'] == 'Arid']
        df = df[df['threshold'] <= -2]
        T.print_head_n(df)
        # exit()
        # df = df[df['scale']=='spi03']
        # df = df[df['drought_type']=='hot-drought']
        # df = df[df['drought_type']==drt]
        df_pix_dict = T.df_groupby(df,'pix')
        spatial_dict = {}
        print(df.columns)
        for pix in tqdm(df_pix_dict):
            df_i = df_pix_dict[pix]
            NDVI_anomaly_detrend = df_i[col_name].values
            # NDVI_anomaly_detrend = df_i['NDVI-anomaly_detrend_post_1'].values
            mean = np.nanmean(NDVI_anomaly_detrend)
            spatial_dict[pix] = mean
        D.pix_dic_to_tif(spatial_dict, outf)

    def tif_hot_normal_drought_rt_merge_extreme_hot_normal(self):
        outdir = join(self.this_class_tif, 'tif_hot_normal_drought_rt_merge_extreme_hot_normal')
        T.mk_dir(outdir, force=1)
        dff = self.dff_merge
        # col_name = 'LT_CFE_Hybrid_NT_origin_rs_pre_baseline_2_GS'
        col_name = 'LT_CFE_Hybrid_NT_origin_rt_pre_baseline_GS'
        drought_type_list = global_drought_type_list
        # for drt in drought_type_list:
        # outf = join(outdir, f'hot.tif')
        # drt = 'hot-drought'
        drt = 'normal-drought'
        outf = join(outdir, f'{col_name}_{drt}.tif')
        df = T.load_df(dff)
        df = df[df['AI_class'] == 'Arid']
        # df = df[df['threshold'] <= -2]
        T_max = df['T_max'].values
        plt.hist(T_max,bins=100)
        plt.show()
        T.print_head_n(df)
        # exit()

        T.print_head_n(df)
        # exit()
        # df = df[df['scale']=='spi03']
        # df = df[df['T_max']>0.5]
        df = df[df['T_max']<0]
        # df = df[df['drought_type']==drt]
        df_pix_dict = T.df_groupby(df,'pix')
        spatial_dict = {}
        print(df.columns)
        for pix in tqdm(df_pix_dict):
            df_i = df_pix_dict[pix]
            NDVI_anomaly_detrend = df_i[col_name].values
            # NDVI_anomaly_detrend = df_i['NDVI-anomaly_detrend_post_1'].values
            mean = np.nanmean(NDVI_anomaly_detrend)
            spatial_dict[pix] = mean
        D.pix_dic_to_tif(spatial_dict, outf)


    def tif_hot_normal_drought_rs_merge(self):
        outdir = join(self.this_class_tif, 'tif_hot_normal_rs_merge')
        T.mk_dir(outdir, force=1)
        dff = self.dff_merge
        df = T.load_df(dff)
        df = df[df['AI_class'] == 'Arid']
        print(df.columns.tolist())
        # col_list = [
        #     'NDVI4g_climatology_percentage_rt_GS',
        #     'NDVI4g_climatology_percentage_rs_1_GS',
        #     'NDVI4g_climatology_percentage_rs_2_GS',
        #     'NDVI4g_climatology_percentage_rs_3_GS',
        #     'NDVI4g_climatology_percentage_rs_4_GS',
        # ]
        col_list = [
            'NDVI4g_climatology_percentage_detrend_post_1_GS',
            'NDVI4g_climatology_percentage_detrend_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_post_3_GS',
            'NDVI4g_climatology_percentage_detrend_post_4_GS',
        ]
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df[df['drought_type']==drt]
            df_pix_dict = T.df_groupby(df_drt, 'pix')
            for col in col_list:
                print(col, drt)
                spatial_dict = {}
                for pix in df_pix_dict:
                    df_i = df_pix_dict[pix]
                    vals = df_i[col].tolist()
                    mean = np.nanmean(vals)
                    spatial_dict[pix] = mean
                outf = join(outdir, f'{drt}_{col}.tif')
                D.pix_dic_to_tif(spatial_dict, outf)


    def tif_hot_normal_drought_rs(self):
        outdir = join(self.this_class_tif, 'tif_hot_normal_drought_rs')
        T.mk_dir(outdir, force=1)
        for f in T.listdir(self.df_dir):
            if not f.endswith('.df'):
                continue
            dff = join(self.df_dir,f)
            df = T.load_df(dff)
            df = df[df['AI_class'] == 'Arid']
            T.mk_dir(outdir, force=1)
            n_list = [1,2,3,4]
            drought_type_list = global_drought_type_list
            for n in n_list:
                for drt in drought_type_list:
                    col_name = f'NDVI4g_climatology_percentage_rs_{n}_GS'
                    print(col_name)
                    outf = join(outdir, f'{f}_{drt}_{n}.tif')
                    df_drt = df[df['drought_type']==drt]
                    df_pix_dict = T.df_groupby(df_drt,'pix')
                    spatial_dict = {}
                    for pix in tqdm(df_pix_dict):
                        df_i = df_pix_dict[pix]
                        NDVI_anomaly_detrend = df_i[col_name].values
                        # NDVI_anomaly_detrend = df_i['NDVI-anomaly_detrend_post_1'].values
                        mean = np.nanmean(NDVI_anomaly_detrend)
                        spatial_dict[pix] = mean
                    D.pix_dic_to_tif(spatial_dict, outf)

    def plot_tif_hot_normal_drought_rs(self):
        fdir = join(self.this_class_tif, 'tif_hot_normal_drought_rs')
        outdir = join(self.this_class_png,'tif_hot_normal_drought_rs')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            Plot().plot_Robinson(fpath,vmin=0.95,vmax=1.05)
            plt.title(f.replace('.tif','')+'_RS')
            plt.savefig(outf,dpi=600)
            plt.close()
        T.open_path_and_file(outdir)
        pass

    def plot_tif_hot_normal_drought_rt_merge(self):
        fdir = join(self.this_class_tif, 'tif_hot_normal_rt_merge')
        outdir = join(self.this_class_png,'tif_hot_normal_rt_merge')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            Plot().plot_Robinson(fpath,vmin=0.9,vmax=1.1)
            plt.title(f.replace('.tif','')+'_Rt')
            plt.savefig(outf,dpi=600)
            plt.close()
        T.open_path_and_file(outdir)
        pass

    def statistic_pft_hot_normal_merge(self):
        import analysis
        fdir = join(self.this_class_tif, 'tif_hot_normal_rs_merge')
        outdir = join(self.this_class_png,'statistic_pft_hot_normal_merge')
        T.mk_dir(outdir)
        spatial_dicts = {}
        col_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            spatial_dict = D.spatial_tif_to_dic(fpath)
            spatial_dicts[f] = spatial_dict
            col_list.append(f)
        df = T.spatial_dics_to_df(spatial_dicts)
        df = analysis.Dataframe_func(df).df
        # T.print_head_n(df)
        # col_name = 'val'
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # # print(lc_list)
        # # exit()
        # # lc_list = global_lc_list
        for col in col_list:
            print(col)
            mean_list = []
            std_list = []
            plt.figure()
            for lc in lc_list:
                df_lc = df[df['landcover_GLC']==lc]
                vals = df_lc[col].tolist()
                mean = np.nanmean(vals)
                std = T.uncertainty_err(vals)[0]
                std = abs(std)
                mean_list.append(mean)
                std_list.append(std)
            plt.bar(lc_list,mean_list,yerr=std_list)
            plt.ylabel(col)
            plt.ylim(0.96,1.02)
            outf = join(outdir,f'{col}.png')
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def statistic_pft_hot_normal_merge_trend(self):
        import analysis
        fdir = join(self.this_class_tif, 'spatial_trend_normal_hot')
        outdir = join(self.this_class_png,'statistic_pft_hot_normal_merge_trend')
        T.mk_dir(outdir)
        spatial_dicts = {}
        col_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            if f.endswith('_p.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            spatial_dict = D.spatial_tif_to_dic(fpath)
            spatial_dicts[f] = spatial_dict
            col_list.append(f)
        df = T.spatial_dics_to_df(spatial_dicts)
        df = analysis.Dataframe_func(df).df
        # T.print_head_n(df)
        # col_name = 'val'
        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        # # print(lc_list)
        # # exit()
        # # lc_list = global_lc_list
        for col in col_list:
            print(col)
            mean_list = []
            std_list = []
            plt.figure()
            for lc in lc_list:
                df_lc = df[df['landcover_GLC']==lc]
                vals = df_lc[col].tolist()
                mean = np.nanmean(vals)
                std = T.uncertainty_err(vals)[0]
                std = abs(std)
                mean_list.append(mean)
                std_list.append(std)
            plt.bar(lc_list,mean_list,yerr=std_list)
            plt.ylabel(col)
            # plt.ylim(0.96,1.02)
            outf = join(outdir,f'{col}.png')
            # plt.show()
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def plot_tif_hot_normal_drought_rs_merge(self):
        fdir = join(self.this_class_tif, 'tif_hot_normal_rs_merge')
        outdir = join(self.this_class_png,'tif_hot_normal_rs_merge')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.tif','.png'))
            Plot().plot_Robinson(fpath,vmin=0.95,vmax=1.05)
            plt.title(f.replace('.tif',''))
            plt.savefig(outf,dpi=600)
            plt.close()
        T.open_path_and_file(outdir)
        pass

    def plot_tif_hot_normal_drought_rs_merge_test(self):
        fpath = '/Volumes/NVME4T/hotdrought_CMIP/results/statistic/Hot_normal_drought/tif/tif_hot_normal_rt/-1.5_hot-drought.tif'
        # Plot().plot_Robinson(fpath,vmin=0.9,vmax=1,res=50000,is_discrete=True)
        Plot().plot_Robinson(fpath,vmin=0.9,vmax=1,res=50000)
        plt.show()

    def matrix_T_SPI(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        # scale = 1
        # scale = 3
        # scale = 6
        # scale = 9
        # scale = 12
        # df = df[df['scale_int']==scale]
        # col_name = 'NDVI4g_climatology_percentage_rt_GS'
        # col_name = 'scale_int'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_1_GS'
        col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_4_GS'
        T_col = 'ERA_Tair_anomaly_detrend'
        # SPI_col = 'intensity'
        SPI_col = 'severity'
        df = df[df['AI_class'] =='Arid']
        df = df[df[SPI_col] >-20]
        SPI_vals = df[SPI_col].values
        # plt.hist(SPI_vals,bins=100)
        # plt.show()
        # SPI_bin = np.arange(-3,-2,0.1)
        SPI_bin = np.arange(-20,0,1)
        T_bin = np.arange(0,2,0.1)
        # T_bin = np.arange(-2,2,0.1)

        df_group_T, bins_list_str_T = T.df_bin(df,T_col, T_bin)

        matrix = []
        y_ticks = []
        x_ticks = []
        for name_T,df_group_i_T in df_group_T:
            y_ticks.append(name_T[0].left)
            df_group_SPI, bins_list_str_SPI = T.df_bin(df_group_i_T, SPI_col, SPI_bin)
            matrix_i = []
            x_ticks = []
            for name_SPI, df_group_i_SPI in df_group_SPI:
                x_ticks.append(name_SPI[0].left)
                vals = df_group_i_SPI[col_name].values
                vals_mean = np.nanmean(vals)
                matrix_i.append(vals_mean)
            matrix.append(matrix_i)
        matrix = np.array(matrix)
        print(matrix.shape)
        # plt.imshow(matrix, cmap='RdBu', interpolation='nearest',vmin=0.9,vmax=1.1,aspect='auto')
        # plt.imshow(matrix, cmap='RdBu_r', interpolation='nearest',aspect='auto')
        plt.imshow(matrix, cmap='RdBu', interpolation='nearest',aspect='auto')
        # add label name on colorbar
        cbar = plt.colorbar()
        cbar.set_label(col_name, rotation=90)


        plt.xticks(np.arange(len(x_ticks)), x_ticks)
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        plt.xlabel(SPI_col)
        plt.ylabel(T_col)
        # plt.title(f'spi{scale}')
        plt.tight_layout()
        plt.show()

    def temporal_change1(self):
        df_dir = self.df_dir
        outdir = join(self.this_class_arr,'temporal_change')
        T.mk_dir(outdir)
        col_name = 'NDVI4g_climatology_percentage_rt_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_1_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_4_GS'
        drought_type_list = global_drought_type_list
        plt.figure(figsize=(10, 5))
        flag = 0
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            flag += 1
            dff = join(df_dir,f)
            df = T.load_df(dff)
            print(df.columns)
            # exit()
            df = df[df['AI_class'] == 'Arid']
            # T.print_head_n(df)
            plt.subplot(2,2,flag)
            year_list = global_year_range_list
            for drt in drought_type_list:
                df_drt = df[df['drought_type']==drt]
                df_year_dict = T.df_groupby(df_drt,'drought_year')
                mean_list = []
                err_list = []
                for year in year_list:
                    df_year = df_year_dict[year]
                    vals = df_year[col_name].tolist()
                    vals_mean = np.nanmean(vals)
                    # vals_std = np.nanstd(vals)
                    vals_std,_,_ = T.uncertainty_err(vals)
                    # mean_list.append(vals_mean)
                    mean_list.append(len(vals))
                    err_list.append(vals_std)
                plt.plot(year_list,mean_list,label=drt)
                # plt.fill_between(year_list, np.array(mean_list) - np.array(err_list),
                #                     np.array(mean_list) + np.array(err_list), alpha=0.2)
            # plt.hlines(1,year_list[0],year_list[-1],linestyles='--')
            plt.legend()
            # plt.ylim(0.85,1.1)
            plt.title(f)
            # plt.suptitle(col_name)
        plt.tight_layout()
        plt.show()

    def temporal_change(self):
        df_dir = self.df_dir
        outdir = join(self.this_class_png,'temporal_change')
        T.mk_dir(outdir)

        # col_name = 'NDVI4g_climatology_percentage_rt_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_1_GS'
        col_name = 'LT_CFE_Hybrid_NT_origin_rs_pre_baseline_2_GS'
        # col_name = 'LT_CFE_Hybrid_NT_origin_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_4_GS'


        # col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_1_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_4_GS'
        plt.figure(figsize=(10, 5))
        flag = 0
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            flag += 1
            dff = join(df_dir,f)
            df = T.load_df(dff)
            print(df.columns)
            # exit()
            df = df[df['AI_class'] == 'Arid']
            # T.print_head_n(df)
            # plt.subplot(2,2,flag)
            df_year_dict = T.df_groupby(df,'drought_year')
            year_list = df_year_dict.keys()
            year_list = sorted(year_list)
            mean_list = []
            err_list = []
            for year in year_list:
                df_year = df_year_dict[year]
                vals = df_year[col_name].tolist()
                vals_mean = np.nanmean(vals)
                # vals_std = np.nanstd(vals)
                vals_std,_,_ = T.uncertainty_err(vals)
                mean_list.append(vals_mean)
                # mean_list.append(len(vals))
                err_list.append(vals_std)
            plt.plot(year_list,mean_list,label=f,lw=2)
            # plt.ylim(0.85,1.1)
            # plt.title(f)
            # plt.suptitle(col_name)
        plt.legend()
        plt.title(col_name)
        # plt.ylim(0.87,1.03)
        # plt.ylim(0.92,1.07)
        outf = join(outdir,f'{col_name}.pdf')
        # plt.savefig(outf,dpi=600)
        # plt.close()
        plt.tight_layout()
        plt.show()

    def temporal_change_percentage_bar(self):
        df_dir = self.df_dir
        outdir = join(self.this_class_png,'temporal_change_percentage')
        T.mk_dir(outdir)

        # col_name = 'NDVI4g_climatology_percentage_rt_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_1_GS'
        col_name = 'NDVI4g_climatology_percentage_rs_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_4_GS'


        # col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_1_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_4_GS'
        flag = 0
        for f in T.listdir(df_dir):
            if not f.endswith('.df'):
                continue
            flag += 1
            dff = join(df_dir,f)
            df = T.load_df(dff)
            print(df.columns)
            # exit()
            # T.print_head_n(df)
            # plt.subplot(2,2,flag)
            df_year_dict = T.df_groupby(df,'drought_year')
            year_list = df_year_dict.keys()
            year_list = sorted(year_list)
            mean_list = []
            err_list = []
            # vals_bins = [-np.inf,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,np.inf]
            vals_bins = [-np.inf,0.8,0.85,0.9,0.95,1]
            ratio_list_all = []
            for year in year_list:
                df_year = df_year_dict[year]
                vals = df_year[col_name].tolist()
                ratio_list = []
                for i in range(len(vals_bins)):
                    if i == len(vals_bins)-1:
                        break
                    bin_start = vals_bins[i]
                    bin_end = vals_bins[i+1]
                    vals_i = []
                    for val in vals:
                        if bin_start<=val<bin_end:
                            vals_i.append(val)
                    ratio = len(vals_i)/len(vals) * 100
                    ratio_list.append(ratio)
                ratio_list_all.append(ratio_list)
            ratio_list_all = np.array(ratio_list_all)
            ratio_list_all = ratio_list_all.T
            # plot_bar
            color_list = T.gen_colors(len(vals_bins)-1,palette='Reds')[::-1]
            plt.figure(figsize=(10, 5))

            for i in range(len(vals_bins)-1):
                print(vals_bins[i],vals_bins[i+1])
                plt.bar(year_list,ratio_list_all[i],bottom=np.sum(ratio_list_all[:i],axis=0),
                        label=f'{vals_bins[i]}-{vals_bins[i+1]}',
                        color=color_list[i])
            # plt.plot(year_list,mean_list,label=f,lw=2)
            plt.legend()
            plt.title(f'{col_name} {f}')
            plt.show()


    def tif_spatial_trend(self):
        dff_merge = self.dff_merge
        df_merge = T.load_df(dff_merge)
        outdir = join(self.this_class_tif,'spatial_trend_merge')
        T.mk_dir(outdir)
        pix_list = T.get_df_unique_val_list(df_merge,'pix')
        print('groupby pix')
        df_group_dict = T.df_groupby(df_merge,'pix')
        print('done')
        col_list = [
            'NDVI4g_climatology_percentage_rt_GS',
            'NDVI4g_climatology_percentage_rs_1_GS',
            'NDVI4g_climatology_percentage_rs_2_GS',
            'NDVI4g_climatology_percentage_rs_3_GS',
            'NDVI4g_climatology_percentage_rs_4_GS',
        ]
        for col in col_list:
            trend_spatial_dict = {}
            trend_p_spatial_dict = {}
            for pix in tqdm(df_group_dict,desc=col):
                df_pix = df_group_dict[pix]
                vals = df_pix[col].tolist()
                drought_range_list = df_pix['drought_range'].tolist()
                drought_start_list = []
                for dr in drought_range_list:
                    s,e = dr
                    drought_start_list.append(s)
                a,b,r,p = T.nan_line_fit(drought_start_list,vals)
                trend_spatial_dict[pix] = a
                trend_p_spatial_dict[pix] = p
            outf = join(outdir,f'{col}.tif')
            outf_p = join(outdir,f'{col}_p.tif')
            D.pix_dic_to_tif(trend_spatial_dict,outf)
            D.pix_dic_to_tif(trend_p_spatial_dict,outf_p)
        T.open_path_and_file(outdir)

    def tif_spatial_trend_normal_hot(self):
        dff_merge = self.dff_merge
        df_merge = T.load_df(dff_merge)
        df_merge = df_merge[df_merge['AI_class']=='Arid']
        outdir = join(self.this_class_tif,'spatial_trend_normal_hot')
        T.mk_dir(outdir)
        pix_list = T.get_df_unique_val_list(df_merge,'pix')
        col_list = [
            'NDVI4g_climatology_percentage_rt_GS',
            'NDVI4g_climatology_percentage_rs_1_GS',
            'NDVI4g_climatology_percentage_rs_2_GS',
            'NDVI4g_climatology_percentage_rs_3_GS',
            'NDVI4g_climatology_percentage_rs_4_GS',
        ]
        drought_type_list = global_drought_type_list
        for drt in drought_type_list:
            df_drt = df_merge[df_merge['drought_type']==drt]
            df_group_dict = T.df_groupby(df_drt, 'pix')
            for col in col_list:
                trend_spatial_dict = {}
                trend_p_spatial_dict = {}
                for pix in tqdm(df_group_dict,desc=col):
                    df_pix = df_group_dict[pix]
                    vals = df_pix[col].tolist()
                    drought_range_list = df_pix['drought_range'].tolist()
                    drought_start_list = []
                    for dr in drought_range_list:
                        s,e = dr
                        drought_start_list.append(s)
                    a,b,r,p = T.nan_line_fit(drought_start_list,vals)
                    trend_spatial_dict[pix] = a
                    trend_p_spatial_dict[pix] = p
                outf = join(outdir,f'{drt}_{col}.tif')
                outf_p = join(outdir,f'{drt}_{col}_p.tif')
                D.pix_dic_to_tif(trend_spatial_dict,outf)
                D.pix_dic_to_tif(trend_p_spatial_dict,outf_p)
        T.open_path_and_file(outdir)

    def plot_sptatial_trend(self):
        fdir = join(self.this_class_tif,'spatial_trend_merge')
        outdir = join(self.this_class_png,'spatial_trend_merge')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if f.endswith('_p.tif'):
                continue
            if not f.endswith('.tif'):
                continue
            print(f)
            fpath = join(fdir,f)
            out_f = join(outdir,f.replace('.tif','.png'))
            Plot().plot_Robinson(fpath,vmin=-0.0002,vmax=0.0002)
            plt.title(f.replace('.tif',''))
            plt.savefig(out_f,dpi=600)
            plt.close()
        T.open_path_and_file(outdir)

    def rs_trend_vs_NDVI_trend_scatter(self):
        NDVI_trend_f = join(Spatial_Trends().this_class_tif,'NDVI','NDVI.tif')
        # rs_trend_f = join(self.this_class_tif,'spatial_trend_merge','NDVI4g_climatology_percentage_detrend_post_2_GS.tif')
        rs_trend_f = join(self.this_class_tif,'spatial_trend_merge','NDVI4g_climatology_percentage_rt_GS.tif')

        spatial_dict_NDVI = D.spatial_tif_to_dic(NDVI_trend_f)
        spatial_dict_rs = D.spatial_tif_to_dic(rs_trend_f)
        df = T.spatial_dics_to_df({'NDVI_trends':spatial_dict_NDVI,'rs_trends':spatial_dict_rs})
        df = df.dropna()
        NDVI_trends = df['NDVI_trends'].tolist()
        rs_trends = df['rs_trends'].tolist()
        a,b,r,p = T.nan_line_fit(NDVI_trends,rs_trends)
        print(a,b,r,p)
        KDE_plot().plot_scatter(NDVI_trends,rs_trends,s=10)
        plt.show()
        # plt.scatter(NDVI_trends,rs_trends)
        # plt.show()

    def hot_normal_delta(self):
        fdir = join(self.this_class_tif,'tif_hot_normal_rt_merge')
        outdir = join(self.this_class_tif,'tif_hot_rt_merge_delta')
        T.mk_dir(outdir)
        hot_f = join(fdir,'hot.tif')
        normal_f = join(fdir,'normal.tif')
        hot_arr = D.spatial_tif_to_arr(hot_f)
        normal_arr = D.spatial_tif_to_arr(normal_f)
        delta = hot_arr - normal_arr
        outf = join(outdir,'delta.tif')
        D.arr_to_tif(delta,outf)

    def plot_hot_normal_delta(self):
        fpath = join(self.this_class_tif,'tif_hot_rt_merge_delta','delta.tif')
        outdir = join(self.this_class_png,'hot_normal_delta')
        T.mk_dir(outdir)
        Plot().plot_Robinson(fpath,vmin=-0.1,vmax=0.1)
        outf = join(outdir,'delta.png')
        plt.title('RT hot-normal')
        plt.savefig(outf,dpi=600)
        plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

    def statistic_pft_hot_normal_delta(self):
        import analysis
        fpath = join(self.this_class_tif,'tif_hot_rt_merge_delta','delta.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.spatial_dics_to_df({'rt':spatial_dict})
        df = analysis.Dataframe_func(df).df
        T.print_head_n(df)
        col_name = 'rt'
        lc_list = global_lc_list
        mean_list = []
        std_list = []
        for lc in lc_list:
            df_lc = df[df['landcover_GLC']==lc]
            vals = df_lc[col_name].tolist()
            mean = np.nanmean(vals)
            std = T.uncertainty_err(vals)[0]
            std = abs(std)
            mean_list.append(mean)
            std_list.append(std)
        plt.bar(lc_list,mean_list,yerr=std_list)
        plt.show()

    def statistic_AI_hot_normal_delta(self):
        import analysis
        fpath = join(self.this_class_tif,'tif_hot_rt_merge_delta','delta.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.spatial_dics_to_df({'rt':spatial_dict})
        df = analysis.Dataframe_func(df).df
        T.print_head_n(df)
        col_name = 'rt'
        AI_bins = np.arange(0.1,0.66,0.02)
        df_group, bins_list_str = T.df_bin(df,'aridity_index',AI_bins)
        AI_list = []
        mean_list = []
        for name,df_group_i in df_group:
            AI = name[0].left
            vals = df_group_i[col_name].tolist()
            mean = np.nanmean(vals)
            AI_list.append(AI)
            mean_list.append(mean)
        plt.plot(AI_list,mean_list)
        plt.ylabel('RT')
        plt.xlabel('AI')
        plt.show()

    def statistic_BNPP_hot_normal_delta(self):
        fpath = join(self.this_class_tif,'tif_hot_rt_merge_delta','delta.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.spatial_dics_to_df({'rt':spatial_dict})
        # df = analysis.Dataframe_func(df).df
        fpath = join(data_root, 'BNPP/tif_025/BNPP_0-200cm.tif')
        spatial_dict = D.spatial_tif_to_dic(fpath)
        df = T.add_spatial_dic_to_df(df, spatial_dict, 'BNPP')
        df = df.dropna()
        T.print_head_n(df)
        # exit()
        col_name = 'rt'
        BNPP_bins = np.arange(1,4,0.1)
        df_group, bins_list_str = T.df_bin(df,'BNPP',BNPP_bins)
        AI_list = []
        mean_list = []
        for name,df_group_i in df_group:
            AI = name[0].left
            vals = df_group_i[col_name].tolist()
            mean = np.nanmean(vals)
            AI_list.append(AI)
            mean_list.append(mean)
        plt.plot(AI_list,mean_list)
        plt.ylabel('RT hot-normal')
        plt.xlabel('BNPP')
        plt.show()


class Matrix_T_SM_rs:

    def __init__(self):
        import analysis
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir(f'Matrix_T_SM_rs', result_root_this_script, mode=2)
        # T.mk_class_dir('Hot_normal_drought_double_detrend', result_root_this_script, mode=2)
        # T.mk_class_dir('Hot_normal_drought', result_root_this_script, mode=2)
        # T.mk_class_dir('Hot_normal_drought_double_detrend', result_root_this_script, mode=2)
        self.dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        self.df_dir = analysis.Dataframe_SM().df_dir
        pass

    def run(self):
        self.rs()
        pass

    def rs(self):
        df_merge = T.load_df(self.dff_merge)
        # pix_list = T.get_df_unique_val_list(df_merge,'pix')
        # spatial_dict = {}
        # for pix in pix_list:
        #     spatial_dict[pix] = 1
        # arr = D.pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        print(df_merge.columns.tolist())
        # exit()
        # T.print_head_n(df_merge)
        threshold_list = T.get_df_unique_val_list(df_merge,'threshold')
        T_level_list = T.get_df_unique_val_list(df_merge,'T_level')
        print(threshold_list)
        print(T_level_list)
        T_level_list = np.array(T_level_list)
        T_level_list = T_level_list[T_level_list>0]
        T_level_list = T_level_list[T_level_list<3]
        # col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        col_name = 'NDVI4g_climatology_percentage_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        matrix = []
        for threshold in threshold_list:
            df_threshold = df_merge[df_merge['threshold']==threshold]
            matrix_i = []
            for T_level in T_level_list:
                df_T_level = df_threshold[df_threshold['T_level']==T_level]
                vals_list = df_T_level[col_name].tolist()
                if len(vals_list) == 0:
                    matrix_i.append(np.nan)
                    continue
                val_mean = np.nanmean(vals_list)
                matrix_i.append(val_mean)
                # matrix_i.append(len(vals_list))
            matrix.append(matrix_i)
        matrix = np.array(matrix)
        # plt.imshow(matrix,aspect='auto',cmap='RdBu',vmin=0.98,vmax=1.02)
        plt.imshow(matrix,aspect='auto',cmap='RdBu',vmin=0.9,vmax=1.)
        # plt.imshow(matrix,aspect='auto',cmap='jet')
        plt.xticks(np.arange(len(T_level_list))[::5],T_level_list[::5])
        plt.yticks(np.arange(len(threshold_list)),threshold_list)
        plt.xlabel('T_anomaly')
        plt.ylabel('SM_anomaly')
        plt.colorbar()
        plt.title(col_name)
        plt.show()


        pass

class Spatial_Trends:
    def __init__(self):
        # import analysis
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Spatial_Trends', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.NDVI_tif()
        self.SM_tif()
        self.Tair_tif()
        # self.NDVI_tif()
        # self.plot_NDVI_tif()
        # self.NDVI_ts()
        # self.NDVI_ts_detrend()
        pass

    def NDVI_tif(self):
        import analysis
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df_merge = T.load_df(dff)
        df_merge = df_merge[df_merge['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_merge,'pix')
        data_dict, data_name = Load_Data().NDVI4g_origin()
        outdir = join(self.this_class_tif,'NDVI')
        T.mk_dir(outdir)
        trend_dict = {}
        for pix in tqdm(pix_list,desc='NDVI'):
            vals = data_dict[pix]
            vals = np.array(vals)
            if not pix in data_dict:
                continue
            if T.is_all_nan(vals):
                continue
            GS = global_get_gs(pix)
            annual_vals = T.monthly_vals_to_annual_val(vals,grow_season=GS)
            a,b,r,p = T.nan_line_fit(np.arange(len(annual_vals)),annual_vals)
            trend_dict[pix] = {
                'a':a,
                'p':p,
            }
        outf = join(outdir,'NDVI.tif')
        outf_p = join(outdir,'NDVI_p.tif')
        df = T.dic_to_df(trend_dict,'pix')
        spatial_dict = T.df_to_spatial_dic(df,'a')
        spatial_dict_p = T.df_to_spatial_dic(df,'p')
        D.pix_dic_to_tif(spatial_dict,outf)
        D.pix_dic_to_tif(spatial_dict_p,outf_p)

    def plot_NDVI_tif(self):
        fdir = join(self.this_class_tif,'NDVI')
        fpath = join(fdir,'NDVI.tif')
        outdir = join(self.this_class_png,'NDVI')
        T.mk_dir(outdir)
        outpath = join(outdir,'NDVI.png')
        Plot().plot_Robinson(fpath,vmin=-30,vmax=30)
        plt.title('NDVI trend')
        plt.savefig(outpath,dpi=600)
        plt.close()
        T.open_path_and_file(outdir)

    def NDVI_ts(self):
        outdir = join(self.this_class_png,'NDVI_ts')
        T.mk_dir(outdir)
        import analysis
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df_merge = T.load_df(dff)
        df_merge = df_merge[df_merge['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_merge, 'pix')
        data_dict, data_name = Load_Data().NDVI4g_origin()
        all_vals = []
        for pix in tqdm(pix_list, desc='NDVI'):
            vals = data_dict[pix]
            vals = np.array(vals)
            if not pix in data_dict:
                continue
            if T.is_all_nan(vals):
                continue
            GS = global_get_gs(pix)
            annual_vals = T.monthly_vals_to_annual_val(vals, grow_season=GS)
            all_vals.append(annual_vals)
        all_vals = np.array(all_vals)
        all_vals_mean = np.nanmean(all_vals,axis=0)
        year_list = global_year_range_list
        plt.figure(figsize=(10, 5))
        plt.plot(year_list,all_vals_mean,c='k',lw=3,label='NDVI',linestyle='dashed',zorder=99)
        # plt.legend()
        # plt.show()
        outf = join(outdir,'NDVI_ts.pdf')
        plt.savefig(outf,dpi=600)
        plt.close()
        T.open_path_and_file(outdir)

    def NDVI_ts_detrend(self):
        outdir = join(self.this_class_png,'NDVI_ts_detrend')
        T.mk_dir(outdir)
        import analysis
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df_merge = T.load_df(dff)
        df_merge = df_merge[df_merge['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_merge, 'pix')
        data_dict, data_name = Load_Data().NDVI4g_origin()
        all_vals = []
        for pix in tqdm(pix_list, desc='NDVI'):
            vals = data_dict[pix]
            vals = np.array(vals)
            if not pix in data_dict:
                continue
            if T.is_all_nan(vals):
                continue
            GS = global_get_gs(pix)
            annual_vals = T.monthly_vals_to_annual_val(vals, grow_season=GS)
            annual_vals_detrend = T.detrend_vals(annual_vals)
            all_vals.append(annual_vals_detrend)
        all_vals = np.array(all_vals)
        all_vals_mean = np.nanmean(all_vals,axis=0)
        year_list = global_year_range_list
        plt.figure(figsize=(10, 5))
        plt.plot(year_list,all_vals_mean,c='k',lw=3,label='NDVI',linestyle='dashed',zorder=99)
        # plt.legend()
        # plt.show()
        outf = join(outdir,'NDVI_ts.pdf')
        plt.savefig(outf,dpi=600)
        plt.close()
        T.open_path_and_file(outdir)

    def SM_tif(self):
        import analysis
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df_merge = T.load_df(dff)
        df_merge = df_merge[df_merge['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_merge,'pix')
        data_dict, data_name = Load_Data().ERA_SM_origin_GS()
        outdir = join(self.this_class_tif,'SM')
        T.mk_dir(outdir)
        trend_dict = {}
        for pix in tqdm(pix_list,desc='SM'):
            if not pix in data_dict:
                continue
            annual_vals = data_dict[pix]
            annual_vals = np.array(annual_vals)
            if not pix in data_dict:
                continue
            if T.is_all_nan(annual_vals):
                continue
            a,b,r,p = T.nan_line_fit(np.arange(len(annual_vals)),annual_vals)
            trend_dict[pix] = {
                'a':a,
                'p':p,
            }
        outf = join(outdir,'a.tif')
        outf_p = join(outdir,'p.tif')
        df = T.dic_to_df(trend_dict,'pix')
        spatial_dict = T.df_to_spatial_dic(df,'a')
        spatial_dict_p = T.df_to_spatial_dic(df,'p')
        D.pix_dic_to_tif(spatial_dict,outf)
        D.pix_dic_to_tif(spatial_dict_p,outf_p)

    def Tair_tif(self):
        import analysis
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df_merge = T.load_df(dff)
        df_merge = df_merge[df_merge['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_merge, 'pix')
        data_dict, data_name = Load_Data().ERA_Tair_origin_GS()
        outdir = join(self.this_class_tif, 'Tair')
        T.mk_dir(outdir)
        trend_dict = {}
        for pix in tqdm(pix_list, desc='Tair'):
            if not pix in data_dict:
                continue
            annual_vals = data_dict[pix]
            annual_vals = np.array(annual_vals)
            if not pix in data_dict:
                continue
            if T.is_all_nan(annual_vals):
                continue
            a, b, r, p = T.nan_line_fit(np.arange(len(annual_vals)), annual_vals)
            trend_dict[pix] = {
                'a': a,
                'p': p,
            }
        outf = join(outdir, 'a.tif')
        outf_p = join(outdir, 'p.tif')
        df = T.dic_to_df(trend_dict, 'pix')
        spatial_dict = T.df_to_spatial_dic(df, 'a')
        spatial_dict_p = T.df_to_spatial_dic(df, 'p')
        D.pix_dic_to_tif(spatial_dict, outf)
        D.pix_dic_to_tif(spatial_dict_p, outf_p)

class Greening_Resilience:

    def __init__(self):
        import analysis
        self.Dataframe_mode = analysis.Dataframe_SM().Dataframe_mode
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir(f'Greening_Resilience/{self.Dataframe_mode}', result_root_this_script, mode=2)
        self.dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        self.df_dir = analysis.Dataframe_SM().df_dir
        pass

    def run(self):
        # self.df_greening_rs_trend_mode()
        # self.tif_greening_rs_trend_mode()
        self.four_mode_NDVI_ts()
        # self.ratio_statistic()
        # self.ratio_statistic2()
        pass

    def df_greening_rs_trend_mode(self):
        outdir = join(self.this_class_arr,'df_greening_rs_trend_mode')
        T.mk_dir(outdir)
        outf = join(outdir,'df_greening_rs_trend_mode.df')
        greening_f = join(Spatial_Trends().this_class_tif,'NDVI/NDVI.tif')
        greening_p_f = join(Spatial_Trends().this_class_tif,'NDVI/NDVI_p.tif')
        rs_trend_f = join(Hot_normal_drought().this_class_tif,'spatial_trend_merge/NDVI4g_climatology_percentage_rs_2_GS.tif')
        rs_trend_p_f = join(Hot_normal_drought().this_class_tif,'spatial_trend_merge/NDVI4g_climatology_percentage_rs_2_GS_p.tif')
        print('building spatial dict')
        spatial_dict_greening = D.spatial_tif_to_dic(greening_f)
        spatial_dict_rs = D.spatial_tif_to_dic(rs_trend_f)
        spatial_dict_greening_p = D.spatial_tif_to_dic(greening_p_f)
        spatial_dict_rs_p = D.spatial_tif_to_dic(rs_trend_p_f)
        dict_all = {
            'greening':spatial_dict_greening,
            'greening_p':spatial_dict_greening_p,
            'rs_trend':spatial_dict_rs,
            'rs_trend_p':spatial_dict_rs_p,
        }
        print('building spatial df')
        df = T.spatial_dics_to_df(dict_all)
        mode_list = []
        for i,row in tqdm(df.iterrows(),total=len(df)):
            greening = row['greening']
            rs_trend = row['rs_trend']
            if greening > 0:
                if rs_trend < 0:
                    mode = 'greening_rs-decline'
                else:
                    mode = 'greening_rs-stable'
            else:
                if rs_trend < 0:
                    mode = 'browning_rs-decline'
                else:
                    mode = 'browning_rs-stable'
            mode_list.append(mode)
        df['mode'] = mode_list
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def tif_greening_rs_trend_mode(self):
        outdir = join(self.this_class_tif,'greening_rs_trend_mode')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'df_greening_rs_trend_mode','df_greening_rs_trend_mode.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        mode_list = T.get_df_unique_val_list(df,'mode')
        for mode in mode_list:
            outf = join(outdir,f'{mode}.tif')
            df_mode = df[df['mode']==mode]
            spatial_dict = {}
            pix_list = df_mode['pix'].tolist()
            for pix in pix_list:
                spatial_dict[pix] = 1
            D.pix_dic_to_tif(spatial_dict,outf)
        T.open_path_and_file(outdir)
        pass

    def four_mode_NDVI_ts(self):

        import analysis
        # col_name = 'NDVI4g_climatology_percentage_rt_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_1_GS'
        col_name = 'NDVI4g_climatology_percentage_rs_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_rs_4_GS'

        # col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_1_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_4_GS'
        outdir = join(self.this_class_png, 'four_mode_NDVI_ts',col_name)
        T.mk_dir(outdir,force=True)
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        NDVI_dict, _ = Load_Data().NDVI4g_climatology_percentage()
        df_pix_dict = T.df_groupby(df, 'pix')
        dff_greening_rs_trend_mode = join(self.this_class_arr,'df_greening_rs_trend_mode','df_greening_rs_trend_mode.df')
        df_greening_rs_trend_mode = T.load_df(dff_greening_rs_trend_mode)
        # T.print_head_n(df_greening_rs_trend_mode)
        mode_list = T.get_df_unique_val_list(df_greening_rs_trend_mode,'mode')
        for mode in mode_list:
            df_mode = df_greening_rs_trend_mode[df_greening_rs_trend_mode['mode']==mode]
            pix_list = T.get_df_unique_val_list(df_mode,'pix')
            NDVI_list = []
            for pix in tqdm(pix_list,desc=mode):
                NDVI = NDVI_dict[pix]
                if True in np.isnan(NDVI):
                    continue
                NDVI_list.append(NDVI)
            NDVI_mean = np.nanmean(NDVI_list,axis=0)
            plt.figure(figsize=(10,5))
            plt.plot(NDVI_mean,label=mode)
            plt.legend()
            num = len(pix_list)
            ratio = f'{num/len(df_greening_rs_trend_mode)*100:.2f} %'
            plt.title(f'{mode} {ratio}')
            outf = join(outdir,f'{mode}.pdf')
            plt.savefig(outf,dpi=600)
            plt.close()
        T.open_path_and_file(outdir)
        # plt.show()


    def ratio_statistic(self):
        import analysis
        dff = join(self.this_class_arr, 'df_greening_rs_trend_mode', 'df_greening_rs_trend_mode.df')
        df = T.load_df(dff)
        df = df.dropna()
        df = analysis.Dataframe_func(df).df
        landcover_GLC_list = T.get_df_unique_val_list(df, 'landcover_GLC')
        for lc in landcover_GLC_list:
            df_lc = df[df['landcover_GLC']==lc]
            ratio_rs_increase_sig, ratio_rs_increase_non_sig, ratio_rs_decline_sig, ratio_rs_decline_non_sig = self.__rs_trend_ratio(df_lc,len(df))
            plt.bar(lc,ratio_rs_increase_sig,bottom=0,color='g')
            plt.bar(lc,ratio_rs_increase_non_sig,bottom=ratio_rs_increase_sig,color='g',alpha=0.5)
            plt.bar(lc,ratio_rs_decline_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig,color='r')
            plt.bar(lc,ratio_rs_decline_non_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig+ratio_rs_decline_sig,color='r',alpha=0.5)
        plt.show()
        exit()


    def ratio_statistic3(self):
        outdir = join(self.this_class_png,'ratio_statistic')
        T.mk_dir(outdir)
        outf = join(outdir,'ratio_statistic.pdf')
        dff = join(self.this_class_arr,'df_greening_rs_trend_mode','df_greening_rs_trend_mode.df')
        df = T.load_df(dff)
        df = df.dropna()
        T.print_head_n(df)
        total_len = len(df)
        df_rs_increase_sig = df[(df['rs_trend_p']<0.05) & (df['rs_trend']>0)]
        ratio_greening_increase_sig, ratio_greening_increase_non_sig, ratio_greening_decline_sig, ratio_greening_decline_non_sig = self.__greening_trend_ratio(df_rs_increase_sig,total_len)
        plt.bar('df_rs_increase_sig',ratio_greening_increase_sig,bottom=0,color='g')
        plt.bar('df_rs_increase_sig',ratio_greening_increase_non_sig,bottom=ratio_greening_increase_sig,color='g',alpha=0.5)
        plt.bar('df_rs_increase_sig',ratio_greening_decline_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig,color='r')
        plt.bar('df_rs_increase_sig',ratio_greening_decline_non_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig+ratio_greening_decline_sig,color='r',alpha=0.5)

        df_rs_increase_non_sig = df[(df['rs_trend_p']>0.05) & (df['rs_trend']>0)]
        ratio_greening_increase_sig, ratio_greening_increase_non_sig, ratio_greening_decline_sig, ratio_greening_decline_non_sig = self.__greening_trend_ratio(df_rs_increase_non_sig,total_len)
        plt.bar('df_rs_increase_non_sig',ratio_greening_increase_sig,bottom=0,color='g')
        plt.bar('df_rs_increase_non_sig',ratio_greening_increase_non_sig,bottom=ratio_greening_increase_sig,color='g',alpha=0.5)
        plt.bar('df_rs_increase_non_sig',ratio_greening_decline_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig,color='r')
        plt.bar('df_rs_increase_non_sig',ratio_greening_decline_non_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig+ratio_greening_decline_sig,color='r',alpha=0.5)

        df_rs_decline_sig = df[(df['rs_trend_p']<0.05) & (df['rs_trend']<0)]
        ratio_greening_increase_sig, ratio_greening_increase_non_sig, ratio_greening_decline_sig, ratio_greening_decline_non_sig = self.__greening_trend_ratio(df_rs_decline_sig,total_len)
        plt.bar('df_rs_decline_sig',ratio_greening_increase_sig,bottom=0,color='g')
        plt.bar('df_rs_decline_sig',ratio_greening_increase_non_sig,bottom=ratio_greening_increase_sig,color='g',alpha=0.5)
        plt.bar('df_rs_decline_sig',ratio_greening_decline_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig,color='r')
        plt.bar('df_rs_decline_sig',ratio_greening_decline_non_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig+ratio_greening_decline_sig,color='r',alpha=0.5)

        df_rs_decline_non_sig = df[(df['rs_trend_p']>0.05) & (df['rs_trend']<0)]
        ratio_greening_increase_sig, ratio_greening_increase_non_sig, ratio_greening_decline_sig, ratio_greening_decline_non_sig = self.__greening_trend_ratio(df_rs_decline_non_sig,total_len)
        plt.bar('df_rs_decline_non_sig',ratio_greening_increase_sig,bottom=0,color='g',label='greening_increase_sig')
        plt.bar('df_rs_decline_non_sig',ratio_greening_increase_non_sig,bottom=ratio_greening_increase_sig,color='g',alpha=0.5,label='greening_increase_non_sig')
        plt.bar('df_rs_decline_non_sig',ratio_greening_decline_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig,color='r',label='greening_decline_sig')
        plt.bar('df_rs_decline_non_sig',ratio_greening_decline_non_sig,bottom=ratio_greening_increase_sig+ratio_greening_increase_non_sig+ratio_greening_decline_sig,color='r',alpha=0.5,label='greening_decline_non_sig')
        plt.legend()
        plt.ylabel('ratio')
        plt.xlabel('mode')
        plt.savefig(outf,dpi=600)
        plt.close()
        T.open_path_and_file(outdir)
        # plt.show()
        exit()
    def ratio_statistic2(self):
        import analysis
        dff = join(self.this_class_arr,'df_greening_rs_trend_mode','df_greening_rs_trend_mode.df')
        df = T.load_df(dff)
        df = df.dropna()
        df = analysis.Dataframe_func(df).df
        landcover_GLC_list = T.get_df_unique_val_list(df,'landcover_GLC')
        print(landcover_GLC_list)
        exit()

        T.print_head_n(df)
        exit()
        total_len = len(df)

        df_greening_sig = df[(df['greening_p']<0.05) & (df['greening']>0)]
        ratio_rs_increase_sig, ratio_rs_increase_non_sig, ratio_rs_decline_sig, ratio_rs_decline_non_sig = self.__rs_trend_ratio(df_greening_sig,total_len)
        plt.bar('df_greening_sig',ratio_rs_increase_sig,bottom=0,color='g')
        plt.bar('df_greening_sig',ratio_rs_increase_non_sig,bottom=ratio_rs_increase_sig,color='g',alpha=0.5)
        plt.bar('df_greening_sig',ratio_rs_decline_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig,color='r')
        plt.bar('df_greening_sig',ratio_rs_decline_non_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig+ratio_rs_decline_sig,color='r',alpha=0.5)
        df_greening_non_sig = df[(df['greening_p']>0.05) & (df['greening']>0)]

        ratio_rs_increase_sig, ratio_rs_increase_non_sig, ratio_rs_decline_sig, ratio_rs_decline_non_sig = self.__rs_trend_ratio(df_greening_non_sig,total_len)
        plt.bar('df_greening_non_sig',ratio_rs_increase_sig,bottom=0,color='g')
        plt.bar('df_greening_non_sig',ratio_rs_increase_non_sig,bottom=ratio_rs_increase_sig,color='g',alpha=0.5)
        plt.bar('df_greening_non_sig',ratio_rs_decline_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig,color='r')
        plt.bar('df_greening_non_sig',ratio_rs_decline_non_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig+ratio_rs_decline_sig,color='r',alpha=0.5)

        df_browning_sig = df[(df['greening_p']<0.05) & (df['greening']<0)]
        ratio_rs_increase_sig, ratio_rs_increase_non_sig, ratio_rs_decline_sig, ratio_rs_decline_non_sig = self.__rs_trend_ratio(df_browning_sig,total_len)
        plt.bar('df_browning_sig',ratio_rs_increase_sig,bottom=0,color='g')
        plt.bar('df_browning_sig',ratio_rs_increase_non_sig,bottom=ratio_rs_increase_sig,color='g',alpha=0.5)
        plt.bar('df_browning_sig',ratio_rs_decline_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig,color='r')
        plt.bar('df_browning_sig',ratio_rs_decline_non_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig+ratio_rs_decline_sig,color='r',alpha=0.5)

        df_browning_non_sig = df[(df['greening_p']>0.05) & (df['greening']<0)]
        ratio_rs_increase_sig, ratio_rs_increase_non_sig, ratio_rs_decline_sig, ratio_rs_decline_non_sig = self.__rs_trend_ratio(df_browning_non_sig,total_len)
        plt.bar('df_browning_non_sig',ratio_rs_increase_sig,bottom=0,color='g',label='rs_increase_sig')
        plt.bar('df_browning_non_sig',ratio_rs_increase_non_sig,bottom=ratio_rs_increase_sig,color='g',alpha=0.5,label='rs_increase_non_sig')
        plt.bar('df_browning_non_sig',ratio_rs_decline_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig,color='r',label='rs_decline_sig')
        plt.bar('df_browning_non_sig',ratio_rs_decline_non_sig,bottom=ratio_rs_increase_sig+ratio_rs_increase_non_sig+ratio_rs_decline_sig,color='r',alpha=0.5,label='rs_decline_non_sig')
        plt.legend()
        plt.show()
        # ratio_greening_sig_rs_increase_sig, ratio_greening_sig_rs_increase_non_sig, ratio_greening_sig_rs_decline_sig, ratio_greening_sig_rs_decline_non_sig \
        #     = self.__rs_trend_ratio(df_greening_sig)
        # plt.bar(['sig_rs_increase_sig','sig_rs_increase_non_sig','sig_rs_decline_sig','sig_rs_decline_non_sig'],
        #         [ratio_greening_sig_rs_increase_sig, ratio_greening_sig_rs_increase_non_sig, ratio_greening_sig_rs_decline_sig, ratio_greening_sig_rs_decline_non_sig])
        # plt.show()
        # df_greening_non_sig = df[(df['greening_mark']==1)&(df['greening_p_mark']==0)]
        #
        #
        # df_browning_sig = df[(df['greening_mark']==0)&(df['greening_p_mark']==1)]
        #
        #
        # df_browning_non_sig = df[(df['greening_mark']==0)&(df['greening_p_mark']==0)]
        exit()

    def __rs_trend_ratio(self,df,total_len):
        df_rs_increase_sig = df[(df['rs_trend_p']<0.05) & (df['rs_trend']>0)]
        df_rs_increase_non_sig = df[(df['rs_trend_p']>0.05) & (df['rs_trend']>0)]
        df_rs_decline_sig = df[(df['rs_trend_p']<0.05) & (df['rs_trend']<0)]
        df_rs_decline_non_sig = df[(df['rs_trend_p']>0.05) & (df['rs_trend']<0)]

        ratio_rs_increase_sig = len(df_rs_increase_sig)/total_len
        ratio_rs_increase_non_sig = len(df_rs_increase_non_sig)/total_len
        ratio_rs_decline_sig = len(df_rs_decline_sig)/total_len
        ratio_rs_decline_non_sig = len(df_rs_decline_non_sig)/total_len
        return ratio_rs_increase_sig,ratio_rs_increase_non_sig,ratio_rs_decline_sig,ratio_rs_decline_non_sig

    def __greening_trend_ratio(self,df,total_len):
        df_greening_increase_sig = df[(df['greening_p']<0.05) & (df['greening']>0)]
        df_greening_increase_non_sig = df[(df['greening_p']>0.05) & (df['greening']>0)]
        df_greening_decline_sig = df[(df['greening_p']<0.05) & (df['greening']<0)]
        df_greening_decline_non_sig = df[(df['greening_p']>0.05) & (df['greening']<0)]

        ratio_greening_increase_sig = len(df_greening_increase_sig)/total_len
        ratio_greening_increase_non_sig = len(df_greening_increase_non_sig)/total_len
        ratio_greening_decline_sig = len(df_greening_decline_sig)/total_len
        ratio_greening_decline_non_sig = len(df_greening_decline_non_sig)/total_len
        return ratio_greening_increase_sig,ratio_greening_increase_non_sig,ratio_greening_decline_sig,ratio_greening_decline_non_sig

    def ratio_statistic1(self):
        dff = join(self.this_class_arr,'df_greening_rs_trend_mode','df_greening_rs_trend_mode.df')
        df = T.load_df(dff)
        df = df.dropna()
        T.print_head_n(df)
        mode_p_list = []
        for i,row in df.iterrows():
            greening = row['greening']
            greening_p = row['greening_p']
            rs_trend = row['rs_trend']
            rs_trend_p = row['rs_trend_p']
            if greening > 0:
                if greening_p < 0.05:
                    if rs_trend < 0:
                        if rs_trend_p < 0.05:
                            mode = 'greening-sig_rs-decline-sig'
                        else:
                            mode = 'greening-sig_rs-decline-non-sig'
                    else:
                        if rs_trend_p < 0.05:
                            mode = 'greening-sig_rs-increase-sig'
                        else:
                            mode = 'greening-sig_rs-increase-non-sig'
                else:
                    if rs_trend < 0:
                        if rs_trend_p < 0.05:
                            mode = 'greening-non-sig_rs-decline-sig'
                        else:
                            mode = 'greening-non-sig_rs-decline-non-sig'
                    else:
                        if rs_trend_p < 0.05:
                            pass
                        else:
                            pass
            else:
                if greening_p < 0.05:
                    if rs_trend < 0:
                        if rs_trend_p < 0.05:
                            pass
                        else:
                            pass
                    else:
                        if rs_trend_p < 0.05:
                            pass
                        else:
                            pass
                else:
                    if rs_trend < 0:
                        if rs_trend_p < 0.05:
                            pass
                        else:
                            pass
                    else:
                        if rs_trend_p < 0.05:
                            pass
                        else:
                            pass

        exit()

class Sensitivity_Analysis:

    def __init__(self):
        self.this_class_arr, self.Sensitivity_Analysis, self.this_class_png = \
            T.mk_class_dir('Sensitivity_Analysis', result_root_this_script, mode=2)
        pass

    def run(self):
        self.NDVI_vs_T()
        self.NDVI_T_sens_vs_trend()
        pass

    def NDVI_vs_T(self):
        outdir = join(self.this_class_arr,'NDVI_vs_T')
        T.mk_dir(outdir)
        NDVI_spatial_dict, NDVI_name = Load_Data().NDVI4g_origin()
        T_spatial_dict, T_name = Load_Data().ERA_Tair_origin()
        import analysis
        dff = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')
        df_merge = T.load_df(dff)
        df_merge = df_merge[df_merge['AI_class'] == 'Arid']
        pix_list = T.get_df_unique_val_list(df_merge, 'pix')

        sensitivity_dict = {}
        for pix in tqdm(pix_list, desc='NDVI_vs_T'):
            NDVI_vals = NDVI_spatial_dict[pix]
            T_vals = T_spatial_dict[pix]
            if T.is_all_nan(NDVI_vals) or T.is_all_nan(T_vals):
                continue
            GS = global_get_gs(pix)
            NDVI_vals = T.monthly_vals_to_annual_val(NDVI_vals, grow_season=GS)
            T_vals = T.monthly_vals_to_annual_val(T_vals, grow_season=GS)
            a, b, r, p = T.nan_line_fit(T_vals, NDVI_vals)
            sensitivity_dict[pix] = {
                'a': a,
                'p': p,
                'r': r,
            }
        outf = join(outdir, 'NDVI_vs_T.df')
        df = T.dic_to_df(sensitivity_dict, 'pix')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)

        pass

    def NDVI_T_sens_vs_trend(self):
        sensitivity_dff = join(self.this_class_arr,'NDVI_vs_T','NDVI_vs_T.df')
        sensitivity_df = T.load_df(sensitivity_dff)
        trend_tif = join(Hot_normal_drought().this_class_tif,'spatial_trend_merge','NDVI4g_climatology_percentage_rt_GS.tif')
        # trend_tif = '/Volumes/NVME4T/hotdrought_CMIP/results/statistic/Hot_normal_drought/tif/spatial_trend_merge/NDVI4g_climatology_percentage_rt_GS.tif'
        NDVI_trend_dict = D.spatial_tif_to_dic(trend_tif)
        df = T.add_spatial_dic_to_df(sensitivity_df,NDVI_trend_dict,'rt_trend')
        NDVI_trend = df['rt_trend'].tolist()
        a = df['a'].tolist()
        KDE_plot().plot_scatter(NDVI_trend,a)
        plt.show()


        pass

class Partial_Dependence_Plots_SM:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Partial_Dependence_Plots_SM', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        self.variable_list()
        pass

    def run(self):
        # self.copy_df()
        # self.check_variables_ranges()
        # self.run_importance_merged()
        # self.run_importance_merged_extreme()
        # self.run_partial_dependence_plots_merged()
        # self.run_partial_dependence_plots_merged_extreme()

        # self.print_importance_merged()
        # self.plot_importance_merged()
        # self.plot_importance_merged_extreme()

        # self.plot_run_partial_dependence_plots_merge()
        # self.plot_run_partial_dependence_plots_merge_extreme()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')

        df = T.load_df(dff_merge)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def check_variables_ranges(self):
        dff = self.dff
        df = T.load_df(dff)
        df = self.__select_extreme(df)
        T.print_head_n(df)
        print(len(self.x_variable_list))
        # exit()
        flag = 1
        for var in self.x_variable_list:
            print(flag,var)
            vals = df[var].tolist()
            plt.subplot(4,4,flag)
            flag += 1
            plt.hist(vals,bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()

        pass

    def variable_list(self):
        self.x_variable_list = ['intensity', 'severity_mean', 'T_max',
                                'BNPP', 'MAP', 'MAT', 'S_SILT', 'SOC','rooting_depth',
                                'water_table_depth', 'Precipitation-origin_CV',
                                'VPD-anomaly_GS'
                                ]

        self.x_variable_range_dict = {
            'intensity':(-3,0),
            'severity_mean':(-2,1),
            'T_max':(-1,3),
            'BNPP':(0,4),
            'MAP':(0,1200),
            'MAT':(0,30),
            'S_SILT':(0,40),
            'SOC':(0,2000),
            'rooting_depth':(0,7),
            'water_table_depth':(0,600),
            # 'ERA_Tair_origin_CV':(0,0.004),
            'Precipitation-origin_CV':(0,0.5),
            'VPD-anomaly_GS':(-2,2),
        }

        self.x_variable_range_dict_extreme = {
            'intensity':(-3,-2),
            'severity_mean':(-2,-1.5),
            'T_max':(1,3),
            'BNPP':(0,4),
            'MAP':(0,1200),
            'MAT':(0,30),
            'S_SILT':(0,40),
            'SOC':(0,2000),
            'rooting_depth':(0,7),
            'water_table_depth':(0,600),
            # 'ERA_Tair_origin_CV':(0,0.004),
            'Precipitation-origin_CV':(0,0.5),
            'VPD-anomaly_GS':(-2,2),
        }



        # self.y_variable_list = [
        #     # 'NDVI4g_climatology_percentage_rs_1_GS',
        #    'NDVI4g_climatology_percentage_rs_2_GS',
        #    # 'NDVI4g_climatology_percentage_rs_3_GS',
        #    # 'NDVI4g_climatology_percentage_rs_4_GS',
        #    'NDVI4g_climatology_percentage_detrend_rt_pre_baseline_GS',
        # ]
        pass

    def valid_range_df(self,df,is_extreme):
        if is_extreme:
            x_variable_range_dict = self.x_variable_range_dict_extreme
        else:
            x_variable_range_dict = self.x_variable_range_dict
        print(len(df))
        for var in x_variable_range_dict:
            min,max = x_variable_range_dict[var]
            df = df[(df[var]>=min)&(df[var]<=max)]
        print(len(df))
        return df

    def run_importance_merged(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        dff = self.dff
        outdir = join(self.this_class_arr, 'importance_merged')
        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list
        # y_variable = self.y_variable_list[-1]
        # y_variable = 'NDVI4g_climatology_percentage_post_2_GS'
        y_variable = 'NDVI4g_climatology_percentage_detrend_GS'
        df = T.load_df(dff)
        df = self.valid_range_df(df,is_extreme=False)
        print(df.columns.tolist())
        print(len(df))
        T.print_head_n(df)
        print('-'*50)
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        all_vars = copy.copy(x_variable_list)  # copy the x variables
        all_vars.append(y_variable)  # add the y variable to the list
        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna()  # drop rows with missing values
        X = all_vars_df[x_variable_list]
        Y = all_vars_df[y_variable]
        model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        coef_ = model.feature_importances_  # get the importance of each variable
        coef_dic = dict(zip(x_variable_list, coef_))  # put the importance of each variable into a dictionary
        print(coef_dic)
        outf = join(outdir, f'{y_variable}.npy')
        T.save_npy(coef_dic, outf)

    def run_importance_merged_extreme(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        dff = self.dff
        outdir = join(self.this_class_arr, 'importance_merged_extreme')
        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list
        # y_variable = self.y_variable_list[-1]
        # y_variable = 'NDVI4g_climatology_percentage_post_2_GS'
        y_variable = 'NDVI4g_climatology_percentage_detrend_GS'
        df = T.load_df(dff)
        df = self.__select_extreme(df)
        df = self.valid_range_df(df,is_extreme=True)
        print(df.columns.tolist())
        print(len(df))
        T.print_head_n(df)
        print('-'*50)
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        all_vars = copy.copy(x_variable_list)  # copy the x variables
        all_vars.append(y_variable)  # add the y variable to the list
        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna()  # drop rows with missing values
        X = all_vars_df[x_variable_list]
        Y = all_vars_df[y_variable]
        model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        coef_ = model.feature_importances_  # get the importance of each variable
        coef_dic = dict(zip(x_variable_list, coef_))  # put the importance of each variable into a dictionary
        print(coef_dic)
        outf = join(outdir, f'{y_variable}.npy')
        T.save_npy(coef_dic, outf)

    def print_importance_merged(self):
        fdir = join(self.this_class_arr,'importance_merged')
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            coef_dic = T.load_npy(fpath)
            df = pd.DataFrame(coef_dic,index=['imp']).T
            df_sort = df.sort_values(by='imp',ascending=False)
            print(f)
            print(df_sort)
            print('-'*50)
            # exit()
        # exit()

    def plot_importance_merged(self):
        fdir = join(self.this_class_arr,'importance_merged')
        outdir = join(self.this_class_png,'importance_merged')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            coef_dic = T.load_npy(fpath)
            df = pd.DataFrame(coef_dic,index=['imp']).T
            df_sort = df.sort_values(by='imp',ascending=False)
            outf = join(outdir,f'{f}.pdf')
            x = df_sort.index.tolist()[::-1]
            y = df_sort['imp'].tolist()[::-1]
            # plt.figure(figsize=(10,5))
            plt.figure(figsize=(2, 5))

            plt.scatter(y, x, s=80)
            for i in range(len(x)):
                plt.plot([0, y[i]], [x[i], x[i]], color='k')
            plt.xlim(0, 0.16)
            plt.title(f)
            # plt.tight_layout()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def sort_importance_merged(self):
        fdir = join(self.this_class_arr,'importance_merged')
        outdir = join(self.this_class_png,'importance_merged')
        T.mk_dir(outdir,force=True)
        importance_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            coef_dic = T.load_npy(fpath)
            df = pd.DataFrame(coef_dic,index=['imp']).T
            df_sort = df.sort_values(by='imp',ascending=False)
            outf = join(outdir,f'{f}.pdf')
            x = df_sort.index.tolist()
            importance_dict[f] = x
        return importance_dict

    def plot_importance_merged_extreme(self):
        fdir = join(self.this_class_arr,'importance_merged_extreme')
        outdir = join(self.this_class_png,'importance_merged_extreme')
        T.mk_dir(outdir,force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            coef_dic = T.load_npy(fpath)
            df = pd.DataFrame(coef_dic,index=['imp']).T
            df_sort = df.sort_values(by='imp',ascending=False)
            outf = join(outdir,f'{f}.pdf')
            x = df_sort.index.tolist()[::-1]
            y = df_sort['imp'].tolist()[::-1]
            plt.figure(figsize=(2,5))

            plt.scatter(y,x,s=80)
            for i in range(len(x)):
                plt.plot([0,y[i]],[x[i],x[i]],color='k')
            plt.xlim(0,0.16)

            plt.title(f)
            # plt.tight_layout()
            # plt.show()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def sort_importance_merged_extreme(self):
        fdir = join(self.this_class_arr,'importance_merged_extreme')
        importance_dict = {}
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            coef_dic = T.load_npy(fpath)
            df = pd.DataFrame(coef_dic,index=['imp']).T
            df_sort = df.sort_values(by='imp',ascending=False)
            x = df_sort.index.tolist()
            importance_dict[f] = x
        return importance_dict


    def run_partial_dependence_plots_merged(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        dff = self.dff
        outdir = join(self.this_class_arr, 'partial_dependence_plots_merged')
        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list
        # y_variable = self.y_variable_list[-1]
        # y_variable = 'NDVI4g_climatology_percentage_post_2_GS'
        y_variable = 'NDVI4g_climatology_percentage_detrend_GS'
        df = T.load_df(dff)
        df = self.valid_range_df(df,is_extreme=False)
        # pix_list = T.get_df_unique_val_list(df,'pix')
        # spatial_dict = {}
        # for pix in pix_list:
        #     spatial_dict[pix] = 1
        # arr = D.pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        # exit()
        print(df.columns.tolist())
        # exit()
        # df = df[df['AI_class']=='Arid']
        print(len(df))
        # pix_list = T.get_df_unique_val_list(df,'pix')
        # spatial_dict = {}
        # for pix in pix_list:
        #     spatial_dict[pix] = 1
        # arr = D.pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr)
        # plt.show()
        T.print_head_n(df)
        print('-'*50)
        result_dic = self.partial_dependence_plots(df, x_variable_list, y_variable)
        outf = join(outdir, f'{y_variable}.npy')
        T.save_npy(result_dic, outf)

    def run_partial_dependence_plots_merged_extreme(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        dff = self.dff
        outdir = join(self.this_class_arr, 'partial_dependence_plots_merged_extreme')
        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list
        # y_variable = self.y_variable_list[-1]
        # y_variable = 'NDVI4g_climatology_percentage_post_2_GS'
        y_variable = 'NDVI4g_climatology_percentage_detrend_GS'
        df = T.load_df(dff)
        df = self.__select_extreme(df)
        df = self.valid_range_df(df,is_extreme=True)
        # pix_list = T.get_df_unique_val_list(df,'pix')
        # spatial_dict = {}
        # for pix in pix_list:
        #     spatial_dict[pix] = 1
        # arr = D.pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr,interpolation='nearest')
        # plt.show()
        # exit()
        print(df.columns.tolist())
        # exit()
        # df = df[df['AI_class']=='Arid']
        print(len(df))
        # pix_list = T.get_df_unique_val_list(df,'pix')
        # spatial_dict = {}
        # for pix in pix_list:
        #     spatial_dict[pix] = 1
        # arr = D.pix_dic_to_spatial_arr(spatial_dict)
        # plt.imshow(arr)
        # plt.show()
        T.print_head_n(df)
        print('-'*50)
        result_dic = self.partial_dependence_plots(df, x_variable_list, y_variable)
        outf = join(outdir, f'{y_variable}.npy')
        T.save_npy(result_dic, outf)


    def plot_run_partial_dependence_plots_merge(self):
        fdir = join(self.this_class_arr,'partial_dependence_plots_merged')
        outdir = join(self.this_class_png,'partial_dependence_plots_merged')
        T.mk_dir(outdir,force=True)
        ylim_dict = {
            'NDVI4g_climatology_percentage_detrend_GS.npy':(0.93,1.05),
            'NDVI4g_climatology_percentage_post_2_GS.npy':(0.98,1.03),
        }
        importance_sorted_dict = self.sort_importance_merged()
        for f in T.listdir(fdir):

            fpath = join(fdir,f)

            result_dict = T.load_npy(fpath)
            print(result_dict.keys())
            # print(result_dict)
            sorted_xlist = importance_sorted_dict[f]
            flag = 1
            plt.figure(figsize=(10,10))
            for key in sorted_xlist:
                if not key in result_dict:
                    continue
                result_dict_i = result_dict[key]
                x = result_dict_i['x']
                y = result_dict_i['y']
                y_std = result_dict_i['y_std']
                # if key in variable_range_dict:
                #     df_temp = pd.DataFrame({'x': x, 'y': y})
                #     bottom,top = variable_range_dict[key]
                #     df_temp = df_temp[(df_temp['x']>=bottom) & (df_temp['x']<=top)]
                #     x = df_temp['x'].tolist()
                #     y = df_temp['y'].tolist()
                y_std = result_dict_i['y_std'] / 4
                plt.subplot(4,4,flag)
                flag += 1
                y = SMOOTH().smooth_convolve(y,window_len=5)
                y_std = SMOOTH().smooth_convolve(y_std,window_len=5)
                plt.plot(x,y)
                plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                # plt.ylim(0.95,1.0)
                plt.xlabel(key)
                # y_std = y_std / 4
                # plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                # plt.legend()
                plt.ylim(ylim_dict[f])
                # plt.xlabel(key.replace('_vs_NDVI-anomaly_detrend_','\nsensitivity\n'))
            plt.suptitle(f)
            # plt.tight_layout()
            outf = join(outdir,f'{f}.pdf')
            # plt.show()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def plot_run_partial_dependence_plots_merge_extreme(self):
        fdir = join(self.this_class_arr,'partial_dependence_plots_merged_extreme')
        outdir = join(self.this_class_png,'partial_dependence_plots_merged_extreme')
        T.mk_dir(outdir,force=True)
        ylim_dict = {
            'NDVI4g_climatology_percentage_detrend_GS.npy':(0.9,1.0),
            'NDVI4g_climatology_percentage_post_2_GS.npy':(0.98,1.03),
        }
        importance_sorted_dict = self.sort_importance_merged_extreme()
        for f in T.listdir(fdir):
            # print(f)
            # exit()

            fpath = join(fdir,f)

            result_dict = T.load_npy(fpath)
            print(result_dict.keys())
            # print(result_dict)
            sorted_xlist = importance_sorted_dict[f]
            flag = 1
            plt.figure(figsize=(10,10))
            for key in sorted_xlist:
                if not key in result_dict:
                    continue
                result_dict_i = result_dict[key]
                x = result_dict_i['x']
                y = result_dict_i['y']
                y_std = result_dict_i['y_std']
                # if key in variable_range_dict:
                #     df_temp = pd.DataFrame({'x': x, 'y': y})
                #     bottom,top = variable_range_dict[key]
                #     df_temp = df_temp[(df_temp['x']>=bottom) & (df_temp['x']<=top)]
                #     x = df_temp['x'].tolist()
                #     y = df_temp['y'].tolist()
                y_std = result_dict_i['y_std'] / 4
                plt.subplot(4,4,flag)
                flag += 1
                y = SMOOTH().smooth_convolve(y,window_len=5)
                y_std = SMOOTH().smooth_convolve(y_std,window_len=5)
                plt.plot(x,y)
                plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                # plt.ylim(0.95,1.0)
                plt.xlabel(key)
                # y_std = y_std / 4
                # plt.fill_between(x,y-y_std,y+y_std,alpha=0.5)
                # plt.legend()
                plt.ylim(ylim_dict[f])
                # plt.xlabel(key.replace('_vs_NDVI-anomaly_detrend_','\nsensitivity\n'))
            plt.suptitle(f)
            # plt.tight_layout()
            outf = join(outdir,f'{f}.pdf')
            # plt.show()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def _random_forest_train(self, X, Y, variable_list):
        '''
        :param X: a dataframe of x variables
        :param Y: a dataframe of y variable
        :param variable_list: a list of x variables
        :return: details of the random forest model and the importance of each variable
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # split the data into training and testing
        clf = RandomForestRegressor(n_estimators=100, n_jobs=7) # build a random forest model
        clf.fit(X_train, Y_train) # train the model
        result = permutation_importance(clf, X_train, Y_train, scoring=None,
                                        n_repeats=50, random_state=1,
                                        n_jobs=7) # calculate the importance of each variable using permutation importance
        importances = result.importances_mean # get the importance of each variable
        importances_dic = dict(zip(variable_list, importances)) # put the importance of each variable into a dictionary
        labels = []
        importance = []
        for key in variable_list:
            labels.append(key)
            importance.append(importances_dic[key])
        y_pred = clf.predict(X_test) # predict the y variable using the testing data
        r_model = stats.pearsonr(Y_test, y_pred)[0] # calculate the correlation between the predicted y variable and the actual y variable
        mse = sklearn.metrics.mean_squared_error(Y_test, y_pred) # calculate the mean squared error
        score = clf.score(X_test, Y_test) # calculate the R^2
        return clf, importances_dic, mse, r_model, score, Y_test, y_pred

    def __add_sensitivity(self,df,season):
        import analysis
        fdir = join(analysis.Long_term_correlation().this_class_tif,
                    'seasonal_correlation')
        print('adding', season)
        for folder in T.listdir(fdir):
            fpath = join(fdir,folder,f'{folder}_{season}.tif')
            spatial_dict = DIC_and_TIF().spatial_tif_to_dic(fpath)
            df = T.add_spatial_dic_to_df(df,spatial_dict,f'{folder}_{season}')
        return df


    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})

    def __select_extreme(self,df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df


class SHAP:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('SHAP', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        self.dff_sample = join(self.this_class_arr, 'sample_dataframe/sample_dataframe.df')
        self.variable_list()
        # self.y_variable = 'NDVI4g_climatology_percentage_detrend_GS'
        self.y_variable = 'NDVI4g_climatology_percentage_post_2_GS'

        pass

    def run(self):
        # self.copy_df()
        # self.check_variables_ranges()
        self.pdp_shap()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')

        df = T.load_df(dff_merge)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)


    def check_variables_ranges(self):
        dff = self.dff
        df = T.load_df(dff)
        df = self.__select_extreme(df)
        T.print_head_n(df)
        print(len(self.x_variable_list))
        # exit()
        flag = 1
        for var in self.x_variable_list:
            print(flag,var)
            vals = df[var].tolist()
            plt.subplot(4,4,flag)
            flag += 1
            plt.hist(vals,bins=100)
            plt.title(var)
        plt.tight_layout()
        plt.show()

        pass

    def variable_list(self):
        self.x_variable_list = ['intensity', 'severity_mean', 'T_max',
                                'BNPP', 'MAP', 'MAT', 'S_SILT', 'SOC','rooting_depth',
                                'water_table_depth', 'Precipitation-origin_CV',
                                'VPD-anomaly_GS'
                                ]

        self.x_variable_range_dict = {
            'intensity':(-3,0),
            'severity_mean':(-2,1),
            'T_max':(-1,3),
            'BNPP':(0,4),
            'MAP':(0,1200),
            'MAT':(0,30),
            'S_SILT':(0,40),
            'SOC':(0,2000),
            'rooting_depth':(0,7),
            'water_table_depth':(0,600),
            # 'ERA_Tair_origin_CV':(0,0.004),
            'Precipitation-origin_CV':(0,0.5),
            'VPD-anomaly_GS':(-2,2),
        }

        self.x_variable_range_dict_extreme = {
            'intensity':(-3,-2),
            'severity_mean':(-2,-1.5),
            'T_max':(-1,3),
            'BNPP':(0,4),
            'MAP':(0,1200),
            'MAT':(0,30),
            'S_SILT':(0,40),
            'SOC':(0,2000),
            'rooting_depth':(0,7),
            'water_table_depth':(0,600),
            # 'ERA_Tair_origin_CV':(0,0.004),
            'Precipitation-origin_CV':(0,0.5),
            'VPD-anomaly_GS':(-2,2),
        }



        # self.y_variable_list = [
        #     # 'NDVI4g_climatology_percentage_rs_1_GS',
        #    'NDVI4g_climatology_percentage_rs_2_GS',
        #    # 'NDVI4g_climatology_percentage_rs_3_GS',
        #    # 'NDVI4g_climatology_percentage_rs_4_GS',
        #    'NDVI4g_climatology_percentage_detrend_rt_pre_baseline_GS',
        # ]
        pass

    def valid_range_df(self,df,is_extreme):
        if is_extreme:
            x_variable_range_dict = self.x_variable_range_dict_extreme
        else:
            x_variable_range_dict = self.x_variable_range_dict
        print(len(df))
        for var in x_variable_range_dict:
            min,max = x_variable_range_dict[var]
            df = df[(df[var]>=min)&(df[var]<=max)]
        print(len(df))
        return df

    def pdp_shap(self):
        # fdir = join(Random_Forests_delta().this_class_arr, 'seasonal_delta')
        dff = self.dff
        # outdir = join(self.this_class_arr, 'model_file')
        # T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list
        # y_variable = self.y_variable_list[-1]
        # y_variable = 'NDVI4g_climatology_percentage_post_2_GS'
        y_variable = self.y_variable
        df = T.load_df(dff)
        df = self.__select_extreme(df)
        df = self.valid_range_df(df, is_extreme=True)
        print(df.columns.tolist())
        print(len(df))
        T.print_head_n(df)
        print('-'*50)
        # exit()
        # model, r2 = self.__train_model(X, Y)  # train a Random Forests model
        all_vars = copy.copy(x_variable_list)  # copy the x variables
        all_vars.append(y_variable)  # add the y variable to the list
        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna()  # drop rows with missing values
        X = all_vars_df[x_variable_list]
        Y = all_vars_df[y_variable]
        Y = Y * 100
        model,y,y_pred = self.__train_model(X, Y)  # train a Random Forests model
        # explainer = shap.Explainer(model)
        explainer = shap.TreeExplainer(model)

        y_base = explainer.expected_value
        print('y_base', y_base)
        print('y_mean', np.mean(y))
        # shap_values = explainer.shap_values(X)
        shap_values = explainer(X)
        flag = 1
        for x_var in x_variable_list:
            shap_values_mat = shap_values[:,x_var]
            data_i = shap_values_mat.data
            value_i = shap_values_mat.values
            df_i = pd.DataFrame({x_var:data_i,'shap_v':value_i})
            x_variable_range_dict = self.x_variable_range_dict
            start,end = x_variable_range_dict[x_var]
            bins = np.linspace(start,end,50)
            df_group, bins_list_str = T.df_bin(df_i,x_var,bins)
            y_mean_list = []
            x_mean_list = []
            y_err_list = []
            for name, df_group_i in df_group:
                x_i = name[0].left
                # print(x_i)
                # exit()
                vals = df_group_i['shap_v'].tolist()

                if len(vals) == 0:
                    continue
                mean = np.nanmean(vals)
                err = np.nanstd(vals)
                y_mean_list.append(mean)
                x_mean_list.append(x_i)
                y_err_list.append(err)
            #     err,_,_ = self.uncertainty_err(SM)
            # print(df_i)
            # exit()
            plt.subplot(3,4,flag)
            plt.scatter(data_i, value_i, alpha=0.2,c='gray',marker='.',s=1,zorder=-1)
            # print(data_i[0])
            # exit()
            interp_model = interpolate.interp1d(x_mean_list, y_mean_list, kind='cubic')
            y_interp = interp_model(x_mean_list)
            plt.plot(x_mean_list,y_interp,c='red',alpha=1)

            # exit()
            # plt.fill_between(x_mean_list, np.array(y_mean_list) - np.array(y_err_list), np.array(y_mean_list) + np.array(y_err_list), alpha=0.3,color='red')
            plt.title(x_var)
            flag += 1
            plt.ylim(-6,6)
        plt.tight_layout()
        plt.show()

    def __select_extreme(self,df):
        # df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df
    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        # rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7) # build a random forest model
        # rf.fit(X_train, y_train) # train the model
        # r2 = rf.score(X_test,y_test)
        model = xgb.XGBRegressor(objective="reg:squarederror")
        model.fit(X, y)
        # model.fit(X_train, y_train)
        # Get predictions
        y_pred = model.predict(X)
        r = stats.pearsonr(y, y_pred)
        r2 = r[0] ** 2
        print('r2:', r2)

        return model,y,y_pred

    def benchmark_model(self,y,y_pred):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
        plt.scatter(y, y_pred)
        plt.plot([0.6, 1.2], [0.6, 1.2], color='r', linestyle='-', linewidth=2)
        plt.ylabel('Predicted', size=20)
        plt.xlabel('Actual', size=20)
        plt.xlim(0.6, 1.2)
        plt.ylim(0.6, 1.2)
        plt.show()

class Critical_P_and_T:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Critical_P_and_T', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe','dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # df = T.load_df(self.dff)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_Tair_juping)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_SM_juping)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_SM_origin)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_Tair_origin)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_SM_juping)
        # self.Temperature_unit_convert(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)
        # self.all_intensity()
        self.all_T_max()
        # self.multiregression()
        # self.MAT_MAP_multiregression_anomaly()
        # self.MAT_MAP_multiregression_origin()
        # self.apply_MAT_MAP_multiregression_anomaly_crit_T()
        # self.apply_MAT_MAP_multiregression_anomaly_crit_SM()
        # self.apply_MAT_MAP_multiregression_origin_crit_T()
        # self.plot_apply_MAT_MAP_multiregression_anomaly_crit_T()
        # self.statistic_crit_T()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')

        df = T.load_df(dff_merge)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

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

    def Temperature_unit_convert(self,df):
        col_name = 'ERA_Tair_origin_GS'
        vals = df[col_name].tolist()
        vals = [x - 273.15 for x in vals]
        df[col_name] = vals
        return df

    def all_intensity(self):
        df = T.load_df(self.dff)
        print(df.columns.tolist())
        exit()
        # T.print_head_n(df)
        rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_3_GS'
        rs_col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'

        intensity = df['intensity'].tolist()
        rs_vals_list = df[rs_col_name].tolist()
        rt_vals_list = df[rt_col_name].tolist()
        plt.figure()

        self.plot_fit_line(intensity, rs_vals_list, rs_col_name,'SM anomaly')
        self.plot_fit_line(intensity, rt_vals_list, rt_col_name,'SM anomaly')
        plt.legend()
        # plt.show()

    def all_T_max(self):
        df = T.load_df(self.dff)
        print(df.columns.tolist())
        # T.print_head_n(df)
        rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'

        T_max = df['T_max'].tolist()
        rs_vals_list = df[rs_col_name].tolist()
        rt_vals_list = df[rt_col_name].tolist()
        plt.figure()

        self.plot_fit_line(T_max, rs_vals_list, rs_col_name,'T_max')
        self.plot_fit_line(T_max, rt_vals_list, rt_col_name,'T_max')
        plt.legend()
        plt.show()

    def plot_fit_line(self,intensity, vals_list, col_name,x_label):
        std = np.nanstd(vals_list)
        a, b, r, p = T.nan_line_fit(intensity, vals_list)
        x_list = np.linspace(np.min(intensity), np.max(intensity), 100)
        y_list = [a * x + b for x in x_list]
        upper_bound_0 = y_list[0] + std
        lower_bound_0 = y_list[0] - std
        upper_bound_1 = y_list[-1] + std
        lower_bound_1 = y_list[-1] - std
        y_upper_list = [upper_bound_0, lower_bound_1]
        y_lower_list = [lower_bound_0, upper_bound_1]
        x_list_bound = [x_list[0], x_list[-1]]
        x_list = np.array(x_list)

        # plt.scatter(intensity,vals_list)
        plt.plot(x_list, y_list, label=f'{col_name}\n{r:.2f}')
        plt.fill_between(x_list_bound, y_upper_list, y_lower_list, alpha=0.5)
        plt.xlabel(x_label)
        pass

    def multiregression(self):
        outdir = join(self.this_class_png,'multiregression')
        T.mk_dir(outdir,force=True)
        df = T.load_df(self.dff)
        print(df.columns.tolist())
        # T.print_head_n(df)
        # rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'

        # df_pix = df[df['pix']==pix]
        # df_pix = df_pix[df_pix[col_name] < 1.5]
        # df_pix = df_pix[df_pix[col_name] > 0.5]
        sm_col = 'intensity'
        T_col = 'ERA_Tair_juping_GS'
        # df = df.dropna(subset=[col_name,'ERA_SM_juping_GS','ERA_Tair_juping_GS'],how='any')
        df = df.dropna(subset=[col_name,sm_col,T_col],how='any')
        intensity = df[sm_col].tolist()
        intensity = np.array(intensity)
        T_max = df[T_col].tolist()

        vals_list = df[col_name].tolist()
        reg = LinearRegression()
        # plt.hist(T_max,bins=100)
        # plt.show()
        X = np.array([intensity,T_max]).T
        y = np.array(vals_list)
        reg.fit(X,y)
        intensity_predict_range = np.linspace(-3,0,26)
        T_max_predict_range = np.linspace(-2.5,2.5,26)
        intensity_predict_range = [round(x,2) for x in intensity_predict_range]
        T_max_predict_range = [round(x,2) for x in T_max_predict_range]
        matrix = []
        for i in range(len(intensity_predict_range)):
            matrix_i = []
            for j in range(len(T_max_predict_range)):
                intensity_predict = intensity_predict_range[i]
                T_max_predict = T_max_predict_range[j]
                X_predict = [intensity_predict,T_max_predict]
                Y_predict = reg.predict([X_predict])[0]
                matrix_i.append(Y_predict)
            matrix.append(matrix_i)
        matrix = matrix[::-1]
        plt.imshow(matrix,cmap='RdBu',vmin=0.9,vmax=1)
        # plt.imshow(matrix,cmap='RdBu',vmin=0.98,vmax=1.0)
        plt.colorbar()
        plt.xticks(np.arange(len(T_max_predict_range)),T_max_predict_range,rotation=90)
        plt.yticks(np.arange(len(intensity_predict_range)),np.array(intensity_predict_range)[::-1])
        plt.xlabel(T_col)
        plt.ylabel(sm_col)
        plt.title(f'{col_name}')
        # plt.show()
        outf = join(outdir,f'{col_name}.pdf')
        plt.savefig(outf)
        plt.close()

    def MAT_MAP_multiregression_anomaly(self):
        outdir = join(self.this_class_arr, 'MAT_MAP_multiregression_anomaly')
        T.mk_dir(outdir, force=True)
        df = T.load_df(self.dff)
        for i in (df.columns.tolist()):
            print(i)
        # exit()
        # T.print_head_n(df)
        # rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # SM_col_name = 'ERA_SM_origin_GS'
        SM_col_name = 'intensity'
        T_col_name = 'ERA_Tair_juping_GS'
        MAT_bins = np.linspace(0,30,31)
        MAP_bins = np.linspace(0,1200,31)
        # print(MAP_bins)
        # exit()
        MAT_bins = [round(x,1) for x in MAT_bins]
        MAP_bins = [round(x,1) for x in MAP_bins]
        df_group_MAT, bins_list_str_MAT = T.df_bin(df,'MAT',MAT_bins)
        flag = -1
        result_dict = {}
        for MAT_name, df_group_MAT_i in tqdm(df_group_MAT):
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i, 'MAP', MAP_bins)
            y_list = []
            for MAP_name, df_group_MAP_i in df_group_MAP:
                if len(df_group_MAP_i) == 0:
                    continue
                MAT_left = MAT_name[0].left
                MAT_right = MAT_name[0].right
                MAP_left = MAP_name[0].left
                MAP_right = MAP_name[0].right
                flag += 1
                df_group_MAP_i = df_group_MAP_i.dropna(subset=[col_name, SM_col_name, T_col_name], how='any')
                SM_vals = df_group_MAP_i[SM_col_name].tolist()
                SM_vals = np.array(SM_vals)
                T_vals = df_group_MAP_i[T_col_name].tolist()

                vals_list = df_group_MAP_i[col_name].tolist()
                reg = LinearRegression()
                # plt.hist(T_vals,bins=100)
                # plt.show()
                X = np.array([SM_vals, T_vals]).T
                y = np.array(vals_list)
                reg.fit(X, y)
                pix_list = df_group_MAP_i['pix'].tolist()
                result_dict[flag] = {
                    'pix':pix_list,
                    'model':reg,
                    'MAT_left':MAT_left,
                    'MAT_right':MAT_right,
                    'MAP_left':MAP_left,
                    'MAP_right':MAP_right,
                }
        outf = join(outdir, f'{col_name}_reg.npy')
        T.save_npy(result_dict, outf)

    def MAT_MAP_multiregression_origin(self):
        outdir = join(self.this_class_arr, 'MAT_MAP_multiregression_origin')
        T.mk_dir(outdir, force=True)
        df = T.load_df(self.dff)
        for i in (df.columns.tolist()):
            print(i)
        # exit()
        # T.print_head_n(df)
        # rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        col_name = 'NDVI4g_climatology_percentage_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # SM_col_name = 'ERA_SM_origin_GS'
        SM_col_name = 'intensity'
        T_col_name = 'ERA_Tair_origin_GS'
        MAT_bins = np.linspace(0,30,31)
        MAP_bins = np.linspace(0,1200,31)
        # print(MAP_bins)
        # exit()
        MAT_bins = [round(x,1) for x in MAT_bins]
        MAP_bins = [round(x,1) for x in MAP_bins]
        df_group_MAT, bins_list_str_MAT = T.df_bin(df,'MAT',MAT_bins)
        flag = -1
        result_dict = {}
        for MAT_name, df_group_MAT_i in tqdm(df_group_MAT):
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i, 'MAP', MAP_bins)
            y_list = []
            for MAP_name, df_group_MAP_i in df_group_MAP:
                if len(df_group_MAP_i) == 0:
                    continue
                flag += 1
                df_group_MAP_i = df_group_MAP_i.dropna(subset=[col_name, SM_col_name, T_col_name], how='any')
                SM_vals = df_group_MAP_i[SM_col_name].tolist()
                T_vals = df_group_MAP_i[T_col_name].tolist()

                vals_list = df_group_MAP_i[col_name].tolist()
                reg = LinearRegression()
                # plt.hist(T_vals,bins=100)
                # plt.show()
                X = np.array([SM_vals, T_vals]).T
                y = np.array(vals_list)
                reg.fit(X, y)
                pix_list = df_group_MAP_i['pix'].tolist()
                result_dict[flag] = {
                    'pix':pix_list,
                    'model':reg,
                }
        outf = join(outdir, f'{col_name}_reg.npy')
        T.save_npy(result_dict, outf)

    def apply_MAT_MAP_multiregression_anomaly_crit_T(self):
        outdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_T')
        T.mk_dir(outdir)
        # col_name = 'NDVI4g_climatology_percentage_GS'
        col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        fpath = join(self.this_class_arr, 'MAT_MAP_multiregression_anomaly', f'{col_name}_reg.npy')
        result_dict = T.load_npy(fpath)
        # y = 0.95 # affected by drought
        y = 0.99 # affected by drought
        # sm_anomaly = -0.5 # mild drought
        sm_anomaly_list = [-0.5,-1,-1.5,-2,-2.5,-3]
        for sm_anomaly in sm_anomaly_list:
            spatial_dict = {}
            for i in result_dict:
                dict_i = result_dict[i]
                model = dict_i['model']
                coef = model.coef_
                a1,a2 = coef[0],coef[1]
                b = model.intercept_
                intercept = model.intercept_
                crit_T = (y - b - a1*sm_anomaly)/a2
                pix_list = dict_i['pix']
                for pix in pix_list:
                    spatial_dict[pix] = crit_T
            arr = D.pix_dic_to_spatial_arr(spatial_dict)
            ouf = join(outdir,f'{col_name}_{sm_anomaly}.tif')
            D.arr_to_tif(arr,ouf)
        T.open_path_and_file(outdir)


    def apply_MAT_MAP_multiregression_anomaly_crit_SM(self):
        outdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_SM')
        T.mk_dir(outdir)
        col_name = 'NDVI4g_climatology_percentage_GS'
        fpath = join(self.this_class_arr, 'MAT_MAP_multiregression_anomaly', f'{col_name}_reg.npy')
        result_dict = T.load_npy(fpath)
        y = 0.95 # affected by drought
        # sm_anomaly = -0.5 # mild drought
        T_anomaly_list = [
            -3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3
        ]
        for T_anomaly in T_anomaly_list:
            spatial_dict = {}
            for i in result_dict:
                dict_i = result_dict[i]
                model = dict_i['model']
                coef = model.coef_
                a1,a2 = coef[0],coef[1]
                b = model.intercept_
                intercept = model.intercept_
                crit_T = (y - b - a2*T_anomaly)/a1
                pix_list = dict_i['pix']
                for pix in pix_list:
                    spatial_dict[pix] = crit_T
            arr = D.pix_dic_to_spatial_arr(spatial_dict)
            outf = join(outdir,f'{col_name}_{T_anomaly}.tif')
            D.arr_to_tif(arr,outf)

        T.open_path_and_file(outdir)

    def apply_MAT_MAP_multiregression_origin_crit_T(self):
        outdir = join(self.this_class_tif,'MAT_MAP_multiregression_origin_crit_T')
        T.mk_dir(outdir)
        col_name = 'NDVI4g_climatology_percentage_GS'
        fpath = join(self.this_class_arr, 'MAT_MAP_multiregression_origin', f'{col_name}_reg.npy')
        result_dict = T.load_npy(fpath)
        y = 0.95 # affected by drought
        # sm_anomaly = -0.5 # mild drought
        sm_anomaly_list = [-0.5,-1,-1.5,-2,-2.5,-3]
        for sm_anomaly in sm_anomaly_list:
            spatial_dict = {}
            for i in result_dict:
                dict_i = result_dict[i]
                model = dict_i['model']
                coef = model.coef_
                a1,a2 = coef[0],coef[1]
                b = model.intercept_
                intercept = model.intercept_
                crit_T = (y - b - a1*sm_anomaly)/a2
                pix_list = dict_i['pix']
                for pix in pix_list:
                    spatial_dict[pix] = crit_T
            arr = D.pix_dic_to_spatial_arr(spatial_dict)
            ouf = join(outdir,f'{col_name}_{sm_anomaly}.tif')
            D.arr_to_tif(arr,ouf)

        T.open_path_and_file(outdir)


    def plot_apply_MAT_MAP_multiregression_anomaly_crit_T(self):

        fdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_T')
        outdir = join(self.this_class_png,'MAT_MAP_multiregression_anomaly_crit_T')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            color_list = [
                '#844000',
                '#fc9831',
                '#fffbd4',
                '#86b9d2',
                '#064c6c',
            ][::-1]
            # Blue represents high values, and red represents low values.
            cmap = Tools().cmap_blend(color_list)
            Plot().plot_Robinson(fpath,vmin=-3,vmax=3,cmap=cmap)
            plt.title(f.replace('.tif',''))
            outf = join(outdir,f.replace('.tif','.png'))
            plt.savefig(outf)
            plt.close()
            # plt.show()
        T.open_path_and_file(outdir)

    def statistic_crit_T(self):
        import analysis
        outdir = join(self.this_class_png,'statistic_crit_T')
        fdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_T')
        all_dict = {}
        col_name_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            col_name = f.replace('.tif','')
            col_name_list.append(col_name)
            spatial_dict = D.spatial_tif_to_dic(join(fdir,f))
            all_dict[col_name] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = analysis.Dataframe_func(df).df
        T.print_head_n(df)
        # print(df.columns.tolist())

        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        result_dict = {}
        for col_name in col_name_list:
            df_clean = df[df[col_name]>-10]
            df_clean = df_clean[df_clean[col_name]<10]
            vals_mean_list = []
            for lc in lc_list:
                df_lc = df_clean[df_clean['landcover_GLC']==lc]
                vals = df_lc[col_name].tolist()
                vals_mean = np.nanmean(vals)
                vals_mean_list.append(vals_mean)
                if not col_name in result_dict:
                    result_dict[col_name] = {}
                result_dict[col_name][lc] = vals_mean
        df = T.dic_to_df(result_dict,'threshold')
        col_name_list = ['NDVI4g_climatology_percentage_GS_-0.5', 'NDVI4g_climatology_percentage_GS_-1', 'NDVI4g_climatology_percentage_GS_-1.5', 'NDVI4g_climatology_percentage_GS_-2', 'NDVI4g_climatology_percentage_GS_-2.5', 'NDVI4g_climatology_percentage_GS_-3']

        for lc in lc_list:
            threshold_list = df['threshold'].tolist()
            val_list = []
            for threshold in col_name_list:
                df_threshold = df[df['threshold']==threshold]
                val = df_threshold[lc].tolist()[0]
                val_list.append(val)
            plt.figure()
            plt.barh(threshold_list,val_list,label=lc)
            plt.title(lc)
            plt.xlim(-3,4)
            plt.tight_layout()
        plt.show()


class Critical_P_and_T_RF:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Critical_P_and_T_RF', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr,'dataframe','dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # df = T.load_df(self.dff)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_Tair_juping)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_SM_juping)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_SM_origin)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_Tair_origin)
        # df = self.add_variables_during_droughts_GS(df,Load_Data().ERA_SM_juping)
        # df = self.add_variables_during_droughts_GS(df,Load_Data1().vpd_anomaly)
        # df = self.add_SOC(df)
        # df = self.add_BNPP(df)
        # self.Temperature_unit_convert(df)
        # T.save_df(df,self.dff)
        # T.df_to_excel(df,self.dff)
        # self.all_intensity()
        # self.all_T_max()
        # self.multiregression()
        # self.MAT_MAP_RF_anomaly_PDP()
        # self.MAT_MAP_RF_anomaly_importance()
        # self.MAT_MAP_RF_anomaly_importance_extreme()
        # self.matrix_MAT_MAP_RF_sample_size()
        self.matrix_MAT_MAP_RF_anomaly_importance()
        self.matrix_MAT_MAP_RF_anomaly_importance_extreme()
        # self.apply_MAT_MAP_RF_anomaly_T_SM_ratio()
        # self.apply_MAT_MAP_observation_anomaly_T_SM_ratio()
        # self.apply_MAT_MAP_multiregression_anomaly_crit_SM()
        # self.apply_MAT_MAP_multiregression_origin_crit_T()
        # self.plot_apply_MAT_MAP_multiregression_anomaly_crit_T()
        # self.statistic_crit_T()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')

        df = T.load_df(dff_merge)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

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

    def add_SOC(self,df):
        soc_tif = join(data_root,'SoilGrids/SOC/tif_sum/SOC_sum.tif')
        spatial_dict = D.spatial_tif_to_dic(soc_tif)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'SOC')
        return df

    def add_BNPP(self,df):
        soc_tif = join(data_root,'BNPP/tif_025/BNPP_0-200cm.tif')
        spatial_dict = D.spatial_tif_to_dic(soc_tif)
        df = T.add_spatial_dic_to_df(df,spatial_dict,'BNPP')
        return df

    def Temperature_unit_convert(self,df):
        col_name = 'ERA_Tair_origin_GS'
        vals = df[col_name].tolist()
        vals = [x - 273.15 for x in vals]
        df[col_name] = vals
        return df

    def all_intensity(self):
        df = T.load_df(self.dff)
        print(df.columns.tolist())
        exit()
        # T.print_head_n(df)
        rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_3_GS'
        rs_col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'

        intensity = df['intensity'].tolist()
        rs_vals_list = df[rs_col_name].tolist()
        rt_vals_list = df[rt_col_name].tolist()
        plt.figure()

        self.plot_fit_line(intensity, rs_vals_list, rs_col_name,'SM anomaly')
        self.plot_fit_line(intensity, rt_vals_list, rt_col_name,'SM anomaly')
        plt.legend()
        # plt.show()

    def all_T_max(self):
        df = T.load_df(self.dff)
        print(df.columns.tolist())
        # T.print_head_n(df)
        rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'

        T_max = df['T_max'].tolist()
        rs_vals_list = df[rs_col_name].tolist()
        rt_vals_list = df[rt_col_name].tolist()
        plt.figure()

        self.plot_fit_line(T_max, rs_vals_list, rs_col_name,'T_max')
        self.plot_fit_line(T_max, rt_vals_list, rt_col_name,'T_max')
        plt.legend()
        plt.show()

    def plot_fit_line(self,intensity, vals_list, col_name,x_label):
        std = np.nanstd(vals_list)
        a, b, r, p = T.nan_line_fit(intensity, vals_list)
        x_list = np.linspace(np.min(intensity), np.max(intensity), 100)
        y_list = [a * x + b for x in x_list]
        upper_bound_0 = y_list[0] + std
        lower_bound_0 = y_list[0] - std
        upper_bound_1 = y_list[-1] + std
        lower_bound_1 = y_list[-1] - std
        y_upper_list = [upper_bound_0, lower_bound_1]
        y_lower_list = [lower_bound_0, upper_bound_1]
        x_list_bound = [x_list[0], x_list[-1]]
        x_list = np.array(x_list)

        # plt.scatter(intensity,vals_list)
        plt.plot(x_list, y_list, label=f'{col_name}\n{r:.2f}')
        plt.fill_between(x_list_bound, y_upper_list, y_lower_list, alpha=0.5)
        plt.xlabel(x_label)
        pass

    def multiregression(self):
        outdir = join(self.this_class_png,'multiregression')
        T.mk_dir(outdir,force=True)
        df = T.load_df(self.dff)
        print(df.columns.tolist())
        # T.print_head_n(df)
        # rt_col_name = 'NDVI4g_climatology_percentage_rt_pre_baseline_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        # rs_col_name = 'NDVI4g_climatology_percentage_rs_pre_baseline_3_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_GS'
        col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_rs_pre_baseline_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_post_2_GS'

        # df_pix = df[df['pix']==pix]
        # df_pix = df_pix[df_pix[col_name] < 1.5]
        # df_pix = df_pix[df_pix[col_name] > 0.5]
        sm_col = 'intensity'
        T_col = 'ERA_Tair_juping_GS'
        # df = df.dropna(subset=[col_name,'ERA_SM_juping_GS','ERA_Tair_juping_GS'],how='any')
        df = df.dropna(subset=[col_name,sm_col,T_col],how='any')
        intensity = df[sm_col].tolist()
        intensity = np.array(intensity)
        T_max = df[T_col].tolist()

        vals_list = df[col_name].tolist()
        reg = LinearRegression()
        # plt.hist(T_max,bins=100)
        # plt.show()
        X = np.array([intensity,T_max]).T
        y = np.array(vals_list)
        reg.fit(X,y)
        intensity_predict_range = np.linspace(-3,0,26)
        T_max_predict_range = np.linspace(-2.5,2.5,26)
        intensity_predict_range = [round(x,2) for x in intensity_predict_range]
        T_max_predict_range = [round(x,2) for x in T_max_predict_range]
        matrix = []
        for i in range(len(intensity_predict_range)):
            matrix_i = []
            for j in range(len(T_max_predict_range)):
                intensity_predict = intensity_predict_range[i]
                T_max_predict = T_max_predict_range[j]
                X_predict = [intensity_predict,T_max_predict]
                Y_predict = reg.predict([X_predict])[0]
                matrix_i.append(Y_predict)
            matrix.append(matrix_i)
        matrix = matrix[::-1]
        plt.imshow(matrix,cmap='RdBu',vmin=0.9,vmax=1)
        # plt.imshow(matrix,cmap='RdBu',vmin=0.98,vmax=1.0)
        plt.colorbar()
        plt.xticks(np.arange(len(T_max_predict_range)),T_max_predict_range,rotation=90)
        plt.yticks(np.arange(len(intensity_predict_range)),np.array(intensity_predict_range)[::-1])
        plt.xlabel(T_col)
        plt.ylabel(sm_col)
        plt.title(f'{col_name}')
        # plt.show()
        outf = join(outdir,f'{col_name}.pdf')
        plt.savefig(outf)
        plt.close()

    def MAT_MAP_RF_anomaly_PDP(self):
        outdir = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_PDP')
        T.mk_dir(outdir, force=True)
        df = T.load_df(self.dff)
        for i in (df.columns.tolist()):
            print(i)
        col_list = df.columns.tolist()
        # print(self.x_variable_list)
        x_variable_list = self.x_variable_list
        for x_var in self.x_variable_list:
            # print(x_var)
            print(x_var in col_list)
        # exit()
        # T.print_head_n(df)

        y_var = 'NDVI4g_climatology_percentage_detrend_GS'
        # y_var = 'NDVI4g_climatology_percentage_post_2_GS'
        MAT_bins = np.linspace(0,30,31)
        MAP_bins = np.linspace(0,1200,31)
        # print(MAP_bins)
        # exit()
        MAT_bins = [round(x,1) for x in MAT_bins]
        MAP_bins = [round(x,1) for x in MAP_bins]
        df_group_MAT, bins_list_str_MAT = T.df_bin(df,'MAT',MAT_bins)
        flag = -1
        result_dict = {}
        for MAT_name, df_group_MAT_i in tqdm(df_group_MAT):
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i, 'MAP', MAP_bins)
            y_list = []
            for MAP_name, df_group_MAP_i in df_group_MAP:
                if len(df_group_MAP_i) == 0:
                    continue
                MAT_left = MAT_name[0].left
                MAT_right = MAT_name[0].right
                MAP_left = MAP_name[0].left
                MAP_right = MAP_name[0].right
                flag += 1
                df_group_MAP_i = df_group_MAP_i.dropna(subset=x_variable_list+[y_var], how='any')
                X = df_group_MAP_i[x_variable_list].values

                # X = np.array(x_variable_list).T
                y_val = df_group_MAP_i[y_var].tolist()
                if len(y_val) == 0:
                    continue
                # print(X)
                # print(y_val)
                result_dic = self.partial_dependence_plots(df_group_MAP_i, x_variable_list, y_var)
                pix_list = df_group_MAP_i['pix'].tolist()
                result_dict[flag] = {
                    'pix':pix_list,
                    'model':result_dic,
                    'MAT_left':MAT_left,
                    'MAT_right':MAT_right,
                    'MAP_left':MAP_left,
                    'MAP_right':MAP_right,
                }
        outf = join(outdir, f'{y_var}_reg.npy')
        T.save_npy(result_dict, outf)

    def MAT_MAP_RF_anomaly_importance(self):
        # y_var = 'NDVI4g_climatology_percentage_detrend_GS'
        y_var = 'NDVI4g_climatology_percentage_post_2_GS'
        outdir = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_importance')
        T.mk_dir(outdir, force=True)
        df = T.load_df(self.dff)
        df = Partial_Dependence_Plots_SM().valid_range_df(df,is_extreme=False)
        for i in (df.columns.tolist()):
            print(i)
        col_list = df.columns.tolist()
        # print(self.x_variable_list)
        x_variable_list = Partial_Dependence_Plots_SM().x_variable_list
        for x_var in x_variable_list:
            # print(x_var)
            print(x_var in col_list)
        # exit()
        # T.print_head_n(df)


        MAT_bins = np.linspace(0,30,31)
        MAP_bins = np.linspace(0,1200,31)
        # print(MAP_bins)
        # exit()
        MAT_bins = [round(x,1) for x in MAT_bins]
        MAP_bins = [round(x,1) for x in MAP_bins]
        df_group_MAT, bins_list_str_MAT = T.df_bin(df,'MAT',MAT_bins)
        flag = -1
        result_dict = {}
        for MAT_name, df_group_MAT_i in tqdm(df_group_MAT):
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i, 'MAP', MAP_bins)
            y_list = []
            for MAP_name, df_group_MAP_i in df_group_MAP:
                if len(df_group_MAP_i) == 0:
                    continue
                MAT_left = MAT_name[0].left
                MAT_right = MAT_name[0].right
                MAP_left = MAP_name[0].left
                MAP_right = MAP_name[0].right
                flag += 1
                df_group_MAP_i = df_group_MAP_i.dropna(subset=x_variable_list+[y_var], how='any')
                X = df_group_MAP_i[x_variable_list].values

                # X = np.array(x_variable_list).T
                y_val = df_group_MAP_i[y_var].tolist()
                if len(y_val) == 0:
                    continue
                # print(X)
                # print(y_val)
                result_dic = self.importance(df_group_MAP_i, x_variable_list, y_var)
                pix_list = df_group_MAP_i['pix'].tolist()
                result_dict[flag] = {
                    'pix':pix_list,
                    'model':result_dic,
                    'MAT_left':MAT_left,
                    'MAT_right':MAT_right,
                    'MAP_left':MAP_left,
                    'MAP_right':MAP_right,
                }
        outf = join(outdir, f'{y_var}_reg.npy')
        T.save_npy(result_dict, outf)

    def MAT_MAP_RF_anomaly_importance_extreme(self):
        # y_var = 'NDVI4g_climatology_percentage_detrend_GS'
        y_var = 'NDVI4g_climatology_percentage_post_2_GS'
        outdir = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_importance_extreme')
        T.mk_dir(outdir, force=True)
        df = T.load_df(self.dff)
        df = self.__select_extreme(df)
        df = Partial_Dependence_Plots_SM().valid_range_df(df,is_extreme=True)
        for i in (df.columns.tolist()):
            print(i)
        col_list = df.columns.tolist()
        # print(self.x_variable_list)
        x_variable_list = Partial_Dependence_Plots_SM().x_variable_list
        for x_var in x_variable_list:
            # print(x_var)
            print(x_var in col_list)
        # exit()
        # T.print_head_n(df)


        MAT_bins = np.linspace(0,30,31)
        MAP_bins = np.linspace(0,1200,31)
        # print(MAP_bins)
        # exit()
        MAT_bins = [round(x,1) for x in MAT_bins]
        MAP_bins = [round(x,1) for x in MAP_bins]
        df_group_MAT, bins_list_str_MAT = T.df_bin(df,'MAT',MAT_bins)
        flag = -1
        result_dict = {}
        for MAT_name, df_group_MAT_i in tqdm(df_group_MAT):
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i, 'MAP', MAP_bins)
            y_list = []
            for MAP_name, df_group_MAP_i in df_group_MAP:
                if len(df_group_MAP_i) == 0:
                    continue
                MAT_left = MAT_name[0].left
                MAT_right = MAT_name[0].right
                MAP_left = MAP_name[0].left
                MAP_right = MAP_name[0].right
                flag += 1
                df_group_MAP_i = df_group_MAP_i.dropna(subset=x_variable_list+[y_var], how='any')
                X = df_group_MAP_i[x_variable_list].values

                # X = np.array(x_variable_list).T
                y_val = df_group_MAP_i[y_var].tolist()
                if len(y_val) == 0:
                    continue
                # print(X)
                # print(y_val)
                result_dic = self.importance(df_group_MAP_i, x_variable_list, y_var)
                pix_list = df_group_MAP_i['pix'].tolist()
                result_dict[flag] = {
                    'pix':pix_list,
                    'model':result_dic,
                    'MAT_left':MAT_left,
                    'MAT_right':MAT_right,
                    'MAP_left':MAP_left,
                    'MAP_right':MAP_right,
                }
        outf = join(outdir, f'{y_var}_reg.npy')
        T.save_npy(result_dict, outf)

    def matrix_MAT_MAP_RF_anomaly_importance_extreme(self):
        fdir = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_importance_extreme')
        outdir = join(self.this_class_png, 'matrix_MAT_MAP_RF_anomaly_importance_extreme')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result_dict = T.load_npy(fpath)
            result_dict_max_imp = {}
            for i in result_dict:
                result_dict_i = result_dict[i]
                # for key in result_dict_i:
                #     print(key)
                # print(result_dict_i['model'])
                if result_dict_i['model'] == None:
                    continue
                max_var = T.get_max_key_from_dict(result_dict_i['model'])
                max_imp = result_dict_i['model'][max_var]
                MAT = result_dict_i['MAT_left']
                MAP = result_dict_i['MAP_left']
                result_dict_max_imp[i] = {
                    'MAT':MAT,
                    'MAP':MAP,
                    'max_var':max_var,
                    'max_imp':max_imp,
                }
            df = T.dic_to_df(result_dict_max_imp,'index')
            T.print_head_n(df)
            x_variable_list = Partial_Dependence_Plots_SM().x_variable_list
            color_list = T.gen_colors(len(x_variable_list))
            color_dict = T.dict_zip(x_variable_list, color_list)
            # print(color_dict)
            # print(color_list)
            # exit()
            plt.figure()
            ax = plt.subplot(1, 1, 1)
            for i,row in df.iterrows():
                MAP = row['MAP']
                MAT = row['MAT']
                max_imp = row['max_imp']
                color = color_dict[row['max_var']]
                color = list(color)
                # color.append(max_imp)
                # print(color)
                # exit()
                plt.scatter(MAT,MAP,color=color,marker='s',s=40)
            # plt.colorbar()
            plt.xlabel('MAT')
            plt.ylabel('MAP')
            plt.title(f.replace('.npy', ''))
            # plt.tight_layout()
            bounds = np.linspace(0, len(color_list), len(color_list)+1)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
            cmap = Tools().cmap_blend(color_list)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.15,shrink=0.5)
            cax,kw = mpl.colorbar.make_axes(ax,location='right',pad=0.05,shrink=0.5)
            # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='vertical')
            # cbar.set_label('Factors')
            cax.set_title('Factors')
            max_var_list = copy.copy(x_variable_list)
            max_var_list.append(' ')
            cbar.ax.set_yticklabels(max_var_list)
            # plt.tight_layout()
            outf = join(outdir, f.replace('.npy', '.pdf'))
            # plt.show()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)
        # plt.show()

    def matrix_MAT_MAP_RF_anomaly_importance(self):
        fdir = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_importance')
        outdir = join(self.this_class_png, 'matrix_MAT_MAP_RF_anomaly_importance')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            result_dict = T.load_npy(fpath)
            result_dict_max_imp = {}
            for i in result_dict:
                result_dict_i = result_dict[i]
                # for key in result_dict_i:
                #     print(key)
                # print(result_dict_i['model'])
                max_var = T.get_max_key_from_dict(result_dict_i['model'])
                max_imp = result_dict_i['model'][max_var]
                MAT = result_dict_i['MAT_left']
                MAP = result_dict_i['MAP_left']
                result_dict_max_imp[i] = {
                    'MAT':MAT,
                    'MAP':MAP,
                    'max_var':max_var,
                    'max_imp':max_imp,
                }
            df = T.dic_to_df(result_dict_max_imp,'index')
            T.print_head_n(df)
            # max_var_list = T.get_df_unique_val_list(df,'max_var')
            # print(max_var_list)
            x_variable_list = Partial_Dependence_Plots_SM().x_variable_list
            color_list = T.gen_colors(len(x_variable_list))
            color_dict = T.dict_zip(x_variable_list,color_list)
            # print(color_dict)
            # print(color_list)
            # exit()
            plt.figure()
            ax = plt.subplot(1, 1, 1)
            for i,row in df.iterrows():
                MAP = row['MAP']
                MAT = row['MAT']
                max_imp = row['max_imp']
                color = color_dict[row['max_var']]
                color = list(color)
                # color.append(max_imp)
                # print(color)
                # exit()
                plt.scatter(MAT,MAP,color=color,marker='s',s=40)
            # plt.colorbar()
            plt.xlabel('MAT')
            plt.ylabel('MAP')
            plt.title(f.replace('.npy', ''))
            # plt.tight_layout()
            bounds = np.linspace(0, len(color_list), len(color_list)+1)
            # norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
            cmap = Tools().cmap_blend(color_list)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.15,shrink=0.5)
            cax,kw = mpl.colorbar.make_axes(ax,location='right',pad=0.05,shrink=0.5)
            # cax,kw = mpl.colorbar.make_axes(ax,location='bottom',pad=0.05)
            cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds, orientation='vertical')
            # cbar.set_label('Factors')
            cax.set_title('Factors')
            max_var_list = copy.copy(x_variable_list)
            max_var_list.append(' ')
            cbar.ax.set_yticklabels(max_var_list)
            outf = join(outdir, f.replace('.npy', '.pdf'))
            # plt.show()
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)
        # plt.show()


    def matrix_MAT_MAP_RF_sample_size(self):
        fdir = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_importance')
        outdir = join(self.this_class_png, 'matrix_MAT_MAP_RF_sample_size')
        T.mk_dir(outdir, force=True)
        for f in T.listdir(fdir):
            fpath = join(fdir, f)
            result_dict = T.load_npy(fpath)
            result_dict_max_imp = {}
            for i in result_dict:
                result_dict_i = result_dict[i]
                # for key in result_dict_i:
                #     print(key)
                # print(result_dict_i['model'])
                pix_list = result_dict_i['pix']
                # print(pix_list)
                max_var = T.get_max_key_from_dict(result_dict_i['model'])
                max_imp = result_dict_i['model'][max_var]
                MAT = result_dict_i['MAT_left']
                MAP = result_dict_i['MAP_left']
                result_dict_max_imp[i] = {
                    'MAT': MAT,
                    'MAP': MAP,
                    'max_var': max_var,
                    'max_imp': max_imp,
                    'pix_len': len(pix_list),
                }
            df = T.dic_to_df(result_dict_max_imp, 'index')
            T.print_head_n(df)
            max_var_list = T.get_df_unique_val_list(df, 'max_var')
            print(max_var_list)
            color_list = T.gen_colors(len(max_var_list))
            color_dict = T.dict_zip(max_var_list, color_list)
            # print(color_dict)
            # print(color_list)
            # exit()
            plt.figure()
            ax = plt.subplot(1, 1, 1)
            for i, row in df.iterrows():
                MAP = row['MAP']
                MAT = row['MAT']
                max_imp = row['max_imp']
                pix_len = row['pix_len']
                # color = color_dict[row['max_var']]
                # color = list(color)
                # color.append(max_imp)
                # print(color)
                # exit()
                plt.scatter(MAT, MAP, c=pix_len, marker='s', s=40,vmin=0,vmax=10000,cmap='Blues')
            # plt.colorbar()
            plt.xlabel('MAT')
            plt.ylabel('MAP')
            plt.title(f.replace('.npy', ''))
            plt.colorbar()
            outf = join(outdir, 'sample_size.pdf')
            plt.savefig(outf)
            T.open_path_and_file(outdir)
            exit()
            # plt.show()

    def apply_MAT_MAP_RF_anomaly_T_SM_ratio(self):
        outdir = join(self.this_class_tif,'MAT_MAP_RF_anomaly_T_SM_ratio')
        T.mk_dir(outdir)
        # col_name = 'NDVI4g_climatology_percentage_GS'
        col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        fpath = join(self.this_class_arr, 'MAT_MAP_RF_anomaly_PDP', f'{col_name}_reg.npy')
        result_dict = T.load_npy(fpath)
        # y = 0.95 # affected by drought
        # sm_anomaly = -0.5 # mild drought
        spatial_dict = {}
        sm_a_list = []
        for i in result_dict:
            dict_i = result_dict[i]
            model = dict_i['model']
            MAT_left = dict_i['MAT_left']
            MAT_right = dict_i['MAT_right']
            MAP_left = dict_i['MAP_left']
            MAP_right = dict_i['MAP_right']
            pix_list = dict_i['pix']
            SM_var = 'intensity'
            T_var = 'T_max'
            sm_x = model[SM_var]['x']
            T_x = model[T_var]['x']
            sm_y = model[SM_var]['y']
            T_y = model[T_var]['y']
            sm_a,_,_,_ = T.nan_line_fit(sm_x,sm_y)
            T_a,_,_,_ = T.nan_line_fit(T_x,T_y)
            ratio = sm_a/T_a
            if ratio >5:
                continue
            if ratio < -5:
                continue
            plt.scatter(MAT_left,MAP_left,c=ratio,cmap='RdBu',vmin=-3,vmax=3,marker='s',s=40)
            # plt.scatter(MAT_left,MAP_left,c=sm_a,cmap='RdBu',vmin=-0.02,vmax=0.02,marker='s',s=40)
            # plt.scatter(MAT_left,MAP_left,c=T_a,cmap='RdBu',vmin=-0.02,vmax=0.02,marker='s',s=40)
            # sm_a_list.append(sm_a)
            for pix in pix_list:
                spatial_dict[pix] = ratio
        # plt.hist(sm_a_list,bins=100)
        # plt.title('T slope')
        # plt.title('SM slope')
        plt.title('SM/T ratio')
        plt.colorbar()
        plt.show()

        vals = spatial_dict.values()
        arr = D.pix_dic_to_spatial_arr(spatial_dict)
        plt.figure()
        plt.imshow(arr,cmap='RdBu',interpolation='nearest',vmin=-3,vmax=3)
        plt.colorbar()
        plt.show()

    def apply_MAT_MAP_observation_anomaly_T_SM_ratio(self):
        outdir = join(self.this_class_tif,'MAT_MAP_observation_anomaly_T_SM_ratio')
        T.mk_dir(outdir)
        # col_name = 'NDVI4g_climatology_percentage_GS'
        col_name = 'NDVI4g_climatology_percentage_post_2_GS'
        # col_name = 'NDVI4g_climatology_percentage_detrend_GS'
        SM_col = 'intensity'
        T_col = 'T_max'
        dff = self.dff
        df = T.load_df(dff)
        # T.print_head_n(df)
        MAT_bins = np.linspace(0, 30, 31)
        MAP_bins = np.linspace(0, 1200, 31)
        # print(MAP_bins)
        # exit()
        MAT_bins = [round(x, 1) for x in MAT_bins]
        MAP_bins = [round(x, 1) for x in MAP_bins]
        df_group_MAT, bins_list_str_MAT = T.df_bin(df, 'MAT', MAT_bins)
        flag = -1
        result_dict = {}
        matrix = []
        for MAT_name, df_group_MAT_i in tqdm(df_group_MAT):
            df_group_MAP, bins_list_str_MAP = T.df_bin(df_group_MAT_i, 'MAP', MAP_bins)
            y_list = []
            matrix_i = []
            for MAP_name, df_group_MAP_i in df_group_MAP:
                if len(df_group_MAP_i) == 0:
                    matrix_i.append(np.nan)
                    continue
                MAT_left = MAT_name[0].left
                MAT_right = MAT_name[0].right
                MAP_left = MAP_name[0].left
                MAP_right = MAP_name[0].right
                flag += 1
                df_group_MAP_i = df_group_MAP_i.dropna(subset=[col_name,SM_col,T_col], how='any')
                # print(df_group_MAP_i)
                col_vals = df_group_MAP_i[col_name].tolist()
                sm_vals = df_group_MAP_i[SM_col].tolist()
                T_vals = df_group_MAP_i[T_col].tolist()

                reg = LinearRegression()
                reg.fit(np.array([sm_vals,T_vals]).T,col_vals)
                sm_k,T_k = reg.coef_
                # ratio = sm_k/T_k
                delta = abs(sm_k) - abs(T_k)
                # if ratio > 5:
                #     matrix_i.append(np.nan)
                #     continue
                # if ratio < -5:
                #     matrix_i.append(np.nan)
                #     continue
                # print(sm_k,T_k)
                # exit()

                # col_mean = np.nanmean(col_vals)
                # sm_mean = np.nanmean(sm_vals)
                # T_mean = np.nanmean(T_vals)
                # ratio = sm_mean/T_mean
                # matrix_i.append(sm_mean)
                # matrix_i.append(ratio)
                # matrix_i.append(sm_k)
                matrix_i.append(delta)

                # print(col_mean)
                # plt.scatter(MAT_left, MAP_left, c=col_mean, cmap='RdBu', vmin=0.98, vmax=1.02, marker='s', s=40)
                # plt.scatter(MAT_left, MAP_left, c=col_mean, cmap='RdBu', vmin=0.9, vmax=1.1, marker='s', s=40)
                plt.scatter(MAT_left, MAP_left, c=delta, cmap='RdBu',vmin=-.04,vmax=0.04, marker='s', s=40)
            matrix.append(matrix_i)
        # plt.imshow(matrix,cmap='RdBu',vmin=-.05,vmax=0.05)
        # plt.imshow(matrix,cmap='RdBu_r',vmin=.8,vmax=1.2)
        # plt.imshow(matrix,cmap='RdBu_r',vmin=-1.4,vmax=-0.8)
        # plt.imshow(matrix,cmap='RdBu_r',vmin=-5,vmax=5)
        # matrix_flatten = np.array(matrix).flatten()
        # plt.hist(matrix_flatten,bins=100)

        # plt.title(col_name)
        # plt.title('SM')
        plt.title(f'SM - T\n{col_name}')
        plt.colorbar()
        plt.show()

    def apply_MAT_MAP_multiregression_anomaly_crit_SM(self):
        outdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_SM')
        T.mk_dir(outdir)
        col_name = 'NDVI4g_climatology_percentage_GS'
        fpath = join(self.this_class_arr, 'MAT_MAP_multiregression_anomaly', f'{col_name}_reg.npy')
        result_dict = T.load_npy(fpath)
        y = 0.95 # affected by drought
        # sm_anomaly = -0.5 # mild drought
        T_anomaly_list = [
            -3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3
        ]
        for T_anomaly in T_anomaly_list:
            spatial_dict = {}
            for i in result_dict:
                dict_i = result_dict[i]
                model = dict_i['model']
                coef = model.coef_
                a1,a2 = coef[0],coef[1]
                b = model.intercept_
                intercept = model.intercept_
                crit_T = (y - b - a2*T_anomaly)/a1
                pix_list = dict_i['pix']
                for pix in pix_list:
                    spatial_dict[pix] = crit_T
            arr = D.pix_dic_to_spatial_arr(spatial_dict)
            outf = join(outdir,f'{col_name}_{T_anomaly}.tif')
            D.arr_to_tif(arr,outf)

        T.open_path_and_file(outdir)

    def apply_MAT_MAP_multiregression_origin_crit_T(self):
        outdir = join(self.this_class_tif,'MAT_MAP_multiregression_origin_crit_T')
        T.mk_dir(outdir)
        col_name = 'NDVI4g_climatology_percentage_GS'
        fpath = join(self.this_class_arr, 'MAT_MAP_multiregression_origin', f'{col_name}_reg.npy')
        result_dict = T.load_npy(fpath)
        y = 0.95 # affected by drought
        # sm_anomaly = -0.5 # mild drought
        sm_anomaly_list = [-0.5,-1,-1.5,-2,-2.5,-3]
        for sm_anomaly in sm_anomaly_list:
            spatial_dict = {}
            for i in result_dict:
                dict_i = result_dict[i]
                model = dict_i['model']
                coef = model.coef_
                a1,a2 = coef[0],coef[1]
                b = model.intercept_
                intercept = model.intercept_
                crit_T = (y - b - a1*sm_anomaly)/a2
                pix_list = dict_i['pix']
                for pix in pix_list:
                    spatial_dict[pix] = crit_T
            arr = D.pix_dic_to_spatial_arr(spatial_dict)
            ouf = join(outdir,f'{col_name}_{sm_anomaly}.tif')
            D.arr_to_tif(arr,ouf)

        T.open_path_and_file(outdir)


    def plot_apply_MAT_MAP_multiregression_anomaly_crit_T(self):

        fdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_T')
        outdir = join(self.this_class_png,'MAT_MAP_multiregression_anomaly_crit_T')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            color_list = [
                '#844000',
                '#fc9831',
                '#fffbd4',
                '#86b9d2',
                '#064c6c',
            ][::-1]
            # Blue represents high values, and red represents low values.
            cmap = Tools().cmap_blend(color_list)
            Plot().plot_Robinson(fpath,vmin=-3,vmax=3,cmap=cmap)
            plt.title(f.replace('.tif',''))
            outf = join(outdir,f.replace('.tif','.png'))
            plt.savefig(outf)
            plt.close()
            # plt.show()
        T.open_path_and_file(outdir)

    def statistic_crit_T(self):
        import analysis
        outdir = join(self.this_class_png,'statistic_crit_T')
        fdir = join(self.this_class_tif,'MAT_MAP_multiregression_anomaly_crit_T')
        all_dict = {}
        col_name_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            col_name = f.replace('.tif','')
            col_name_list.append(col_name)
            spatial_dict = D.spatial_tif_to_dic(join(fdir,f))
            all_dict[col_name] = spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        df = analysis.Dataframe_func(df).df
        T.print_head_n(df)
        # print(df.columns.tolist())

        lc_list = T.get_df_unique_val_list(df,'landcover_GLC')
        result_dict = {}
        for col_name in col_name_list:
            df_clean = df[df[col_name]>-10]
            df_clean = df_clean[df_clean[col_name]<10]
            vals_mean_list = []
            for lc in lc_list:
                df_lc = df_clean[df_clean['landcover_GLC']==lc]
                vals = df_lc[col_name].tolist()
                vals_mean = np.nanmean(vals)
                vals_mean_list.append(vals_mean)
                if not col_name in result_dict:
                    result_dict[col_name] = {}
                result_dict[col_name][lc] = vals_mean
        df = T.dic_to_df(result_dict,'threshold')
        col_name_list = ['NDVI4g_climatology_percentage_GS_-0.5', 'NDVI4g_climatology_percentage_GS_-1', 'NDVI4g_climatology_percentage_GS_-1.5', 'NDVI4g_climatology_percentage_GS_-2', 'NDVI4g_climatology_percentage_GS_-2.5', 'NDVI4g_climatology_percentage_GS_-3']

        for lc in lc_list:
            threshold_list = df['threshold'].tolist()
            val_list = []
            for threshold in col_name_list:
                df_threshold = df[df['threshold']==threshold]
                val = df_threshold[lc].tolist()[0]
                val_list.append(val)
            plt.figure()
            plt.barh(threshold_list,val_list,label=lc)
            plt.title(lc)
            plt.xlim(-3,4)
            plt.tight_layout()
        plt.show()

    def partial_dependence_plots(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        flag = 0
        result_dic = {}
        for var in x_vars:
            flag += 1
            df_PDP = self.__get_PDPvalues(var, X, model) # get the partial dependence plot values
            ppx = df_PDP[var]
            ppy = df_PDP['PDs']
            ppy_std = df_PDP['PDs_std']
            result_dic[var] = {'x':ppx,
                               'y':ppy,
                               'y_std':ppy_std,
                               'r2':r2}
        return result_dic

    def importance(self,df,x_vars,y_var):
        '''
        :param df: a dataframe
        :param x_vars: a list of x variables
        :param y_var: a y variable
        :return:
        '''
        all_vars = copy.copy(x_vars) # copy the x variables
        all_vars.append(y_var) # add the y variable to the list
        all_vars_df = df[all_vars] # get the dataframe with the x variables and the y variable
        all_vars_df = all_vars_df.dropna() # drop rows with missing values
        X = all_vars_df[x_vars]
        Y = all_vars_df[y_var]
        if len(X) < len(x_vars):
            return None
        model, r2 = self.__train_model(X, Y) # train a Random Forests model
        importance = model.feature_importances_
        result_dic = {}

        for i in range(len(x_vars)):
            var = x_vars[i]
            importance_i = importance[i]
            result_dic[var] = importance_i
        return result_dic

    def __train_model(self,X,y):
        '''
        :param X: a dataframe of x variables
        :param y: a dataframe of y variable
        :return: a random forest model and the R^2
        '''
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=1, test_size=0.2) # split the data into training and testing
        rf = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=7) # build a random forest model
        rf.fit(X_train, y_train) # train the model
        r2 = rf.score(X_test,y_test)
        return rf,r2

    def __get_PDPvalues(self, col_name, data, model, grid_resolution=50):
        '''
        :param col_name: a variable
        :param data: a dataframe of x variables
        :param model: a random forest model
        :param grid_resolution: the number of points in the partial dependence plot
        :return: a dataframe of the partial dependence plot values
        '''
        Xnew = data.copy()
        sequence = np.linspace(np.min(data[col_name]), np.max(data[col_name]), grid_resolution) # create a sequence of values
        Y_pdp = []
        Y_pdp_std = []
        for each in sequence:
            Xnew[col_name] = each
            Y_temp = model.predict(Xnew)
            Y_pdp.append(np.mean(Y_temp))
            Y_pdp_std.append(np.std(Y_temp))
        return pd.DataFrame({col_name: sequence, 'PDs': Y_pdp, 'PDs_std': Y_pdp_std})

    def __select_extreme(self,df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df


class Historical_Critical_P_T_change:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Historical_Critical_P_T_change', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        self.scatter()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')

        df = T.load_df(dff_merge)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def scatter(self):
        outdir = join(self.this_class_png,'scatter')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        # df_pix_dict = T.df_groupby(df,'pix')
        SM_col = 'intensity'
        T_col = 'ERA_Tair_juping_GS'
        # for pix in df_pix_dict:
        #     df_pix = df_pix_dict[pix]
        SM_list = df[SM_col].tolist()[::50]
        T_list = df[T_col].tolist()[::50]
        year_list = df['drought_year'].tolist()[::50]
        plt.scatter(T_list,SM_list,c=year_list,alpha=0.5,s=4,cmap='Spectral_r')
        plt.colorbar()
        plt.ylabel(SM_col)
        plt.xlabel(T_col)
        plt.xlim(-2.5,2.5)
        plt.ylim(-3,0)
        outf = join(outdir,f'{SM_col}_{T_col}.png')
        plt.savefig(outf,dpi=300)
        plt.close()
        T.open_path_and_file(outdir)

class T_vs_rs_under_different_drought_intensity:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('T_vs_rs_under_different_drought_intensity', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        self.foo()

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        if isfile(self.dff):
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            print('Warning: this function will overwrite the dataframe')
            pause()
            pause()
        dff_merge = join(analysis.Dataframe_SM().this_class_arr, 'dataframe_merge', 'dataframe_merge.df')

        df = T.load_df(dff_merge)
        T.save_df(df, self.dff)
        T.df_to_excel(df, self.dff)

    def foo(self):
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        threshold_list = T.get_df_unique_val_list(df,'threshold')
        rs_col = 'NDVI4g_climatology_percentage_detrend_post_2_GS'
        # rs_col = 'NDVI4g_climatology_percentage_GS'
        # rs_col = 'NDVI4g_climatology_percentage_detrend_GS'

        T_col = 'ERA_Tair_juping_GS'
        T_bin = np.linspace(-2.5, 2.5, 26)
        color_list = T.gen_colors(len(threshold_list))
        flag = 0
        for threshold in threshold_list:
            df_threshold = df[df['threshold']==threshold]
            df_group, bins_list_str = T.df_bin(df_threshold,T_col,T_bin)
            mean_list = []
            for name,df_group_i in df_group:
                vals = df_group_i[rs_col].tolist()
                mean = np.nanmean(vals)
                mean_list.append(mean)
            plt.plot(bins_list_str,mean_list,'-o',label=threshold,color=color_list[flag],alpha=0.7)
            flag += 1
        plt.legend()
        plt.title(f'{rs_col}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

        pass

def main():

    # Hot_normal_drought().run()
    # Matrix_T_SM_rs().run()
    # Spatial_Trends().run()
    # Greening_Resilience().run()
    # Sensitivity_Analysis().run()
    # Random_Forests().run()
    # Random_Forests_SM().run()
    # Partial_Dependence_Plots_SM().run()
    SHAP().run()
    # Critical_P_and_T().run()
    # Critical_P_and_T_RF().run()
    # Historical_Critical_P_T_change().run()
    # T_vs_rs_under_different_drought_intensity().run()

    pass


if __name__ == '__main__':
    main()