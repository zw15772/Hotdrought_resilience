# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np

from meta_info import *

result_root_this_script = join(results_root, 'Plot_result')

class Rt_Rs:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Rt_Rs', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        self.__constant__()
        pass

    def run(self):
        # self.copy_df()
        # self.gen_tif()
        # self.plot_png()
        # self.gen_tif_extreme()
        # self.plot_png_extreme()
        self.statistic_extreme()
        # self.gen_tif_normal_drought()
        # self.plot_png_normal_drought()
        # self.AI_gradient()
        # self.AI_gradient_not_rs_rt_percentage()
        # self.AI_gradient_boxplot()
        # self.AI_hist()
        # self.PFTs()
        # self.PFTs_not_rs_rt_percentage()
        # self.plot_variables_hist()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr, 'dataframe')
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
        pass


    def gen_tif(self):
        outdir = join(self.this_class_tif, 'Rt_Rs')
        T.mk_dir(outdir)
        df = self.load_df()
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS':(0.98,1.02),
            'NDVI4g_climatology_percentage_detrend_GS':(0.9,1.1),
        }

        for col in col_name_list:
            df_group_dict = T.df_groupby(df,'pix')
            spatial_dict = {}
            for pix in tqdm(df_group_dict):
                df_pix = df_group_dict[pix]
                vals = df_pix[col].tolist()
                vals_mean = np.nanmean(vals)
                spatial_dict[pix] = vals_mean
            outf = join(outdir,f'{col}.tif')
            D.pix_dic_to_tif(spatial_dict,outf)
        pass

    def plot_png(self):
        fdir = join(self.this_class_tif, 'Rt_Rs')
        outdir = join(self.this_class_png, 'Rt_Rs')
        T.mk_dir(outdir)
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': (0.98, 1.02),
            'NDVI4g_climatology_percentage_detrend_GS': (0.95, 1),
        }
        for f in T.listdir(fdir):
            col_name = f.split('.')[0]
            fpath = join(fdir, f)
            outf = join(outdir, f.replace('.tif', '.png'))
            plt.figure(figsize=(10, 5))
            Plot().plot_Robinson(fpath, vmin=col_range_dict[col_name][0], vmax=col_range_dict[col_name][1])
            plt.title(col_name)
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)


    def gen_tif_extreme(self):
        outdir = join(self.this_class_tif, 'Rt_Rs_extreme')
        T.mk_dir(outdir)
        df = self.load_df()
        df = self.__select_extreme(df)
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS':(0.98,1.02),
            'NDVI4g_climatology_percentage_detrend_GS':(0.9,1.1),
        }

        for col in col_name_list:
            df_group_dict = T.df_groupby(df,'pix')
            spatial_dict = {}
            for pix in tqdm(df_group_dict):
                df_pix = df_group_dict[pix]
                vals = df_pix[col].tolist()
                vals_mean = np.nanmean(vals)
                spatial_dict[pix] = vals_mean
            outf = join(outdir,f'{col}.tif')
            D.pix_dic_to_tif(spatial_dict,outf)
        pass

    def plot_png_extreme(self):
        fdir = join(self.this_class_tif, 'Rt_Rs_extreme')
        outdir = join(self.this_class_png, 'Rt_Rs_extreme')
        T.mk_dir(outdir)
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': (0.92, 1.02),
            'NDVI4g_climatology_percentage_detrend_GS': (0.9, 1),
        }
        for f in T.listdir(fdir):
            col_name = f.split('.')[0]
            fpath = join(fdir, f)
            outf = join(outdir, f.replace('.tif', '.png'))
            plt.figure(figsize=(10, 5))
            Plot().plot_Robinson(fpath, vmin=col_range_dict[col_name][0], vmax=col_range_dict[col_name][1])
            plt.title(col_name)
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def statistic_extreme(self):
        fdir = join(self.this_class_tif, 'Rt_Rs_extreme')
        outdir = join(self.this_class_png, 'Rt_Rs_extreme')
        T.mk_dir(outdir)
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': (0.92, 1.02),
            'NDVI4g_climatology_percentage_detrend_GS': (0.9, 1),
        }
        all_dict = {}
        for f in T.listdir(fdir):
            col_name = f.split('.')[0]
            fpath = join(fdir, f)
            spatial_dict = D.spatial_tif_to_dic(fpath)
            all_dict[col_name] = spatial_dict

        df = T.spatial_dics_to_df(all_dict)
        threshold = 0.95
        for col in col_range_dict:
            df_low = df[df[col]<threshold]
            ratio = len(df_low) / len(df)
            print(col,ratio)



    def gen_tif_normal_drought(self):
        outdir = join(self.this_class_tif, 'Rt_Rs_normal_drought')
        T.mk_dir(outdir)
        df = self.load_df()
        df = self.__select_normal_drought(df)
        # df = self.__select_extreme(df)
        # print(len(df))
        # exit()
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS':(0.98,1.02),
            'NDVI4g_climatology_percentage_detrend_GS':(0.9,1.1),
        }

        for col in col_name_list:
            df_group_dict = T.df_groupby(df,'pix')
            spatial_dict = {}
            for pix in tqdm(df_group_dict):
                df_pix = df_group_dict[pix]
                vals = df_pix[col].tolist()
                vals_mean = np.nanmean(vals)
                spatial_dict[pix] = vals_mean
            outf = join(outdir,f'{col}.tif')
            D.pix_dic_to_tif(spatial_dict,outf)
        pass

    def plot_png_normal_drought(self):
        fdir = join(self.this_class_tif, 'Rt_Rs_normal_drought')
        outdir = join(self.this_class_png, 'Rt_Rs_normal_drought')
        T.mk_dir(outdir)
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': (0.95, 1.05),
            'NDVI4g_climatology_percentage_detrend_GS': (0.9, 1),
        }
        for f in T.listdir(fdir):
            col_name = f.split('.')[0]
            fpath = join(fdir, f)
            outf = join(outdir, f.replace('.tif', '.png'))
            plt.figure(figsize=(10, 5))
            Plot().plot_Robinson(fpath, vmin=col_range_dict[col_name][0], vmax=col_range_dict[col_name][1])
            plt.title(col_name)
            plt.savefig(outf, dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

    def AI_hist(self):
        outdir = join(self.this_class_png, 'AI_gradient')
        T.mk_dir(outdir)
        df = self.load_df()
        AI_vals = df['aridity_index'].tolist()
        x,y = Plot().plot_hist_smooth(AI_vals, bins=100, alpha=0)
        # plt.plot(x,y)
        plt.fill_between(x, y, alpha=0.2)
        # plt.legend()
        plt.savefig(join(outdir, 'AI_hist.pdf'))
        T.open_path_and_file(outdir)
        # plt.show()
        pass

    def AI_gradient(self):
        outdir = join(self.this_class_png,'AI_gradient')
        T.mk_dir(outdir)
        df = self.load_df()
        aridity_index_bin = np.arange(0.0, 0.66, 0.01)
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        df_group, bins_list_str = T.df_bin(df, 'aridity_index', aridity_index_bin)
        for col in col_name_list:
            x_list = []
            y_list = []
            err_list = []
            for name,df_group_i in df_group:
                left = name[0].left
                vals = df_group_i[col].tolist()
                mean = np.nanmean(vals)
                std = np.nanstd(vals) / 8.
                # std,_,_ = T.uncertainty_err(vals)
                std = abs(std)
                x_list.append(left)
                y_list.append(mean)
                err_list.append(std)
            plt.plot(x_list,y_list,label=col)
            plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list), alpha=0.2)
        plt.legend()
        plt.savefig(join(outdir,'AI_gradient.pdf'))
        T.open_path_and_file(outdir)
        # plt.show()
        pass

    def AI_gradient_not_rs_rt_percentage(self):
        outdir = join(self.this_class_png,'AI_gradient_not_rs_rt_percentage')
        T.mk_dir(outdir)
        df = self.load_df()
        df = self.__select_extreme(df)
        aridity_index_bin = np.arange(0.0, 0.66, 0.01)
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        not_rs_rt_threshold_dict = {
            'NDVI4g_climatology_percentage_post_2_GS':0.95,
            'NDVI4g_climatology_percentage_detrend_GS':0.95,
        }
        df_group, bins_list_str = T.df_bin(df, 'aridity_index', aridity_index_bin)
        for col in col_name_list:
            x_list = []
            y_list = []
            err_list = []
            for name,df_group_i in tqdm(df_group):
                left = name[0].left
                vals = df_group_i[col].tolist()
                vals = np.array(vals)
                vals_not_rs = vals[vals<not_rs_rt_threshold_dict[col]]
                if len(vals) == 0:
                    ratio = np.nan
                else:
                    ratio = len(vals_not_rs) / len(vals) * 100
                x_list.append(left)
                y_list.append(ratio)
                ratio_random_list = []
                for i in range(10000):
                    vals_random_choice = np.random.choice(vals, size=int(len(vals) * 0.5), replace=False)
                    vals_not_rs_random = vals_random_choice[vals_random_choice < not_rs_rt_threshold_dict[col]]
                    if len(vals_random_choice) == 0:
                        ratio_random = np.nan
                    else:
                        ratio_random = len(vals_not_rs_random) / len(vals_random_choice) * 100
                    ratio_random_list.append(ratio_random)
                # print(np.nanmean(ratio_random_list))
                # print(ratio)
                err = np.nanmax(ratio_random_list) - np.nanmin(ratio_random_list)
                err_list.append(err)

            plt.plot(x_list,y_list,label=col)
            plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list), alpha=0.2)
        plt.ylim(0,70)
        plt.legend()
        plt.savefig(join(outdir,'AI_gradient_extreme.pdf'))
        # plt.savefig(join(outdir,'AI_gradiente.pdf'))
        T.open_path_and_file(outdir)
        # plt.show()
        pass

    def AI_gradient_boxplot(self):
        # outdir = join(self.this_class_png,'AI_gradient')
        # T.mk_dir(outdir)
        df = self.load_df()
        aridity_index_bin = np.arange(0.0, 0.66, 0.005)
        # print(aridity_index_bin)
        # exit()
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        df_group, bins_list_str = T.df_bin(df, 'aridity_index', aridity_index_bin)
        for col in col_name_list:
            x_list = []
            y_list = []
            err_list = []
            box_list = []
            for name,df_group_i in df_group:
                left = name[0].left
                vals = df_group_i[col].tolist()
                vals = np.array(vals)
                vals = T.remove_np_nan(vals)
                mean = np.nanmean(vals)
                std = np.nanstd(vals) / 8.
                # std,_,_ = T.uncertainty_err(vals)
                std = abs(std)
                x_list.append(left)
                y_list.append(mean)
                err_list.append(std)
                box_list.append(vals)
            plt.figure()
            plt.boxplot(box_list,labels=x_list,showfliers=False)
            plt.title(col)
            # plt.plot(x_list,y_list,label=col)
            # plt.fill_between(x_list, np.array(y_list) - np.array(err_list), np.array(y_list) + np.array(err_list), alpha=0.2)
        # plt.legend()
        # plt.savefig(join(outdir,'AI_gradient.pdf'))
        # T.open_path_and_file(outdir)
        plt.show()
        pass

    def PFTs(self):
        outdir = join(self.this_class_png,'PFTs')
        T.mk_dir(outdir)
        df = self.load_df()
        print(df.columns.tolist())
        # print(aridity_index_bin)
        # exit()
        col_range_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': (0.98, 1.02),
            'NDVI4g_climatology_percentage_detrend_GS': (0.95, 1),
        }
        col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]
        pft_list = global_lc_list
        for col in col_name_list:
            x_list = []
            y_list = []
            err_list = []
            box_list = []
            for lc in pft_list:
                df_lc = df[df['landcover_GLC']==lc]
                vals = df_lc[col].tolist()
                vals = np.array(vals)
                vals = T.remove_np_nan(vals)
                mean = np.nanmean(vals)
                std = np.nanstd(vals) / 8.
                # std,_,_ = T.uncertainty_err(vals)
                std = abs(std)
                x_list.append(lc)
                y_list.append(mean)
                err_list.append(std)
                box_list.append(vals)
            # plt.boxplot(box_list,labels=x_list,showfliers=False)
            plt.figure()
            plt.bar(x_list,y_list,label=col)
            plt.errorbar(x_list,y_list,yerr=err_list,fmt='none',ecolor='k')
            plt.ylim(col_range_dict[col])
            plt.title(col)
            outf = join(outdir,f'{col}.pdf')
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def PFTs_not_rs_rt_percentage(self):
        outdir = join(self.this_class_png,'PFTs_not_rs_rt_percentage','extreme')
        # outdir = join(self.this_class_png,'PFTs_not_rs_rt_percentage','normal')
        # outdir = join(self.this_class_png,'PFTs_not_rs_rt_percentage','all')
        T.mk_dir(outdir,force=True)
        df = self.load_df()
        df = self.__select_extreme(df)
        # df = self.__select_normal_drought(df)
        print(df.columns.tolist())

        col_name_list = self.col_name_list
        # pft_list = global_lc_list
        pft_list = T.get_df_unique_val_list(df,'landcover_GLC')
        print(pft_list)
        pft_list = ['crop', 'deciduous', 'evergreen', 'grass', 'shrubs']
        # exit()
        for col in col_name_list:
            x_list = []
            y_list = []
            err_list = []
            for lc in pft_list:
                df_lc = df[df['landcover_GLC']==lc]
                vals = df_lc[col].tolist()
                vals = np.array(vals)
                threshold = self.not_rs_rt_threshold_dict[col]
                vals = T.remove_np_nan(vals)
                vals_not_rs = vals[vals<threshold]
                if len(vals) == 0:
                    ratio = np.nan
                else:
                    ratio = len(vals_not_rs) / len(vals) * 100
                x_list.append(lc)
                y_list.append(ratio)
                ratio_random_list = []
                for i in range(10000):
                    vals_random_choice = np.random.choice(vals, size=int(len(vals) * 0.5), replace=False)
                    vals_not_rs_random = vals_random_choice[vals_random_choice < threshold]
                    if len(vals_random_choice) == 0:
                        ratio_random = np.nan
                    else:
                        ratio_random = len(vals_not_rs_random) / len(vals_random_choice) * 100
                    ratio_random_list.append(ratio_random)
                # print(np.nanmean(ratio_random_list))
                # print(ratio)
                err = np.nanmax(ratio_random_list) - np.nanmin(ratio_random_list)
                err_list.append(err)
            plt.figure()
            plt.bar(x_list,y_list,label=col)
            plt.errorbar(x_list,y_list,yerr=err_list,fmt='none',ecolor='k')
            plt.ylim(0,60)
            plt.title(col)
            outf = join(outdir,f'{col}.pdf')
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

    def plot_variables_hist(self):
        dff = self.dff
        df = T.load_df(dff)
        col_list = df.columns.tolist()
        for col in col_list:
            print(col)
            # vals = df[col].tolist()
            # x,y = Plot().plot_hist_smooth(vals,bins=100,alpha=0)
            # plt.plot(x,y,label=col)
            # plt.legend()
            # plt.show()
        df = self.__select_extreme(df)
        # vpd = df['VPD-anomaly_GS'].tolist()
        vpd = df['T_max'].tolist()
        plt.hist(vpd,bins=100)
        plt.show()
        exit()

        pass

    def load_df(self):
        df = T.load_df(self.dff)
        T.print_head_n(df)
        col_list = df.columns.tolist()
        for col in col_list:
            print(col)
        return df

    def __select_extreme(self,df):
        df = df[df['T_max'] > 1.5]
        df = df[df['intensity'] < -2]
        return df

    def __select_normal_drought(self,df):
        df = df[df['intensity'] < -2]
        df = df[df['T_max'] < 1.5]
        return df

    def __constant__(self):

        self.not_rs_rt_threshold_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': 1,
            # 'NDVI4g_climatology_percentage_post_2_GS': 0.95,
            'NDVI4g_climatology_percentage_detrend_GS': 1,
            # 'NDVI4g_climatology_percentage_detrend_GS': 0.95,
        }

        self.col_name_list = [
            'NDVI4g_climatology_percentage_post_2_GS',
            'NDVI4g_climatology_percentage_detrend_GS',
        ]

        self.col_lim_dict = {
            'NDVI4g_climatology_percentage_post_2_GS': (0.98, 1.02),
            'NDVI4g_climatology_percentage_detrend_GS': (0.95, 1),
        }
        pass


class Constant_plot:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Constant_plot', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # self.gen_Aridity_index_tif()
        self.plot_Aridity_index_tif()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr, 'dataframe')
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
        pass

    def gen_Aridity_index_tif(self):
        outdir = join(self.this_class_tif, 'Aridity_index')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        T.print_head_n(df)
        spatial_dict = {}
        df_pix_dict = T.df_groupby(df,'pix')
        for pix in df_pix_dict:
            df_pix = df_pix_dict[pix]
            vals = df_pix['aridity_index'].tolist()
            vals_mean = np.nanmean(vals)
            spatial_dict[pix] = vals_mean
        outf = join(outdir,'aridity_index.tif')
        D.pix_dic_to_tif(spatial_dict,outf)

        pass

    def plot_Aridity_index_tif(self):
        fdir = join(self.this_class_tif, 'Aridity_index')
        outdir = join(self.this_class_png, 'Aridity_index')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            color_list = [
                '#844000',
                '#fc9831',
                '#fffbd4',
                '#ffffff',
                # '#86b9d2',
                # '#064c6c',
            ]
            # Blue represents high values, and red represents low values.
            cmap = Tools().cmap_blend(color_list)
            plt.figure(figsize=(10, 5))
            Plot().plot_Robinson(fpath,cmap=cmap,res=100000)
            # outf = join(outdir,f.replace('.tif','.png'))
            outf = join(outdir,f.replace('.tif','.pdf'))
            plt.savefig(outf,dpi=300)
            plt.close()
        T.open_path_and_file(outdir)

        pass


class Drought_characteristic:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Drought_characteristic', result_root_this_script, mode=2)
        self.dff = join(self.this_class_arr, 'dataframe', 'dataframe.df')
        pass

    def run(self):
        # self.copy_df()
        # self.tif_drought_frequency()
        # self.plot_tif_drought_frequency()
        # self.plot_drought_affected_area_ts()
        # self.plot_temperature_sm_anomaly_ts()
        self.plot_temperature_anomaly_ts()
        pass

    def copy_df(self):
        import analysis
        outdir = join(self.this_class_arr, 'dataframe')
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
        pass

    def tif_drought_frequency(self):
        outdir = join(self.this_class_tif, 'drought_frequency')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        # T.print_head_n(df)
        df = self.__select_extreme(df)
        df_pix_dict = T.df_groupby(df,'pix')
        spatial_dict = {}
        for pix in tqdm(df_pix_dict):
            df_pix = df_pix_dict[pix]
            drought_num = len(df_pix)
            spatial_dict[pix] = drought_num
        outf = join(outdir,'drought_frequency.tif')
        D.pix_dic_to_tif(spatial_dict,outf)
        pass

    def plot_tif_drought_frequency(self):
        fdir = join(self.this_class_tif, 'drought_frequency')
        outdir = join(self.this_class_png, 'drought_frequency')
        T.mk_dir(outdir)
        fpath = join(fdir,'drought_frequency.tif')
        color_list = [
            '#844000',
            '#fc9831',
            '#fffbd4',
            '#ffffff',
            # '#86b9d2',
            # '#064c6c',
        ][::-1]
        # Blue represents high values, and red represents low values.
        cmap = Tools().cmap_blend(color_list)
        plt.figure(figsize=(10, 5))
        Plot().plot_Robinson(fpath, cmap=cmap, res=25000,vmin=0,vmax=3)
        outf = join(outdir,'drought_frequency.png')
        # outf = join(outdir, 'drought_frequency.pdf')
        plt.savefig(outf, dpi=300)
        plt.close()
        T.open_path_and_file(outdir)
        pass


    def plot_drought_affected_area_ts(self):
        outdir = join(self.this_class_png, 'drought_affected_area_ts')
        T.mk_dir(outdir)
        dff = self.dff
        df = T.load_df(dff)
        # df = self.__select_extreme(df)

        pix_list = T.get_df_unique_val_list(df,'pix')
        year_list = T.get_df_unique_val_list(df,'drought_year')

        print(df.columns)
        intensity_bins = [-3,-2,-1.5,-1]
        df_group, bins_list_str = T.df_bin(df,'intensity',intensity_bins)
        bottom = 0
        plt.figure(figsize=(10,4))
        for name,df_group_i in df_group:
            left = name[0].left
            df_year_dict = T.df_groupby(df_group_i, 'drought_year')
            x = []
            y = []
            for year in year_list:
                df_year = df_year_dict[year]
                pix_year = T.get_df_unique_val_list(df_year,'pix')
                # ratio = len(df_year)
                # pix_year = T.get_df_unique_val_list(df_year,'pix')
                ratio = len(pix_year) / len(pix_list) * 100
                x.append(year)
                y.append(ratio)
            x = np.array(x)
            y = np.array(y)
            plt.bar(x,y,bottom=bottom,label=left)
            bottom += y
        # plt.legend()
        outf = join(outdir,'drought_affected_area_ts.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir)


    def plot_temperature_sm_anomaly_ts(self):
        dff = self.dff
        df = T.load_df(dff)
        # df = self.__select_extreme(df)
        print(df.columns.tolist())
        T_col = 'T_max'
        SM_col = 'intensity'
        year_list = T.get_df_unique_val_list(df,'drought_year')
        df_year_dict = T.df_groupby(df,'drought_year')
        x = []
        y1 = []
        y2 = []
        y1_err = []
        y2_err = []
        for year in df_year_dict:
            df_year = df_year_dict[year]
            intensity = df_year[SM_col].tolist()
            T_max = df_year[T_col].tolist()
            intensity_mean = np.nanmean(intensity)
            T_max_mean = np.nanmean(T_max)
            itensity_std = np.nanstd(intensity)
            T_max_std = np.nanstd(T_max)
            x.append(year)
            y1.append(intensity_mean)
            y2.append(T_max_mean)
            y1_err.append(itensity_std)
            y2_err.append(T_max_std)
        x = np.array(x)
        y1 = np.array(y1)
        y2 = np.array(y2)
        y1_err = np.array(y1_err) / 2.
        y2_err = np.array(y2_err) / 2.
        plt.figure(figsize=(10,4))
        plt.plot(x,y1,label=SM_col,color='b')
        plt.fill_between(x,y1-y1_err,y1+y1_err,alpha=0.2,color='b')

        # plt.twinx()
        plt.plot(x,y2,label=T_col,color='r')
        plt.fill_between(x,y2-y2_err,y2+y2_err,alpha=0.2,color='r')
        plt.ylim(-2,2)
        plt.show()
        pass

    def plot_temperature_anomaly_ts(self):
        outdir_df = join(self.this_class_arr, 'dataframe')
        outdir_png = join(self.this_class_png, 'temperature_anomaly_ts')
        T.mk_dir(outdir_png)
        T.mk_dir(outdir_df)
        outf = join(outdir_df,'temperature_anomaly_ts.df')
        if isfile(outf):
            df = T.load_df(outf)
        else:

            import analysis
            spatial_dict,_ = Load_Data().ERA_Tair_anomaly_GS()

            all_dict = {'T':spatial_dict}
            df = T.spatial_dics_to_df(all_dict)
            df = analysis.Dataframe_func(df).df
            T.save_df(df,outf)
            T.df_to_excel(df,outf)
        T_col = 'T'
        temperature_anomaly = df[T_col].tolist()
        temperature_anomaly = np.array(temperature_anomaly)
        temperature_anomaly_new = []
        for ta in temperature_anomaly:
            len_ta = len(ta)
            if len_ta == 38:
                ta = np.concatenate([[np.nan],ta])
            temperature_anomaly_new.append(ta)
        temperature_anomaly_mean = np.nanmean(temperature_anomaly_new,axis=0)
        temperature_anomaly_std = np.nanstd(temperature_anomaly_new,axis=0)
        plt.figure(figsize=(10,4))
        plt.plot(temperature_anomaly_mean)
        plt.fill_between(range(len(temperature_anomaly_mean)),temperature_anomaly_mean-temperature_anomaly_std,temperature_anomaly_mean+temperature_anomaly_std,alpha=0.2)
        outf = join(outdir_png,'temperature_anomaly_ts.pdf')
        plt.savefig(outf)
        plt.close()
        T.open_path_and_file(outdir_png)
        pass

    def __select_extreme(self,df):
        df = df[df['T_max'] > 1]
        df = df[df['intensity'] < -2]
        return df

def main():
    Rt_Rs().run()
    # Constant_plot().run()
    # Drought_characteristic().run()
    pass

if __name__ == '__main__':
    main()

    pass