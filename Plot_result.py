import matplotlib.pyplot as plt
import numpy as np

from meta_info import *

this_root=rf'F:\\Hotdrought_Resilience\\'
data_root = join(this_root,'data')
results_root = join(this_root,'results')
temp_root = join(this_root,'temp')

result_root_this_script = join(results_root, 'Plot_result')

class Rt_Rs:
    def __init__(self):


        self.dff = join(results_root,  'analysis\Dataframe\drought_dataframe.df')

        self.outdir=join(results_root, 'analysis','Plot_result')
        T.mk_dir(self.outdir)
        # self.map_width = 8.2 * centimeter_factor
        # self.map_height = 8.2 * centimeter_factor

        self.map_width = 16 * centimeter_factor
        self.map_height = 16 * centimeter_factor



        pass
    def run(self):
        # self.plot_Rt_Rs()
        self.plot_Rt_Rs_spatial_map()
        # self.plot_ratio_unrecovered()
        # self.moving_window_extraction()
        # self.heatmap()

        pass
    def plot_Rt_Rs(self):
        df=''
        print(len(df))
        df=self.df_clean(df)
        print(len(df))

        df_normal=df[df['drought_type'] == 'normal-drought']

        ## heatmap

        post_n_years = [1,2,3,4]

        aridity_col='Aridity'

        aridity_bin = np.linspace(0, 2.5, 25)
        df_groupe1, bin_list_str = T.df_bin(df_normal, aridity_col, aridity_bin)
        for n in post_n_years:
            val_list = []

            bin_centers = []
            name_list = []

            for name, group in df_groupe1:
                bin_left = name[0].left
                bin_right = name[0].right
                bin_centers.append((bin_left + bin_right) / 2)
                name_list.append(f"{bin_left:.2f}-{bin_right:.2f}")
                val = np.nanmean(group[f'GS_NDVI_post_{n}_relative_change'])
                # val=np.nanmean(group['rs_4years'])
                val_list.append(val)


            x = np.arange(len(val_list))  #
            width = 0.6

            plt.figure(figsize=(6, 4))
            plt.bar(x, val_list, width=width, color='steelblue', edgecolor='k', alpha=0.8)

            plt.xticks(x, name_list, rotation=45, ha='right', fontsize=12)
            plt.xlabel('Aridity Index', fontsize=12)
            plt.ylabel('Mean GS NDVI', fontsize=12)
            plt.yticks(fontsize=12)

            plt.title(f'post_{n} years NDVI anomaly (Hot Drought)', fontsize=13)
            plt.ylim(-5, 2)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()


    def plot_Rt_Rs_spatial_map(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))
        spatial_dic={}

        df_hot=df[df['Temp'] <-0.5]
        df_group=T.df_groupby(df_hot,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list=df_pix['GS_NDVI_relative_change'].tolist()
            val_mean=np.nanmean(val_list)
            spatial_dic[pix]=val_mean

        array=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir=join(self.outdir,'hot_drought')
        T.mk_dir(outdir)
        outf=join(outdir,'GS_NDVI_relative_change.tif')
        print(outf)

        DIC_and_TIF().arr_to_tif(array,outf)

    def plot_ratio_unrecovered(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)

        print(len(df))



        df_hot = df[df["drought_type"] == "hot-drought"]
        koppen_list = [1, 2, 3, 4]
        dic_koppen_name = {1: "Tropical", 2: "Arid", 3: "Temperate", 4: "Cold"}

        # 要统计的 post 年份
        post_years = [1, 2, 3, 4]
        result_dic = {koppen: [] for koppen in koppen_list}

        for koppen in koppen_list:
            df_koppen = df_hot[df_hot["Koppen"] == koppen]

            grouped = df_koppen.groupby("pix").mean(numeric_only=True)

            for yr in post_years:
                col_name = f"GS_NDVI_post_{yr}_relative_change"
                vals=grouped[col_name].tolist()
                vals = np.array(vals)
                vals[vals<-999]=np.nan
                vals=vals[~np.isnan(vals)]


                if len(vals) == 0:
                    result_dic[koppen].append(np.nan)
                    continue

                unrecovered_ratio = len(vals[vals <- 20]) / len(vals) * 100
                result_dic[koppen].append(unrecovered_ratio)

        # --- 绘图 ---
        plt.figure(figsize=(5, 3))
        bar_width = 0.2
        x = np.arange(len(koppen_list))

        for i, yr in enumerate(post_years):
            vals = [result_dic[k][i] for k in koppen_list]
            plt.bar(x + i * bar_width, vals, width=bar_width,
                    label=f"Post {yr} yr", alpha=0.8)

        plt.xticks(x + bar_width * 1.5, [dic_koppen_name[k] for k in koppen_list])
        plt.ylim(0, 40)
        plt.ylabel("Unrecovered percentage")

        plt.legend(title="Years after Drought")

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()






    def df_clean(self, df):
        T.print_head_n(df)
        # df = df.dropna(subset=[self.y_variable])
        # T.print_head_n(df)
        # exit()
        # df = df[df['row'] > 60]
        # df = df[df['Aridity'] < 0.65]
        # df = df[df['LC_max'] < 10]
        df=df[df['MODIS_LUCC']!=12]
        df=df[df['Koppen']!=5]


        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['landcover_classfication'] != 'Bare']

        return df

    def moving_window_extraction(self):
        dff=self.dff
        df=T.load_df(dff)
        df=self.df_clean(df)
        T.print_head_n(df)
        years=list(range(1982,2021))
        years=np.array(years)
        step=1

        df_hot = df[df["drought_type"] == "hot-drought"]
        koppen_list = [1, 2, 3, 4]
        dic_koppen_name = {1: "Tropical", 2: "Temperate", 3: "Arid", 4: "Cold"}

        # 要统计的 post 年份
        post_years = [1, 2, 3, 4]
        result_dic = {koppen: [] for koppen in koppen_list}


        for koppen in koppen_list:
            df_koppen = df_hot[df_hot["Koppen"] == koppen]

            grouped = df_koppen.groupby("pix").mean(numeric_only=True)

            for yr in post_years:
                window = 10


                col_name = f"GS_NDVI_post_{yr}_relative_change"
                grouped.loc[grouped[col_name] < -999, col_name] = np.nan

                # 提取年份序列

                window_results = []

                for start in range(int(years.min()), int(years.max()) - window + 2, step):
                    end = start + window - 1
                    df_window = grouped[(grouped['drought_year'] >= start) & (grouped['drought_year'] <= end)]

                    # 每个窗口取平均 NDVI
                    vals = df_window[col_name].dropna().values
                    if len(vals) == 0:
                        mean_val = np.nan
                    else:
                        mean_val = np.nanmean(vals)

                    window_results.append({
                        "koppen": koppen,
                        "post_year": yr,
                        "window_start": start,
                        "window_end": end,

                        "mean_NDVI": mean_val,
                        "n_samples": len(vals)
                    })

    pass


    def heatmap(self):  ## plot trend as function of Aridity and precipitation trend
        ## plot trends as function of inter precipitaiton CV and intra precipitation CV
        dff=results_root+rf'\\analysis\\Dataframe\\drought_dataframe.df'
        print(dff)

        df=T.load_df(dff)
        df=self.df_clean(df)
        print(len(df))
        # df=df[df['Aridity'] > 0.65]
        df = df[df['Aridity'] < 0.65]

        T.print_head_n(df)
        x_var = 'drought_serverity'
        y_var = 'Temp'
        z_var = 'GS_NDVI_post_4_relative_change'
        # z_var = 'GS_NDVI_relative_change'


        plt.hist(df[x_var])
        plt.show()
        plt.hist(df[y_var])
        plt.show()

        bin_x = np.linspace(-3, -1.6,4, )

        bin_y = np.linspace(-3, 3, 9)
        # percentile_list=np.linspace(0,100,7)
        # bin_x=np.percentile(df[x_var],percentile_list)
        # print(bin_x)
        # bin_y=np.percentile(df[y_var],percentile_list)
        plt.figure(figsize=(self.map_width, self.map_height))

        matrix_dict,x_ticks_list,y_ticks_list = T.df_bin_2d(df,val_col_name=z_var,
                    col_name_x=x_var,
                    col_name_y=y_var,bin_x=bin_x,bin_y=bin_y,round_x=4,round_y=4)
        # pprint(matrix_dict);exit()

        my_cmap = T.cmap_blend(color_list = ['#000000','r', 'b'])
        my_cmap = 'RdBu'
        self.plot_df_bin_2d_matrix(matrix_dict,-3,3,x_ticks_list,y_ticks_list,cmap=my_cmap,
                              is_only_return_matrix=False)
        plt.colorbar()
        plt.xticks(rotation=45)
        plt.tight_layout()
        # pprint(matrix_dict)
        # plt.show()


        matrix_dict_count, x_ticks_list, y_ticks_list = self.df_bin_2d_count(df, val_col_name=z_var,
                                                              col_name_x=x_var,
                                                              col_name_y=y_var, bin_x=bin_x, bin_y=bin_y)
        # pprint(matrix_dict_count)
        scatter_size_dict = {
            (1,20): 5,
            (20,50): 20,
            (50,100): 50,
            (100,200): 75,
            (200,400): 100,
            (400,800): 200,
            (800,np.inf): 250
        }
        matrix_dict_count_normalized = {}
        # Normalize counts for circle size
        for key in matrix_dict_count:
            num = matrix_dict_count[key]
            for key2 in scatter_size_dict:
                if num >= key2[0] and num < key2[1]:
                    matrix_dict_count_normalized[key] = scatter_size_dict[key2]
                    break
        # pprint(matrix_dict_count_normalized)
        reverse_x = list(range(len(bin_y)-1))[::-1]
        reverse_x_dict = {}
        for i in range(len(bin_y)-1):
            reverse_x_dict[i] = reverse_x[i]
        # print(reverse_x_dict);exit()
        # for x,y in matrix_dict_count_normalized:
        #     plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='gray',edgecolors='none',alpha=.5)
        # for x,y in matrix_dict_count_normalized:
        #     plt.scatter(y,reverse_x_dict[x],s=matrix_dict_count_normalized[(x,y)],c='none',edgecolors='gray',alpha=1)

        plt.xlabel('Drought severity (%)',fontsize=12)
        plt.ylabel('temperature (%)',fontsize=12)

        plt.show()
        # plt.savefig(outf)
        # plt.close()

    def df_bin_2d_count(self,df,val_col_name,col_name_x,col_name_y,bin_x,bin_y,round_x=2,round_y=2):
        df_group_y, _ = self.df_bin(df, col_name_y, bin_y)
        matrix_dict = {}
        y_ticks_list = []
        x_ticks_dict = {}
        flag1 = 0
        for name_y, df_group_y_i in df_group_y:
            matrix_i = []
            y_ticks = (name_y[0].left + name_y[0].right) / 2
            y_ticks = np.round(y_ticks, round_y)
            y_ticks_list.append(y_ticks)
            df_group_x, _ = self.df_bin(df_group_y_i, col_name_x, bin_x)
            flag2 = 0
            for name_x, df_group_x_i in df_group_x:
                vals = df_group_x_i[val_col_name].tolist()
                rt_mean = len(vals)
                matrix_i.append(rt_mean)
                x_ticks = (name_x[0].left + name_x[0].right) / 2
                x_ticks = np.round(x_ticks, round_x)
                x_ticks_dict[x_ticks] = 0
                key = (flag1, flag2)
                matrix_dict[key] = rt_mean
                flag2 += 1
            flag1 += 1
        x_ticks_list = list(x_ticks_dict.keys())
        x_ticks_list.sort()
        return matrix_dict,x_ticks_list,y_ticks_list
    def df_bin(self, df, col, bins):
        df_copy = df.copy()
        df_copy[f'{col}_bins'] = pd.cut(df[col], bins=bins)
        df_group = df_copy.groupby([f'{col}_bins'],observed=True)
        bins_name = df_group.groups.keys()
        bins_name_list = list(bins_name)
        bins_list_str = [str(i) for i in bins_name_list]
        # for name,df_group_i in df_group:
        #     vals = df_group_i[col].tolist()
        #     mean = np.nanmean(vals)
        #     err,_,_ = self.uncertainty_err(SM)
        #     # x_list.append(name)
        #     y_list.append(mean)
        #     err_list.append(err)
        return df_group, bins_list_str
    def plot_df_bin_2d_matrix(self, matrix_dict, vmin, vmax, x_ticks_list, y_ticks_list, cmap='RdBu',
                              is_only_return_matrix=False):
        print(x_ticks_list)
        keys = list(matrix_dict.keys())
        r_list = []
        c_list = []
        for r, c in keys:
            r_list.append(r)
            c_list.append(c)
        r_list = set(r_list)
        c_list = set(c_list)

        row = len(r_list)
        col = len(c_list)
        spatial = []
        for r in range(row):
            temp = []
            for c in range(col):
                key = (r, c)
                if key in matrix_dict:
                    val_pix = matrix_dict[key]
                    temp.append(val_pix)
                else:
                    temp.append(np.nan)
            spatial.append(temp)

        matrix = np.array(spatial, dtype=float)
        matrix = matrix[::-1]
        if is_only_return_matrix:
            return matrix
        plt.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(range(len(c_list)), x_ticks_list)
        plt.yticks(range(len(r_list)), y_ticks_list[::-1])
        # plt.colorbar()
        # plt.show()








def main():
    Rt_Rs().run()
    pass





if __name__ == '__main__':
    main()
    pass