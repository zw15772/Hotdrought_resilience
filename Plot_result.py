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
        self.class_dir=rf'F:\Hotdrought_Resilience\results\analysis\Dataframe\arr\dataframe\\'
        print(self.class_dir)

        self.dff = join(self.class_dir,  'dataframe.df')

        self.outdir=join(results_root, 'Plot_result')

        pass
    def run(self):
        # self.plot_Rt_Rs()
        self.plot_Rt_Rs_spatial_map()
        # self.plot_ratio_unrecovered()
        # self.moving_window_extraction()

        pass
    def plot_Rt_Rs(self):
        df=T.load_df(self.dff)
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

        df_normal=df[df['drought_type'] == 'hot-drought']
        df_group=T.df_groupby(df_normal,'pix')
        for pix in df_group:
            df_pix=df_group[pix]
            val_list=df_pix['GS_NDVI_relative_change'].tolist()
            val_mean=np.nanmean(val_list)
            spatial_dic[pix]=val_mean

        array=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic)
        outdir=join(self.outdir,'normal')
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
        dic_koppen_name = {1: "Tropical", 2: "Temperate", 3: "Arid", 4: "Cold"}

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






def main():
    Rt_Rs().run()
    pass





if __name__ == '__main__':
    main()
    pass