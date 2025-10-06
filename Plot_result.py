import numpy as np

from meta_info import *

this_root='/Volumes/SSD1T/Hotdrought_Resilience/'
data_root = join(this_root,'data')
results_root = join(this_root,'results')
temp_root = join(this_root,'temp')

result_root_this_script = join(results_root, 'Plot_result')

class Rt_Rs:
    def __init__(self):
        self.class_dir=join(results_root,'analysis/Dataframe/arr/dataframe/')
        print(self.class_dir)

        self.dff = join(self.class_dir,  'dataframe.df')

        self.outdir=join(results_root, 'Plot_result')

        pass
    def run(self):
        self.plot_Rt_Rs()

        pass
    def plot_Rt_Rs(self):
        df=T.load_df(self.dff)
        print(len(df))
        df=self.df_clean(df)
        print(len(df))

        df_normal=df[df['drought_type'] == 'normal-drought']

        ## heatmap
        val_list=[]
        aridity=df['Aridity'].tolist()
        # plt.hist(aridity)
        # plt.show()
        # exit()
        aridity_col='Aridity'

        aridity_bin=np.linspace(0,2.5,20)
        df_groupe1,bin_list_str=T.df_bin(df_normal,aridity_col,aridity_bin)
        for name1,df_groupe_i1 in df_groupe1:
            name1_=name1[0].left
            val=np.nanmean(df_groupe_i1['GS_NDVI'])
            val_list.append(val)
        plt.bar(bin_list_str,val_list)
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


        df = df[df['landcover_classfication'] != 'Cropland']
        df = df[df['landcover_classfication'] != 'Bare']

        return df




def main():
    Rt_Rs().run()
    pass





if __name__ == '__main__':
    main()
    pass