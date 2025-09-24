# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from lytools import *
import shutil
T = Tools()
D_005 = DIC_and_TIF(pixelsize=0.05)
D_008 = DIC_and_TIF(pixelsize=0.08)
# result_root_this_script = '/Volumes/NVME4T/temp_work_files/GPP_compare'
# data_root = join(result_root_this_script,'data')
this_root = '/Users/liyang/Projects_data/GPP_compare/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
temp_root = this_root + 'temp/'
result_root_this_script = join(results_root, 'GPP_compare')

class GLC2000:
    def __init__(self):
        self.datadir = join(data_root,'GLC2000')
        pass

    def run(self):
        self.resample()
        # self.unify()
        # self.reclass_lc()
        # self.check_reclass()
        self.reverse_lc_dict()
        pass

    def resample(self):
        fpath = join(self.datadir,'origin','glc2000_v1_1.tif')
        outdir = join(self.datadir,'resample')
        T.mk_dir(outdir)
        # mojority resample
        outf = join(outdir,'glc2000_005.tif')
        ToRaster().resample_reproj(fpath,outf,0.05)

    def unify(self):
        fpath = join(self.datadir,'resample','glc2000_005.tif')
        outdir = join(self.datadir,'unify')
        T.mk_dir(outdir)
        outf = join(outdir,'glc2000_005.tif')
        DIC_and_TIF().unify_raster1(fpath,outf,0.05)
        pass

    def reclass_lc(self):
        outdir = join(self.datadir, 'reclass_lc')
        T.mk_dir(outdir)
        outf = join(outdir,'glc2000_005.npy')
        excel = join(self.datadir,'origin','Global_Legend.xls')
        tif = join(self.datadir,'unify','glc2000_005.tif')
        legend_df = pd.read_excel(excel)
        val_dic = T.df_to_dic(legend_df,'VALUE')
        spatial_dic = DIC_and_TIF(pixelsize=0.05).spatial_tif_to_dic(tif)
        lc_dict = {
            'evergreen':1,
            'deciduous':2,
            'shrubs':3,
            'grass':4,
            'crop':5,
            'mixed':5,
        }
        reclass_dic = {}
        reclass_num_dic = {}
        for pix in tqdm(spatial_dic):
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            val = int(val)
            if not val in val_dic:
                continue
            lc = val_dic[val]['reclass_1']
            # lc = val_dic[val]['reclass_2']
            if type(lc) == float:
                continue
            # if not lc in lc_dict:
            #     continue
            reclass_dic[pix] = lc
            # val = lc_dict[lc]
            # reclass_num_dic[pix] = val
        T.save_npy(reclass_dic,outf)

    def check_reclass(self):
        fpath = join(self.datadir, 'reclass_lc','glc2000_005.npy')
        spatial_dict = T.load_npy(fpath)
        lc_dict = {
            'evergreen': 1,
            'deciduous': 2,
            'shrubs': 3,
            'grass': 4,
            'crop': 5,
            'mixed': 6,
        }
        spatial_dict_val = {}
        for pix in tqdm(spatial_dict):
            lc = spatial_dict[pix]
            val = lc_dict[lc]
            spatial_dict_val[pix] = val
        arr = DIC_and_TIF(pixelsize=0.05).pix_dic_to_spatial_arr(spatial_dict_val)
        plt.imshow(arr,interpolation='nearest')
        plt.colorbar()
        plt.show()

    def reverse_lc_dict(self):
        fpath = join(self.datadir, 'reclass_lc','glc2000_005.npy')
        spatial_dict = T.load_npy(fpath)
        reverse_spatial_dict = T.reverse_dic(spatial_dict)
        return reverse_spatial_dict

class MODIS_LC:

    def __init__(self):
        self.datadir = join(data_root,'MODIS_LC')
        pass

    def run(self):
        # self.hdf_to_tif()
        self.reclass_lc()
        pass

    def hdf_to_tif(self):
        fpath = join(self.datadir,'hdf','MCD12C1.A2001001.061.2022146170409.hdf')
        outdir = join(self.datadir,'tif')
        T.mk_dir(outdir)
        outf = join(outdir,'MCD12C1.A2001001.061.2022146170409.tif')
        hdf_ds = gdal.Open(fpath, gdal.GA_ReadOnly)
        # print(hdf_ds.GetSubDatasets())
        band_ds = gdal.Open(hdf_ds.GetSubDatasets()[0][0], gdal.GA_ReadOnly)
        arr = band_ds.ReadAsArray()
        ToRaster().array2raster(outf,-180,90,0.05,-0.05,arr)

    def reclass_lc(self):
        outdir = join(self.datadir, 'reclass_lc')
        T.mk_dir(outdir)
        outf = join(outdir,'2001_lc.npy')
        excel = join(self.datadir,'Global_Legend.xls')
        tif = join(self.datadir,'tif/2001_lc.tif')
        legend_df = pd.read_excel(excel)
        val_dic = T.df_to_dic(legend_df,'VALUE')
        spatial_dic = DIC_and_TIF(pixelsize=0.05).spatial_tif_to_dic(tif)
        reclass_dic = {}
        reclass_num_dic = {}
        for pix in tqdm(spatial_dic):
            val = spatial_dic[pix]
            if np.isnan(val):
                continue
            val = int(val)
            if not val in val_dic:
                continue
            lc = val_dic[val]['reclass_1']
            # lc = val_dic[val]['reclass_2']
            if type(lc) == float:
                continue
            # if not lc in lc_dict:
            #     continue
            reclass_dic[pix] = lc
            # val = lc_dict[lc]
            # reclass_num_dic[pix] = val
        T.save_npy(reclass_dic,outf)

    def reverse_lc_dict(self):
        fpath = join(self.datadir, 'reclass_lc','2001_lc.npy')
        spatial_dict = T.load_npy(fpath)
        reverse_spatial_dict = T.reverse_dic(spatial_dict)
        picked_lc_list = [
            'Closed Shrublands',
            'Grasslands',
            'Open Shrublands',
            'Savannas',
            'Woody Savannas',
        ]
        picked_reverse_spatial_dict = {}
        for lc in picked_lc_list:
            picked_reverse_spatial_dict[lc] = reverse_spatial_dict[lc]
        return picked_reverse_spatial_dict

class Aridity_index:

    def __init__(self):
        self.datadir = join(data_root,'Aridity_index')
        pass

    def run(self):
        # self.resample()
        self.dryland_pix()
        pass

    def resample(self):
        fpath = join(self.datadir,'Aridity_index.tif')
        outpath = join(self.datadir,'Aridity_index_005.tif')
        ToRaster().resample_reproj(fpath,outpath,0.05)

    def dryland_pix(self):
        fpath = join(self.datadir,'Aridity_index_005.tif')
        arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
        arr = np.array(arr)
        arr[arr<0] = np.nan
        arr[arr>0.65] = np.nan
        dryland_pix = {}
        for r in tqdm(range(len(arr)),desc='dryland_pix'):
            for c in range(len(arr[0])):
                val = arr[r][c]
                if np.isnan(val):
                    continue
                pix = (r,c)
                dryland_pix[pix] = val
        outf = join(self.datadir,'dryland_pix.npy')
        T.save_npy(dryland_pix,outf)

class GPP_preprocess:

    def __init__(self):
        self.datadir = join(data_root,'GPP')
        # self.get_product_name()
        self.product_list = ['LT_CFE-Hybrid_NT']

        # self.product = 'LT_Baseline_NT'
        # self.datarange = '1982-2020'
        pass

    def run(self):
        for product in self.product_list:
            self.product = product
            # self.nc_to_tif()
            # self.average()
            self.longterm_average_picked_LC()
            # self.annual_per_pix_unit_change()
            # self.perpix()
        pass

    def get_product_name(self):
        fdir = join(self.datadir, 'zips')
        product_list = []
        for product in T.listdir(fdir):
            product = product.split('-')[:-1]
            product = '-'.join(product)
            product_list.append(product)
        self.product_list = product_list

    def nc_to_tif(self):
        fdir = join(self.datadir,'nc',self.product)
        outdir = join(self.datadir,'tif',self.product)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir),desc=self.product):
            fpath = join(fdir,f)
            outf = join(outdir,f.replace('.nc','.tif'))
            nc = Dataset(fpath)
            arr = nc.variables['GPP_mean'][:][0]
            arr = np.array(arr) * 0.01
            arr[arr<-9] = np.nan
            longitude_start = -180
            latitude_start = 90
            pixelWidth = 0.05
            pixelHeight = -0.05
            ToRaster().array2raster(outf,longitude_start, latitude_start, pixelWidth, pixelHeight, arr)


    def perpix(self):
        fdir = join(self.datadir,'tif025',self.product,self.datarange)
        outdir = join(self.datadir,'per_pix',self.product,self.datarange)
        T.mk_dir(outdir,force=True)
        Pre_Process().data_transform(fdir,outdir)

    def get_year_list(self):
        fdir = join(self.this_class_tif,'annual_average')
        year_dict = {}
        for product in T.listdir(fdir):
            fdir_i = join(fdir,product)
            year_list = []
            for f in T.listdir(fdir_i):
                year = f.replace('.tif','')
                year_list.append(year)
            year_list = list(set(year_list))
            year_list.sort()
            year_dict[product] = year_list
        return year_dict

    def average(self):
        product_list = GPP_preprocess().product_list
        fdir = join(GPP_preprocess().datadir,'tif')
        outdir = join(self.datadir,'average')
        T.mk_dir(outdir)
        for product in product_list:
            fdir_i = join(fdir,product)
            arr_sum = 0
            flag = 0
            for f in tqdm(T.listdir(fdir_i),desc=product):
                fpath = join(fdir_i,f)
                arr,originX,originY,pixelWidth,pixelHeight = ToRaster().raster2array(fpath)
                arr_sum += arr
                flag += 1
            arr_mean = arr_sum / flag
            arr_mean = arr_mean * 365 / 1000
            outf = join(outdir,product+'.tif')
            ToRaster().array2raster(outf,originX,originY,pixelWidth,pixelHeight,arr_mean)

    def plot_average(self):
        fdir = join(self.datadir,'average')
        outdir = join(self.this_class_png,'average')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir),desc='plot_average'):
            if not f.endswith('.tif'):
                continue
            title = f.replace('.tif','')
            fpath = join(fdir,f)
            Plot().plot_Robinson(fpath,vmin=0,vmax=9,res=5000)
            plt.title(title)
            outf = join(outdir,f.replace('.tif','.png'))
            plt.savefig(outf,dpi=1200)
            plt.close()
        T.open_path_and_file(outdir)

    def tif_annual_average(self):
        fdir = join(GPP_preprocess().datadir,'tif')
        outdir = join(self.datadir,'annual_average')
        T.mk_dir(outdir)
        product_list = GPP_preprocess().product_list
        for product in product_list:
            fdir_i = join(fdir,product)
            outdir_i = join(outdir,product)
            T.mk_dir(outdir_i,force=True)
            year_list = []
            for f in T.listdir(fdir_i):
                date = f.split('.')[0].split('_')[-1]
                year = date[:4]
                year_list.append(year)
            year_list = list(set(year_list))
            year_list.sort()

            for year in tqdm(year_list,desc=product):
                picked_list = []
                for f in T.listdir(fdir_i):
                    date = f.split('.')[0].split('_')[-1]
                    year_str = date[:4]
                    if year_str == year:
                        picked_list.append(join(fdir_i,f))
                outf = join(outdir_i,year+'.tif')
                self.compose_annual_tif(picked_list,outf)
        pass

    def compose_annual_tif(self,flist,outf):
        arr_sum = 0
        flag = 0
        for fpath in flist:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr_sum += arr
            flag += 1
        arr_mean = arr_sum / flag
        # outf = join(outdir, product + '.tif')
        ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, arr_mean)

    def annual_per_pix(self):
        fdir = join(self.this_class_tif, 'annual_average')
        outdir = join(self.this_class_arr, 'annual_per_pix')
        T.mk_dir(outdir)
        product_list = GPP_preprocess().product_list
        for product in product_list:
            fdir_i = join(fdir, product)
            outdir_i = join(outdir, product)
            T.mk_dir(outdir_i, force=True)
            Pre_Process().data_transform(fdir_i, outdir_i,n=1000000)

    def annual_per_pix_unit_change(self):
        fdir = join(self.datadir, 'annual_per_pix',self.product)
        outdir = join(self.datadir, 'annual_per_pix_unit_change',self.product)
        T.mk_dir(outdir,force=True)
        for f in tqdm(T.listdir(fdir)):
            fpath = join(fdir,f)
            spatial_dict = T.load_npy(fpath)
            spatial_dict_new = {}
            for pix in spatial_dict:
                vals = spatial_dict[pix]
                if T.is_all_nan(vals):
                    continue
                vals_new = vals * 365 / 1000
                spatial_dict_new[pix] = vals_new
            outf = join(outdir,f)
            T.save_npy(spatial_dict_new,outf)

    def longterm_average_picked_LC(self):
        fpath = join(self.datadir, 'average',f'{self.product}.tif')
        outdir = join(self.datadir, 'average_LC')
        T.mkdir(outdir)
        outpath = join(outdir,f'{self.product}.tif')
        lc_dict = MODIS_LC().reverse_lc_dict()
        picked_lc_list = []
        for lc in tqdm(lc_dict):
            pixlist = lc_dict[lc]
            print(lc)
            for pix in pixlist:
                picked_lc_list.append(pix)
            # break
        # exit()
        # print(len(picked_lc_list))
        # exit()
        spatial_dict = DIC_and_TIF(pixelsize=0.05).spatial_tif_to_dic(fpath)
        spatial_dict_picked = {}
        for pix in tqdm(picked_lc_list):
            # if not pix in spatial_dict:
            #     continue
            spatial_dict_picked[pix] = spatial_dict[pix]
            # a=spatial_dict[pix]
        arr = DIC_and_TIF(pixelsize=0.05).pix_dic_to_spatial_arr(spatial_dict_picked)
        ToRaster().array2raster(outpath,-180,90,0.05,-0.05,arr)

class LAI4g_preprocess:
    def __init__(self):
        self.datadir = join(data_root,'LAI4g')
        pass

    def run(self):
        # self.rename()
        # self.resample()
        # self.clean_tif()
        # self.tif_every_year_split()
        # self.MVC()
        self.tif_annual_average()
        # self.tif_to_perpix_1982_2020()
        # self.longterm_average()
        # self.longterm_average_picked_LC()

        pass

    def rename(self):
        fdir = join(self.datadir,'tif')
        for f in tqdm(T.listdir(fdir)):
            date = f.split('.')[-2].split('_')[-1]
            year = date[:4]
            mon = date[4:6]
            day = date[6:]
            if day == '02':
                day = '15'
            newf = f'{year}{mon}{day}.tif'
            new_f_path = join(fdir,newf)
            os.rename(join(fdir,f),new_f_path)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_005')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            if os.path.isfile(outf):
                continue
            f_path = join(fdir,f)
            ToRaster().resample_reproj(f_path,outf,0.05)
        pass

    def clean_tif(self):
        # fdir = join(self.datadir,'tif_005')
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_008_clean')
        # outdir = join(self.datadir,'tif_005_clean')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            f_path = join(fdir,f)
            outf = join(outdir,f)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_path)
            arr = np.array(arr,dtype=float)
            arr[arr<0] = np.nan
            arr[arr>1000] = np.nan
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, arr)

    def tif_every_year_split(self):
        fdir = join(self.datadir,'tif_008_clean')
        outdir = join(self.datadir,'tif_008_every_year_split')
        T.mk_dir(outdir)
        year_list = []
        for f in T.listdir(fdir):
            date = f.split('.')[0]
            year = date[:4]
            year_list.append(year)
        year_list = list(set(year_list))
        year_list.sort()
        for year in year_list:
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i)
            picked_list = []
            for f in T.listdir(fdir):
                date = f.split('.')[0]
                year_str = date[:4]
                if year_str == year:
                    picked_list.append(join(fdir,f))
            for fpath in tqdm(picked_list,desc=year):
                outf = join(outdir_i,fpath.split('/')[-1])
                shutil.copy(fpath,outf)

    def MVC(self):
        fdir = join(self.datadir,'tif_every_year_split')
        outdir = join(self.datadir,'tif_MVC')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            fdir_i = join(fdir,year)
            Pre_Process().monthly_compose(fdir_i, outdir, method='max')
            exit()

    def kernel_MVC(self,param):
        fdir,outdir = param
        Pre_Process().monthly_compose(fdir,outdir,method='max')

    def tif_annual_average(self):
        # fdir = join(self.datadir,'tif_every_year_split')
        fdir = join(self.datadir,'tif_008_every_year_split')
        outdir = join(self.datadir,'tif_008_annual_average')
        # outdir = join(self.datadir,'tif_annual_average')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            picked_list = []
            for f in T.listdir(join(fdir,year)):
                date = f.split('.')[0].split('_')[-1]
                year_str = date[:4]
                if year_str == year:
                    picked_list.append(join(fdir,year,f))
            # print(picked_list)
            # exit()
            outf = join(outdir,year+'.tif')
            self.compose_annual_tif(picked_list,outf)
            # exit()
        pass
    def compose_annual_tif(self,flist,outf):
        arr_sum = 0
        flag = 0
        for fpath in flist:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = np.array(arr)
            arr[arr>1000] = np.nan
            arr[np.isnan(arr)] = 0
            arr_sum += arr
            flag += 1
        arr_mean = arr_sum / flag
        # outf = join(outdir, product + '.tif')
        arr_mean[arr_mean==0] = np.nan
        ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, arr_mean)

    def tif_to_perpix_1982_2020(self):
        # fdir = join(self.datadir, 'tif_MVC_annual_average')
        fdir = join(self.datadir, 'tif_annual_average')
        outdir = join(self.datadir, 'per_pix/1982-2020')
        T.mk_dir(outdir, force=True)
        selected_tif_list = []
        for y in range(1982, 2021):
            f = '{}.tif'.format(y)
            selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list, n=1000000)

    def longterm_average(self):
        fdir = join(self.datadir, 'tif_annual_average')
        outdir = join(self.datadir, 'longterm_average')
        T.mk_dir(outdir)
        flist = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            flist.append(join(fdir,f))
        self.compose_annual_tif(flist,join(outdir,'LAI4g.tif'))

        pass


    def longterm_average_picked_LC(self):
        fpath = join(self.datadir, 'longterm_average','LAI4g.tif')
        outpath = join(self.datadir,'longterm_average','LAI4g_LC.tif')
        lc_dict = MODIS_LC().reverse_lc_dict()
        picked_lc_list = []
        for lc in tqdm(lc_dict):
            pixlist = lc_dict[lc]
            print(lc)
            for pix in pixlist:
                picked_lc_list.append(pix)
            # break
        # exit()
        # print(len(picked_lc_list))
        # exit()
        spatial_dict = DIC_and_TIF(pixelsize=0.05).spatial_tif_to_dic(fpath)
        spatial_dict_picked = {}
        for pix in tqdm(picked_lc_list):
            # if not pix in spatial_dict:
            #     continue
            spatial_dict_picked[pix] = spatial_dict[pix]
            # a=spatial_dict[pix]
        arr = DIC_and_TIF(pixelsize=0.05).pix_dic_to_spatial_arr(spatial_dict_picked)
        ToRaster().array2raster(outpath,-180,90,0.05,-0.05,arr)

class NDVI4g_preprocess:
    def __init__(self):
        self.datadir = join(data_root,'NDVI4g')
        pass

    def run(self):
        # self.rename()
        # self.resample()
        # self.clean_tif()
        # self.tif_every_year_split()
        # self.MVC()
        self.tif_annual_average()
        # self.tif_to_perpix_1982_2020()
        # self.longterm_average()
        # self.longterm_average_picked_LC()

        pass

    def rename(self):
        fdir = join(self.datadir,'tif')
        for f in tqdm(T.listdir(fdir)):
            date = f.split('.')[-2].split('_')[-1]
            year = date[:4]
            mon = date[4:6]
            day = date[6:]
            if day == '02':
                day = '15'
            newf = f'{year}{mon}{day}.tif'
            new_f_path = join(fdir,newf)
            os.rename(join(fdir,f),new_f_path)
        pass

    def resample(self):
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_005')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            outf = join(outdir,f)
            if os.path.isfile(outf):
                continue
            f_path = join(fdir,f)
            ToRaster().resample_reproj(f_path,outf,0.05)
        pass

    def clean_tif(self):
        # fdir = join(self.datadir,'tif_005')
        fdir = join(self.datadir,'tif')
        outdir = join(self.datadir,'tif_008_clean')
        # outdir = join(self.datadir,'tif_005_clean')
        T.mk_dir(outdir)
        for f in tqdm(T.listdir(fdir)):
            f_path = join(fdir,f)
            outf = join(outdir,f)
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(f_path)
            arr = np.array(arr,dtype=float)
            arr[arr<0] = np.nan
            arr[arr>10000] = np.nan
            ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, arr)

    def tif_every_year_split(self):
        fdir = join(self.datadir,'tif_008_clean')
        outdir = join(self.datadir,'tif_008_every_year_split')
        T.mk_dir(outdir)
        year_list = []
        for f in T.listdir(fdir):
            date = f.split('.')[0]
            year = date[:4]
            year_list.append(year)
        year_list = list(set(year_list))
        year_list.sort()
        for year in year_list:
            outdir_i = join(outdir,year)
            T.mk_dir(outdir_i)
            picked_list = []
            for f in T.listdir(fdir):
                date = f.split('.')[0]
                year_str = date[:4]
                if year_str == year:
                    picked_list.append(join(fdir,f))
            for fpath in tqdm(picked_list,desc=year):
                outf = join(outdir_i,fpath.split('/')[-1])
                shutil.copy(fpath,outf)

    def MVC(self):
        fdir = join(self.datadir,'tif_every_year_split')
        outdir = join(self.datadir,'tif_MVC')
        T.mk_dir(outdir)
        for year in T.listdir(fdir):
            fdir_i = join(fdir,year)
            Pre_Process().monthly_compose(fdir_i, outdir, method='max')
            exit()

    def kernel_MVC(self,param):
        fdir,outdir = param
        Pre_Process().monthly_compose(fdir,outdir,method='max')

    def tif_annual_average(self):
        # fdir = join(self.datadir,'tif_every_year_split')
        fdir = join(self.datadir,'tif_008_every_year_split')
        outdir = join(self.datadir,'tif_008_annual_average')
        # outdir = join(self.datadir,'tif_annual_average')
        T.mk_dir(outdir)
        for year in tqdm(T.listdir(fdir)):
            picked_list = []
            for f in T.listdir(join(fdir,year)):
                date = f.split('.')[0].split('_')[-1]
                year_str = date[:4]
                if year_str == year:
                    picked_list.append(join(fdir,year,f))
            # print(picked_list)
            # exit()
            outf = join(outdir,year+'.tif')
            self.compose_annual_tif(picked_list,outf)
            # exit()
        pass
    def compose_annual_tif(self,flist,outf):
        arr_sum = 0
        flag = 0
        for fpath in flist:
            arr, originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(fpath)
            arr = np.array(arr)
            arr[arr>10000] = np.nan
            arr[np.isnan(arr)] = 0
            arr_sum += arr
            flag += 1
        arr_mean = arr_sum / flag
        # outf = join(outdir, product + '.tif')
        arr_mean[arr_mean==0] = np.nan
        ToRaster().array2raster(outf, originX, originY, pixelWidth, pixelHeight, arr_mean)

    def tif_to_perpix_1982_2020(self):
        # fdir = join(self.datadir, 'tif_MVC_annual_average')
        fdir = join(self.datadir, 'tif_annual_average')
        outdir = join(self.datadir, 'per_pix/1982-2020')
        T.mk_dir(outdir, force=True)
        selected_tif_list = []
        for y in range(1982, 2021):
            f = '{}.tif'.format(y)
            selected_tif_list.append(f)
        Pre_Process().data_transform_with_date_list(fdir, outdir, selected_tif_list, n=1000000)

    def longterm_average(self):
        fdir = join(self.datadir, 'tif_annual_average')
        outdir = join(self.datadir, 'longterm_average')
        T.mk_dir(outdir)
        flist = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            flist.append(join(fdir,f))
        self.compose_annual_tif(flist,join(outdir,'LAI4g.tif'))

        pass


    def longterm_average_picked_LC(self):
        fpath = join(self.datadir, 'longterm_average','LAI4g.tif')
        outpath = join(self.datadir,'longterm_average','LAI4g_LC.tif')
        lc_dict = MODIS_LC().reverse_lc_dict()
        picked_lc_list = []
        for lc in tqdm(lc_dict):
            pixlist = lc_dict[lc]
            print(lc)
            for pix in pixlist:
                picked_lc_list.append(pix)
            # break
        # exit()
        # print(len(picked_lc_list))
        # exit()
        spatial_dict = DIC_and_TIF(pixelsize=0.05).spatial_tif_to_dic(fpath)
        spatial_dict_picked = {}
        for pix in tqdm(picked_lc_list):
            # if not pix in spatial_dict:
            #     continue
            spatial_dict_picked[pix] = spatial_dict[pix]
            # a=spatial_dict[pix]
        arr = DIC_and_TIF(pixelsize=0.05).pix_dic_to_spatial_arr(spatial_dict_picked)
        ToRaster().array2raster(outpath,-180,90,0.05,-0.05,arr)

class GPP_analysis:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('GPP_analysis', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.ts_LC()
        # self.plot_ts_LC()
        pass

    def ts_LC(self):
        outdir = join(self.this_class_arr,'ts_LC')
        T.mk_dir(outdir)
        lc_dict = GLC2000().reverse_lc_dict()
        product_list = GPP_preprocess().product_list
        per_pix_dir = join(self.this_class_arr, 'annual_per_pix')
        params_list = []
        for product in product_list:
            params = (per_pix_dir,product,lc_dict,outdir)
            params_list.append(params)
        MULTIPROCESS(self.kernel_ts_LC,params_list).run()

    def kernel_ts_LC(self,params):
        per_pix_dir,product,lc_dict,outdir = params
        fdir = join(per_pix_dir, product)
        result_dict = {}
        for lc in lc_dict:
            pix_list = lc_dict[lc]
            pix_list = set(pix_list)
            vals_list = []
            std_list = []
            err_list = []
            for f in tqdm(T.listdir(fdir), desc=f'{product}-{lc}'):
                fpath = join(fdir, f)
                spatial_dict = T.load_npy(fpath)
                for pix in pix_list:
                    if not pix in spatial_dict:
                        continue
                    vals = spatial_dict[pix]
                    if T.is_all_nan(vals):
                        continue
                    vals_list.append(vals)
            vals_list_mean = np.nanmean(vals_list, axis=0)
            vals_list_std = np.nanstd(vals_list, axis=0)
            err = T.uncertainty_err_2d(vals_list, axis=0)
            result_dict[lc] = {
                'mean_list': vals_list_mean,
                'std_list': vals_list_std,
                'err_list': err,
            }
        outf = join(outdir, product + '.npy')
        T.save_npy(result_dict, outf)

    def plot_ts_LC(self):
        fdir = join(self.this_class_arr,'ts_LC')
        for f in T.listdir(fdir):
            fpath = join(fdir,f)
            product = f.replace('.npy','')
            year_list = self.get_year_list()[product]
            result_dict = T.load_npy(fpath)
            # plt.figure()
            for lc in result_dict:
                result = result_dict[lc]
                mean_list = result['mean_list']
                std_list = result['std_list']
                # err_list = result['err_list']
                plt.plot(year_list,mean_list,label=f'{lc}-{product}')
                plt.fill_between(np.arange(len(mean_list)),mean_list-std_list,mean_list+std_list,alpha=0.5)
                # plt.fill_between(np.arange(len(mean_list)),mean_list-err_list,mean_list+err_list,alpha=0.5)
            plt.legend()
            plt.xticks(rotation=90)
            # plt.title(product)
            plt.tight_layout()
            plt.show()

class Products_compare:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Products_compare', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.GPP_ts_LC()
        # self.LAI_ts_LC()
        # self.plot_ts_LC()
        self.ts_LC_slope()
        # self.plot_spatial_LC_LAI4g()
        # self.plot_spatial_LC_GPP()
        # self.spatial_trend_LAI()
        # self.spatial_trend_GPP()
        # self.tif_spatial_trend()
        # self.tif_spatial_trend_pick_LC()
        # self.plot_spatial_trend_pick_LC()
        # self.statistic_global_spatial_trend_pick_LC()
        # self.statistic_global_spatial_trend_significance_pick_LC()
        # self.spatial_trend_LC_statistic_df()
        # self.spatial_trend_LC_statistic()
        pass

    def GPP_ts_LC(self):
        outdir = join(self.this_class_arr,'ts_LC')
        T.mk_dir(outdir)
        lc_dict = MODIS_LC().reverse_lc_dict()
        product_list = [
            # 'LT_Baseline_DT',
            # 'LT_Baseline_NT',
            # 'LT_CFE-Hybrid_DT',
            'LT_CFE-Hybrid_NT',
        ]
        GPP_per_pix_dir = join(GPP_preprocess().datadir, 'annual_per_pix_unit_change')
        # params_list = []
        for product in product_list:
            fdir = join(GPP_per_pix_dir, product)
            result_dict = {}
            for lc in lc_dict:
                pix_list = lc_dict[lc]
                pix_list = set(pix_list)
                vals_list = []
                std_list = []
                err_list = []
                for f in tqdm(T.listdir(fdir), desc=f'{product}-{lc}'):
                    fpath = join(fdir, f)
                    spatial_dict = T.load_npy(fpath)
                    for pix in pix_list:
                        if not pix in spatial_dict:
                            continue
                        vals = spatial_dict[pix]
                        if T.is_all_nan(vals):
                            continue
                        vals_list.append(vals)
                vals_list_mean = np.nanmean(vals_list, axis=0)
                vals_list_std = np.nanstd(vals_list, axis=0)
                err = T.uncertainty_err_2d(vals_list, axis=0)
                result_dict[lc] = {
                    'mean_list': vals_list_mean,
                    'std_list': vals_list_std,
                    'err_list': err,
                }
            outf = join(outdir, product + '.npy')
            T.save_npy(result_dict, outf)

    def LAI_ts_LC(self):
        outdir = join(self.this_class_arr,'ts_LC')
        T.mk_dir(outdir)
        lc_dict = MODIS_LC().reverse_lc_dict()
        LAI_per_pix_dir = join(LAI4g_preprocess().datadir, 'per_pix','1982-2020')
        # params_list = []
        fdir = LAI_per_pix_dir
        result_dict = {}
        product = 'LAI4g'
        for lc in lc_dict:
            pix_list = lc_dict[lc]
            pix_list = set(pix_list)
            vals_list = []
            std_list = []
            err_list = []
            for f in tqdm(T.listdir(fdir), desc=f'{product}-{lc}'):
                fpath = join(fdir, f)
                spatial_dict = T.load_npy(fpath)
                for pix in pix_list:
                    if not pix in spatial_dict:
                        continue
                    vals = spatial_dict[pix]
                    if T.is_all_nan(vals):
                        continue
                    vals_list.append(vals)
            vals_list_mean = np.nanmean(vals_list, axis=0)
            vals_list_std = np.nanstd(vals_list, axis=0)
            err = T.uncertainty_err_2d(vals_list, axis=0)
            result_dict[lc] = {
                'mean_list': vals_list_mean,
                'std_list': vals_list_std,
                'err_list': err,
            }
        outf = join(outdir, product + '.npy')
        T.save_npy(result_dict, outf)

    def plot_ts_LC(self):
        fdir = join(self.this_class_arr,'ts_LC')
        outdir = join(self.this_class_png,'ts_LC')
        T.mk_dir(outdir)
        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            print(f)
            fpath = join(fdir,f)
            product = f.replace('.npy','')
            year_list = range(1982,2021)
            result_dict = T.load_npy(fpath)
            # plt.figure()
            for lc in result_dict:
                result = result_dict[lc]
                mean_list = result['mean_list']
                std_list = result['std_list']
                # err_list = result['err_list']
                plt.plot(year_list,mean_list,label=f'{lc}-{product}')
                plt.scatter(year_list,mean_list)
                # plt.fill_between(np.arange(len(mean_list)),mean_list-std_list,mean_list+std_list,alpha=0.5)
                # plt.fill_between(np.arange(len(mean_list)),mean_list-err_list,mean_list+err_list,alpha=0.5)
            # plt.legend()
            plt.xticks(rotation=90)
            plt.title(f)
            plt.tight_layout()
            # outf = join(outdir,f.replace('.npy','legend.pdf'))
            outf = join(outdir,f.replace('.npy','.pdf'))
            # plt.savefig(outf)
            # plt.close()
            plt.show()
        T.open_path_and_file(outdir)

    def ts_LC_slope(self):
        fdir = join(self.this_class_arr,'ts_LC')
        for f in T.listdir(fdir):
            if not f.endswith('.npy'):
                continue
            # print(f)
            fpath = join(fdir,f)
            product = f.replace('.npy','')
            year_list = range(1982,2021)
            result_dict = T.load_npy(fpath)
            print('----')
            for lc in result_dict:
                result = result_dict[lc]
                mean_list = result['mean_list']
                a,b,r,p = T.nan_line_fit(year_list,mean_list)
                print(f,lc,f'{a*1000:.6f}',f'{p:.3f}')
                # print(mean_list)
                # exit()


    def plot_spatial_LC_LAI4g(self):
        fpath = join(LAI4g_preprocess().datadir,'longterm_average/LAI4g_LC.tif')
        outdir = join(self.this_class_png,'spatial_LC_LAI4g')
        T.mk_dir(outdir)
        Plot().plot_Robinson(fpath,vmin=0,vmax=300,res=5000)
        plt.title('LAI4g_LC')
        outf = join(outdir,'LAI4g_LC.png')
        plt.savefig(outf,dpi=1200)
        plt.close()

    def plot_spatial_LC_GPP(self):
        fdir = join(GPP_preprocess().datadir,'average_LC')
        outdir = join(self.this_class_png, 'spatial_LC_GPP')
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            fpath = join(fdir,f)
            T.mk_dir(outdir)
            Plot().plot_Robinson(fpath,vmin=0,vmax=2,res=5000)
            plt.title(f.replace('.tif',''))
            outf = join(outdir,f'{f}.png')
            plt.savefig(outf,dpi=1200)
            plt.close()

    def spatial_trend_LAI(self):
        product = 'LAI4g'
        fdir = join(LAI4g_preprocess().datadir, 'per_pix/1982-2020')
        outdir = join(self.this_class_arr, 'spatial_trend',product)
        T.mk_dir(outdir,force=True)
        param_list = []
        for f in T.listdir(fdir):
            param_list.append((fdir, f, outdir))
        MULTIPROCESS(self.kernel_cal_trend, param_list).run(process=7)

    def spatial_trend_GPP(self):
        product_list = [
            # 'LT_Baseline_NT',
            'LT_CFE-Hybrid_NT',
        ]
        fdir_father = join(GPP_preprocess().datadir, 'annual_per_pix_unit_change')
        for product in product_list:
            fdir = join(fdir_father, product)
            outdir = join(self.this_class_arr, 'spatial_trend', product)
            T.mk_dir(outdir, force=True)
            param_list = []
            for f in T.listdir(fdir):
                param_list.append((fdir,f,outdir))
            #     self.kernel_cal_trend((fdir, f, outdir))
            MULTIPROCESS(self.kernel_cal_trend,param_list).run(process=7)

    def kernel_cal_trend(self,params):
        fdir,f,outdir = params
        fpath = join(fdir, f)
        spatial_dict = T.load_npy(fpath)
        spatial_dict_result = {}
        for pix in spatial_dict:
            vals = spatial_dict[pix]
            if T.is_all_nan(vals):
                continue
            try:
                a, b, r, p = T.nan_line_fit(list(range(len(vals))), vals)
                spatial_dict_result[pix] = {'a':a,'p':p}
            except:
                pass
        outf = join(outdir,f)
        T.save_npy(spatial_dict_result,outf)

    def tif_spatial_trend(self):
        fdir = join(self.this_class_arr,'spatial_trend')
        outdir = join(self.this_class_tif,'spatial_trend')
        T.mk_dir(outdir)
        for product in T.listdir(fdir):
            print(product)
            # exit()
            # if not product == 'LT_CFE-Hybrid_NT':
            #     continue
            fdir_i = join(fdir,product)
            outdir_i = join(outdir,product)
            T.mk_dir(outdir_i)
            spatial_dict_a = {}
            spatial_dict_p = {}
            for f in tqdm(T.listdir(fdir_i),desc=product):
                fpath = join(fdir_i,f)
                spatial_dict_i = T.load_npy(fpath)
                for pix in spatial_dict_i:
                    a = spatial_dict_i[pix]['a']
                    p = spatial_dict_i[pix]['p']
                    spatial_dict_a[pix] = a
                    spatial_dict_p[pix] = p
            outf_a = join(outdir_i,'a.tif')
            outf_p = join(outdir_i,'p.tif')
            D_005.pix_dic_to_tif(spatial_dict_a,outf_a)
            D_005.pix_dic_to_tif(spatial_dict_p,outf_p)

    def tif_spatial_trend_pick_LC(self):
        fdir = join(self.this_class_tif,'spatial_trend')
        outdir = join(self.this_class_tif,'tif_spatial_trend_pick_LC')

        for product in T.listdir(fdir):
            fdir_i = join(fdir,product)
            outdir_i = join(outdir,product)
            T.mk_dir(outdir_i,force=True)
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir_i,f)
                outpath = join(outdir_i,f)
                self.kernel_tif_spatial_trend_pick_LC(fpath,outpath)

    def statistic_global_spatial_trend_pick_LC(self):
        fdir = join(self.this_class_tif, 'tif_spatial_trend_pick_LC')

        for product in T.listdir(fdir):
            fdir_i = join(fdir, product)
            for f in T.listdir(fdir_i):
                if not f.endswith('.tif'):
                    continue
                fpath = join(fdir_i, f)
                arr = ToRaster().raster2array(fpath)[0]
                arr[arr<-1] = np.nan
                arr[arr>1] = np.nan
                # arr[arr<-999] = np.nan
                mean = np.nanmean(arr)
                std = np.nanstd(arr)
                print(product,f)
                print('mean',f'{mean:.4f}')
                print('std',f'{std:.4f}')
                print('----')

    def statistic_global_spatial_trend_significance_pick_LC(self):
        fdir = join(self.this_class_tif, 'tif_spatial_trend_pick_LC')

        for product in T.listdir(fdir):
            fdir_i = join(fdir, product)
            f_a = join(fdir_i,'a.tif')
            f_p = join(fdir_i,'p.tif')
            arr_a = ToRaster().raster2array(join(fdir_i,f_a))[0]
            arr_p = ToRaster().raster2array(join(fdir_i,f_p))[0]
            arr_a[arr_a<-999] = np.nan
            arr_p[arr_p<-999] = np.nan
            father = 0
            sig_pos = 0
            sig_neg = 0
            for r in tqdm(range(arr_a.shape[0])):
                for c in range(arr_a.shape[1]):
                    val_a = arr_a[r,c]
                    val_p = arr_p[r,c]
                    if np.isnan(val_a):
                        continue
                    # print(val_a,val_p)
                    if val_p < 0.05:
                        if val_a > 0:
                            sig_pos += 1
                        else:
                            sig_neg += 1
                    father += 1
            print(product)
            print('father',father)
            sig_pos_ratio = sig_pos / father * 100
            sig_neg_ratio = sig_neg / father * 100
            print('sig_pos_ratio',sig_pos_ratio)
            print('sig_neg_ratio',sig_neg_ratio)
            print('----')


    def kernel_tif_spatial_trend_pick_LC(self,fpath,outpath):
        # fpath_a = join(fdir, product, 'a.tif')
        # outpath = join(self.datadir, 'longterm_average', 'LAI4g_LC.tif')
        lc_dict = MODIS_LC().reverse_lc_dict()
        picked_lc_list = []
        for lc in tqdm(lc_dict):
            pixlist = lc_dict[lc]
            print(lc)
            for pix in pixlist:
                picked_lc_list.append(pix)
            # break
        # exit()
        # print(len(picked_lc_list))
        # exit()
        spatial_dict = DIC_and_TIF(pixelsize=0.05).spatial_tif_to_dic(fpath)
        spatial_dict_picked = {}
        for pix in tqdm(picked_lc_list):
            # if not pix in spatial_dict:
            #     continue
            spatial_dict_picked[pix] = spatial_dict[pix]
            # a=spatial_dict[pix]
        arr = DIC_and_TIF(pixelsize=0.05).pix_dic_to_spatial_arr(spatial_dict_picked)
        ToRaster().array2raster(outpath, -180, 90, 0.05, -0.05, arr)


    def plot_spatial_trend_pick_LC(self):
        fdir = join(self.this_class_tif,'tif_spatial_trend_pick_LC')
        outdir = join(self.this_class_png,'tif_spatial_trend_pick_LC')
        T.mk_dir(outdir)

        for product in T.listdir(fdir):
            print(product)
            if product == 'LAI4g':
                continue
            plt.figure(figsize=(10, 5))
            fpath_a = join(fdir, product, 'a.tif')
            fpath_p = join(fdir, product, 'p.tif')
            if 'LAI' in product:
                m, ret = Plot().plot_Robinson(fpath_a,vmin=-1,vmax=1,res=5000)
            else:
                m, ret = Plot().plot_Robinson(fpath_a, vmin=-0.007, vmax=0.007, res=5000)
            Plot().plot_Robinson_significance_scatter(m,fpath_p, temp_root,s=5, linewidths=0.3)
            # exit()
            plt.title(product)
            # plt.show()
            outf = join(outdir,product+'.png')
            plt.savefig(outf,dpi=1200)
            plt.close()
        T.open_path_and_file(outdir)

    def spatial_trend_LC_statistic_df(self):
        outdir = join(self.this_class_arr,'spatial_trend_LC_statistic_df')
        T.mk_dir(outdir)
        picked_reverse_spatial_dict = MODIS_LC().reverse_lc_dict()
        LC_spatial_dict = {}
        for lc in picked_reverse_spatial_dict:
            pix_list = picked_reverse_spatial_dict[lc]
            for pix in pix_list:
                LC_spatial_dict[pix] = lc
        fdir = join(self.this_class_tif,'tif_spatial_trend_pick_LC')
        all_dict = {}
        for product in T.listdir(fdir):
            fpath = join(fdir,product,'a.tif')
            spatial_dict = D_005.spatial_tif_to_dic(fpath)
            all_dict[product] = spatial_dict
        all_dict['LC'] = LC_spatial_dict
        df = T.spatial_dics_to_df(all_dict)
        outf = join(outdir,'dataframe.df')
        T.save_df(df,outf)
        T.df_to_excel(df,outf)

    def spatial_trend_LC_statistic(self):
        outdir = join(self.this_class_png,'spatial_trend_LC_statistic')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'spatial_trend_LC_statistic_df','dataframe.df')
        df = T.load_df(dff)
        product_list = [
            'LAI4g',
            # 'LT_Baseline_NT',
            'LT_CFE-Hybrid_NT',
        ]
        lc_list = T.get_df_unique_val_list(df,'LC')
        for product in product_list:
            mean_list = []
            plt.figure(figsize=(10,5))
            for lc in lc_list:
                df_lc = df[df['LC']==lc]
                vals = df_lc[product].tolist()
                mean = np.nanmean(vals)
                mean_list.append(mean)
            plt.bar(lc_list,mean_list)
            plt.title(product)
            outf = join(outdir,f'{product}.pdf')
            plt.savefig(outf)
            plt.close()
        T.open_path_and_file(outdir)

class Pick_points:

    def __init__(self):
        self.this_class_arr, self.this_class_tif, self.this_class_png = \
            T.mk_class_dir('Pick_points', result_root_this_script, mode=2)
        pass

    def run(self):
        # self.NDVI4g()
        # self.LAI4g()
        # self.GPP()
        self.compose_dataframe()
        # self.check_data()
        pass

    def get_sites_info(self):
        excel_f = join(data_root,'Sites/data.xlsx')
        df = pd.read_excel(excel_f)
        T.print_head_n(df)
        lat_list = df['Latitude (degrees)'].tolist()
        lon_list = df['Longitude (degrees)'].tolist()
        Site_ID = df['Site ID'].tolist()
        pix_list_008 = D_008.lon_lat_to_pix(lon_list,lat_list)
        pix_list_005 = D_005.lon_lat_to_pix(lon_list,lat_list)

        return Site_ID,pix_list_005,pix_list_008,lon_list,lat_list


    def NDVI4g(self):
        outdir = join(self.this_class_arr,'NDVI4g')
        T.mk_dir(outdir)
        Site_ID, _, pix_list_008,lon_list,lat_list = self.get_sites_info()
        fdir = join(data_root,'NDVI4g/tif_008_annual_average')
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            year = int(f.replace('.tif',''))
            fpath = join(fdir,f)
            arr = ToRaster().raster2array(fpath)[0]
            picked_vals_dict = {}
            for i in range(len(Site_ID)):
                site_name = Site_ID[i]
                pix = pix_list_008[i]
                lon = lon_list[i]
                lat = lat_list[i]
                picked_val = arr[pix]
                if not site_name in picked_vals_dict:
                    picked_vals_dict[site_name] = {}
                picked_vals_dict[site_name] = {
                    'year':year,
                    # 'pix':pix,
                    'lon':lon,
                    'lat':lat,
                    'NDVI4g':picked_val/10000.,
                }
            df = T.dic_to_df(picked_vals_dict,key_col_str='site_name')
            df_list.append(df)
        df_all = pd.concat(df_list)
        df_all = df_all.sort_values(by=['site_name','year'])
        outf = join(outdir,'NDVI4g.df')
        T.save_df(df_all,outf)
        T.df_to_excel(df_all,outf,n=1000000)

    def LAI4g(self):
        outdir = join(self.this_class_arr,'LAI4g')
        T.mk_dir(outdir)
        Site_ID, _, pix_list_008,lon_list,lat_list = self.get_sites_info()
        fdir = join(data_root,'LAI4g/tif_008_annual_average')
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            year = int(f.replace('.tif',''))
            fpath = join(fdir,f)
            arr = ToRaster().raster2array(fpath)[0]
            picked_vals_dict = {}
            for i in range(len(Site_ID)):
                site_name = Site_ID[i]
                pix = pix_list_008[i]
                lon = lon_list[i]
                lat = lat_list[i]
                picked_val = arr[pix]
                if not site_name in picked_vals_dict:
                    picked_vals_dict[site_name] = {}
                picked_vals_dict[site_name] = {
                    'year':year,
                    # 'pix':pix,
                    'lon':lon,
                    'lat':lat,
                    'LAI4g':picked_val/100.,
                }
            df = T.dic_to_df(picked_vals_dict,key_col_str='site_name')
            df_list.append(df)
        df_all = pd.concat(df_list)
        df_all = df_all.sort_values(by=['site_name','year'])
        outf = join(outdir,'LAI4g.df')
        T.save_df(df_all,outf)
        T.df_to_excel(df_all,outf,n=1000000)

    def GPP(self):
        outdir = join(self.this_class_arr, 'GPP')
        T.mk_dir(outdir)
        Site_ID, pix_list_005, _, lon_list, lat_list = self.get_sites_info()
        fdir = join(data_root, 'GPP/annual_average/LT_CFE-Hybrid_NT')
        df_list = []
        for f in T.listdir(fdir):
            if not f.endswith('.tif'):
                continue
            year = int(f.replace('.tif', ''))
            fpath = join(fdir, f)
            arr = ToRaster().raster2array(fpath)[0]
            picked_vals_dict = {}
            for i in range(len(Site_ID)):
                site_name = Site_ID[i]
                pix = pix_list_005[i]
                lon = lon_list[i]
                lat = lat_list[i]
                picked_val = arr[pix]
                if not site_name in picked_vals_dict:
                    picked_vals_dict[site_name] = {}
                picked_vals_dict[site_name] = {
                    'year': year,
                    # 'pix':pix,
                    'lon': lon,
                    'lat': lat,
                    'GPP': picked_val,
                }
            df = T.dic_to_df(picked_vals_dict, key_col_str='site_name')
            df_list.append(df)
        df_all = pd.concat(df_list)
        df_all = df_all.sort_values(by=['site_name', 'year'])
        outf = join(outdir, 'GPP.df')
        T.save_df(df_all, outf)
        T.df_to_excel(df_all, outf, n=1000000)
        pass

    def compose_dataframe(self):
        outdir = join(self.this_class_arr,'dataframe')
        T.mk_dir(outdir)
        NDVI_4g_dff = join(self.this_class_arr,'NDVI4g','NDVI4g.df')
        LAI_4g_dff = join(self.this_class_arr,'LAI4g','LAI4g.df')
        GPP_dff = join(self.this_class_arr,'GPP','GPP.df')

        NDVI_4g_df = T.load_df(NDVI_4g_dff)
        LAI_4g_df = T.load_df(LAI_4g_dff)
        GPP_df = T.load_df(GPP_dff)
        site_name_list = T.get_df_unique_val_list(NDVI_4g_df,'site_name')
        Site_ID_list = self.get_sites_info()[0]
        # print(Site_ID)
        # exit()
        # df_1 =
        site_dict = {}
        for site in Site_ID_list:
            df_site_NDVI = NDVI_4g_df[NDVI_4g_df['site_name']==site]
            df_site_LAI = LAI_4g_df[LAI_4g_df['site_name']==site]
            df_site_GPP = GPP_df[GPP_df['site_name']==site]

            lat = df_site_NDVI['lat'].tolist()[0]
            lon = df_site_NDVI['lon'].tolist()[0]
            year_list = df_site_NDVI['year'].tolist()

            NDVI_vals_list = df_site_NDVI['NDVI4g'].tolist()
            LAI_vals_list = df_site_LAI['LAI4g'].tolist()
            GPP_vals_list = df_site_GPP['GPP'].tolist()

            NDVI_mean = np.nanmean(NDVI_vals_list)
            LAI_mean = np.nanmean(LAI_vals_list)
            GPP_mean = np.nanmean(GPP_vals_list)

            site_dict[site] = {
                'lat':lat,
                'lon':lon,
                'NDVI4g':NDVI_vals_list,
                'LAI4g':LAI_vals_list,
                'GPP':GPP_vals_list,
                'NDVI4g_mean':NDVI_mean,
                'LAI4g_mean':LAI_mean,
                'GPP_mean':GPP_mean,
                'year':year_list,
            }
        df_1 = T.dic_to_df(site_dict,key_col_str='Site_ID')
        outf = join(outdir,'site_extract.df')
        T.save_df(df_1,outf)
        T.df_to_excel(df_1,outf,n=1000000)
        pass

    def check_data(self):
        outdir = join(self.this_class_arr,'point_shp')
        T.mk_dir(outdir)
        dff = join(self.this_class_arr,'dataframe','site_extract.df')
        df = T.load_df(dff)
        T.print_head_n(df)
        lon_list = df['lon'].tolist()
        lat_list = df['lat'].tolist()
        NDVI4g_mean_list = df['NDVI4g_mean'].tolist()
        LAI4g_mean_list = df['LAI4g_mean'].tolist()
        GPP_mean_list = df['GPP_mean'].tolist()

        point_list = []
        for i in range(len(lon_list)):
            lon = lon_list[i]
            lat = lat_list[i]
            NDVI4g_mean = NDVI4g_mean_list[i]
            LAI4g_mean = LAI4g_mean_list[i]
            GPP_mean = GPP_mean_list[i]
            point_list.append([lon,lat,{
                'NDVI4g':NDVI4g_mean,
                'LAI4g':LAI4g_mean,
                'GPP':GPP_mean,
            }])
        outf = join(outdir,'point.shp')
        T.point_to_shp(point_list,outf)

def main():
    # GLC2000().run()
    # Aridity_index().run()
    # MODIS_LC().run()
    # LAI4g_preprocess().run()
    # NDVI4g_preprocess().run()
    # GPP_preprocess().run()
    # GPP_analysis().run()
    # Products_compare().run()
    Pick_points().run()
    pass


if __name__ == '__main__':
    main()
