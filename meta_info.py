# coding=utf-8
from global_init import *
D = DIC_and_TIF(pixelsize=0.5)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 8
centimeter_factor = 1 / 2.54

this_root=rf'F:\Hotdrought_Resilience\\'
data_root = join(this_root,'data')
results_root = join(this_root,'results')
temp_root = join(this_root,'temp')

global_drought_type_list = ['normal-drought', 'hot-drought']
global_drought_type_color_dict = {
    'normal-drought': 'blue',
    'hot-drought': 'red',
}
global_ELI_class = ['Energy-Limited', 'Water-Limited']
global_ELI_class_color_dict = {
    'Energy-Limited': 'blue',
    'Water-Limited': 'red',
}
global_AI_class = ['Humid', 'Arid']
global_land_tif = join(this_root, 'conf/land.tif')
global_land_tif_reproj = join(this_root, 'conf/land_reproj.tif')
global_year_range = '1982-2020'
global_start_year, global_end_year = global_year_range.split('-')
global_start_year = int(global_start_year)
global_end_year = int(global_end_year)
global_year_range_list = list(range(global_start_year, global_end_year + 1))
# data_path_dict = Meta_information().path(global_year_range)

def global_get_gs(pix): # for .25 degree
    global_northern_hemi_gs = (5, 6, 7, 8, 9, 10)
    global_southern_hemi_gs = (11, 12, 1, 2, 3, 4)
    tropical_gs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12)
    r,c = pix
    if r < 240:
        return global_northern_hemi_gs
    elif 240 <= r < 480:
        return tropical_gs
    elif r >= 480:
        return global_southern_hemi_gs
    else:
        raise ValueError('r not in range')

def global_get_gs_CMIP(pix): # for 1 degree
    global_northern_hemi_gs = (5, 6, 7, 8, 9, 10)
    global_southern_hemi_gs = (11, 12, 1, 2, 3, 4)
    tropical_gs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12)
    r,c = pix
    if r < 60:
        return global_northern_hemi_gs
    elif 60 <= r < 120:
        return tropical_gs
    elif r >= 120:
        return global_southern_hemi_gs


global_lc_list = ('deciduous', 'evergreen', 'grass', 'shrubs')
global_lc_marker_dict = {
    'deciduous': 'o',
    'evergreen': 's',
    'grass': 'v',
    'shrubs': 'D',
}
global_koppen_list = ('arid', 'cold arid', 'cold humid', 'hot arid', 'hot humid')
global_koppen_color_dict = {
    'arid': '#EB6100',
    'cold arid': '#601986',
    'cold humid': 'b',
    'hot arid': 'r',
    'hot humid': 'g',
}
global_ELI_class_list = ('Energy-Limited', 'Water-Limited')
global_AI_class_list = ('Humid', 'Arid')
global_threshold = 0.05

global_VIs_year_range_dict = {
    'NDVI3g': '1982-2015',
    'NDVI4g': '1982-2020',
    'CSIF': '2000-2020',
}
global_color_list = [
    '#844000',
    '#fc9831',
    '#fffbd4',
    '#86b9d2',
    '#064c6c',
]
global_cmap = T.cmap_blend(global_color_list, )
global_cmap_r = T.cmap_blend(global_color_list[::-1])
global_selected_spei_list = ['spei01', 'spei03', 'spei06', 'spei09', 'spei12', ]
global_selected_spi_list = [i.replace('spei', 'spi') for i in global_selected_spei_list]

global_all_spei_list = [
    'spei01', 'spei02', 'spei03', 'spei04', 'spei05', 'spei06', 'spei07', 'spei08', 'spei09', 'spei10', 'spei11',
    'spei12',
    'spei13', 'spei14', 'spei15', 'spei16', 'spei17', 'spei18', 'spei19', 'spei20', 'spei21', 'spei22', 'spei23',
    'spei24',
]
global_all_spi_list = [i.replace('spei', 'spi') for i in global_all_spei_list]
global_ELI_bins = np.arange(-0.8, 0.75, .05)
global_lag_list = list(range(0, 7))
global_drought_season_list = ['spring', 'summer', 'autumn']
global_drought_season_dict = {
    'spring': [5, 6],
    'summer': [7, 8],
    'autumn': [9, 10],
}

global_season_mon_dict = {
    5: 'spring',
    6: 'spring',
    7: 'summer',
    8: 'summer',
    9: 'autumn',
    10: 'autumn',
    1: 'other',
    2: 'other',
    3: 'other',
    4: 'other',
    11: 'other',
    12: 'other',
}


class Load_Data1:

    def __init__(self):

        pass

    def SPI(self, year_range=global_year_range):
        data_path = join(data_root, 'SPI/per_pix', year_range)
        path_type = 'multi-files'
        var_name = 'SPI'
        return self.__load_data(data_path, path_type), var_name

    def SPI_scale(self, scale, year_range=global_year_range):
        data_path = join(data_root, 'SPI/per_pix', year_range, f'{scale}.npy')
        path_type = 'file'
        var_name = f'{scale}'
        return self.__load_data(data_path, path_type), var_name

    def CCI_SM_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'CCI_SM_v7/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'CCI-SM-origin'
        return self.__load_data(data_path, path_type), var_name

    def CCI_SM_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'CCI_SM_v7/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'CCI-SM-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def CCI_SM_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'CCI_SM_v7/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'CCI-SM-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'ERA-SM/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'ERA-SM-anomaly'
        return self.__load_data(data_path, path_type), var_name


    def NDVI_origin(self, year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/per_pix', year_range)
        path_type = 'dir'
        var_name = 'NDVI-origin'
        return self.__load_data(data_path, path_type), var_name

    def NDVI_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/per_pix_anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'NDVI-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def NDVI_3g_origin(self, year_range=global_VIs_year_range_dict['NDVI3g']):
        data_path = join(data_root, 'NDVI3g/per_pix', year_range)
        path_type = 'dir'
        var_name = 'NDVI3g-origin'
        return self.__load_data(data_path, path_type), var_name

    def NDVI_3g_anomaly_detrend(self, year_range=global_VIs_year_range_dict['NDVI3g']):
        data_path = join(data_root, 'NDVI3g/per_pix_anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'NDVI3g-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-origin'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_origin_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/per_pix_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-origin_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_anomaly_juping_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_tmp/anomaly_juping_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature-anomaly_juping_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/detrend/{year_range}/precip.npy')
        path_type = 'file'
        var_name = 'Precipitation-detrend'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'Precipitation-origin'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'Precipitation-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'CRU_precip/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'Precipitation-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Srad_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'Terraclimate/srad/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'Radiation-origin'
        return self.__load_data(data_path, path_type), var_name

    def Srad_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'Terraclimate/srad/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'Radiation-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def Srad_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'Terraclimate/srad/anomaly_detrend/{year_range}/srad.npy')
        path_type = 'file'
        var_name = 'Radiation-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def FAPAR_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'FAPAR/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'FAPAR-origin'
        return self.__load_data(data_path, path_type), var_name

    def FAPAR_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'FAPAR/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'FAPAR-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def FAPAR_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'FAPAR/anomaly_detrend/{year_range}')
        path_type = 'dir'
        var_name = 'FAPAR-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def CSIF_origin(self, year_range=global_VIs_year_range_dict['CSIF']):
        data_path = join(data_root, 'CSIF/per_pix', year_range)
        path_type = 'dir'
        var_name = 'CSIF-origin'
        return self.__load_data(data_path, path_type), var_name

    def CSIF_anomaly(self, year_range=global_VIs_year_range_dict['CSIF']):
        data_path = join(data_root, 'CSIF/anomaly', year_range)
        path_type = 'dir'
        var_name = 'CSIF-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def CSIF_anomaly_detrend(self, year_range=global_VIs_year_range_dict['CSIF']):
        data_path = join(data_root, 'CSIF/anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'CSIF-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def vpd_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'TerraClimate/vpd/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'VPD-origin'
        return self.__load_data(data_path, path_type), var_name

    def vpd_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'TerraClimate/vpd/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'VPD-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def P_ET_diff_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'P_ET_diff/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'P_ET_diff-origin'
        return self.__load_data(data_path, path_type), var_name

    def P_ET_diff_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'P_ET_diff/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'P_ET_diff-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def __load_data(self, data_path, path_type):
        if path_type == 'file':
            spatial_dict = T.load_npy(data_path)
        elif path_type == 'dir':
            spatial_dict = T.load_npy_dir(data_path)
        elif path_type == 'multi-files':
            print('loading multi-files')
            spatial_dict = {}
            for f in T.listdir(data_path):
                print(f'loading {f}')
                key = f.split('.')[0]
                spatial_dict_i = T.load_npy(join(data_path, f))
                spatial_dict[key] = spatial_dict_i
        else:
            raise ValueError('path_type not recognized')
        return spatial_dict

class Load_Data:

    def __init__(self):

        pass

    # ------------------------------------------------------

    def LT_Baseline_NT_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/per_pix/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_Baseline_NT_origin'
        return self.__load_data(data_path, path_type), var_name

    def LT_Baseline_NT_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/anomaly/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_Baseline_NT_anomaly'
        return self.__load_data(data_path, path_type), var_name

    def LT_Baseline_NT_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/anomaly_detrend/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_Baseline_NT_anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def LT_Baseline_NT_anomaly_detrend_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/anomaly_detrend_GS/LT_Baseline_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_Baseline_NT_anomaly_detrend_GS'
        return self.__load_data(data_path, path_type), var_name

    def LT_Baseline_NT_origin_GS_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/origin_GS_detrend/LT_Baseline_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_Baseline_NT_origin_GS_detrend'
        return self.__load_data(data_path, path_type), var_name
    # ------------------------------------------------------

    def LT_CFE_Hybrid_NT_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/per_pix/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_CFE_Hybrid_NT_origin'
        return self.__load_data(data_path, path_type), var_name

    def LT_CFE_Hybrid_NT_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/anomaly/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_CFE_Hybrid_NT_anomaly'
        return self.__load_data(data_path, path_type), var_name

    def LT_CFE_Hybrid_NT_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/anomaly_detrend/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_CFE_Hybrid_NT_anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def LT_CFE_Hybrid_NT_anomaly_detrend_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/anomaly_detrend_GS/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_CFE_Hybrid_NT_anomaly_detrend_GS'
        return self.__load_data(data_path, path_type), var_name

    def LT_CFE_Hybrid_NT_origin_GS_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'GPP/origin_GS_detrend/LT_CFE-Hybrid_NT', year_range)
        path_type = 'dir'
        var_name = 'LT_CFE_Hybrid_NT_origin_GS_detrend'
        return self.__load_data(data_path, path_type), var_name
    # ------------------------------------------------------

    def NDVI4g_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/per_pix', year_range)
        path_type = 'dir'
        var_name = 'NDVI4g_origin'
        return self.__load_data(data_path, path_type), var_name

    def NDVI4g_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/anomaly', year_range)
        path_type = 'dir'
        var_name = 'NDVI4g_anomaly'
        return self.__load_data(data_path, path_type), var_name

    def NDVI4g_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'NDVI4g_anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def NDVI4g_climatology_percentage(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/climatology_percentage', year_range)
        path_type = 'dir'
        var_name = 'NDVI4g_climatology_percentage'
        return self.__load_data(data_path, path_type), var_name

    def NDVI4g_climatology_percentage_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/climatology_percentage_detrend', year_range)
        path_type = 'dir'
        var_name = 'NDVI4g_climatology_percentage_detrend'
        return self.__load_data(data_path, path_type), var_name


    def NDVI4g_anomaly_detrend_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'NDVI4g/anomaly_detrend_GS', year_range)
        path_type = 'dir'
        var_name = 'NDVI4g_anomaly_detrend_GS'
        return self.__load_data(data_path, path_type), var_name
    # ------------------------------------------------------

    def ERA_SM_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/per_pix', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_origin'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/anomaly', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_anomaly'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_anomaly_detrend_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/anomaly_detrend_GS', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_anomaly_detrend_GS'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_origin_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/origin_GS', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_origin_GS'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_anomaly_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/anomaly_GS', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_anomaly_GS'
        return self.__load_data(data_path, path_type), var_name

    def ERA_SM_juping(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/SM/climatology_juping', year_range)
        path_type = 'dir'
        var_name = 'ERA_SM_juping'
        return self.__load_data(data_path, path_type), var_name

    # ------------------------------------------------------

    def ERA_Tair_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/per_pix', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_origin'
        return self.__load_data(data_path, path_type), var_name

    def ERA_Tair_anomaly(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/anomaly', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_anomaly'
        return self.__load_data(data_path, path_type), var_name

    def ERA_Tair_anomaly_detrend(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/anomaly_detrend', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def ERA_Tair_anomaly_detrend_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/anomaly_detrend_GS', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_anomaly_detrend_GS'
        return self.__load_data(data_path, path_type), var_name

    def ERA_Tair_origin_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/origin_GS', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_origin_GS'
        return self.__load_data(data_path, path_type), var_name

    def ERA_Tair_anomaly_GS(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/anomaly_GS', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_anomaly_GS'
        return self.__load_data(data_path, path_type), var_name

    def ERA_Tair_juping(self,year_range=global_year_range):
        data_path = join(data_root, 'ERA5/Tair/climatology_juping', year_range)
        path_type = 'dir'
        var_name = 'ERA_Tair_juping'
        return self.__load_data(data_path, path_type), var_name

    def SPI(self,year_range=global_year_range):
        data_path = join(data_root, 'SPI/per_pix', year_range)
        path_type = 'multi-files'
        var_name = 'SPI'
        return self.__load_data(data_path, path_type), var_name

    def SPI_scale(self,scale,year_range=global_year_range):
        # data_path = join(data_root, 'SPI/per_pix', year_range,f'SPI{scale:02d}.npy')
        data_path = join(data_root, 'SPI/dic', f'{scale}')
        # path_type = 'file'
        path_type = 'dir'
        var_name = f'{scale}'
        return self.__load_data(data_path, path_type), var_name

    def Precipitation_origin(self,year_range=global_year_range):
        data_path = join(data_root, 'TerraClimate/ppt/per_pix', year_range)
        path_type = 'dir'
        var_name = 'Precipitation-origin'
        return self.__load_data(data_path, path_type), var_name

    # ------------------------------------------------------

    def GLEAM_SMRoot_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'GLEAM_SMRoot/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'GLEAM-SMRoot-origin'
        return self.__load_data(data_path, path_type), var_name

    def GLEAM_SMRoot_anomaly(self, year_range=global_year_range):
        data_path = join(data_root, f'GLEAM/anomaly/{year_range}')
        path_type = 'dir'
        var_name = 'GLEAM-SMRoot-anomaly'
        return self.__load_data(data_path, path_type), var_name

    def GLEAM_SMRoot_anomaly_detrend(self, year_range=global_year_range):
        data_path = join(data_root, f'GLEAM/SMsurf/detrend/{year_range}')
        path_type = 'dir'
        var_name = 'GLEAM-SMRoot-anomaly_detrend'
        return self.__load_data(data_path, path_type), var_name

    def Temperature_max_origin(self, year_range=global_year_range):
        data_path = join(data_root, f'TerraClimate/tmax/per_pix/{year_range}')
        path_type = 'dir'
        var_name = 'Temperature_max_origin'
        return self.__load_data(data_path, path_type), var_name

    def __load_data(self, data_path,path_type):
        if path_type == 'file':
            spatial_dict = T.load_npy(data_path)
        elif path_type == 'dir':
            spatial_dict = T.load_npy_dir(data_path)
        elif path_type == 'multi-files':
            spatial_dict = {}
            for f in T.listdir(data_path):
                key = f.split('.')[0]
                spatial_dict_i = T.load_npy_dir(join(data_path, f))
                spatial_dict[key] = spatial_dict_i
        else:
            raise ValueError('path_type not recognized')
        return spatial_dict


def get_rs_rt_cols():
    post_n_list = [1, 2, 3, 4]
    cols = ['rt']
    for n in post_n_list:
        cols.append('rs_{}'.format(n))
    return cols


def year_range_str_to_list(year_range_str):
    year_range = year_range_str.split('-')
    year_range = list(range(int(year_range[0]), int(year_range[1]) + 1))
    return year_range

def global_date_list(year_range_str):
    year_range = year_range_str_to_list(year_range_str)
    date_list = []
    for year in year_range:
        for month in range(1, 13):
            date_list.append(f'{year}-{month:02d}')
    return date_list

def global_date_obj_list(year_range_str=global_year_range):
    year_range = year_range_str_to_list(year_range_str)
    date_list = []
    for year in year_range:
        for month in range(1, 13):
            date = datetime.datetime(year, month, 1)
            date_list.append(date)
            # date_list.append(f'{year}-{month:02d}')
    return date_list