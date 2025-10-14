from PIL.ImageChops import screen

from global_init import *

# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


import matplotlib
matplotlib.use('TkAgg')

class calculating_carry_over_effect:

    def __init__(self, ):
        self.data_root = data_root
        self.results_root = results_root+'\\analysis_multi_year_drought\\carry_over_effect\\'
        T.mk_dir(self.results_root)

    def run(self):
        self.produce_df()
        pass


    def produce_df(self):

        spi_fdir = data_root + rf'\SPI\dic\spi12\\'
        spi_dic = T.load_npy_dir(spi_fdir)
        NDVI_fdir = data_root + rf'\NDVI4g\annual_growth_season_NDVI_detrend\\'
        NDVI_dic = T.load_npy_dir(NDVI_fdir)
        threshold = -1.5
        min_len = 2
        years = np.arange(1982, 2021)



        results = []
        for pix in spi_dic:
            spi = spi_dic[pix]
            if pix not in NDVI_dic:
                continue
            lai = NDVI_dic[pix]
            spi = np.array(spi, dtype=float)
            spei_series_reshape = np.reshape(spi, (-1, 12))
            spi_2d = np.array(spei_series_reshape, dtype=float)  # shape: (n_years, 12)

            events=self.identify_multidroughts(spi_2d, years, threshold, min_len)
            screened=self.screen_multidroughts_with_data(events, spi_2d, min_years=8, year_list=years)


            result = self.calculate_carryover_change(lai, screened, year_list=years)
            if result is not np.nan:
                result['pix'] = pix
                result['duration'] = events[0]['duration']
                result['start'] = events[0]['start']
                result['end'] = events[0]['end']
                results.append(result)

        df = pd.DataFrame(results)
        print(df.head())
        T.save_df(df, join(self.results_root, 'carry_over_df.df'))
        T.df_to_excel(df, join(self.results_root, 'carry_over_df.xlsx'))



        pass



    def identify_multidroughts(self,spei_series, year_list=None, threshold=-1.5, min_len=2 ):
        """
                Identify multi-year droughts based on SPEI time series.

                Parameters
                ----------
                spei_series : array-like
                    1D array of annual SPEI values (e.g., SPEI-12 for each year).
                year_list : list, optional
                    List of years corresponding to the series. If None, assumes index = 0,1,2,...
                threshold : float, optional
                    Drought threshold (default -1). Drought defined when SPEI < threshold.
                min_len : int, optional
                    Minimum consecutive years to be considered a multi-year drought (default 2).

                Returns
                -------
                events : list of dict
                    Each dict: {'start': start_year, 'end': end_year, 'duration': duration, 'severity': mean SPEI during event}


        """


        # === Step 1: 年均 SPI12 ===
        spi_annual = np.nanmin(spei_series, axis=1)

        # Find years where SPEI < threshold
        drought_years = np.where(spi_annual < threshold)[0]
        if len(drought_years) == 0:
            return []

        # --- Group consecutive drought years ---
        groups = []
        start = prev = drought_years[0]
        for i in drought_years[1:]:
            if i == prev + 1:
                prev = i
            else:
                if prev - start + 1 >= min_len:
                    groups.append((start, prev))
                start = prev = i
        if prev - start + 1 >= min_len:
            groups.append((start, prev))

        # --- Build event dictionary ---
        events = []
        for start, end in groups:
            years = year_list[start:end + 1]
            mean_spei = np.nanmean(spi_annual[start:end + 1])
            events.append({
                'start': years[0],
                'end': years[-1],
                'duration': len(years),
                'severity': mean_spei
            })
        events = [e for e in events if e['start'] > year_list[0] + 8 and e['end'] < year_list[-1] - 8]

        print(events)
        return events


    def screen_multidroughts_with_data(self,events, spei_series, valid_mask=None, min_years=8, year_list=None):
        """
        Screen drought events ensuring at least N valid data years before and after each event.

        Parameters
        ----------
        events : list of dict
            Output from identify_multidroughts()
        spei_series : array-like
            Original SPEI time series (same length as NDVI/LAI series)
        valid_mask : array-like, optional
            Boolean mask (True if data valid). If None, all non-NaN in SPEI are valid.
        min_years : int, optional
            Minimum number of valid years required before and after (default 8)
        year_list : list, optional
            List of years corresponding to the series

        Returns
        -------
        screened_events : list of dict
            Events that have enough data before and after the drought
        """
        spei_series = np.array(spei_series, dtype=float)
        # print(spei_series);exit()
        n = len(spei_series)
        # print(n);exit()
        if year_list is None:
            year_list = np.arange(n)
        if valid_mask is None:
            valid_mask = ~np.isnan(spei_series)


        screened_events = []

        for e in events:
            start_idx = np.where(year_list == e['start'])[0][0]
            end_idx = np.where(year_list == e['end'])[0][0]

            # (0) boundary check
            if e['start'] <= year_list[0] + min_years or e['end'] >= year_list[-1] - min_years:
                continue

            # (1) define pre/post indices safely
            pre_start = start_idx - min_years
            post_end = end_idx + 1 + min_years
            if pre_start < 0 or post_end > n:
                continue

            pre_idx = np.arange(pre_start, start_idx)
            post_idx = np.arange(end_idx + 1, post_end)

            # (2) enforce strict window length
            if len(pre_idx) < min_years or len(post_idx) < min_years:
                continue

            # (3) check valid data
            valid_pre = np.sum(valid_mask[pre_idx])
            valid_post = np.sum(valid_mask[post_idx])
            if valid_pre < min_years or valid_post < min_years:
                continue

            # (4) check window cleanliness
            pre_years = list(year_list[pre_idx])
            post_years = list(year_list[post_idx])
            if not self.is_window_clean(e, pre_years, post_years, events, spei_series, year_list, threshold=-1.5):
                continue

            # (5) keep event
            e['pre_years'] = pre_years
            e['post_years'] = post_years
            e['n_pre'] = int(valid_pre)
            e['n_post'] = int(valid_post)
            screened_events.append(e)

        return screened_events

        # helper: check if window overlaps any other drought years
    def is_window_clean(self,event, pre_years, post_years, all_events, spei, year_list, threshold):
        # combine both windows
        window_years = set(pre_years) | set(post_years)
        spei=np.array(spei)
        spei_reshape = np.reshape(spei, (-1, 12))
        spi_2d = np.array(spei_reshape, dtype=float)  # shape: (n_years, 12)
        spi_annual = np.nanmin(spi_2d, axis=1)

        # (1) check SPEI threshold
        for y in window_years:
            if y not in year_list:
                continue
            idx = np.where(year_list == y)[0][0]

            if spi_annual[idx] < threshold:  # another drought year
                return False

        # (2) check overlaps with other multi-drought events
        for e2 in all_events:
            if e2 is event:
                continue
            other_years = set(range(e2['start'], e2['end'] + 1))
            if other_years & window_years:  # overlap
                return False
        return True

    def calculate_carryover_change(self,lai_series, screened_events, year_list=None):
        """
        Calculate change in vegetation carry-over (lag-1 autocorrelation)
        before and after the first qualified multi-year drought.

        Parameters
        ----------
        lai_series : array-like
            1D array of annual LAI (or NDVI) values.
        screened_events : list of dict
            Output from screen_multidroughts_with_data(), including pre_years and post_years.
        year_list : list, optional
            List of years corresponding to the LAI series.
            If None, assumes np.arange(len(lai_series)).

        Returns
        -------
        result : dict or None
            {'start': drought_start_year,
             'end': drought_end_year,
             'rho_pre': lag1_before,
             'rho_post': lag1_after,
             'delta_rho': difference,
             'n_pre': pre_years_count,
             'n_post': post_years_count}
            or None if no valid drought available.
        """
        lai_series = np.array(lai_series, dtype=float)
        n = len(lai_series)
        if year_list is None:
            year_list = np.arange(n)

        if lai_series.ndim != 1 or len(lai_series) < 10:
            return np.nan

        if len(screened_events) == 0:
            return np.nan

        # --- only use the first qualified drought event ---
        event = screened_events[0]


        pre_idx = [np.where(year_list == y)[0][0] for y in event['pre_years'] if y in year_list]
        post_idx = [np.where(year_list == y)[0][0] for y in event['post_years'] if y in year_list]

        lai_pre = lai_series[pre_idx]
        lai_post = lai_series[post_idx]

        # remove NaN
        lai_pre = lai_pre[~np.isnan(lai_pre)]
        lai_post = lai_post[~np.isnan(lai_post)]

        def lag1_autocorr(x):
            if len(x) < 3 or np.all(np.isnan(x)):
                return np.nan
            x = np.array(x)
            return np.corrcoef(x[:-1], x[1:])[0, 1]

        correlation_pre = lag1_autocorr(lai_pre)
        correlation_post = lag1_autocorr(lai_post)
        delta_corr = correlation_post - correlation_pre if np.isfinite(correlation_pre) and np.isfinite(correlation_post) else np.nan

        result = {
            'start': event['start'],
            'end': event['end'],
            'correlation_pre': correlation_pre,
            'correlation_post': correlation_post,
            'delta_correlation': delta_corr,
            'n_pre': len(lai_pre),
            'n_post': len(lai_post),
        }
        return result

class PLOT_results():
    def __init__(self):
        self.dff=join(results_root,rf'analysis_multi_year_drought/carry_over_effect/Dataframe/carry_over_df.df')
        self.outdir=join(results_root, 'analysis_multi_year_drought','carry_over_effect')
        pass
    def run(self):
        # self.plot_spatial()
        # self.plot_aridity()
        self.plot_histogram()
        pass
    def plot_spatial(self):
        df = T.load_df(self.dff)
        print(len(df))
        # df=df[df['duration']>2]
        # df = self.df_clean(df)
        print(len(df))
        spatial_dic_delta = {}
        spatial_pre = {}
        spatial_post = {}
        spatial_duration = {}

        df_group = T.df_groupby(df, 'pix')
        for pix in df_group:
            df_pix = df_group[pix]
            val_list_delta = df_pix['delta_correlation'].tolist()
            val_list_pre=df_pix['correlation_pre'].tolist()
            val_list_post=df_pix['correlation_post'].tolist()
            val_list_duration=df_pix['duration'].tolist()



            val_mean = np.nanmean(val_list_delta)
            val_pre=np.nanmean(val_list_pre)
            val_post=np.nanmean(val_list_post)
            vals_duration=np.nanmean(val_list_duration)
            spatial_pre[pix]=val_pre
            spatial_post[pix]=val_post
            spatial_duration[pix]=vals_duration


            spatial_dic_delta[pix] = val_mean


        array_mean = DIC_and_TIF().pix_dic_to_spatial_arr(spatial_dic_delta)

        outdir = join(self.outdir, 'tiff',)
        T.mk_dir(outdir)

        outf = join(outdir, 'carry_over_change.tif')
        DIC_and_TIF().arr_to_tif(array_mean, outf)
        print(outf)

        array_pre=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_pre)
        outf = join(outdir, 'carry_over_pre.tif')
        DIC_and_TIF().arr_to_tif(array_pre, outf)
        print(outf)

        array_post=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_post)
        outf = join(outdir, 'carry_over_post.tif')
        DIC_and_TIF().arr_to_tif(array_post, outf)
        print(outf)

        array_duration=DIC_and_TIF().pix_dic_to_spatial_arr(spatial_duration)
        outf = join(outdir, 'carry_over_duration.tif')
        DIC_and_TIF().arr_to_tif(array_duration, outf)
        print(outf)

        pass
    def plot_aridity(self):
        dff=join(results_root,rf'analysis_multi_year_drought/carry_over_effect/Dataframe/carry_over_df.df')
        df = T.load_df(dff)

        print(len(df))


        aridity_col = 'Aridity'

        aridity_bin = np.linspace(0, 2.5, 10)
        df_groupe1, bin_list_str = T.df_bin(df, aridity_col, aridity_bin)

        val_list = []

        bin_centers = []
        name_list = []

        for name, group in df_groupe1:
            bin_left = name[0].left
            bin_right = name[0].right
            bin_centers.append((bin_left + bin_right) / 2)
            name_list.append(f"{bin_left:.2f}-{bin_right:.2f}")
            val = np.nanmean(group['delta_correlation'])
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


        plt.ylim(-.5, .5)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    pass

    def plot_histogram(self):
        dff=join(results_root,rf'analysis_multi_year_drought/carry_over_effect/Dataframe/carry_over_df.df')
        df = T.load_df(dff)
        print(len(df))
        # df=df[df['Aridity']<0.65]
        # df=df[df['duration']>2]
        df = self.df_clean(df)
        print(len(df))
        val_list = df['correlation_post'].tolist()
        val_array=np.array(val_list)
        val_array=val_array[~np.isnan(val_array)]
        plt.axvline(0, color='k', linestyle='--', linewidth=1.2)
        plt.hist(val_array, bins=20, color='lightblue', edgecolor='k', alpha=0.8)
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


def main():

    # calculating_carry_over_effect().run()
    PLOT_results().run()



if __name__ == '__main__':
    main()
    pass