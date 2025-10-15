from global_init import *
# coding=utf-8
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intra_CV_anaysis import Check_data
from meta_info import *
from pprint import pprint
mpl.use('TkAgg')

# this_root='/Users/liyang/Projects_data/Hotdrought_Resilience/'
# data_root = join(this_root,'data')
# results_root = join(this_root,'results')
# temp_root = join(this_root,'temp')

result_root_this_script = join(results_root, 'analysis_repeat_drought')



class Pick_multi_year_drought_events_year:

    def __init__(self):
        self.outdir = r'F:\Hotdrought_Resilience\results\analysis_multi_year_drought\\Multiyear_droughts_voxels\\'
        T.mk_dir(self.outdir, True)

        pass
    def run(self):
        ## step 1
        # SPEI_3d=self.build_SPEI_3d()
        # ## step 2
        # drought_cube = self.build_spatiotemporal_cube(SPEI_3d)
        # ## step 3
        # labels, num_features = self.identify_spatiotemporal_clusters(drought_cube)
        # #
        # #
        # # ## step 4
        # df=self.summarize_myd_events(labels)
        # # # step 5
        # self.filter_multiyear_events(self.outdir,df)
        self.plot_event_map()




    def build_SPEI_3d(self):
        """
        将 {pix: SPI月序列} 转为 (time, lat, lon) 格式的三维数组"""

        spi_12_dir = join(data_root, 'SPI/per_pix/spi12')
        SPI_dict = T.load_npy_dir(spi_12_dir)
        years = np.arange(1982, 2021)
        lat_list = np.arange(360)
        lon_list = np.arange(720)
        n_years = len(years)
        n_lat = len(lat_list)
        n_lon = len(lon_list)
        SPEI_3d = np.full((n_years, n_lat, n_lon), np.nan)

        # 逐像元填充
        for pix in tqdm(SPI_dict):
            r, c = pix
            # if not c==612:
            #     continue
            # if not r==33:
            #     continue

            vals=SPI_dict[pix]

            vals[vals < -999] = np.nan
            if np.isnan(np.nanmean(vals)):
                continue
            spi_annual = np.reshape(vals, (-1, 12))
            spi_min = np.array(spi_annual, dtype=float)  # shape: (n_years, 12)

            spi_min=np.nanmin(spi_min, axis=1)
            # print(spi_min)
            # 写入 3D 数组
            SPEI_3d[:, r, c] = spi_min
            # print(SPEI_3d);exit()

        return SPEI_3d


    def build_spatiotemporal_cube(self,SPEI_3d, ):
        """
        Step 2: 将 SPEI 三维数组转为时空干旱立方体 (0/1)

        Parameters
        ----------
        SPEI_3d : np.ndarray
            shape = (n_years, n_lat, n_lon)
            各年份对应的 SPEI 数值
        threshold : float
            干旱阈值 (默认 -1.1)

        Returns
        -------
        drought_cube : np.ndarray
            0/1 二值立方体 (n_years, n_lat, n_lon)
            1 表示干旱, 0 表示非干旱
        """
        threshold = -1.1
        keep_nan = True
        # print(SPEI_3d)

        drought_cube = np.full_like(SPEI_3d, np.nan) if keep_nan else np.zeros_like(SPEI_3d, dtype=np.uint8)

        valid_mask = ~np.isnan(SPEI_3d)

        drought_cube[valid_mask] = (SPEI_3d[valid_mask] < threshold).astype(np.uint8)

        # print(drought_cube[:, 33, 612]);exit()

        ## show

        # plt.imshow(drought_cube[10, :, :], cmap='Reds')
        # plt.title("Drought Mask (Year = 1992)")
        # plt.colorbar(label="Drought = 1")
        # plt.show()

        return drought_cube



    def identify_spatiotemporal_clusters(self,drought_cube):

        from scipy.ndimage import label
        """
        Step 3 – 识别时空连通的 MYD 事件
        ----------
        drought_cube : np.ndarray
            shape = (time, lat, lon) 的 0/1/NaN 数组
        返回：
            labels : np.ndarray，与输入同形状；每个事件有唯一 ID
            num_features : int，检测到的事件数量
        """
        # 将 NaN 设为 0 但保留 mask 用于后续还原
        nan_mask = np.isnan(drought_cube)
        data = np.where(nan_mask, 0, drought_cube).astype(np.uint8)

        # 3×3×3 连通结构 (时间、纬度、经度 三个方向均相邻)
        structure = np.ones((3, 3, 3), dtype=int)

        # 执行 label 操作
        labels, num_features = label(data, structure=structure)

        # 将 NaN 区域恢复为 NaN
        labels = labels.astype(float)  # 转为 float 以便可以写入 NaN
        labels[nan_mask] = np.nan


        # year_idx = 10  # 例如 1992
        # plt.imshow(labels[year_idx, :, :], cmap="tab20")
        # plt.title(f"Spatiotemporal MYD Labels (Year = {1982 + year_idx})")
        # plt.colorbar(label="MYD Event ID")
        # plt.show()
        #
        # print(labels, num_features);exit()

        return labels, num_features



    def summarize_myd_events(self,labels):
        """
        Step 4 – 提取每个 MYD 事件的基本属性
        """
        years=np.arange(1982,2021)
        event_info = []
        valid_ids = np.unique(labels[~np.isnan(labels)])
        valid_ids = valid_ids[valid_ids != 0]  # 0 = 非干旱

        for eid in valid_ids:
            mask = labels == eid
            t_idx, y_idx, x_idx = np.where(mask)

            start_year = years[t_idx.min()]
            end_year = years[t_idx.max()]
            duration = end_year - start_year + 1
            n_voxels = len(t_idx)

            # === Step 2: 计算空间面积 ===
            area_pixels = np.sum(mask)  # 该事件包含的总像素数

            # === 在这里加过滤条件 ===
            if area_pixels < 100:
                continue  # 太小的事件，跳过

            event_info.append({
                "event_id": int(eid),
                "start_year": int(start_year),
                "end_year": int(end_year),
                "duration": int(duration),
                "n_voxels": int(n_voxels),
                "lat_min": int(y_idx.min()),
                "lat_max": int(y_idx.max()),
                "lon_min": int(x_idx.min()),
                "lon_max": int(x_idx.max())
            })

        df = pd.DataFrame(event_info)
        df = df.sort_values("start_year").reset_index(drop=True)

        return df

    def filter_multiyear_events(self,outdir,df, min_duration=2):
        df_filtered = df[df["duration"] >= min_duration].copy()
        print(f"Retained {len(df_filtered)} multiyear events (≥{min_duration} years")

        ## step 6
        outdir=join(outdir,'Dataframe')
        T.mk_dir(outdir, True)
        T.save_df(df_filtered, join(outdir, 'multiyear_droughts_voxels.df'))
        T.df_to_excel(df_filtered, join(outdir, 'multiyear_droughts_voxels'))

        labels_path = join(outdir, 'labels.npy')
        labels = np.load(labels_path)

        # === 4. 获取保留的事件ID ===
        keep_ids = df_filtered["event_id"].unique()
        print(f"✅ Keeping {len(keep_ids)} event IDs")

        # === 5. 生成新的 label 数组 ===
        labels_filtered = np.copy(labels)
        all_ids = np.unique(labels_filtered)
        drop_ids = [i for i in all_ids if (i not in keep_ids and i > 0)]

        for drop_id in drop_ids:
            labels_filtered[labels_filtered == drop_id] = 0  # 或 np.nan

        # === 6. 保存新的 labels 文件 ===
        labels_filtered_path = join(outdir, 'labels_filtered.npy')
        np.save(labels_filtered_path, labels_filtered)


    def plot_event_map(self):
        """可视化某个事件在不同年份的空间分布"""
        labels_f=join(self.outdir,'Dataframe','labels_filtered.npy')
        labels = np.load(labels_f)
        years=np.arange(1982,2021)


        event_ids = np.unique(labels[labels > 0])
        print(f"✅ 共检测到 {len(event_ids)} 个事件")


        # === 3. 输出目录 ===
        save_dir = os.path.join(self.outdir, "Event_Maps")
        T.mk_dir(save_dir, True)

        # === 4. 遍历每个事件 ===
        for event_id in event_ids:
            mask = labels == event_id
            t_idx, y_idx, x_idx = np.where(mask)
            if len(t_idx) == 0:
                continue  # 该ID未检测到事件

            # 该事件涉及的年份
            t_unique = np.unique(t_idx)
            n = len(t_unique)
            ncols = min(n, 5)
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
            axes = np.atleast_1d(axes).ravel()

            for i, t in enumerate(t_unique):
                axes[i].imshow(mask[t, :, :], cmap="Reds", vmin=0, vmax=1)
                axes[i].set_title(f"Year = {years[t]}")
                axes[i].axis("off")

            for ax in axes[n:]:
                ax.axis("off")

            plt.suptitle(f"Event ID {event_id}: {years[t_unique[0]]}-{years[t_unique[-1]]}")
            plt.tight_layout()

            # === 5. 保存 ===
            out_path = os.path.join(save_dir, f"event_{event_id}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

    def add_total_NDVI_during_and_post_drought(self):

        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')
        temp_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend_relative_change')

        # === 1. 读取数据 ===
        df = T.load_df(dff)
        ndvi_dic = T.load_npy_dir(temp_dic_path)

        # === 2. 初始化结果 ===
        # === 2. 初始化结果 ===
        total_drought_list = []
        post1_list, post2_list, post3_list, post4_list = [], [], [], []
        drought_len_list = []

        base_year = 1982

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_years = row['drought_years']
            print(pix, drought_years)

            # --- 转换 drought_years ---
            if isinstance(drought_years, str):
                try:
                    drought_years = eval(drought_years)
                except:
                    drought_years = []
            elif not isinstance(drought_years, list):
                drought_years = []

            if len(drought_years) == 0:
                total_drought_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            # --- NDVI 缺失 ---
            if pix not in ndvi_dic:
                total_drought_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            ndvi_arr = np.array(ndvi_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(ndvi_arr))

            # --- 提取干旱期 ---
            drought_indices = [np.where(all_years == y)[0][0] for y in drought_years if y in all_years]
            if len(drought_indices) == 0:
                total_drought_list.append(np.nan)
                post1_list.append(np.nan)
                post2_list.append(np.nan)
                post3_list.append(np.nan)
                post4_list.append(np.nan)
                drought_len_list.append(np.nan)
                continue

            ndvi_drought = ndvi_arr[drought_indices]
            total_drought = np.nansum(ndvi_drought)
            drought_len = len(drought_indices)
            drought_end_year = max(drought_years)

            # --- 提取干旱后 1–4 年 ---
            post_vals = []
            for offset in [1, 2, 3, 4]:
                year_target = drought_end_year + offset
                if year_target in all_years:
                    idx = np.where(all_years == year_target)[0][0]
                    post_vals.append(ndvi_arr[idx])
                else:
                    post_vals.append(np.nan)

            # === 累积 ===
            post1 = post_vals[0]
            post2 = np.nansum(post_vals[:2])  # 1 + 2
            post3 = np.nansum(post_vals[:3])  # 1 + 2 + 3
            post4 = np.nansum(post_vals[:4])  # 1 + 2 + 3 + 4

            # === 存储结果 ===
            total_drought_list.append((total_drought))
            post1_list.append(abs(post1)-abs(total_drought))
            post2_list.append(abs(post2)-abs(total_drought))
            post3_list.append(abs(post3)-abs(total_drought))
            post4_list.append(abs(post4)-abs(total_drought))
            drought_len_list.append(drought_len)

        # === 3. 写入 DataFrame ===
        df["NDVI_total_drought"] = total_drought_list
        df["NDVI_post1_total"] = post1_list
        df["NDVI_post2_total"] = post2_list
        df["NDVI_post3_total"] = post3_list
        df["NDVI_post4_total"] = post4_list


        # === 4. 保存 ===
        outf=join(self.outdir,'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)

    def safe_divide(self,a, b):
        """安全除法：当分母为0或NaN时返回np.nan"""
        if np.isnan(a) or np.isnan(b) or abs(b) < 1e-8:
            return np.nan
        return abs(a / b)

    def add_post_NDVI_to_df(self):
        """
        向干旱事件表中添加 NDVI 恢复期（post-drought）数据：
            - post1_NDVI: 干旱结束后1年的NDVI
            - post2_NDVI: 干旱结束后2年的NDVI
            - post12_NDVI_mean: post1和post2的平均
        """

        # === 1. 读取数据 ===
        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')

        NDVI_dic_path = join(data_root, r'NDVI4g\annual_growth_season_NDVI_detrend_relative_change')

        df = T.load_df(dff)
        ndvi_dic = T.load_npy_dir(NDVI_dic_path)

        # === 2. 初始化结果列表 ===
        post1_mean_list,post12_mean_list,post123_mean_list, post1234_mean_list = [], [], [],[]

        base_year = 1982  # NDVI起始年份

        for i, row in tqdm(df.iterrows(), total=len(df)):
            pix = row['pix']
            drought_years = row['drought_years']

            # 把字符串转换成list
            if isinstance(drought_years, str):
                try:
                    drought_years = eval(drought_years)
                except:
                    drought_years = []

            # 没有NDVI数据的像元
            if pix not in ndvi_dic:
                post1_mean_list.append(np.nan)

                post12_mean_list.append(np.nan)
                post123_mean_list.append(np.nan)
                post1234_mean_list.append(np.nan)
                continue

            ndvi_arr = np.array(ndvi_dic[pix], dtype=float)
            all_years = np.arange(base_year, base_year + len(ndvi_arr))

            if len(drought_years) == 0:
                post1_mean_list.append(np.nan)

                post12_mean_list.append(np.nan)
                post123_mean_list.append(np.nan)
                post1234_mean_list.append(np.nan)
                continue

            # === 干旱结束年份 ===
            drought_end_year = max(drought_years)

            # === post1 和 post2 年份 ===
            post1_year = drought_end_year + 1
            post2_year = drought_end_year + 2
            post3_year=drought_end_year+3
            post4_year=drought_end_year+4

            # 检查年份是否在范围内
            if post1_year not in all_years:
                post1_ndvi = np.nan
            else:
                post1_ndvi = ndvi_arr[np.where(all_years == post1_year)[0][0]]

            if post2_year not in all_years:
                post2_ndvi = np.nan
            else:
                post2_ndvi = ndvi_arr[np.where(all_years == post2_year)[0][0]]

            if post3_year not in all_years:
                post3_ndvi = np.nan
            else:
                post3_ndvi = ndvi_arr[np.where(all_years == post3_year)[0][0]]

            if post4_year not in all_years:
                post4_ndvi = np.nan
            else:
                post4_ndvi = ndvi_arr[np.where(all_years == post4_year)[0][0]]

            # === 平均值 ===
            post1_mean = np.nanmean([post1_ndvi])
            post12_mean = np.nanmean([post1_ndvi, post2_ndvi])
            post123_mean = np.nanmean([post1_ndvi, post2_ndvi, post3_ndvi])
            post1234_mean = np.nanmean([post1_ndvi, post2_ndvi, post3_ndvi, post4_ndvi])


            post12_mean_list.append(post12_mean)
            post1_mean_list.append(post1_mean)
            post123_mean_list.append(post123_mean)
            post1234_mean_list.append(post1234_mean)
        # === 3. 添加到表中 ===
        df['post1_NDVI'] = post1_mean_list

        df['post12_NDVI'] = post12_mean_list
        df['post123_NDVI'] = post123_mean_list
        df['post1234_NDVI'] = post1234_mean_list

        # === 4. 保存结果 ===
        outf=join(self.outdir,'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf, random=True)

    def add_hot_drought(self):
        dff = join(self.outdir, 'Dataframe/multiyear_droughts.df')

        df = T.load_df(dff)


        df['drought_type'] = np.where(df['max_temp'] > 1, 'Hot', 'Normal')

        outf = join(self.outdir, 'Dataframe/multiyear_droughts.df')
        T.save_df(df, outf)
        T.df_to_excel(df, outf)


    def generation_drought_character_df(self):

        import ast
        dff=join(self.outdir,'Dataframe/multiyear_droughts.df')


        df = T.load_df(dff)

        all_records = []

        # === Step 1. group by pix ===
        grouped = df.groupby('pix')

        for pix, group in tqdm(grouped, total=len(grouped)):
            durations = []
            hot_count, normal_count = 0, 0

            for _, row in group.iterrows():
                # --- drought years ---
                drought_years_raw = row.get("drought_years", [])
                if isinstance(drought_years_raw, str):
                    try:
                        drought_years = ast.literal_eval(drought_years_raw)
                    except:
                        drought_years = []
                elif isinstance(drought_years_raw, list):
                    drought_years = drought_years_raw
                else:
                    drought_years = []

                dur = len(drought_years)
                if dur > 0:
                    durations.append(dur)

                # --- drought type ---
                dtype = str(row.get("drought_type", "")).lower()
                if "hot" in dtype:
                    hot_count += 1
                elif "normal" in dtype:
                    normal_count += 1

            # --- aggregate ---
            if len(durations) == 0:
                mean_dur = np.nan
                max_dur = np.nan
            else:
                mean_dur = np.mean(durations)
                max_dur = np.max(durations)

            all_records.append({
                "pix": pix,
                "multi_drought_count": len(group),
                "hot_drought_count": hot_count,
                "normal_drought_count": normal_count,
                "mean_duration": mean_dur,
                "max_duration": max_dur
            })

        df_out = pd.DataFrame(all_records)
        outpath = join(self.outdir, 'Dataframe/drought_characterastic.df')
        T.save_df(df_out, outpath)
        T.df_to_excel(df_out, outpath.replace(".df", ".xlsx"))


def main():
    Pick_multi_year_drought_events_year().run()
    pass






if __name__ == '__main__':
    main()
    pass