# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lytools import *
from sklearn.ensemble import RandomForestRegressor
from scipy.special import softmax
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pprint import pprint
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

T = Tools()
results_root = rf'F:\Hotdrought_Resilience\results\\RF\\'


class Random_Forest():
    pass

    def __init__(self):
        self.outdir = results_root
        pass

    def run(self):
        pass


    def pdp_shap(self):
        dff = self.dff
        outdir = join(self.this_class_png, 'pdp_shap_beta_anomaly')

        T.mk_dir(outdir, force=True)
        x_variable_list = self.x_variable_list_CRU

        y_variable = self.y_variable
        # plt.hist(T.load_df(dff)[y_variable].tolist(),bins=100)
        # plt.show()
        df = T.load_df(dff)
        df = self.df_clean(df)



        pix_list = df['pix'].tolist()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}

        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()

        T.print_head_n(df)
        # print(len(df))
        # T.print_head_n(df)
        print('-' * 50)
        ## text select df the first 1000


        # df = self.valid_range_df(df)
        all_vars = copy.copy(x_variable_list)

        all_vars.append(y_variable)  # add the y variable to the list
        all_vars.append('pix')



        all_vars_df = df[all_vars]  # get the dataframe with the x variables and the y variable

        all_vars_df = all_vars_df.dropna(subset=x_variable_list, how='any')
        all_vars_df = all_vars_df.dropna(subset=self.y_variable, how='any')
        # print('len(all_vars_df):', len(all_vars_df));exit()


        ######

        pix_list = all_vars_df['pix'].tolist()
        # print(len(pix_list));exit()
        unique_pix_list = list(set(pix_list))
        spatial_dic = {}
        #
        for pix in unique_pix_list:
            spatial_dic[pix] = 1
        arr = DIC_and_TIF(pixelsize=0.5).pix_dic_to_spatial_arr(spatial_dic)
        plt.imshow(arr, vmin=-0.5, vmax=0.5, cmap='jet', interpolation='nearest')
        plt.colorbar()
        plt.show()

        X = all_vars_df[x_variable_list]
        Y = all_vars_df[y_variable]
        train_data_X_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.X.df')
        train_data_y_path = join(self.this_class_png, 'pdp_shap_CV', self.y_variable + '.y.df')

        # exit()

        ## save selected df for future ploting
        # T.print_head_n(X)
        # X = X.dropna()
        # print(len(X));exit()

        # model, y, y_pred = self.__train_model(X, Y)  # train a Random Forests model
        model, y, y_pred = self.__train_model_bootstrap(X, Y)
        # plt.scatter(y, y_pred)
        # plt.xlabel('y')
        # plt.ylabel('y_pred')
        # plt.show()
        # exit()
        imp_dict_xgboost = {}
        for i in range(len(x_variable_list)):
            imp_dict_xgboost[x_variable_list[i]] = model.feature_importances_[i]
        #     plt.barh(x_variable_list[i], model.feature_importances_[i])
        # plt.show()
        sorted_imp = sorted(imp_dict_xgboost.items(), key=lambda x: x[1], reverse=True)

        x_ = []
        y_ = []
        for key, value in sorted_imp:
            x_.append(key)
            y_.append(value)
        print(x_)
        plt.figure()
        plt.bar(x_, y_)
        plt.xticks(rotation=45)
        # plt.tight_layout()
        plt.title('Xgboost feature importance')
        plt.show()
        # exit()
        # plt.figure()

        ## random sample
        seed = np.random.seed(1)
        # [35318, 84714, 74782, ..., 58371, 99838, 53471]

        # sample_indices = np.random.choice(X.shape[0], size=500, replace=False)
        # pprint(sample_indices)
        # exit()
        # X_sample = X.iloc[sample_indices]
        explainer = shap.TreeExplainer(model)
        # pprint(X_sample);exit()

        # shap_values = explainer.shap_values(X) ##### not use!!!
        shap_values = explainer(X)
        # shap_values = explainer(X_sample)
        outf_shap = join(outdir, self.y_variable + '.shap')

        T.save_dict_to_binary(shap_values, outf_shap)

        ## save model

        T.save_dict_to_binary(model, join(outdir, self.y_variable + '.model'))

def main():

    Random_Forest().run()


if __name__ == '__main__':
    main()
    pass
# exit()