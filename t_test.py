import numpy as np
from scipy.stats import ttest_ind


fusion_la = np.array([2.315, 3.861, 6.250, 
                      4.050, 5.945
                      ])
fusion_df = np.array([5.596, 6.302, 8.817, 
                      11.796, 11.205
                      ])

classi_la = np.array([10.726, 18.527, 12.040,
                       13.046, 13.573
                       ])
classi_df = np.array([7.149, 11.534, 5.192, 
                      7.270, 10.485
                      ])

classi_la_2 = np.array([17.691, 10.097, 11.467,
                         9.961, 14.788
                         ])
classi_df_2 = np.array([8.898, 6.544, 7.606, 
                        6.517, 10.651
                        ])

fla_m = np.mean(fusion_la)
fdf_m = np.mean(fusion_df)


cla_m = np.mean(classi_la)
cdf_m = np.mean(classi_df)

cla_m_2 = np.mean(classi_la_2)
cdf_m_2 = np.mean(classi_df_2)

print('Mean LA: Fusion: {} Clasificacion:{} Clasif_train_dev:{}'.format(fla_m,cla_m,cla_m_2))
print('Mean DF: Fusion: {} Clasificacion:{} Clasif_train_dev:{}'.format(fdf_m,cdf_m,cdf_m_2))
# Ejecutamos Welch's t-test ya que no podemos asumir que las varianzas de las
# dos poblaciones sean iguales.

test_la = ttest_ind(fusion_la, classi_la, axis=0, equal_var=False)
test_df = ttest_ind(fusion_df, classi_df, axis=0, equal_var=False)

test_la_2 = ttest_ind(fusion_la, classi_la_2, axis=0, equal_var=False)
test_df_2 = ttest_ind(fusion_df, classi_df_2, axis=0, equal_var=False)

print(test_la.pvalue)
print(test_df.pvalue)

print(test_la_2.pvalue)
print(test_df_2.pvalue)