import os

# # Merge server and local
# folders = list(filter(lambda f: f in os.listdir('reports_server'), os.listdir('reports')))

# for folder in folders:
#     print(folder)
#     try:
#         with open(os.path.join('reports', folder, 'report.csv'), 'a') as dest:
#             with open(os.path.join('reports_server', folder, 'report.csv'), 'r') as orig:
#                 dest.writelines(orig.readlines()[1:])
#     except Exception:
#         continue


def format_notes(notes):
    formatted = notes.readlines()
    formatted = str(formatted).replace("[", "").replace("]", "")
    formatted = " ".join(formatted.split())
    formatted = formatted.replace(",", "") 
    formatted = formatted.replace("#", "") 
    formatted = formatted.replace("\\n", "") 
    formatted = formatted.replace("'", "") 
    return formatted


def form(s):
    formatted = str(s).replace("[", "").replace("]", "")
    formatted = " ".join(formatted.split())
    formatted = formatted.replace(",", "") 
    formatted = formatted.replace("#", "") 
    formatted = formatted.replace("\\n", "") 
    formatted = formatted.replace("'", "") 
    return formatted


def get_info(notes):
    infos = notes.readlines()
    bal = infos[1].split(" ")[-1]
    norm_type = infos[3].split(" ")[-1]
    norm_by = infos[2].split(" ")[-1]
    return form(bal), form(norm_type), form(norm_by)
    

# Merge all
with open("report.csv", "w") as report:
    report.write("TRIAL,BALANCEAMENTO,NORM_TYPE,NORM_BY,DATA,NOME,TOPOLOGIA,DATASET,VAL_IS_TEST,"
    "TARGET,AUGMENTATION,EPOCAS,LR,LOSSFUNC,TRAIN_LOSS,TRAIN_MSE,TRAIN_R2,"
    "TRAIN_PEARSON,VAL_LOSS,VAL_MSE,VAL_R2,VAL_PEARSON,TEST_LOSS,TEST_MSE,"
    "TEST_R2,TEST_PEARSON\n")
    for folder in os.listdir('reports'):
        try:
            if "YieldTSC" not in folder:
                with open(os.path.join('reports', folder, "report.csv"), "r") as parcial:
                    with open(os.path.join('reports', folder, "notes.md"), "r") as notes:
                        # formatted_notes = format_notes(notes)
                        bal, norm_type, norm_by = get_info(notes)
                        for line in parcial.readlines()[1:]:
                            report.write(f"{folder},{bal},{norm_type},{norm_by},{line}")
        except Exception as e:
            print(e)

# with open("report_yield_tsc.csv", "w") as report:
#     report.write("TRIAL,BALANCEAMENTO,NORM_TYPE,NORM_BY,DATA,NOME,TOPOLOGIA,DATASET,VAL_IS_TEST,"
#     "TARGET,AUGMENTATION,EPOCAS,LR,LOSSFUNC,train_loss,train_Yield_loss,"
#     "train_TSC_loss,train_Yield_mean_squared_error,train_Yield_r2_keras,"
#     "train_Yield_R_squared,train_Yield_pearson_r,train_TSC_mean_squared_error,"
#     "train_TSC_r2_keras,train_TSC_R_squared,train_TSC_pearson_r,val_loss,"
#     "val_Yield_loss,val_TSC_loss,val_Yield_mean_squared_error,"
#     "val_Yield_r2_keras,val_Yield_R_squared,val_Yield_pearson_r,"
#     "val_TSC_mean_squared_error,val_TSC_r2_keras,val_TSC_R_squared,"
#     "val_TSC_pearson_r,test_loss,test_Yield_loss,test_TSC_loss,"
#     "test_Yield_mean_squared_error,test_Yield_r2_keras,test_Yield_R_squared,"
#     "test_Yield_pearson_r,test_TSC_mean_squared_error,test_TSC_r2_keras,"
#     "test_TSC_R_squared,test_TSC_pearson_r,SK_TRAIN_YIELD_R2,SK_VAL_YIELD_R2,"
#     "SK_TEST_YIELD_R2,SK_TRAIN_YIELD_R2,SK_VAL_YIELD_R2,SK_TEST_YIELD_R2\n")
#     for folder in os.listdir('reports'):
#         try:
#             if "YieldTSC" in folder:
#                 with open(os.path.join('reports', folder, "report_yield_tsc.csv"), "r") as parcial:
#                     with open(os.path.join('reports', folder, "notes.md"), "r") as notes:
#                         # formatted_notes = format_notes(notes)
#                         bal, norm_type, norm_by = get_info(notes)
#                         for line in parcial.readlines()[1:]:
#                             report.write(f"{folder},{bal},{norm_type},{norm_by},{line}")
#         except Exception as e:
#             print(e )

import pandas as pd

def to_ptbr(report):
    data = pd.read_csv(report + ".csv")
    data.to_csv(f'{report}_ptbr.csv', sep=';', decimal=',')

# to_ptbr("report_yield_tsc")
to_ptbr("report")

 