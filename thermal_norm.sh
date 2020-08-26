
#############################################################################
#############################################################################
#                                   THERMAL
#############################################################################
#############################################################################
python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            default
Topologia                   CNN_DAS_leakyrelu_heavy3_pool2
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               SIM
" TSC default CNN_DAS_leakyrelu_heavy3_pool2 TRIAL_1_TSC_THERMAL 1 MinMax Date None --balance

python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu_heavy
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" TSC simple CNN_DAS_leakyrelu_heavy TRIAL_1_TSC_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu_heavy2_pool
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               SIM
" TSC simple CNN_DAS_leakyrelu_heavy2_pool TRIAL_1_TSC_THERMAL 1 MinMax Date None --balance

python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            default
Topologia                   CNN_DAS_leakyrelu_heavy2_pool
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" TSC default CNN_DAS_leakyrelu_heavy2_pool TRIAL_1_TSC_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            default
Topologia                   CNN_DAS_leakyrelu_heavy3_pool2
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" TSC default CNN_DAS_leakyrelu_heavy3_pool2 TRIAL_1_TSC_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu_heavy2_pool
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" TSC simple CNN_DAS_leakyrelu_heavy2_pool TRIAL_1_TSC_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - Yield
Target                      Yield
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu_heavy2_pool
Trial                       TRIAL_1_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" Yield simple CNN_DAS_leakyrelu_heavy2_pool TRIAL_1_Yield_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - Yield
Target                      Yield
Aumento de dados            None
Topologia                   CNN_DAS_leakyrelu2
Trial                       TRIAL_1_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" Yield None CNN_DAS_leakyrelu2 TRIAL_1_Yield_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - Yield
Target                      Yield
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu2
Trial                       TRIAL_1_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" Yield simple CNN_DAS_leakyrelu2 TRIAL_1_Yield_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - Yield
Target                      Yield
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu_pool
Trial                       TRIAL_1_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" Yield simple CNN_DAS_leakyrelu_pool TRIAL_1_Yield_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - Yield
Target                      Yield
Aumento de dados            None
Topologia                   CNN_DAS_leakyrelu_heavy
Trial                       TRIAL_1_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO 
" Yield None CNN_DAS_leakyrelu_heavy TRIAL_1_Yield_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 1 - TSC
Target                      TSC
Aumento de dados            default
Topologia                   CNN_DAS_leakyrelu_heavy3_pool2
Trial                       TRIAL_1_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               SIM
" TSC default CNN_DAS_leakyrelu_heavy3_pool2 TRIAL_1_TSC_THERMAL 1 MinMax Date None --balance




python train_thermal.py "# RODADA 2 - TSC
Target                      TSC
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu_heavy2_pool
Trial                       TRIAL_2_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" TSC simple CNN_DAS_leakyrelu_heavy2_pool TRIAL_2_TSC_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 2 - TSC
Target                      TSC
Aumento de dados            simple
Topologia                   CNN_DAS_leakyrelu
Trial                       TRIAL_2_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" TSC simple CNN_DAS_leakyrelu TRIAL_2_TSC_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 2 - TSC
Target                      TSC
Aumento de dados            None
Topologia                   CNN_DAS_leakyrelu_pool
Trial                       TRIAL_2_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               SIM
" TSC None CNN_DAS_leakyrelu_pool TRIAL_2_TSC_THERMAL 1 MinMax Date None --balance

python train_thermal.py "# RODADA 2 - TSC
Target                      TSC
Aumento de dados            default
Topologia                   CNN_DAS_leakyrelu_heavy2_pool
Trial                       TRIAL_2_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               SIM
" TSC default CNN_DAS_leakyrelu_heavy2_pool TRIAL_2_TSC_THERMAL 1 MinMax Date None --balance

python train_thermal.py "# RODADA 2 - TSC
Target                      TSC
Aumento de dados            default
Topologia                   CNN_DAS_leakyrelu_heavy3_pool2
Trial                       TRIAL_2_TSC_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               SIM
" TSC default CNN_DAS_leakyrelu_heavy3_pool2 TRIAL_2_TSC_THERMAL 1 MinMax Date None --balance

python train_thermal.py "# RODADA 2 - Yield
Target                      Yield
Aumento de dados            None
Topologia                   CNN_DAS_leakyrelu
Trial                       TRIAL_2_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" Yield None CNN_DAS_leakyrelu TRIAL_2_Yield_THERMAL 1 MinMax Date None

python train_thermal.py "# RODADA 2 - Yield
Target                      Yield
Aumento de dados            None
Topologia                   MLP_DAS_leakyrelu
Trial                       TRIAL_2_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" Yield None MLP_DAS_leakyrelu TRIAL_2_Yield_THERMAL 1 MinMax Date None 

python train_thermal.py "# RODADA 2 - Yield
Target                      Yield
Aumento de dados            None
Topologia                   CNN_DAS_leakyrelu_pool
Trial                       TRIAL_2_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" Yield None CNN_DAS_leakyrelu_pool TRIAL_2_Yield_THERMAL 1 MinMax Date None 

python train_thermal.py "# RODADA 2 - Yield
Target                      Yield
Aumento de dados            None
Topologia                   MLP_DAS_leakyrelu
Trial                       TRIAL_2_Yield_THERMAL
Split                       1     
Normalização                DATA
Tipo de normalização        MinMax
Tipo                        THERMAL
Balanceamento               NÃO
" Yield None MLP_DAS_leakyrelu TRIAL_2_Yield_THERMAL 1 MinMax Date None