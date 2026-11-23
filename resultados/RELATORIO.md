# Relatório - Chikungunya em Recife (2025)

## 1. Principais achados

- **1.942 casos** analisados.
- Os 5 bairros com mais casos concentram **33.6%** do total: Cohab (218), Ibura (159), Nova Descoberta (117), Vasco Da Gama (96), Jordao (63).
- Pico epidêmico na semana de **2025-09-27** com **69 casos**.
- Idade mediana: **26 anos**; 55.0% dos casos são mulheres.
- Taxa de hospitalização: **29.80%**.

## 2. Estatística inferencial

| Hipótese | Teste | p-valor | Efeito | Conclusão |
|---|---|---|---|---|
| Sexo está associado à hospitalização? | Qui-quadrado | 0.1694 | V de Cramér = 0.056 | Associação não significativa (V=0.056; efeito pequeno). |
| A idade difere entre hospitalizados e não hospitalizados? | Mann-Whitney U | 0.002471 | r bisserial = 0.146; mediana 20 vs 27 anos | Distribuições diferentes de idade entre os grupos. |
| Hospitalização varia entre faixas etárias? | Qui-quadrado | 5.319e-05 | V de Cramér = 0.199 | Taxa de hospitalização varia entre faixas. |
| A idade dos casos difere entre os 8 bairros mais afetados? | Kruskal-Wallis | 7.078e-05 | epsilon² = 0.030 | Perfil etário difere entre bairros. |
| A idade difere entre mulheres e homens? | Mann-Whitney U | 1.588e-10 | mediana F=29 vs M=23 | Diferença significativa. |
| Taxa de hospitalização (IC 95% bootstrap) | Bootstrap (3000 reamostragens) | - | 29.80% [26.45%; 33.14%] | Estimativa pontual com intervalo de confiança. |

## 3. Machine Learning (hospitalização)

Melhor modelo na validação cruzada: **Random Forest**. Avaliação em *holdout cronológico* (n=207, prevalência=42.5%): ROC-AUC=0.590, PR-AUC=0.496, recall=0.932, F1=0.619.

| Modelo | ROC-AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| Baseline (prevalência) | 0.500 ± 0.000 | 0.298 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Regressão Logística | 0.677 ± 0.019 | 0.458 ± 0.031 | 0.654 ± 0.061 | 0.529 ± 0.034 |
| Random Forest | 0.742 ± 0.020 | 0.533 ± 0.030 | 0.610 ± 0.082 | 0.542 ± 0.028 |
| Gradient Boosting | 0.715 ± 0.036 | 0.527 ± 0.052 | 0.546 ± 0.103 | 0.511 ± 0.044 |

## 4. Previsão de séries temporais

| Modelo | MAE | RMSE |
|---|---|---|
| GradientBoosting | 26.87 | 28.85 |
| Ingênuo (lag-1) | 7.5 | 9.87 |
| Média móvel 4 | 9.31 | 12.17 |

## 5. Clusters de bairros

|   cluster |   casos |   idade_media |   pct_idosos |   pct_fem |   pct_hosp |
|----------:|--------:|--------------:|-------------:|----------:|-----------:|
|         0 |    26.2 |          33.1 |         11.3 |      55.8 |       55.2 |
|         1 |    30.6 |          25.2 |          5.8 |      50.7 |       14.3 |
|         2 |    32   |          24.5 |          2.7 |      58.7 |       52.2 |
|         3 |    39   |          32.6 |          4.5 |      67.3 |       18.7 |
|         4 |    66   |          31.2 |          6.5 |      55.6 |       34.2 |
|         5 |   188.5 |          25.1 |          3   |      49.3 |       15.2 |

## 6. Limitações
- Dados de notificação: subnotificação e atraso de registro são prováveis.
- Casos confirmados ≠ incidência real; sem denominador populacional a comparação entre bairros é limitada.
- Modelos usam apenas variáveis disponíveis na notificação; **não** foram usadas variáveis de desfecho (evitando data leakage).
- Associação estatística não implica causalidade.
- Dados sensíveis de saúde: nenhuma informação identificável foi exportada.
