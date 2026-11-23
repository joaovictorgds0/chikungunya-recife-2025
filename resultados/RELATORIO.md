# Relatório - Chikungunya em Recife (2025)

## 1. Principais achados

- **1.942 casos** analisados.
- Os 5 bairros com mais casos concentram **33.6%** do total: Cohab (218), Ibura (159), Nova Descoberta (117), Vasco Da Gama (96), Jordao (63).
- Pico epidêmico na semana de **2025-09-27** com **69 casos**.
- Idade mediana: **26 anos**; 55.0% dos casos são mulheres.
- Taxa de hospitalização: **29.80%**, calculada apenas entre os **688 casos** (35% do total) com desfecho de hospitalização preenchido.

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

Modelagem com **688 casos** que têm hospitalização conhecida. Melhor modelo na validação cruzada: **Random Forest**. Avaliação em *holdout cronológico* (n=207, prevalência=42.5%): ROC-AUC=0.721, PR-AUC=0.645. Com limiar escolhido apenas no treino (0.47): recall=0.795, F1=0.636 (referência: marcar todos como positivos dá F1=0.597).

| Modelo | ROC-AUC | PR-AUC | Recall | F1 |
|---|---|---|---|---|
| Baseline (prevalência) | 0.500 ± 0.000 | 0.298 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Regressão Logística | 0.777 ± 0.020 | 0.613 ± 0.056 | 0.702 ± 0.036 | 0.586 ± 0.021 |
| Random Forest | 0.816 ± 0.014 | 0.669 ± 0.025 | 0.654 ± 0.050 | 0.612 ± 0.029 |
| Gradient Boosting | 0.802 ± 0.016 | 0.658 ± 0.020 | 0.590 ± 0.102 | 0.580 ± 0.055 |

## 4. Previsão de séries temporais

Validação *rolling-origin* (12 origens de teste, série de 52 semanas). Melhor modelo: **ARIMA(1,1,1) log** (MAE a 4 semanas 2% maior que o ingênuo).

| Modelo | MAE 1 sem. | RMSE 1 sem. | MAE 4 sem. | RMSE 4 sem. |
|---|---|---|---|---|
| Ingênuo (semana anterior) | 8.67 | 10.94 | 14.25 | 16.28 |
| Média móvel 4 sem. | 8.25 | 10.03 | 15.21 | 16.84 |
| ETS (Holt amortecido) | 11.9 | 13.1 | 19.59 | 20.53 |
| ARIMA(1,1,1) log | 8.08 | 10.27 | 14.6 | 16.37 |
| Gradient Boosting (Δ semanal) | 10.83 | 12.74 | 15.95 | 18.49 |

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
