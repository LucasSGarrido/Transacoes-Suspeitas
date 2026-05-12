# Detecção Inteligente de Transações Suspeitas

Projeto de Machine Learning para priorizar transações financeiras com maior risco de fraude.

https://transacoes-suspeitas-a6u4mdcdg2z5tezkqpmm6k.streamlit.app

O objetivo não é apenas treinar um classificador, mas montar um pequeno produto analítico: análise exploratória, comparação de modelos, explicabilidade, simulador de custo e uma fila de investigação para apoiar decisões antifraude.

## Status

MVP com dataset real do Kaggle e dashboard Streamlit:

- análise exploratória real do dataset;
- pipeline de dados com fallback sintético;
- engenharia de features;
- treino de modelos supervisionados;
- experimento com Isolation Forest;
- seleção de threshold por F2-score;
- validação temporal para simular uso mais próximo de produção;
- simulador de custo por threshold;
- cenários conservador, balanceado e agressivo;
- explicabilidade global e local com SHAP;
- resumo executivo automático;
- página de conclusão executiva;
- registro estruturado do experimento;
- testes automatizados com `unittest`.

Resultados atuais com o dataset real do Kaggle:

| Modelo campeão | Threshold | PR-AUC teste | Recall teste | Precision teste | F2 teste |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest balanceada | 0.12 | 0.868 | 0.867 | 0.746 | 0.840 |

O XGBoost ficou próximo no F2, mas a Random Forest venceu no critério usado pelo pipeline. A decisão final do threshold deve considerar custo de erro, capacidade da equipe e validação temporal.

## Dataset

O projeto usa o dataset `Credit Card Fraud Detection` do Kaggle.

Arquivo esperado:

```text
data/raw/creditcard.csv
```

Situação atual:

- fonte dos artefatos: Kaggle;
- linhas: 284.807;
- fraudes: 492;
- taxa de fraude: aproximadamente 0,173%;
- células ausentes: 0;
- linhas duplicadas no recorte bruto: verificadas no dashboard.

Enquanto o arquivo real não existir, o comando de treino gera uma base sintética apenas para desenvolvimento.

## Como baixar o dataset real

Opção manual:

1. Acesse https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Baixe o arquivo do dataset.
3. Extraia o CSV.
4. Coloque o arquivo como:

```text
data/raw/creditcard.csv
```

Opção via Kaggle CLI:

1. Crie um token em https://www.kaggle.com/settings
2. Salve o token em:

```text
C:\Users\Lucas\.kaggle\access_token
```

Ou, se o Kaggle baixar um arquivo legado, salve como:

```text
C:\Users\Lucas\.kaggle\kaggle.json
```

3. Instale o CLI:

```bash
python -m pip install kaggle
```

4. Rode o script:

```powershell
.\scripts\download_dataset.ps1
```

5. Refaça o treino:

```bash
python -m src.train
```

## Como executar

Instale as dependências:

```bash
pip install -r requirements.txt
```

Treine os modelos:

```bash
python -m src.train
```

Gere explicabilidade SHAP:

```bash
python -m src.explain
```

Abra o dashboard:

```bash
streamlit run app/streamlit_app.py
```

Execute os testes:

```bash
python -m unittest discover -s tests
```

## Modelos do MVP

- Regressão logística balanceada.
- Random Forest balanceada.
- XGBoost com `scale_pos_weight`.
- Isolation Forest como abordagem de anomalia.

## Métricas principais

- PR-AUC.
- Recall da classe fraude.
- Precision da classe fraude.
- F2-score.
- Matriz de confusão.
- Custo estimado por threshold.

## Dashboard

### Resumo executivo automático

O topo do dashboard resume o recorte atual:

- quantas fraudes foram capturadas;
- quantos falsos positivos foram gerados;
- recomendação de uso como fila de investigação, não como bloqueio automático.

Essa camada transforma métricas técnicas em leitura de produto.

### Análise dos Dados

A aba de análise exploratória mostra:

- taxa de fraude;
- desbalanceamento da classe;
- distribuição de `Amount`;
- comparação fraude vs legítima;
- transações e taxa de fraude ao longo do tempo;
- dados ausentes e duplicados.

Essa etapa deixa claro que o dataset foi entendido antes da modelagem. Como a fraude representa menos de 0,2% da base, acurácia isolada não é apropriada.

### Modelos

A aba compara os modelos por PR-AUC, F2, recall e precisão. Também mostra a validação temporal, que é importante porque fraude muda com o tempo e um split aleatório pode deixar o problema artificialmente fácil.

### Investigação

A tabela ordenada por `risk_score` simula uma fila antifraude. O uso realista é priorizar analistas, medir `Precision@K` e ajustar o threshold à capacidade diária da equipe.

### Custos

O simulador transforma erro do modelo em decisão de negócio:

- custo por falso positivo;
- custo por falso negativo;
- custo de revisar fraude capturada;
- threshold de menor custo;
- comparação entre cenários conservador, balanceado e agressivo.

O cenário conservador reduz atrito em clientes legítimos. O balanceado busca compromisso entre captura e operação. O agressivo prioriza captura de fraude mesmo com mais alertas.

### Explicabilidade

A aba usa SHAP para explicar o modelo:

- gráfico global das variáveis mais influentes;
- explicação local por transação;
- texto interpretando as variáveis principais.

Como o dataset possui variáveis anonimizadas (`V1` a `V28`), a explicação é técnica. Ela mostra quais componentes latentes influenciam o risco, não atributos de negócio diretamente nomeáveis.

### Mapa Temporal

O mapa temporal cruza score de risco, valor e hora. Ele ajuda a observar concentrações de risco e possíveis janelas com comportamento anormal.

### Conclusão

A página final resume:

- problema;
- modelo escolhido;
- resultado;
- limitações;
- recomendação final.

A conclusão atual é que o modelo funciona bem como ferramenta de priorização, mas não deve ser apresentado como sistema de bloqueio automático. Ainda existem falsos positivos, falsos negativos, variáveis anonimizadas e necessidade de monitoramento contínuo.

## Próximos refinamentos

- Aumentar cobertura de testes para a interface.
- Criar GIF curto do simulador de threshold.
- Adicionar monitoramento de drift.
- Registrar experimentos em MLflow ou ferramenta equivalente.
- Incluir uma camada de calibração de probabilidade.
