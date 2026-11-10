# Q-learning em ambientes Toy Text do Gymnasium

Este repositório reúne a implementação tabular do algoritmo **Q-learning** e os
experimentos realizados nos ambientes **FrozenLake-v1**, **FrozenLake8x8-v1**,
**CliffWalking-v1** e **Taxi-v4**, todos do conjunto *Toy Text* do Gymnasium.

O objetivo é investigar, de forma reproduzível, como o Q-learning se comporta
em ambientes com diferentes níveis de estocasticidade, tamanho do espaço de
estados e estrutura de recompensa, além de avaliar o efeito dos principais
hiperparâmetros (taxa de aprendizado, fator de desconto e exploração).

## Ambientes

| Ambiente | Estados | Ações | Recompensa | Dinâmica |
|---|---:|---:|---|---|
| FrozenLake 4x4 | 16 | 4 | +1 no objetivo; 0 nos demais casos | Estocástica (`is_slippery=True`) |
| FrozenLake 4x4 determinístico | 16 | 4 | +1 no objetivo; 0 nos demais casos | Determinística (`is_slippery=False`) |
| FrozenLake 8x8 | 64 | 4 | +1 no objetivo; 0 nos demais casos | Estocástica (`is_slippery=True`) |
| CliffWalking | 48 | 4 | -1 por passo; -100 ao cair no penhasco | Determinística |
| Taxi | 500 | 6 | -1 por passo; +20 no desembarque correto; -10 em ações ilegais | Determinística |

## Método

A atualização tabular utilizada é a de Watkins (1989):

```
Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

A seleção de ações durante o treinamento segue a política **epsilon-gulosa**,
com decaimento multiplicativo de `epsilon` ao final de cada episódio. A
avaliação é feita com a política gulosa pura, sem exploração, em um ambiente
separado.

## Estrutura do projeto

```
.
├── scripts/
│   ├── run_experiments.py     # executa os experimentos e salva as métricas
│   ├── analyze_results.py     # agrega resultados e calcula convergência
│   ├── make_figures.py        # gera as figuras do relatório
│   ├── build_report.py        # monta o relatório em .docx
│   ├── convert_to_pdf.ps1     # atualiza o sumário e converte o relatório
│   └── export_pdf.ps1         # exporta o .docx para .pdf (usado pelo anterior)
├── src/
│   ├── qlearning.py           # agente Q-learning tabular
│   ├── environments.py        # especificação dos ambientes
│   ├── experiment.py          # treinamento, avaliação e persistência
│   ├── plotting.py            # utilitários de visualização
│   └── report_utils.py        # formatação do relatório (.docx)
├── results/
│   ├── curves/                # curvas por execução (.npz)
│   ├── metrics/               # tabelas agregadas (.csv)
│   └── figures/               # figuras (.png)
├── report/                    # relatório final (.docx e .pdf)
├── requirements.txt
└── LICENSE
```

## Como reproduzir

Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt

# 1) Executa os experimentos principais e os três estudos de hiperparâmetros
python scripts/run_experiments.py

# 2) Agrega os resultados e calcula as métricas de convergência
python scripts/analyze_results.py

# 3) Gera as figuras
python scripts/make_figures.py

# 4) Monta o relatório em .docx
python scripts/build_report.py

# 5) (Opcional, requer Microsoft Word) gera o .pdf
powershell -ExecutionPolicy Bypass -File scripts/convert_to_pdf.ps1
```

Para uma execução curta de validação:

```bash
python scripts/run_experiments.py --quick
```

## Configuração dos experimentos

- **Treinamento principal:** 10 sementes independentes por ambiente;
  `alpha=0.1`, `gamma=0.99`, `epsilon` de 1,0 até 0,01 com fator de decaimento
  0,995; 5.000 episódios (10.000 no FrozenLake 8x8; 3.000 no CliffWalking).
- **Estudo de `alpha`:** 0,01; 0,05; 0,1; 0,5; 1,0 (5 sementes, 4.000 episódios).
- **Estudo de `gamma`:** 0,8; 0,9; 0,95; 0,99; 0,999 (5 sementes, 4.000 episódios).
- **Estudo de exploração:** `epsilon` fixo em 0,05; 0,1; 0,3 e decaimento com
  fatores 0,99; 0,995; 0,999 (5 sementes, 4.000 episódios).
- **Reexecução com a melhor configuração:** os cinco ambientes foram treinados
  novamente com `alpha=0,1`, `gamma=0,99` e `epsilon` fixo em 0,1 (10 sementes,
  mesmos números de episódios do treinamento principal):
  `python scripts/run_experiments.py --only tuned`.
- **Avaliação:** a cada 100 episódios, 10 episódios gulosos em ambiente separado.

## Resultados

As métricas completas estão em `results/metrics/`, as curvas em
`results/curves/` e as figuras em `results/figures/`. O relatório com a análise
crítica está disponível em `report/`.

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE`.
