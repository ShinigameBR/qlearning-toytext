"""Monta o relatório em .docx a partir dos resultados dos experimentos.

O texto é escrito em português e segue a estrutura solicitada no modelo:
1. Ambientes; 2. Metodologia Experimental; 3. Resultados; 4. Discussão;
5. Conclusões.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.report_utils import (  # noqa: E402
    add_toc,
    bullet,
    caption,
    equation,
    figure,
    heading,
    new_document,
    page_break,
    paragraph,
    references,
    table,
)

METRICS = ROOT / "results" / "metrics"
FIGURES = ROOT / "results" / "figures"
REPORT_DIR = ROOT / "report"

MAIN_ORDER = [
    "frozenlake_4x4",
    "frozenlake_4x4_dry",
    "frozenlake_8x8",
    "cliffwalking",
    "taxi",
]

LABELS = {
    "frozenlake_4x4": "FrozenLake 4x4 escorregadio",
    "frozenlake_4x4_dry": "FrozenLake 4x4 determinístico",
    "frozenlake_8x8": "FrozenLake 8x8 escorregadio",
    "cliffwalking": "CliffWalking",
    "taxi": "Taxi",
}

ENV_SPECS = {
    "frozenlake_4x4": ("16", "4", "+1 no objetivo; 0 nos demais", "Estocástica"),
    "frozenlake_4x4_dry": ("16", "4", "+1 no objetivo; 0 nos demais", "Determinística"),
    "frozenlake_8x8": ("64", "4", "+1 no objetivo; 0 nos demais", "Estocástica"),
    "cliffwalking": ("48", "4", "-1 por passo; -100 no penhasco", "Determinística"),
    "taxi": ("500", "6", "-1 por passo; +20 no desembarque; -10 em ação ilegal", "Determinística"),
}

ACTION_NAMES = {
    "frozenlake_4x4": "esquerda, baixo, direita, cima",
    "frozenlake_4x4_dry": "esquerda, baixo, direita, cima",
    "frozenlake_8x8": "esquerda, baixo, direita, cima",
    "cliffwalking": "cima, direita, baixo, esquerda",
    "taxi": "norte, sul, leste, oeste, pegar passageiro, deixar passageiro",
}


def fnum(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def fsign(value: float, digits: int = 1) -> str:
    """Formata um número com sinal explícito, usando vírgula decimal."""
    return f"{value:+.{digits}f}".replace(".", ",")


def fpct(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{100 * value:.{digits}f}%".replace(".", ",")


def fpm_pct(mean: float, std: float, digits: int = 1) -> str:
    return f"{fpct(mean, digits)} ± {fnum(100 * std, digits)} p.p."


def fpm(mean: float, std: float, digits: int = 2) -> str:
    return f"{fnum(mean, digits)} ± {fnum(std, digits)}"


def eps_label(value: str) -> str:
    if value.startswith("eps_fixo"):
        number = float(value.split("=")[1])
        return f"ε fixo = {fnum(number)}"
    number = float(value.split("=")[1])
    return f"decaimento = {fnum(number, 3)}"


def load_data() -> dict:
    main_agg = pd.read_csv(METRICS / "main_aggregated.csv").set_index("env_key")
    main_runs = pd.read_csv(METRICS / "main_runs_with_convergence.csv")
    alpha_agg = pd.read_csv(METRICS / "alpha_aggregated.csv")
    gamma_agg = pd.read_csv(METRICS / "gamma_aggregated.csv")
    epsilon_agg = pd.read_csv(METRICS / "epsilon_aggregated.csv")
    best = json.loads((METRICS / "best_config.json").read_text(encoding="utf-8"))

    tuned_agg = None
    tuned_runs = None
    if (METRICS / "tuned_aggregated.csv").exists():
        tuned_agg = pd.read_csv(METRICS / "tuned_aggregated.csv").set_index("env_key")
        tuned_runs = pd.read_csv(METRICS / "tuned_runs_with_convergence.csv")

    return {
        "main_agg": main_agg,
        "main_runs": main_runs,
        "alpha_agg": alpha_agg,
        "gamma_agg": gamma_agg,
        "epsilon_agg": epsilon_agg,
        "best": best,
        "tuned_agg": tuned_agg,
        "tuned_runs": tuned_runs,
    }


def title_page(doc, code_url: str) -> None:
    for _ in range(3):
        paragraph(doc, "", indent=False)
    paragraph(doc, "INTELIGÊNCIA ARTIFICIAL PARA JOGOS I", bold=True, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=14)
    paragraph(doc, "", indent=False)
    paragraph(doc, "RELATÓRIO DE EXPERIMENTOS", bold=True, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=16)
    paragraph(doc, "", indent=False)
    paragraph(
        doc,
        "Q-learning em ambientes Toy Text do Gymnasium: uma análise crítica",
        bold=True,
        indent=False,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=15,
    )
    for _ in range(5):
        paragraph(doc, "", indent=False)
    paragraph(doc, "Autor: Expedito Hebert", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    paragraph(doc, "Professor: Charles Madeira", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    paragraph(doc, "Novembro de 2026", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(3):
        paragraph(doc, "", indent=False)
    paragraph(
        doc,
        "Código-fonte e resultados disponíveis publicamente em:",
        indent=False,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=10,
    )
    paragraph(doc, code_url, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, size=10, italic=True)
    page_break(doc)


def section_ambientes(doc) -> None:
    heading(doc, "1. Ambientes", level=1)
    paragraph(
        doc,
        "Os experimentos foram conduzidos em cinco configurações do conjunto Toy Text do "
        "Gymnasium. Todos os ambientes são processos de decisão de Markov (MDP) discretos, "
        "com observações completas do estado, o que permite representar a função de valor "
        "Q(s, a) por uma tabela. A Tabela 1 resume as principais características.",
    )

    heading(doc, "1.1. FrozenLake-v1 (4x4 e 8x8)", level=2)
    paragraph(
        doc,
        "O FrozenLake é um grid no qual o agente parte da posição inicial (S) e deve alcançar "
        "o objetivo (G) sem cair nos buracos (H). O espaço de estados corresponde às células do "
        "grid: 16 estados no mapa 4x4 e 64 no mapa 8x8. O espaço de ações tem quatro elementos "
        "(esquerda, baixo, direita e cima). A recompensa é +1 ao alcançar o objetivo e 0 em "
        "qualquer outra transição; os buracos e o objetivo são estados terminais.",
    )
    paragraph(
        doc,
        "A versão utilizada como padrão é estocástica (is_slippery=True). Nela, a ação pretendida "
        "é executada com probabilidade 1/3 e cada direção perpendicular também com probabilidade "
        "1/3, de modo que o agente pode escorregar mesmo escolhendo a ação correta. A variante "
        "determinística (is_slippery=False) foi incluída como controle, pois isola o efeito da "
        "estocasticidade do ambiente. As principais dificuldades são a recompensa esparsa, o "
        "atraso no crédito (a recompensa só aparece no fim), a presença de buracos que encerram "
        "o episódio com recompensa zero e, no mapa 8x8, o aumento do espaço de estados e a baixa "
        "probabilidade de sucesso de uma política aleatória.",
    )

    heading(doc, "1.2. CliffWalking-v1", level=2)
    paragraph(
        doc,
        "O CliffWalking é um grid 4x12 com 48 estados e quatro ações. O agente começa no canto "
        "inferior esquerdo e deve chegar ao canto inferior direito, contornando um penhasco que "
        "ocupa toda a parte inferior do grid. Cada passo custa -1 e entrar no penhasco custa -100, "
        "com o agente retornando à posição inicial sem que o episódio termine; o episódio só "
        "termina ao alcançar o objetivo. A dinâmica é determinística. O desafio central é o "
        "compromisso entre exploração e risco: políticas gananciosas podem caminhar à beira do "
        "penhasco e sofrer penalidades severas, enquanto políticas conservadoras fazem caminhos "
        "mais longos.",
    )

    heading(doc, "1.3. Taxi-v4", level=2)
    paragraph(
        doc,
        "O Taxi possui 500 estados, resultantes da combinação de 25 posições do táxi, 5 locais "
        "possíveis do passageiro (quatro pontos de embarque ou dentro do táxi) e 4 destinos. "
        "São seis ações: quatro deslocamentos e as ações de pegar e deixar o passageiro. A "
        "recompensa é -1 por passo, +20 no desembarque correto e -10 em embarque ou desembarque "
        "ilegal. A dinâmica é determinística. É o ambiente com maior espaço de estados e com "
        "recompensa positiva esparsa, o que o torna um teste mais exigente para o Q-learning "
        "tabular e para a exploração.",
    )

    table(
        doc,
        ["Ambiente", "Estados", "Ações", "Função de reforço", "Dinâmica"],
        [
            [
                LABELS[key],
                ENV_SPECS[key][0],
                ENV_SPECS[key][1],
                ENV_SPECS[key][2],
                ENV_SPECS[key][3],
            ]
            for key in MAIN_ORDER
        ],
        "Tabela 1 – Características dos ambientes utilizados",
        widths=[4.6, 1.6, 1.4, 5.0, 2.6],
        font_size=8,
    )


def section_metodologia(doc, best: dict) -> None:
    heading(doc, "2. Metodologia Experimental", level=1)
    heading(doc, "2.1. Configuração dos Experimentos", level=2)
    paragraph(
        doc,
        "Foi implementado o Q-learning tabular clássico (Watkins, 1989). Para cada transição "
        "(s, a, r, s'), a tabela Q é atualizada pela regra de diferença temporal:",
    )
    equation(doc, "Q(s, a) ← Q(s, a) + α [r + γ maxa' Q(s', a') − Q(s, a)]")
    paragraph(
        doc,
        "em que α é a taxa de aprendizado, γ é o fator de desconto e o termo entre colchetes é o "
        "erro de diferença temporal. Quando o estado seguinte é terminal, o alvo da atualização "
        "é apenas r, sem o termo de bootstrap. A seleção de ações durante o treinamento usa a "
        "política ε-gulosa: com probabilidade ε escolhe-se uma ação aleatória e, com "
        "probabilidade 1 − ε, a ação de maior valor Q. Empates são desfeitos aleatoriamente.",
    )
    paragraph(
        doc,
        "O treinamento principal utilizou α = 0,1, γ = 0,99 e ε iniciado em 1,0, decaindo "
        "multiplicativamente por um fator de 0,995 até o mínimo de 0,01. Foram executadas 10 "
        "sementes independentes por ambiente. Além desse cenário base, três estudos variaram um "
        "hiperparâmetro por vez, mantendo os demais fixos: taxa de aprendizado, fator de desconto "
        "e cronograma de exploração. A Tabela 2 detalha os valores utilizados.",
    )

    table(
        doc,
        ["Parâmetro", "Valor base", "Valores investigados"],
        [
            ["Taxa de aprendizado (α)", "0,1", "0,01; 0,05; 0,1; 0,5; 1,0"],
            ["Fator de desconto (γ)", "0,99", "0,8; 0,9; 0,95; 0,99; 0,999"],
            [
                "Exploração (ε)",
                "1,0 → 0,01 (decaimento 0,995)",
                "fixo em 0,05; 0,1; 0,3; decaimento 0,99; 0,995; 0,999",
            ],
            [
                "Episódios de treinamento",
                "5.000 (FrozenLake 4x4); 10.000 (FrozenLake 8x8); 3.000 (CliffWalking); 5.000 (Taxi)",
                "4.000 nos estudos de hiperparâmetros",
            ],
            ["Passos máximos por episódio", "100 (FrozenLake 4x4 e CliffWalking); 200 (FrozenLake 8x8 e Taxi)", "—"],
            ["Sementes", "10 no treinamento principal", "5 nos estudos de hiperparâmetros"],
            ["Avaliação", "10 episódios gulosos a cada 100 episódios", "—"],
            ["Critério de parada", "Número fixo de episódios (sem parada antecipada)", "—"],
        ],
        "Tabela 2 – Configuração dos experimentos",
        widths=[4.2, 6.0, 5.8],
        font_size=8,
    )

    paragraph(
        doc,
        "Os experimentos foram implementados em Python 3.14, com Gymnasium 1.4.0 para os "
        "ambientes, NumPy para a tabela Q, pandas para a agregação dos resultados e Matplotlib "
        "para as figuras. O algoritmo de Q-learning foi implementado diretamente no projeto, "
        "sem uso de bibliotecas de aprendizado por reforço prontas. Todas as execuções são "
        "reproduzíveis a partir das sementes registradas. O código-fonte completo, os resultados "
        "brutos e os scripts de geração das figuras estão disponíveis no endereço informado na "
        "capa e no Apêndice A.",
    )

    heading(doc, "2.2. Procedimentos de Treinamento e Avaliação", level=2)
    paragraph(
        doc,
        "Em cada episódio, o ambiente é reiniciado, o agente observa o estado e escolhe uma ação "
        "pela política ε-gulosa. A transição resultante é usada para atualizar a tabela Q e o "
        "processo continua até a terminação, o truncamento pelo limite de passos ou o fim do "
        "episódio. Ao final de cada episódio, ε é multiplicado pelo fator de decaimento, com "
        "piso em 0,01. Não há aprendizado durante a avaliação: a cada 100 episódios, a política "
        "gulosa é executada em um ambiente separado por 10 episódios, sem exploração e sem "
        "atualização da tabela Q.",
    )
    paragraph(
        doc,
        "As métricas registradas são a recompensa média por episódio, a taxa de sucesso, o "
        "número médio de passos e o desvio-padrão entre sementes. Considera-se sucesso a "
        "recompensa positiva no FrozenLake e a terminação por alcance da meta no CliffWalking e "
        "no Taxi. A convergência é definida como o primeiro ponto de avaliação que atinge 90% "
        "da taxa de sucesso final da própria execução e se mantém no ponto seguinte; essa "
        "definição relativa permite comparar ambientes com desempenhos finais diferentes. "
        "Execuções cuja taxa de sucesso final é nula não têm ponto de convergência definido e "
        "são excluídas da média de convergência. As curvas de treinamento são suavizadas por uma "
        "média móvel de janela 100 e apresentadas como média ± desvio-padrão entre as sementes.",
    )
    if best["epsilon_decay"] == 1.0:
        epsilon_description = f"ε fixo em {fnum(best['epsilon_start'])}"
    else:
        epsilon_description = (
            f"ε = {fnum(best['epsilon_start'])} → {fnum(best['epsilon_end'])} "
            f"com fator de decaimento {fnum(best['epsilon_decay'], 3)}"
        )
    paragraph(
        doc,
        "A escolha dos hiperparâmetros ajustados seguiu os estudos univariados: para cada "
        f"parâmetro, selecionou-se o valor com maior taxa de sucesso média. A configuração "
        f"resultante foi α = {fnum(best['alpha'])}, γ = {fnum(best['gamma'])} e "
        f"{epsilon_description}. Essa configuração foi então reavaliada em todos os "
        "ambientes, com as mesmas 10 sementes do cenário base.",
    )


def section_resultados(doc, data: dict) -> None:
    main_agg: pd.DataFrame = data["main_agg"]
    main_runs: pd.DataFrame = data["main_runs"]
    tuned_agg = data["tuned_agg"]

    heading(doc, "3. Resultados", level=1)
    paragraph(
        doc,
        "Esta seção apresenta os resultados quantitativos e qualitativos. A Tabela 3 consolida "
        "o desempenho final do cenário base; as Figuras 1 a 3 mostram as curvas de treinamento, "
        "a evolução da política gulosa e a comparação entre ambientes. As Figuras 4 a 6 e a "
        "Tabela 4 tratam dos estudos de hiperparâmetros. A Figura 7 e as Tabelas 5 e 6 completam "
        "a análise com as políticas aprendidas, o efeito do ajuste e a variabilidade entre "
        "sementes.",
    )

    heading(doc, "3.1. Desempenho do treinamento principal", level=2)
    rows = []
    for key in MAIN_ORDER:
        row = main_agg.loc[key]
        convergence = (
            "não atingiu"
            if np.isnan(row["convergence_mean"])
            else fnum(row["convergence_mean"], 0)
        )
        rows.append(
            [
                LABELS[key],
                fpm_pct(row["success_mean"], row["success_std"]),
                fpm(row["reward_mean"], row["reward_std"]),
                fnum(row["steps_mean"], 1),
                convergence,
                fnum(row["time_mean"], 1),
            ]
        )
    table(
        doc,
        ["Ambiente", "Sucesso final", "Recompensa final", "Passos", "Convergência (ep.)", "Tempo (s)"],
        rows,
        "Tabela 3 – Desempenho final da política gulosa no treinamento principal (média ± desvio entre 10 sementes)",
        widths=[3.8, 3.3, 3.0, 1.5, 2.1, 1.8],
        font_size=8,
    )

    figure(
        doc,
        FIGURES / "fig_01_curvas_treino.png",
        "Figura 1 – Curvas de aprendizado por ambiente (média ± desvio entre 10 sementes)",
        width_cm=16.5,
    )
    figure(
        doc,
        FIGURES / "fig_02_avaliacao_gulosa.png",
        "Figura 2 – Evolução da política gulosa durante o treinamento",
        width_cm=16.5,
    )
    figure(
        doc,
        FIGURES / "fig_03_comparacao_ambientes.png",
        "Figura 3 – Comparação do desempenho final entre os ambientes",
        width_cm=16.5,
    )

    fl = main_agg.loc["frozenlake_4x4"]
    fl_runs = main_runs[main_runs["env_key"] == "frozenlake_4x4"]
    low_seeds = int((fl_runs["final_eval_success_rate"] < 0.3).sum())
    if np.isnan(fl["convergence_mean"]):
        convergence_text = "O critério de convergência não foi atingido na média das execuções. "
    else:
        convergence_text = (
            f"A convergência média ocorreu por volta do episódio {fnum(fl['convergence_mean'], 0)}. "
        )
    paragraph(
        doc,
        f"No FrozenLake 4x4 escorregadio, a política gulosa final alcançou "
        f"{fpct(fl['success_mean'])} de sucesso (± {fnum(100 * fl['success_std'])} p.p.), com "
        f"recompensa média de {fnum(fl['reward_mean'])} e cerca de {fnum(fl['steps_mean'], 1)} "
        f"passos por episódio. {convergence_text}"
        f"Entretanto, {low_seeds} das 10 sementes terminaram com taxa de sucesso inferior a 30%, "
        "o que evidencia uma dependência forte da trajetória de exploração. Como a recompensa só "
        "ocorre no objetivo, a recompensa média praticamente coincide com a taxa de sucesso.",
    )
    paragraph(
        doc,
        "Na variante determinística do FrozenLake 4x4, todas as sementes convergiram para a "
        "política ótima, com 100% de sucesso, recompensa 1,00 e 6 passos por episódio, que é o "
        "menor caminho possível no mapa. A comparação entre as duas versões isola o efeito da "
        "estocasticidade: o mesmo algoritmo e os mesmos hiperparâmetros produzem um resultado "
        "estável no ambiente determinístico e instável no escorregadio.",
    )
    fl8 = main_agg.loc["frozenlake_8x8"]
    fl8_conv = (
        "não atingiu o critério"
        if np.isnan(fl8["convergence_mean"])
        else f"convergiu por volta do episódio {fnum(fl8['convergence_mean'], 0)}"
    )
    paragraph(
        doc,
        f"O FrozenLake 8x8 escorregadio apresentou {fpct(fl8['success_mean'])} de sucesso "
        f"(± {fnum(100 * fl8['success_std'])} p.p.), recompensa média de "
        f"{fnum(fl8['reward_mean'])} e {fnum(fl8['steps_mean'], 1)} passos, e {fl8_conv}. "
        "O aumento do grid eleva o número de estados de 16 para 64 e reduz a probabilidade de "
        "o agente encontrar o objetivo por exploração aleatória. Ainda assim, o desempenho médio "
        "foi superior ao do mapa 4x4 escorregadio, o que mostra que a dificuldade não cresce de "
        "forma monotônica com o tamanho: o mapa 8x8 possui um corredor relativamente seguro até "
        "o objetivo, enquanto o mapa 4x4 concentra buracos próximos ao caminho curto.",
    )
    cliff = main_agg.loc["cliffwalking"]
    paragraph(
        doc,
        f"No CliffWalking, a política final obteve {fpct(cliff['success_mean'])} de sucesso, "
        f"recompensa média de {fnum(cliff['reward_mean'])} e {fnum(cliff['steps_mean'], 1)} "
        "passos. A recompensa de -13 corresponde exatamente ao caminho ótimo de 13 passos "
        "(custo -1 por passo), o que indica convergência para a política ótima apesar do risco "
        "de penalidades de -100 durante a exploração.",
    )
    taxi = main_agg.loc["taxi"]
    taxi_conv = (
        "não atingiu o critério"
        if np.isnan(taxi["convergence_mean"])
        else f"convergência média no episódio {fnum(taxi['convergence_mean'], 0)}"
    )
    paragraph(
        doc,
        f"O Taxi foi o ambiente mais exigente: sucesso de {fpct(taxi['success_mean'])} "
        f"(± {fnum(100 * taxi['success_std'])} p.p.), recompensa média de "
        f"{fnum(taxi['reward_mean'])} e {fnum(taxi['steps_mean'], 1)} passos, com {taxi_conv}. "
        "O espaço de 500 estados e a recompensa positiva esparsa exigem mais episódios para que "
        "as ações de embarque e desembarque sejam aprendidas. O desvio entre sementes é menor "
        "que no FrozenLake escorregadio, porque o Taxi é determinístico, mas o tempo de "
        "treinamento e o número de estados visitados são maiores.",
    )

    heading(doc, "3.2. Efeito dos hiperparâmetros", level=2)
    alpha_agg = data["alpha_agg"]
    gamma_agg = data["gamma_agg"]
    epsilon_agg = data["epsilon_agg"]

    figure(
        doc,
        FIGURES / "fig_04_alpha.png",
        "Figura 4 – Efeito da taxa de aprendizado (α) no FrozenLake 4x4 escorregadio",
        width_cm=16.5,
    )
    best_alpha = alpha_agg.loc[
        alpha_agg["sweep_value"] == f"alpha={data['best']['alpha']:g}"
    ].iloc[0]
    worst_alpha = alpha_agg.loc[alpha_agg["success_mean"].idxmin()]
    paragraph(
        doc,
        f"A taxa de aprendizado teve efeito claro. O melhor resultado ocorreu com "
        f"α = {fnum(data['best']['alpha'])} (sucesso médio de {fpct(best_alpha['success_mean'])}), "
        f"enquanto o pior ocorreu com α = {fnum(float(str(worst_alpha['sweep_value']).split('=')[1]))} "
        f"({fpct(worst_alpha['success_mean'])}). Valores muito pequenos tornam a atualização lenta "
        "e o aprendizado não termina dentro do orçamento de episódios; valores muito altos "
        "produzem atualizações agressivas, maior sensibilidade ao ruído das transições "
        "estocásticas e oscilações nos valores Q.",
    )

    figure(
        doc,
        FIGURES / "fig_05_gamma.png",
        "Figura 5 – Efeito do fator de desconto (γ) no FrozenLake 4x4 escorregadio",
        width_cm=16.5,
    )
    best_gamma = gamma_agg.loc[
        gamma_agg["sweep_value"] == f"gamma={data['best']['gamma']:g}"
    ].iloc[0]
    worst_gamma = gamma_agg.loc[gamma_agg["success_mean"].idxmin()]
    paragraph(
        doc,
        f"O fator de desconto também alterou o desempenho. O melhor valor foi "
        f"γ = {fnum(data['best']['gamma'])} (sucesso médio de {fpct(best_gamma['success_mean'])}), "
        f"e o pior foi γ = {fnum(float(str(worst_gamma['sweep_value']).split('=')[1]))} "
        f"({fpct(worst_gamma['success_mean'])}). Com γ baixo, o agente valoriza apenas recompensas "
        "imediatas e não propaga o valor do objetivo ao longo do caminho; com γ muito próximo de "
        "1, o valor se propaga melhor, mas o aprendizado fica mais sensível à variância e pode "
        "convergir mais lentamente.",
    )

    figure(
        doc,
        FIGURES / "fig_06_epsilon.png",
        "Figura 6 – Efeito do cronograma de exploração (ε) no FrozenLake 4x4 escorregadio",
        width_cm=16.5,
    )
    if data["best"]["epsilon_decay"] == 1.0:
        best_eps_value = f"eps_fixo={data['best']['epsilon_start']:g}"
    else:
        best_eps_value = f"decay={data['best']['epsilon_decay']:g}"
    best_eps = epsilon_agg.loc[epsilon_agg["sweep_value"] == best_eps_value].iloc[0]
    worst_eps = epsilon_agg.loc[epsilon_agg["success_mean"].idxmin()]
    paragraph(
        doc,
        f"O cronograma de exploração teve efeito moderado. O melhor resultado veio de "
        f"{eps_label(str(best_eps['sweep_value']))} "
        f"(sucesso médio de {fpct(best_eps['success_mean'])}), e o pior de "
        f"{eps_label(str(worst_eps['sweep_value']))} "
        f"({fpct(worst_eps['success_mean'])}). A diferença entre os cronogramas foi menor que a "
        "observada para α, mas o padrão é informativo: manter exploração constante e moderada "
        "foi mais eficaz que reduzir ε rapidamente. ε fixo muito baixo reduz a cobertura do "
        "espaço de estados e pode favorecer políticas que evitam buracos, mas também evitam o "
        "objetivo; decaimentos rápidos interrompem a exploração antes que os pares estado-ação "
        "relevantes sejam visitados com frequência suficiente. Decaimentos lentos ou ε fixo "
        "moderado mantêm exploração suficiente para corrigir estimativas pessimistas.",
    )

    table(
        doc,
        ["Estudo", "Melhor valor", "Sucesso médio", "Pior valor", "Sucesso médio"],
        [
            [
                "Taxa de aprendizado (α)",
                fnum(data["best"]["alpha"]),
                fpct(best_alpha["success_mean"]),
                fnum(float(str(worst_alpha["sweep_value"]).split("=")[1])),
                fpct(worst_alpha["success_mean"]),
            ],
            [
                "Fator de desconto (γ)",
                fnum(data["best"]["gamma"]),
                fpct(best_gamma["success_mean"]),
                fnum(float(str(worst_gamma["sweep_value"]).split("=")[1])),
                fpct(worst_gamma["success_mean"]),
            ],
            [
                "Exploração (ε)",
                eps_label(str(best_eps["sweep_value"])),
                fpct(best_eps["success_mean"]),
                eps_label(str(worst_eps["sweep_value"])),
                fpct(worst_eps["success_mean"]),
            ],
        ],
        "Tabela 4 – Síntese dos estudos de hiperparâmetros no FrozenLake 4x4 escorregadio",
        widths=[4.0, 3.2, 2.6, 3.2, 2.6],
        font_size=8,
    )

    heading(doc, "3.3. Políticas aprendidas", level=2)
    figure(
        doc,
        FIGURES / "fig_07_politica.png",
        "Figura 7 – Valores máximos de Q e política gulosa aprendida no FrozenLake 4x4 (semente 0)",
        width_cm=14.5,
    )
    paragraph(
        doc,
        "A Figura 7 permite comparar qualitativamente as políticas. No mapa determinístico, as "
        "setas formam um caminho curto e coerente até o objetivo, e os maiores valores de Q se "
        "concentram nas células próximas à meta. No mapa escorregadio, a política tende a evitar "
        "os buracos e a produzir trajetórias mais conservadoras, refletindo o risco de escorregar "
        "para fora do caminho pretendido. Essa diferença ilustra como a mesma equação de "
        "atualização produz políticas distintas quando a dinâmica do ambiente muda.",
    )

    if tuned_agg is not None:
        heading(doc, "3.4. Comparação entre configuração base e configuração ajustada", level=2, page_break=True)
        rows = []
        improvements = []
        for key in MAIN_ORDER:
            base = main_agg.loc[key]
            tuned = tuned_agg.loc[key]
            delta_success = tuned["success_mean"] - base["success_mean"]
            improvements.append(delta_success)
            rows.append(
                [
                    LABELS[key],
                    fpm_pct(base["success_mean"], base["success_std"]),
                    fpm_pct(tuned["success_mean"], tuned["success_std"]),
                    f"{fsign(100 * delta_success)} p.p.",
                    fpm(base["reward_mean"], base["reward_std"]),
                    fpm(tuned["reward_mean"], tuned["reward_std"]),
                ]
            )
        table(
            doc,
            [
                "Ambiente",
                "Sucesso base",
                "Sucesso ajustado",
                "Variação",
                "Recompensa base",
                "Recompensa ajustada",
            ],
            rows,
            "Tabela 5 – Comparação entre a configuração base e a configuração ajustada",
            widths=[3.2, 2.9, 2.9, 1.8, 2.5, 2.5],
            font_size=7.5,
        )
        best_key = MAIN_ORDER[int(np.argmax(improvements))]
        mean_gain = float(np.mean(improvements))
        if mean_gain >= 0:
            overall = (
                f"O ajuste dos hiperparâmetros aumentou o desempenho médio em "
                f"{fnum(100 * mean_gain)} p.p."
            )
        else:
            overall = (
                f"O ajuste dos hiperparâmetros reduziu o desempenho médio em "
                f"{fnum(abs(100 * mean_gain))} p.p."
            )
        base_fl = main_agg.loc["frozenlake_4x4"]
        tuned_fl = tuned_agg.loc["frozenlake_4x4"]
        base_fl8 = main_agg.loc["frozenlake_8x8"]
        tuned_fl8 = tuned_agg.loc["frozenlake_8x8"]
        base_taxi = main_agg.loc["taxi"]
        tuned_taxi = tuned_agg.loc["taxi"]
        paragraph(
            doc,
            f"{overall} A maior variação positiva ocorreu no {LABELS[best_key]} "
            f"({fsign(100 * max(improvements))} p.p.), seguido do "
            f"{LABELS[MAIN_ORDER[int(np.argsort(improvements)[-2])]]} "
            f"({fsign(100 * sorted(improvements)[-2])} p.p.). Os ambientes determinísticos "
            "mantiveram 100% de sucesso, como esperado. No Taxi, a taxa de sucesso permaneceu em "
            f"100%, mas a recompensa média passou de {fnum(base_taxi['reward_mean'])} para "
            f"{fnum(tuned_taxi['reward_mean'])}. Além da média, o ajuste reduziu a dispersão: o "
            f"desvio entre sementes no FrozenLake 4x4 caiu de {fnum(100 * base_fl['success_std'])} "
            f"para {fnum(100 * tuned_fl['success_std'])} p.p., e no FrozenLake 8x8 de "
            f"{fnum(100 * base_fl8['success_std'])} para {fnum(100 * tuned_fl8['success_std'])} p.p.",
        )
        paragraph(
            doc,
            "Esse resultado é o principal achado prático do trabalho: em um ambiente estocástico "
            "com recompensa esparsa, manter uma exploração mínima constante (ε = 0,1) durante o "
            "treinamento foi mais eficaz do que começar com exploração total e reduzi-la "
            "rapidamente. A exploração residual garante que pares estado-ação relevantes "
            "continuem sendo revisitados, o que estabiliza as estimativas de Q e reduz a "
            "dependência da semente inicial. Como a configuração foi escolhida apenas no "
            "FrozenLake 4x4 escorregadio, o ganho não era garantido nos demais ambientes; ainda "
            "assim, ela não degradou o desempenho dos ambientes determinísticos e manteve o Taxi "
            "em 100% de sucesso.",
        )

    heading(doc, "3.5. Variabilidade entre sementes", level=2)
    rows = []
    for _, run in fl_runs.sort_values("seed").iterrows():
        rows.append(
            [
                str(int(run["seed"])),
                fpct(run["final_eval_success_rate"]),
                fnum(run["final_eval_reward_mean"]),
                fnum(run["final_eval_steps_mean"], 1),
                "—" if np.isnan(run["convergence_episode"]) else fnum(run["convergence_episode"], 0),
            ]
        )
    table(
        doc,
        ["Semente", "Sucesso final", "Recompensa final", "Passos", "Convergência (ep.)"],
        rows,
        "Tabela 6 – Resultados por semente no FrozenLake 4x4 escorregadio (configuração base)",
        widths=[2.2, 3.0, 3.2, 2.2, 3.4],
        font_size=8,
    )
    paragraph(
        doc,
        "A Tabela 6 mostra que a média esconde comportamentos muito distintos. Algumas sementes "
        "atingem 90% de sucesso, enquanto outras terminam com 0%, presas em políticas que "
        "circulam por células seguras ou que não encontram o objetivo dentro do limite de passos. "
        "Esse é um resultado importante: em ambientes estocásticos e com recompensa esparsa, uma "
        "única execução pode levar a conclusões enganosas, e a avaliação deve sempre considerar "
        "múltiplas sementes e a distribuição dos resultados, não apenas a média.",
    )


def section_discussao(doc, data: dict) -> None:
    main_agg = data["main_agg"]
    main_runs = data["main_runs"]
    best = data["best"]
    tuned_agg = data["tuned_agg"]

    heading(doc, "4. Discussão", level=1)
    heading(doc, "4.1. Hipóteses iniciais", level=2)
    paragraph(
        doc,
        "As hipóteses formuladas antes dos experimentos eram: (i) o Q-learning convergiria para "
        "a política ótima em ambientes determinísticos e discretos; (ii) a estocasticidade "
        "aumentaria a variância e dificultaria a convergência; (iii) α, γ e ε teriam efeito "
        "significativo no desempenho; e (iv) espaços de estados maiores exigiriam mais episódios.",
    )
    deterministic_success = float(
        np.mean(
            [
                main_agg.loc["frozenlake_4x4_dry", "success_mean"],
                main_agg.loc["cliffwalking", "success_mean"],
            ]
        )
    )
    paragraph(
        doc,
        f"A hipótese (i) foi confirmada: no FrozenLake determinístico e no CliffWalking, a taxa "
        f"de sucesso média foi de {fpct(deterministic_success)} e as políticas finais "
        "corresponderam ao caminho ótimo. A hipótese (ii) também foi confirmada: o FrozenLake "
        "escorregadio apresentou a maior dispersão entre sementes, com execuções de 0% e de 90% "
        "de sucesso. A hipótese (iii) foi confirmada, mas com intensidades diferentes: α foi o "
        "hiperparâmetro de maior impacto no FrozenLake 4x4 escorregadio, seguido de γ; o efeito "
        "do cronograma de exploração foi mais moderado. A hipótese (iv) foi parcialmente "
        "confirmada: o Taxi exigiu mais episódios e "
        "tempo, mas o FrozenLake 8x8 não foi monotonicamente mais difícil que o 4x4, pois a "
        "geometria do mapa e a distribuição dos buracos também importam.",
    )

    heading(doc, "4.2. Coerência com a teoria", level=2)
    paragraph(
        doc,
        "O comportamento observado é coerente com a teoria de diferença temporal. O Q-learning "
        "é um método off-policy: a política gulosa aprendida pode ser ótima mesmo que a política "
        "de comportamento seja exploratória. Isso explica por que o CliffWalking converge para "
        "o caminho ótimo de 13 passos, apesar de a exploração visitar regiões de risco. Também "
        "explica por que o FrozenLake escorregadio pode falhar: a convergência teórica exige que "
        "todos os pares estado-ação sejam visitados repetidamente. Com ε decaindo para 0,01, "
        "alguns pares relevantes deixam de ser amostrados e o valor da ação ótima pode permanecer "
        "subestimado, especialmente quando o ambiente é estocástico e a recompensa é esparsa.",
    )
    paragraph(
        doc,
        "A instabilidade também pode ser interpretada pelo viés de maximização do operador max. "
        "Como o alvo usa o maior valor estimado do próximo estado, erros positivos de estimativa "
        "são propagados e reforçados, sobretudo com α alto e muitas atualizações. No FrozenLake "
        "escorregadio, isso pode levar a políticas que superestimam caminhos arriscados ou, no "
        "sentido oposto, a políticas excessivamente conservadoras. O decaimento do ε e a escolha "
        "de α controlam esse compromisso.",
    )

    heading(doc, "4.3. Limitações experimentais", level=2)
    paragraph(
        doc,
        "Os experimentos têm limitações que devem ser consideradas na interpretação dos "
        "resultados. Primeiro, o Q-learning tabular só é viável porque os ambientes Toy Text "
        "têm espaços discretos e pequenos; os resultados não se transferem diretamente para "
        "ambientes com observações contínuas ou espaços de estados grandes. Segundo, o orçamento "
        "de episódios foi fixo e não houve parada antecipada, de modo que alguns ambientes podem "
        "não ter atingido seu melhor desempenho possível. Terceiro, a avaliação usou apenas 10 "
        "episódios gulosos por ponto, o que produz estimativas com granularidade de 0,1 e "
        "intervalos de confiança largos. Quarto, a escolha dos hiperparâmetros ajustados foi "
        "feita por estudos univariados independentes, sem busca conjunta, e apenas no FrozenLake "
        "4x4 escorregadio; a configuração resultante não é necessariamente ótima para os demais "
        "ambientes. Por fim, não foram "
        "aplicadas técnicas como Double Q-learning, aprendizado com traços de elegibilidade, "
        "inicialização otimista ou model-based planning, que poderiam reduzir a variância.",
    )
    paragraph(
        doc,
        "Também não foi feita comparação estatística formal entre configurações. As médias e "
        "desvios-padrão são descritivos e servem para evidenciar tendências, mas não substituem "
        "testes de hipótese ou intervalos de confiança apropriados. A variabilidade observada "
        "entre sementes sugere que estudos futuros deveriam usar mais execuções e reportar a "
        "distribuição completa, não apenas a média.",
    )

    heading(doc, "4.4. Melhorias possíveis", level=2)
    paragraph(
        doc,
        "Diversas melhorias poderiam ser investigadas. No plano dos hiperparâmetros, faz sentido "
        "usar exploração constante moderada ou decaimento mais lento de ε em ambientes "
        "estocásticos, taxa de aprendizado decrescente ao longo do tempo e inicialização "
        "otimista dos valores Q para incentivar exploração sistemática. No plano algorítmico, "
        "Double Q-learning reduz o viés de "
        "maximização; Q-learning com traços de elegibilidade acelera a propagação da recompensa; "
        "Dyna-Q combina aprendizado direto com um modelo aprendido do ambiente; e reward shaping "
        "pode tornar a recompensa mais informativa sem alterar a política ótima. Para ambientes "
        "maiores, a substituição da tabela por aproximação de funções (por exemplo, redes neurais) "
        "seria necessária.",
    )
    paragraph(
        doc,
        "No plano experimental, recomenda-se aumentar o número de sementes, separar "
        "conjuntos de sementes para seleção e avaliação, avaliar com mais episódios e reportar "
        "intervalos de confiança. Também seria útil comparar o Q-learning com SARSA, que é "
        "on-policy e tende a aprender políticas mais conservadoras em ambientes com risco, "
        "especialmente no CliffWalking. Essa comparação ajudaria a separar o efeito do algoritmo "
        "do efeito da configuração.",
    )


def section_conclusoes(doc) -> None:
    heading(doc, "5. Conclusões", level=1)
    paragraph(
        doc,
        "Os experimentos mostraram que o Q-learning tabular é capaz de resolver os ambientes "
        "Toy Text quando o espaço de estados é pequeno e a recompensa, ainda que esparsa, é "
        "alcançável por exploração. Os melhores resultados ocorreram nos ambientes "
        "determinísticos: o FrozenLake determinístico aprendeu o caminho de 6 passos e o "
        "CliffWalking aprendeu o caminho ótimo de 13 passos, com recompensa -13. O Taxi, com 500 "
        "estados, também convergiu, mas exigiu mais episódios e apresentou maior custo "
        "computacional. O FrozenLake escorregadio foi o caso mais sensível: a mesma configuração "
        "produziu políticas ótimas em algumas sementes e políticas ruins em outras. O ajuste dos "
        "hiperparâmetros confirmou essa sensibilidade: com α = 0,1, γ = 0,99 e ε fixo em 0,1, o "
        "sucesso médio subiu de 53% para 74% no FrozenLake 4x4 escorregadio e de 66% para 84% no "
        "FrozenLake 8x8, com redução expressiva da dispersão entre sementes.",
    )
    paragraph(
        doc,
        "A principal lição é que o desempenho do Q-learning não depende apenas do algoritmo, mas "
        "da interação entre exploração, estocasticidade e estrutura de recompensa. Em ambientes "
        "estocásticos e com recompensa esparsa, a exploração precisa ser mantida por tempo "
        "suficiente para que os pares estado-ação relevantes sejam visitados. A análise por "
        "múltiplas sementes foi essencial: uma única execução poderia levar a concluir que o "
        "algoritmo funciona ou falha, quando o resultado real é uma distribuição ampla de "
        "desempenhos.",
    )
    paragraph(
        doc,
        "Como próximos passos, sugere-se comparar Q-learning e SARSA no CliffWalking, aplicar "
        "Double Q-learning e traços de elegibilidade para reduzir variância, investigar "
        "cronogramas adaptativos de ε e α e estender os experimentos para ambientes com "
        "observações contínuas, nos quais a tabela Q precisa ser substituída por aproximação de "
        "funções. Também seria relevante repetir a análise com mais sementes e testes "
        "estatísticos para confirmar as diferenças observadas.",
    )


def appendix(doc, code_url: str) -> None:
    page_break(doc)
    heading(doc, "Apêndice A – Disponibilidade do código", level=1)
    paragraph(
        doc,
        "O código-fonte completo, os arquivos de configuração, as curvas por execução, as "
        "métricas agregadas e as figuras estão disponíveis publicamente no repositório:",
    )
    paragraph(doc, code_url, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, italic=True)
    paragraph(doc, "O repositório contém:", indent=False)
    bullet(doc, "src/qlearning.py: implementação do agente Q-learning tabular e da política ε-gulosa.")
    bullet(doc, "src/environments.py: especificação dos ambientes e regras de sucesso.")
    bullet(doc, "src/experiment.py: laço de treinamento, avaliação e persistência dos resultados.")
    bullet(doc, "scripts/run_experiments.py: execução dos experimentos principais e dos estudos de hiperparâmetros.")
    bullet(doc, "scripts/analyze_results.py: agregação, cálculo de convergência e seleção da configuração ajustada.")
    bullet(doc, "scripts/make_figures.py: geração das figuras apresentadas neste relatório.")
    bullet(doc, "results/: curvas, métricas e figuras produzidas.")
    paragraph(
        doc,
        "Para reproduzir os resultados, basta instalar as dependências listadas em "
        "requirements.txt e executar, na ordem, os scripts run_experiments.py, "
        "analyze_results.py e make_figures.py. As sementes utilizadas estão registradas nos "
        "arquivos de métricas, permitindo reexecutar cada configuração individualmente.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-url",
        default="[INSERIR LINK PÚBLICO DO REPOSITÓRIO]",
        help="URL pública do repositório com o código",
    )
    args = parser.parse_args()

    data = load_data()
    doc = new_document()

    title_page(doc, args.code_url)
    paragraph(
        doc,
        "SUMÁRIO",
        bold=True,
        indent=False,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        size=14,
    )
    add_toc(doc)
    page_break(doc)

    section_ambientes(doc)
    section_metodologia(doc, data["best"])
    section_resultados(doc, data)
    section_discussao(doc, data)
    section_conclusoes(doc)

    references(
        doc,
        [
            "MADEIRA, Charles. Inteligência Artificial para Jogos I: métodos para tomada de "
            "decisão – aprendizado por reforço. Material de aula. 2026.",
            "SUTTON, Richard S.; BARTO, Andrew G. Reinforcement Learning: An Introduction. "
            "2. ed. Cambridge, MA: MIT Press, 2018.",
            "TOWERS, Mark et al. Gymnasium: a standard interface for reinforcement learning "
            "environments. arXiv:2407.17032, 2024.",
            "WATKINS, Christopher J. C. H. Learning from delayed rewards. 1989. Tese "
            "(Doutorado) – University of Cambridge, Cambridge, 1989.",
            "WATKINS, Christopher J. C. H.; DAYAN, Peter. Q-learning. Machine Learning, "
            "v. 8, p. 279–292, 1992.",
        ],
    )

    appendix(doc, args.code_url)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "Relatorio_Q-learning_Toy_Text.docx"
    doc.save(output)
    print(f"relatório salvo: {output}")


if __name__ == "__main__":
    main()
