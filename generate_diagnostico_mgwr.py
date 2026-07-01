#!/usr/bin/env python3
"""Gera diagnostico_mgwr.pdf com análise completa do modelo MGWR."""

import os
import sys
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Paleta de cores ─────────────────────────────────────────────────────────
AZUL_ESCURO  = colors.HexColor("#1a3a5c")
AZUL_MEDIO   = colors.HexColor("#2563a8")
AZUL_CLARO   = colors.HexColor("#dbeafe")
VERDE        = colors.HexColor("#16a34a")
LARANJA      = colors.HexColor("#ea580c")
CINZA_TITULO = colors.HexColor("#374151")
CINZA_CLARO  = colors.HexColor("#f3f4f6")
CINZA_BORDA  = colors.HexColor("#d1d5db")
BRANCO       = colors.white
AMARELO_BG   = colors.HexColor("#fef9c3")

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

# ── Estilos ──────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

sTitle = S("sTitle",
    fontName="Helvetica-Bold", fontSize=26, leading=34,
    textColor=BRANCO, alignment=TA_CENTER, spaceAfter=8)

sSubtitle = S("sSubtitle",
    fontName="Helvetica", fontSize=13, leading=18,
    textColor=AZUL_CLARO, alignment=TA_CENTER, spaceAfter=4)

sMeta = S("sMeta",
    fontName="Helvetica-Oblique", fontSize=10, leading=14,
    textColor=AZUL_CLARO, alignment=TA_CENTER)

sH1 = S("sH1",
    fontName="Helvetica-Bold", fontSize=15, leading=20,
    textColor=AZUL_ESCURO, spaceBefore=18, spaceAfter=6)

sH2 = S("sH2",
    fontName="Helvetica-Bold", fontSize=12, leading=16,
    textColor=AZUL_MEDIO, spaceBefore=12, spaceAfter=4)

sBody = S("sBody",
    fontName="Helvetica", fontSize=10, leading=15,
    textColor=CINZA_TITULO, alignment=TA_JUSTIFY, spaceAfter=6)

sBullet = S("sBullet",
    fontName="Helvetica", fontSize=10, leading=14,
    textColor=CINZA_TITULO, leftIndent=14, firstLineIndent=-8,
    spaceAfter=3)

sCaption = S("sCaption",
    fontName="Helvetica-Oblique", fontSize=9, leading=12,
    textColor=colors.HexColor("#6b7280"), alignment=TA_CENTER, spaceAfter=4)

sEq = S("sEq",
    fontName="Helvetica", fontSize=10.5, leading=16,
    textColor=CINZA_TITULO, alignment=TA_CENTER,
    spaceBefore=6, spaceAfter=6)

sTOC = S("sTOC",
    fontName="Helvetica", fontSize=10.5, leading=16,
    textColor=CINZA_TITULO)

sTOCitem = S("sTOCitem",
    fontName="Helvetica", fontSize=10, leading=14,
    textColor=AZUL_MEDIO, leftIndent=12, spaceAfter=2)

sFootnote = S("sFootnote",
    fontName="Helvetica-Oblique", fontSize=8.5, leading=12,
    textColor=colors.HexColor("#9ca3af"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(color=CINZA_BORDA, thickness=0.6):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceAfter=6, spaceBefore=2)

def section_header(num, title):
    return [
        hr(AZUL_MEDIO, 1.2),
        Paragraph(f"{num}. {title}", sH1),
        hr(),
    ]

def bullet(text):
    return Paragraph(f"• {text}", sBullet)

def table_default(data, col_widths, header_bg=AZUL_ESCURO):
    t = Table(data, colWidths=col_widths)
    n_rows = len(data)
    style = TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), header_bg),
        ("TEXTCOLOR",   (0,0), (-1,0), BRANCO),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 9),
        ("ALIGN",       (0,0), (-1,0), "CENTER"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("ALIGN",       (1,1), (-1,-1), "CENTER"),
        ("ALIGN",       (0,1), (0,-1), "LEFT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [BRANCO, CINZA_CLARO]),
        ("GRID",        (0,0), (-1,-1), 0.4, CINZA_BORDA),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
    ])
    t.setStyle(style)
    return t


# ── Capa ─────────────────────────────────────────────────────────────────────

def make_cover():
    """Capa com fundo azul via canvas — retornamos uma table que simula o bloco."""
    cover_table = Table(
        [[
            Paragraph("MGWR", S("ct1", fontName="Helvetica-Bold", fontSize=38,
                                 leading=46, textColor=BRANCO, alignment=TA_CENTER)),
        ]],
        colWidths=[PAGE_W - 2*MARGIN],
        rowHeights=[60]
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), AZUL_ESCURO),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    return cover_table


# ── Página de capa como flowables ────────────────────────────────────────────

def cover_flowables():
    story = []

    # Bloco principal da capa
    cover_data = [
        [Paragraph(
            "MGWR<br/>Regressão Geograficamente<br/>Ponderada Multiescala",
            S("cov_title", fontName="Helvetica-Bold", fontSize=22, leading=32,
              textColor=BRANCO, alignment=TA_CENTER))],
        [Spacer(1, 8)],
        [Paragraph(
            "Diagnóstico Espacial — Dataset RJ_WEEKLY",
            S("cov_sub", fontName="Helvetica", fontSize=13, leading=18,
              textColor=AZUL_CLARO, alignment=TA_CENTER))],
        [Spacer(1, 4)],
        [Paragraph(
            "Preditores: Intercepto, TEM_AVG, RAIN",
            S("cov_pred", fontName="Helvetica-Oblique", fontSize=11, leading=15,
              textColor=AZUL_CLARO, alignment=TA_CENTER))],
        [Spacer(1, 4)],
        [Paragraph(
            "N = 307 unidades de saúde · Snapshot de pico (top 10% semanas)",
            S("cov_n", fontName="Helvetica-Oblique", fontSize=10, leading=14,
              textColor=AZUL_CLARO, alignment=TA_CENTER))],
    ]
    cov_t = Table(cover_data, colWidths=[PAGE_W - 2*MARGIN])
    cov_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), AZUL_ESCURO),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 14),
        ("BOTTOMPADDING",(0,0),(-1,-1), 14),
        ("LEFTPADDING",(0,0), (-1,-1), 20),
        ("RIGHTPADDING",(0,0),(-1,-1), 20),
    ]))
    story.append(cov_t)
    story.append(Spacer(1, 1.2*cm))

    # Caixas de info
    info_data = [
        [
            Paragraph("<b>Modelo</b><br/>GWR e MGWR (pysal/mgwr)",
                      S("info", fontName="Helvetica", fontSize=10, leading=14,
                        textColor=CINZA_TITULO, alignment=TA_CENTER)),
            Paragraph("<b>Variável resposta</b><br/>log(y_pico + 0,5)",
                      S("info2", fontName="Helvetica", fontSize=10, leading=14,
                        textColor=CINZA_TITULO, alignment=TA_CENTER)),
            Paragraph("<b>Data</b><br/>" + date.today().strftime("%d/%m/%Y"),
                      S("info3", fontName="Helvetica", fontSize=10, leading=14,
                        textColor=CINZA_TITULO, alignment=TA_CENTER)),
        ]
    ]
    info_t = Table(info_data, colWidths=[(PAGE_W - 2*MARGIN)/3]*3)
    info_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), CINZA_CLARO),
        ("GRID",       (0,0), (-1,-1), 0.5, CINZA_BORDA),
        ("TOPPADDING", (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(info_t)
    story.append(Spacer(1, 1.5*cm))

    # Sumário
    story.append(Paragraph("Sumário", S("sum_h", fontName="Helvetica-Bold",
                                         fontSize=13, leading=18, textColor=AZUL_ESCURO,
                                         spaceAfter=8)))
    story.append(hr(AZUL_MEDIO, 1))
    sections = [
        ("1", "Motivação — por que MGWR?"),
        ("2", "Formulação matemática"),
        ("3", "Dados e snapshot espacial"),
        ("4", "Bandwidths ótimos por preditor"),
        ("5", "Comparação GWR vs MGWR"),
        ("6", "Coeficientes locais do MGWR"),
        ("7", "Painel de mapas"),
        ("8", "Implicações metodológicas"),
        ("9", "Conclusão e cadeia de evidências"),
    ]
    toc_data = [[Paragraph(f"<b>{n}.</b>  {t}", sTOCitem)] for n, t in sections]
    toc_t = Table(toc_data, colWidths=[PAGE_W - 2*MARGIN])
    toc_t.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [BRANCO, CINZA_CLARO]),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ]))
    story.append(toc_t)
    story.append(PageBreak())
    return story


# ── Seção 1: Motivação ───────────────────────────────────────────────────────

def sec1():
    s = []
    s += section_header("1", "Motivação — por que MGWR?")

    s.append(Paragraph(
        "O modelo GWR (Geographically Weighted Regression) representa um avanço "
        "fundamental sobre a regressão global ao permitir que os coeficientes variem "
        "geograficamente. Contudo, o GWR possui uma limitação estrutural importante: "
        "utiliza um <b>único bandwidth</b> para todos os preditores, assumindo que cada "
        "relação espacial opera na mesma escala geográfica. Esta suposição é raramente "
        "defensável em fenômenos complexos como a dengue.",
        sBody))

    s.append(Paragraph(
        "O MGWR (Multiscale GWR, Fotheringham et al., 2017) supera essa limitação ao "
        "permitir que <b>cada preditor tenha seu próprio bandwidth ótimo</b>, determinado "
        "por AICc. Variáveis com efeito regional (ex.: temperatura) podem operar em "
        "escalas maiores, enquanto variáveis com dinâmica local (ex.: chuva intensa, "
        "concentração de criadouros) podem ter bandwidths menores.",
        sBody))

    s.append(Paragraph("<b>Contexto com resultados anteriores:</b>", sH2))
    s.append(bullet(
        "Moran's I ≈ 0,35–0,55 (semanas epidêmicas) confirmou autocorrelação "
        "espacial significativa nos casos de dengue."))
    s.append(bullet(
        "Variograma experimental indicou alcance (range) ≈ 15 km — dependência "
        "espacial intensa até essa distância."))
    s.append(bullet(
        "GWR com BW único = 126 obteve AICc = 491,0 e R² = 0,130, sendo ponto "
        "de partida para o MGWR."))

    s.append(Spacer(1, 6))
    note = Table([[Paragraph(
        "<b>Hipótese central:</b> os três preditores do modelo (Intercept, TEM_AVG, RAIN) "
        "operam em escalas espaciais distintas — o que o MGWR confirmou empiricamente "
        "com bandwidths 304, 220 e 126 respectivamente.",
        S("note", fontName="Helvetica", fontSize=10, leading=14,
          textColor=CINZA_TITULO, alignment=TA_JUSTIFY))]],
        colWidths=[PAGE_W - 2*MARGIN])
    note.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), AMARELO_BG),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 1, colors.HexColor("#ca8a04")),
    ]))
    s.append(note)
    s.append(PageBreak())
    return s


# ── Seção 2: Formulação Matemática ───────────────────────────────────────────

def sec2():
    s = []
    s += section_header("2", "Formulação Matemática")

    s.append(Paragraph("<b>GWR — modelo de referência:</b>", sH2))
    s.append(Paragraph(
        "y<sub>i</sub> = β<sub>0</sub>(u<sub>i</sub>, v<sub>i</sub>) + "
        "Σ<sub>k</sub> β<sub>k</sub>(u<sub>i</sub>, v<sub>i</sub>) · x<sub>ik</sub> + ε<sub>i</sub>",
        sEq))
    s.append(Paragraph(
        "onde (u<sub>i</sub>, v<sub>i</sub>) são as coordenadas da unidade i e todos os "
        "coeficientes β<sub>k</sub> são estimados com o <b>mesmo</b> bandwidth h.",
        sBody))

    s.append(Paragraph("<b>MGWR — formulação multiescala:</b>", sH2))
    s.append(Paragraph(
        "y<sub>i</sub> = β<sub>0,bw0</sub>(u<sub>i</sub>, v<sub>i</sub>) + "
        "Σ<sub>k</sub> β<sub>k,bwk</sub>(u<sub>i</sub>, v<sub>i</sub>) · x<sub>ik</sub> + ε<sub>i</sub>",
        sEq))
    s.append(Paragraph(
        "Cada preditor k possui seu próprio bandwidth h<sub>k</sub>, determinado por "
        "minimização do AICc. Os coeficientes locais são estimados por:",
        sBody))
    s.append(Paragraph(
        "β̂<sub>k</sub>(u<sub>i</sub>,v<sub>i</sub>) = "
        "(X<sup>T</sup> W<sub>i</sub><sup>(k)</sup> X)<sup>-1</sup> "
        "X<sup>T</sup> W<sub>i</sub><sup>(k)</sup> y",
        sEq))

    s.append(Paragraph(
        "onde W<sub>i</sub><sup>(k)</sup> é a matriz diagonal de pesos kernel "
        "específica para o preditor k e a localização i. A função kernel Gaussiana "
        "adaptativa é:",
        sBody))
    s.append(Paragraph(
        "w<sub>ij</sub><sup>(k)</sup> = exp[ − (d<sub>ij</sub> / h<sub>k</sub>)<sup>2</sup> ]",
        sEq))

    s.append(Paragraph(
        "com d<sub>ij</sub> sendo a distância euclidiana entre as unidades i e j "
        "(em graus decimais de lat/lng), e h<sub>k</sub> o bandwidth adaptativo do "
        "preditor k (em número de vizinhos mais próximos quando adaptativo, ou em "
        "distância quando fixo).",
        sBody))

    s.append(Paragraph("<b>Variável resposta e preditores:</b>", sH2))
    eq_data = [
        ["Símbolo", "Descrição"],
        ["y_i", "log(ȳ_pico,i + 0,5) — log-transformada da média de casos nas semanas de pico"],
        ["TEM_AVG", "Temperatura média (°C) — média nas semanas de pico da unidade i"],
        ["RAIN",    "Precipitação total (mm) — média nas semanas de pico da unidade i"],
        ["Intercept", "Intercepto local — captura nível basal de casos não explicado pelos preditores"],
    ]
    cw = [3.2*cm, PAGE_W - 2*MARGIN - 3.2*cm]
    s.append(table_default(eq_data, cw))
    s.append(PageBreak())
    return s


# ── Seção 3: Dados ────────────────────────────────────────────────────────────

def sec3():
    s = []
    s += section_header("3", "Dados e Snapshot Espacial")

    s.append(Paragraph(
        "O MGWR opera sobre um <b>snapshot espacial</b> — uma única observação por "
        "unidade de saúde — em vez do painel completo de 190.647 observações "
        "semanais. Essa agregação é necessária porque o MGWR estima coeficientes "
        "geograficamente variados e requer que cada ponto espacial seja representado "
        "por uma única linha de dados.",
        sBody))

    s.append(Paragraph("<b>Procedimento de construção do snapshot:</b>", sH2))
    s.append(bullet(
        "<b>Semanas de pico:</b> identificaram-se as semanas epidemiológicas no top 10% "
        "de casos totais (somando todas as unidades). Resultado: ≈ 21 semanas de pico."))
    s.append(bullet(
        "<b>Agregação por unidade:</b> para cada unidade i, calculou-se a média de "
        "casos (ȳ_pico,i) e a média de cada preditor nessas semanas de pico."))
    s.append(bullet(
        "<b>Transformação da resposta:</b> y_i = log(ȳ_pico,i + 0,5) para estabilizar "
        "a variância e aproximar a distribuição de uma Normal."))
    s.append(bullet(
        "<b>Coordenadas:</b> LAT/LNG de cada unidade obtidas via join com "
        "cluster_units_summary.csv (coluna ID_UNIDADE). Unidades sem coordenadas "
        "(78 das 397 no arquivo mestre) foram excluídas."))

    s.append(Spacer(1, 8))
    s.append(Paragraph("<b>Resumo do snapshot final:</b>", sH2))
    snap_data = [
        ["Atributo", "Valor"],
        ["N unidades no snapshot",          "307"],
        ["Semanas de pico utilizadas",       "≈ 21 (top 10% semanas)"],
        ["Preditores no modelo",             "3 (Intercept, TEM_AVG, RAIN)"],
        ["Variável resposta",                "log(ȳ_pico + 0,5)"],
        ["Coordenadas",                      "LAT/LNG em graus decimais (WGS-84)"],
        ["Fonte das coordenadas",            "cluster_units_summary.csv via ID_UNIDADE"],
        ["Kernel",                           "Gaussiano adaptativo (vizinhos mais próximos)"],
        ["Seleção de bandwidth",             "Minimização do AICc (Sel_BW do mgwr)"],
    ]
    cw = [5.5*cm, PAGE_W - 2*MARGIN - 5.5*cm]
    s.append(table_default(snap_data, cw))
    s.append(PageBreak())
    return s


# ── Seção 4: Bandwidths ───────────────────────────────────────────────────────

def sec4():
    s = []
    s += section_header("4", "Bandwidths Ótimos por Preditor")

    s.append(Paragraph(
        "A tabela abaixo apresenta os bandwidths ótimos selecionados pelo critério "
        "AICc para cada preditor no modelo MGWR. O bandwidth representa o número de "
        "vizinhos mais próximos utilizados na estimação local de cada coeficiente "
        "(kernel Gaussiano adaptativo).",
        sBody))

    bw_data = [
        ["Preditor", "Bandwidth", "Fração de N", "Escala espacial", "Interpretação"],
        ["Intercept", "304", f"{304/307:.1%}", "Quase global (N=307)",
         "Nível basal de casos varia pouco — estrutura de fundo uniforme"],
        ["TEM_AVG",   "220", f"{220/307:.1%}", "Regional (≈72% de N)",
         "Efeito da temperatura é regional — clima meso-escala"],
        ["RAIN",      "126", f"{126/307:.1%}", "Local-regional (≈41% de N)",
         "Chuva tem efeito mais local — topografia e drenagem urbana"],
    ]
    cw = [3.0*cm, 2.2*cm, 2.2*cm, 3.5*cm, PAGE_W - 2*MARGIN - 10.9*cm]
    t = table_default(bw_data, cw)
    # Destacar linha de RAIN (índice 3)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,3), (-1,3), colors.HexColor("#fef2f2")),
        ("TEXTCOLOR",  (1,3), (1,3),  LARANJA),
        ("FONTNAME",   (1,3), (1,3),  "Helvetica-Bold"),
    ]))
    s.append(t)

    s.append(Spacer(1, 10))
    s.append(Paragraph(
        "<b>Interpretação dos bandwidths:</b> um bandwidth menor indica que o efeito "
        "do preditor varia mais abruptamente no espaço — requer vizinhança menor para "
        "capturar a variação local. Um bandwidth próximo de N indica efeito "
        "praticamente estacionário (próximo à regressão global).",
        sBody))

    s.append(Paragraph("<b>Comparação dos bandwidths:</b>", sH2))
    s.append(bullet(
        "<b>RAIN (BW = 126)</b> — o preditor mais local. A chuva intensa tem efeito "
        "concentrado em áreas específicas devido à heterogeneidade da rede pluvial "
        "carioca. Corresponde a ≈ 41% das unidades na vizinhança."))
    s.append(bullet(
        "<b>TEM_AVG (BW = 220)</b> — escala regional. A temperatura média varia "
        "suavemente entre zonas climáticas (Zona Oeste, Zona Norte, maciços "
        "montanhosos), justificando um raio de influência maior."))
    s.append(bullet(
        "<b>Intercept (BW = 304)</b> — quase global. O intercepto local representa "
        "fatores não observados que variam lentamente (infraestrutura de saúde, "
        "cobertura de saneamento, densidade demográfica estrutural)."))

    s.append(Spacer(1, 8))
    # Barra visual de bandwidths
    bar_data = [
        ["", "Bandwidth relativo (proporção de N = 307)"],
        ["Intercept", "■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■  304 / 307 = 99%"],
        ["TEM_AVG",   "■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■         220 / 307 = 72%"],
        ["RAIN",      "■■■■■■■■■■■■■■■■■■■■                     126 / 307 = 41%"],
    ]
    bar_t = Table(bar_data, colWidths=[2.5*cm, PAGE_W - 2*MARGIN - 2.5*cm])
    bar_t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), AZUL_ESCURO),
        ("TEXTCOLOR",    (0,0), (-1,0), BRANCO),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[BRANCO, CINZA_CLARO]),
        ("GRID",         (0,0), (-1,-1), 0.4, CINZA_BORDA),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("TEXTCOLOR",    (1,1), (1,1),   AZUL_ESCURO),
        ("TEXTCOLOR",    (1,2), (1,2),   AZUL_MEDIO),
        ("TEXTCOLOR",    (1,3), (1,3),   LARANJA),
        ("FONTNAME",     (0,1), (0,-1),  "Helvetica-Bold"),
    ]))
    s.append(bar_t)
    s.append(PageBreak())
    return s


# ── Seção 5: Comparação GWR vs MGWR ──────────────────────────────────────────

def sec5():
    s = []
    s += section_header("5", "Comparação GWR vs MGWR")

    # Valores reais dos CSVs
    # GWR: AICc=491.014, R²=0.130, BW único=126
    # MGWR: AICc=488.930, R²=0.119
    delta_aicc = 491.014 - 488.930  # = 2.084

    s.append(Paragraph(
        "A comparação entre GWR e MGWR utiliza o AICc (Akaike Information Criterion "
        "corrigido) como critério principal, além do R² de ajuste. Um ΔAICc > 2 a "
        "favor do MGWR indica preferência empírica pelo modelo multiescala.",
        sBody))

    s.append(Paragraph("<b>Tabela comparativa — estatísticas globais:</b>", sH2))
    comp_data = [
        ["Métrica", "GWR", "MGWR", "Diferença (MGWR − GWR)"],
        ["AICc",              "491,014", "488,930", f"−{delta_aicc:.3f}  ✓ MGWR melhor"],
        ["R²",                "0,130",   "0,119",   "−0,011"],
        ["Bandwidth(s)",      "126 (único)", "304 / 220 / 126", "—"],
        ["N observações",     "307",     "307",     "—"],
        ["N preditores",      "3",       "3",       "—"],
        ["Parâmetros efetivos","variável","variável (por BW)", "maior flexibilidade"],
    ]
    cw = [3.8*cm, 2.8*cm, 2.8*cm, PAGE_W - 2*MARGIN - 9.4*cm]
    t = table_default(comp_data, cw)
    t.setStyle(TableStyle([
        ("BACKGROUND", (3,1), (3,1), colors.HexColor("#dcfce7")),
        ("TEXTCOLOR",  (3,1), (3,1), VERDE),
        ("FONTNAME",   (3,1), (3,1), "Helvetica-Bold"),
    ]))
    s.append(t)

    s.append(Spacer(1, 10))
    s.append(Paragraph(
        f"<b>ΔAICc = {delta_aicc:.3f}</b> a favor do MGWR. Pela regra de Burnham &amp; "
        f"Anderson (2002), ΔAICc > 2 indica suporte empírico substancial para o modelo "
        f"com menor AICc. O MGWR melhora o ajuste ao permitir escalas distintas por "
        f"preditor, mesmo com o mesmo número de variáveis explicativas.",
        sBody))

    s.append(Paragraph(
        "O R² ligeiramente menor no MGWR (0,119 vs 0,130) reflete a penalização "
        "por maior complexidade paramétrica efetiva — o modelo com mais graus de "
        "liberdade pode ajustar melhor localmente mas com maior variância. O AICc "
        "equilibra ajuste e parcimônia, e favorece o MGWR.",
        sBody))

    s.append(Paragraph("<b>Coeficientes locais — GWR (BW único = 126):</b>", sH2))
    gwr_coef = [
        ["Preditor", "Média", "Desvio Padrão", "P25", "P75", "Mínimo", "Máximo"],
        ["Intercept", "−0,350", "0,125", "−0,452", "−0,261", "−0,604", "−0,041"],
        ["TEM_AVG",   " 0,104", "0,121", " 0,001", " 0,189", "−0,106", " 0,370"],
        ["RAIN",      "−0,126", "0,131", "−0,219", "−0,027", "−0,444", " 0,104"],
    ]
    cw2 = [2.8*cm] + [(PAGE_W - 2*MARGIN - 2.8*cm)/6]*6
    s.append(table_default(gwr_coef, cw2))

    s.append(Paragraph("<b>Coeficientes locais — MGWR (bandwidths por preditor):</b>", sH2))
    mgwr_coef = [
        ["Preditor (BW)", "Média", "Desvio Padrão", "P25", "P75", "Mínimo", "Máximo"],
        ["Intercept (304)", "−0,388", "0,009", "−0,397", "−0,380", "−0,410", "−0,375"],
        ["TEM_AVG (220)",   " 0,066", "0,027", " 0,055", " 0,089", "−0,021", " 0,101"],
        ["RAIN (126)",      "−0,119", "0,128", "−0,159", "−0,029", "−0,481", " 0,050"],
    ]
    cw3 = [3.2*cm] + [(PAGE_W - 2*MARGIN - 3.2*cm)/6]*6
    s.append(table_default(mgwr_coef, cw3))
    s.append(PageBreak())
    return s


# ── Seção 6: Coeficientes locais ─────────────────────────────────────────────

def sec6():
    s = []
    s += section_header("6", "Coeficientes Locais do MGWR")

    s.append(Paragraph(
        "Os coeficientes locais do MGWR variam por unidade de saúde, "
        "evidenciando não-estacionariedade espacial nos efeitos dos preditores. "
        "A dispersão (desvio padrão) de cada coeficiente indica o grau de "
        "heterogeneidade geográfica.",
        sBody))

    s.append(Paragraph("<b>Análise de não-estacionariedade:</b>", sH2))

    coef_analysis = [
        ["Preditor", "BW", "Média", "D. Padrão", "IQR", "Amplitude",
         "Não-estac.?"],
        ["Intercept", "304", "−0,388", "0,009", "0,017", "0,035", "Baixa"],
        ["TEM_AVG",   "220", " 0,066", "0,027", "0,034", "0,122", "Moderada"],
        ["RAIN",      "126", "−0,119", "0,128", "0,130", "0,531", "Alta"],
    ]
    cw = [2.8*cm, 1.4*cm, 1.8*cm, 2.0*cm, 1.6*cm, 2.2*cm,
          PAGE_W - 2*MARGIN - 11.8*cm]
    t = table_default(coef_analysis, cw)
    t.setStyle(TableStyle([
        ("BACKGROUND", (6,1), (6,1), AZUL_CLARO),
        ("BACKGROUND", (6,2), (6,2), AMARELO_BG),
        ("BACKGROUND", (6,3), (6,3), colors.HexColor("#fee2e2")),
        ("TEXTCOLOR",  (6,3), (6,3), LARANJA),
        ("FONTNAME",   (6,3), (6,3), "Helvetica-Bold"),
    ]))
    s.append(t)

    s.append(Spacer(1, 10))
    s.append(Paragraph("<b>Interpretação por preditor:</b>", sH2))

    s.append(Paragraph(
        "<b>Intercept (BW=304, D.P.=0,009):</b> coeficiente quase constante em todo "
        "o território (amplitude total de apenas 0,035). O intercepto local captura "
        "o nível basal de casos não explicado pelos preditores — sua homogeneidade "
        "sugere que fatores estruturais de fundo (saneamento, densidade) variam "
        "lentamente no espaço e são bem capturados por um BW grande.",
        sBody))

    s.append(Paragraph(
        "<b>TEM_AVG (BW=220, D.P.=0,027):</b> variação moderada. O efeito da "
        "temperatura sobre os casos de dengue é positivo na média (β̄ = 0,066), "
        "mas varia geograficamente: em áreas mais frescas (maciços montanhosos, "
        "Zona Sul litorânea) o efeito tende a ser menor, enquanto em zonas "
        "quentes e úmidas da Zona Oeste e Zona Norte o efeito é maior. "
        "O IQR de 0,034 indica variação sistemática, não ruído.",
        sBody))

    s.append(Paragraph(
        "<b>RAIN (BW=126, D.P.=0,128):</b> a maior heterogeneidade espacial do modelo. "
        "O coeficiente médio é negativo (β̄ = −0,119), sugerindo que, nas semanas de "
        "pico, a chuva não amplia necessariamente os casos — possivelmente porque "
        "chuvas intensas reduzem temporariamente a atividade humana ao ar livre. "
        "Contudo, a amplitude de 0,531 (de −0,481 a 0,050) revela que o efeito "
        "é positivo em algumas regiões (áreas de drenagem precária) e fortemente "
        "negativo em outras. Esta não-estacionariedade justifica plenamente o "
        "bandwidth local mais restrito.",
        sBody))

    s.append(Spacer(1, 6))
    interp_note = Table([[Paragraph(
        "<b>Nota metodológica:</b> o desvio padrão dos coeficientes locais no MGWR "
        "é sistematicamente menor que no GWR para Intercept e TEM_AVG, pois o MGWR "
        "usa bandwidths maiores (mais suavização), reduzindo ruído. Para RAIN, "
        "ambos apresentam D.P. similar (~0,128–0,131), pois o bandwidth coincide (126).",
        S("note2", fontName="Helvetica", fontSize=9.5, leading=13,
          textColor=CINZA_TITULO, alignment=TA_JUSTIFY))]],
        colWidths=[PAGE_W - 2*MARGIN])
    interp_note.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), AZUL_CLARO),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("BOX",           (0,0), (-1,-1), 1, AZUL_MEDIO),
    ]))
    s.append(interp_note)
    s.append(PageBreak())
    return s


# ── Seção 7: Painel de mapas ──────────────────────────────────────────────────

def sec7(panel_path):
    s = []
    s += section_header("7", "Painel de Mapas dos Coeficientes Locais")

    s.append(Paragraph(
        "O painel abaixo apresenta a distribuição geográfica dos coeficientes "
        "locais do MGWR para cada preditor. Cada ponto representa uma unidade de "
        "saúde (N = 307), colorida pelo valor do coeficiente local estimado.",
        sBody))

    if panel_path and os.path.exists(panel_path):
        img_w = PAGE_W - 2*MARGIN
        img = Image(panel_path, width=img_w, height=img_w * 0.52)
        s.append(img)
        s.append(Spacer(1, 4))
        s.append(Paragraph(
            "Figura 1 — Coeficientes locais do MGWR por unidade de saúde (N=307). "
            "Da esquerda para direita: Intercept (BW=304), TEM_AVG (BW=220) e "
            "RAIN (BW=126). Coloração: azul escuro = coeficiente negativo, "
            "vermelho/laranja = coeficiente positivo.",
            sCaption))
    else:
        s.append(Paragraph(
            "[Arquivo mgwr_panel.png não encontrado no caminho especificado]",
            sCaption))

    s.append(Paragraph("<b>Interpretação dos padrões geográficos:</b>", sH2))
    s.append(bullet(
        "<b>Intercept:</b> coeficientes negativos em toda a área — consistente com "
        "o nível basal baixo em escala log. A variação é mínima (conforme D.P.=0,009), "
        "refletindo a homogeneidade capturada pelo BW=304."))
    s.append(bullet(
        "<b>TEM_AVG:</b> gradiente suave — valores mais positivos nas zonas de "
        "baixada (Zona Oeste, Zona Norte) onde temperaturas elevadas coincidem com "
        "maior proliferação do Aedes aegypti. Valores menores nas regiões serranas."))
    s.append(bullet(
        "<b>RAIN:</b> maior heterogeneidade espacial. Coeficientes negativos "
        "concentram-se em áreas com boa drenagem pluvial; coeficientes menos "
        "negativos (ou positivos) ocorrem em regiões de alagamento frequente, "
        "onde a chuva acumula água parada e favorece criadouros."))

    s.append(PageBreak())
    return s


# ── Seção 8: Implicações metodológicas ───────────────────────────────────────

def sec8():
    s = []
    s += section_header("8", "Implicações Metodológicas")

    s.append(Paragraph(
        "Os resultados do MGWR se articulam com a cadeia completa de análise "
        "espacial conduzida nesta dissertação, triangulando evidências e "
        "fundamentando as escolhas de raios de influência no modelo final.",
        sBody))

    s.append(Paragraph("<b>Triangulação com Moran's I e variograma:</b>", sH2))
    triang_data = [
        ["Análise", "Resultado", "Implicação para o MGWR"],
        ["Moran's I\n(semanas epidêmicas)",
         "I ≈ 0,35–0,55\n(p < 0,001)",
         "Confirma autocorrelação espacial — "
         "justifica modelagem geograficamente ponderada em vez de OLS global"],
        ["Variograma",
         "Range ≈ 15 km",
         "Escala de dependência local — RAIN (BW=126) opera em escala "
         "compatível com ~8–15 km de influência efetiva"],
        ["GWR (BW único)",
         "AICc = 491,0\nR² = 0,130",
         "Modelo de referência; BW único = 126 captura apenas a escala de RAIN, "
         "ignorando as escalas maiores de TEM_AVG e Intercept"],
        ["MGWR",
         "AICc = 488,9\nBW: 304/220/126",
         "Melhora o AICc e revela estrutura multiescala consistente com "
         "os achados do variograma e Moran's I"],
    ]
    cw = [2.8*cm, 2.5*cm, PAGE_W - 2*MARGIN - 5.3*cm]
    t = table_default(triang_data, cw)
    t.setStyle(TableStyle([
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",    (0,1), (-1,-1), 8.5),
        ("LEADING",     (0,1), (-1,-1), 12),
    ]))
    s.append(t)

    s.append(Paragraph("<b>Justificativa dos raios [3,8 / 7,6 / 15,2 km]:</b>", sH2))
    s.append(Paragraph(
        "O bandwidth de RAIN (126 unidades de 307) representa aproximadamente 41% "
        "do território e corresponde a uma escala de influência de ~8–15 km, "
        "consistente com o range do variograma (≈ 15 km). Usando RAIN como "
        "âncora de calibração, os três raios de influência são definidos como:",
        sBody))
    s.append(bullet(
        "<b>3,8 km</b> (escala hiperlocal) — raio mínimo de criadouros do Aedes; "
        "bairros individuais."))
    s.append(bullet(
        "<b>7,6 km</b> (escala local) — ≈ metade do range do variograma; "
        "zona de forte dependência espacial."))
    s.append(bullet(
        "<b>15,2 km</b> (escala regional) — coincide com o range do variograma; "
        "compatível com o BW de RAIN no MGWR."))

    s.append(Paragraph("<b>Limitações do modelo MGWR:</b>", sH2))
    s.append(bullet(
        "<b>Snapshot temporal:</b> agrega a dimensão temporal em uma única "
        "observação por unidade, perdendo a dinâmica sazonal e as tendências "
        "interanuais presentes no painel completo."))
    s.append(bullet(
        "<b>Log-transformação vs Poisson:</b> y = log(ȳ + 0,5) é uma aproximação; "
        "dados de contagem seriam mais rigorosamente modelados por GWR-Poisson "
        "(ainda em desenvolvimento no ecossistema pysal/mgwr)."))
    s.append(bullet(
        "<b>Preditores limitados:</b> o modelo utilizou apenas 3 preditores "
        "(Intercept, TEM_AVG, RAIN) de um total de 47 features disponíveis. "
        "A inclusão de lags de casos (CASES_LAG_1, CASES_LAG_2) e médias móveis "
        "poderia melhorar substancialmente o R²."))
    s.append(bullet(
        "<b>N reduzido:</b> 307 unidades representam uma amostra espacial "
        "relativamente pequena para MGWR. A convergência do algoritmo iterativo "
        "pode ser sensível à inicialização dos bandwidths."))
    s.append(bullet(
        "<b>Coordenadas em graus decimais:</b> distâncias em graus decimais "
        "(lat/lng) não são isométricas — a distância em km varia com a latitude. "
        "Para análises de precisão, recomenda-se reprojetar para UTM (SIRGAS 2000 "
        "zona 23S) antes do ajuste."))

    s.append(PageBreak())
    return s


# ── Seção 9: Conclusão ────────────────────────────────────────────────────────

def sec9():
    s = []
    s += section_header("9", "Conclusão e Cadeia de Evidências")

    s.append(Paragraph(
        "Este diagnóstico demonstra que o MGWR é superior ao GWR no contexto "
        "da previsão de dengue no Rio de Janeiro, tanto pela melhora do AICc "
        "(ΔAICc = 2,08) quanto pela revelação de uma estrutura espacial multiescala "
        "teoricamente coerente com a epidemiologia da dengue urbana.",
        sBody))

    s.append(Paragraph(
        "O achado mais relevante é que <b>cada preditor opera em sua escala espacial "
        "natural</b>: fatores climáticos de meso-escala (temperatura) têm BW=220; "
        "fatores de escala local-regional (chuva) têm BW=126; e a estrutura de "
        "fundo (intercepto) é essencialmente global (BW=304). Esta hierarquia "
        "de escalas é consistente com a teoria ecológica do Aedes aegypti.",
        sBody))

    s.append(Paragraph("<b>Tabela-resumo: Cadeia de Evidências Espaciais</b>", sH2))
    chain_data = [
        ["Etapa", "Método", "Resultado-chave", "Implicação"],
        ["1",
         "Moran's I\nglobal",
         "I ≈ 0,35–0,55\np < 0,001",
         "Autocorrelação espacial significativa → modelos globais são inadequados"],
        ["2",
         "Variograma\nexperimental",
         "Range ≈ 15 km\nSill estabilizado",
         "Dependência espacial forte até 15 km → define raio de vizinhança"],
        ["3",
         "GWR\n(BW único)",
         "AICc = 491,0\nR² = 0,130",
         "BW único = 126 captura RAIN, mas sub-otimiza TEM_AVG e Intercept"],
        ["4",
         "MGWR\n(BW por preditor)",
         "AICc = 488,9\nBW: 304/220/126",
         "ΔAICc = 2,08 — estrutura multiescala confirmada empiricamente"],
        ["5",
         "Raios de\ninfluência",
         "3,8 / 7,6 / 15,2 km",
         "Calibrados pelo range do variograma e BW de RAIN — triângulo de evidências"],
    ]
    cw = [1.2*cm, 2.4*cm, 3.0*cm, PAGE_W - 2*MARGIN - 6.6*cm]
    t = table_default(chain_data, cw, header_bg=VERDE)
    t.setStyle(TableStyle([
        ("VALIGN",  (0,0), (-1,-1), "TOP"),
        ("FONTSIZE",(0,1), (-1,-1), 8.5),
        ("LEADING", (0,1), (-1,-1), 12),
        ("BACKGROUND", (0,4), (-1,4), colors.HexColor("#f0fdf4")),
        ("FONTNAME",   (0,4), (-1,4), "Helvetica-Bold"),
    ]))
    s.append(t)

    s.append(Spacer(1, 12))
    final = Table([[Paragraph(
        "<b>Síntese final:</b> Moran's I confirmou a necessidade de modelagem "
        "espacial. O variograma calibrou a escala de dependência (~15 km). "
        "O GWR verificou que os coeficientes variam geograficamente. "
        "O MGWR revelou que essa variação ocorre em escalas distintas por preditor. "
        "Em conjunto, estas evidências fundamentam os raios [3,8; 7,6; 15,2 km] "
        "utilizados na construção das features espaciais do modelo de aprendizado de "
        "máquina desta dissertação.",
        S("fin", fontName="Helvetica", fontSize=10.5, leading=15,
          textColor=CINZA_TITULO, alignment=TA_JUSTIFY))]],
        colWidths=[PAGE_W - 2*MARGIN])
    final.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
        ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ("TOPPADDING",    (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
        ("BOX",           (0,0), (-1,-1), 1.5, VERDE),
    ]))
    s.append(final)

    s.append(Spacer(1, 16))
    s.append(hr(CINZA_BORDA))
    s.append(Paragraph(
        "Fotheringham, A.S., Yang, W., Kang, W. (2017). Multiscale Geographically "
        "Weighted Regression (MGWR). Annals of the American Association of "
        "Geographers, 107(6), 1247–1265.",
        sFootnote))
    s.append(Paragraph(
        "Burnham, K.P., Anderson, D.R. (2002). Model Selection and Multimodel "
        "Inference: A Practical Information-Theoretic Approach (2nd ed.). Springer.",
        sFootnote))
    return s


# ── Header/Footer ─────────────────────────────────────────────────────────────

def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Cabeçalho
    canvas.setFillColor(AZUL_ESCURO)
    canvas.rect(0, h - 1.1*cm, w, 1.1*cm, fill=1, stroke=0)
    canvas.setFillColor(BRANCO)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(MARGIN, h - 0.72*cm,
                      "MGWR — Regressão Geograficamente Ponderada Multiescala")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - MARGIN, h - 0.72*cm, "Dissertação de Mestrado · RJ Dengue")
    # Rodapé
    canvas.setFillColor(AZUL_ESCURO)
    canvas.rect(0, 0, w, 0.9*cm, fill=1, stroke=0)
    canvas.setFillColor(BRANCO)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN, 0.3*cm, f"Gerado em {date.today().strftime('%d/%m/%Y')}")
    canvas.drawCentredString(w/2, 0.3*cm, "Dataset: RJ_WEEKLY · N=307 unidades")
    canvas.drawRightString(w - MARGIN, 0.3*cm, f"Página {doc.page}")
    canvas.restoreState()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output_path = r"C:\@work\mcs\diagnostico_mgwr.pdf"
    panel_path  = r"C:\@work\mcs\outputs\mgwr_results\mgwr_panel.png"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.4*cm, bottomMargin=1.2*cm,
        title="MGWR — Diagnóstico Espacial RJ Dengue",
        author="Dissertação de Mestrado",
    )

    story = []
    story += cover_flowables()
    story += sec1()
    story += sec2()
    story += sec3()
    story += sec4()
    story += sec5()
    story += sec6()
    story += sec7(panel_path)
    story += sec8()
    story += sec9()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"PDF gerado: {output_path}")


if __name__ == "__main__":
    main()
