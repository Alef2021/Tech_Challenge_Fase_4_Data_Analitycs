import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import streamlit as st


st.set_page_config(
    page_title="Dashboard - Análise Exploratória de Obesidade",
    layout="wide"
)
# =============================
# BOTÃO DE NAVEGAÇÃO (TOPO DIREITO)
# =============================

col1, col2, col3 = st.columns([8, 1, 1])

with col3:
    if st.button("⚙️ Modelo ML de Previsão"):
        st.switch_page("pages/app.py")



# =============================
# CONFIGURAÇÃO GLOBAL DE ESTILO
# =============================
colors = ['black', 'gray', 'lightgray']
sns.set_style("whitegrid")

# =============================
# CONFIGURAÇÃO DA PÁGINA
# =============================
st.set_page_config(
    page_title="Dashboard - Análise Exploratória de Obesidade",
    layout="wide"
)

st.title("📊 Análise Exploratória — Obesidade")

# =============================
# CARREGAMENTO DOS DADOS
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/Obesity.csv")

    df = df.rename(columns={
        'Gender': 'Gênero',
        'Age': 'Idade',
        'Height': 'Altura',
        'Weight': 'Peso',
        'family_history': 'Histórico_Familiar_Obesidade',
        'FAVC': 'Frequencia_Consumo_Alimento_Calorico',
        'FCVC': 'Frequencia_Consumo_Vegetais',
        'NCP': 'Numero_Refeicoes_Principais',
        'CAEC': 'Consumo_Alimento_Entre_Refeicoes',
        'SMOKE': 'Fumante',
        'CH2O': 'Consumo_Agua',
        'SCC': 'Monitoramento_Calorico',
        'FAF': 'Frequencia_Atividade_Fisica',
        'TUE': 'Tempo_Uso_Tecnologia',
        'CALC': 'Consumo_Alcool',
        'MTRANS': 'Meio_Transporte',
        'Obesity': 'Status_Obesidade'
    })

    mapa_consumo_alcool = {
    'no': 'Não consome',
    'Sometimes': 'Consumo ocasional',
    'Frequently': 'Consumo frequente',
    'Always': 'Consumo diário'
    }

    df['Consumo_Alcool'] = df['Consumo_Alcool'].map(mapa_consumo_alcool)

    cols_cat = [
        'Tempo_Uso_Tecnologia',
        'Frequencia_Atividade_Fisica',
        'Consumo_Agua',
        'Numero_Refeicoes_Principais',
        'Frequencia_Consumo_Vegetais'
    ]

    for col in cols_cat:
        df[col] = df[col].astype(str).str[0].astype(int)

    df['Idade'] = df['Idade'].astype(int)
    df['Peso'] = df['Peso'].astype(float)
    df['Altura'] = df['Altura'].astype(float)

    def normalize(level):
        if level == 'Insufficient_Weight':
            return "Abaixo do peso"
        elif level == 'Normal_Weight':
            return "Peso normal"
        elif level in ['Overweight_Level_I', 'Overweight_Level_II']:
            return "Sobrepeso"
        else:
            return "Obeso"

    df['Status_Obesidade'] = df['Status_Obesidade'].apply(normalize)
    df['IMC'] = df['Peso'] / (df['Altura'] ** 2)

    return df


df = load_data()

# =============================
# SIDEBAR — FILTROS
# =============================
st.sidebar.markdown("## 🎛️ Filtros")

status_opcoes = df['Status_Obesidade'].unique()
status_filtro = [
    s for s in status_opcoes
    if st.sidebar.checkbox(s, value=True)
]

if not status_filtro:
    status_filtro = status_opcoes

genero = st.sidebar.radio(
    "🚻 Gênero",
    ["Todos"] + list(df['Gênero'].unique()),
    horizontal=True
)

genero_filtro = df['Gênero'].unique() if genero == "Todos" else [genero]

idade_min, idade_max = st.sidebar.slider(
    "🎂 Faixa Etária",
    int(df['Idade'].min()),
    int(df['Idade'].max()),
    (
        int(df['Idade'].quantile(0.05)),
        int(df['Idade'].quantile(0.95))
    )
)

agua_map = {
    "Baixo": [1],
    "Médio": [2],
    "Alto": [3],
    "Todos": [1, 2, 3]
}

nivel_agua = st.sidebar.radio(
    "💧 Consumo de Água",
    list(agua_map.keys()),
    horizontal=True,
    index=3
)

# =============================
# DATAFRAME FILTRADO ÚNICO
# =============================
df_filtrado = df[
    (df['Status_Obesidade'].isin(status_filtro)) &
    (df['Gênero'].isin(genero_filtro)) &
    (df['Idade'].between(idade_min, idade_max)) &
    (df['Consumo_Agua'].isin(agua_map[nivel_agua]))
]

# =============================
# HISTOGRAMAS
# =============================
st.markdown("---")
st.subheader("📊 Distribuição de Indicadores")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, col in zip(axes, ['IMC', 'Idade', 'Consumo_Agua']):
    sns.histplot(
        df_filtrado[col],
        bins=20,
        kde=True,
        ax=ax,
        color='black'
    )
    ax.set_title(col)

st.pyplot(fig)

# =============================
# KPIs
# =============================
st.markdown("### 🧠 Insights")

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Indivíduos", len(df_filtrado))
c2.metric("📏 IMC Médio", f"{df_filtrado['IMC'].mean():.2f}")
c3.metric("🎂 Idade Média", f"{df_filtrado['Idade'].mean():.1f}")
c4.metric("💧 Água Média", f"{df_filtrado['Consumo_Agua'].mean():.1f}")

# =============================
# GRÁFICOS DE PIZZA
# =============================
st.markdown("---")
st.subheader("🍕 Perfil Comportamental")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

cols_pizza = [
    'Gênero',
    'Histórico_Familiar_Obesidade',
    'Frequencia_Consumo_Alimento_Calorico'
]

for ax, col in zip(axes, cols_pizza):
    cont = df_filtrado[col].value_counts()
    ax.pie(
        cont.values,
        labels=cont.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=list(itertools.islice(itertools.cycle(colors), len(cont))),
        wedgeprops={"edgecolor": "black"}
    )
    ax.set_title(col.replace("_", " "))
    ax.axis('equal')

st.pyplot(fig)

# ==================================
# KPIs — PERFIL COMPORTAMENTAL (%)
# ==================================
st.markdown("---")
st.subheader("### 🧠 Insights — Perfil Comportamental")

def kpi_percentual(coluna, titulo, icone):
    dist = df_filtrado[coluna].value_counts(normalize=True) * 100

    st.markdown(f"**{icone} {titulo}**")
    cols = st.columns(len(dist))

    for col_ui, (categoria, valor) in zip(cols, dist.items()):
        col_ui.metric(
            label=categoria,
            value=f"{valor:.1f}%"
        )


# Histórico Familiar
kpi_percentual(
    coluna='Histórico_Familiar_Obesidade',
    titulo='Histórico Familiar de Obesidade',
    icone='🧬'
)

# Consumo de Alimento Calórico
kpi_percentual(
    coluna='Frequencia_Consumo_Alimento_Calorico',
    titulo='Consumo de Alimentos Calóricos',
    icone='🍔'
)

# Consumo de Álcool
kpi_percentual(
    coluna='Consumo_Alcool',
    titulo='Consumo de Álcool',
    icone='🍺'
)

# =============================
# BARRAS POR GÊNERO
# =============================
st.markdown("---")
st.subheader("📊 Distribuição Percentual por Gênero")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, col in zip(
    axes,
    ['Consumo_Alcool', 'Frequencia_Atividade_Fisica', 'Status_Obesidade']
):
    tabela = pd.crosstab(
        df_filtrado[col],
        df_filtrado['Gênero'],
        normalize='columns'
    ) * 100

    tabela.plot(
        kind='bar',
        ax=ax,
        color=colors
    )

    ax.set_title(col.replace("_", " "))
    ax.set_ylabel("Percentual (%)")
    ax.legend(title="Gênero")

st.pyplot(fig)

st.markdown("---")
st.subheader("### 🚻   Insights Distribuição Percentual por Gênero")

total = len(df_filtrado)

c1, c2 = st.columns(2)

for genero in df_filtrado['Gênero'].unique():
    percentual = (df_filtrado['Gênero'] == genero).sum() / total * 100

    if genero.lower().startswith("m"):
        c1.metric(
            label=f"👨 {genero}",
            value=f"{percentual:.1f}%",
            delta=f"{(percentual - 50):+.1f} pp"
        )
    else:
        c2.metric(
            label=f"👩 {genero}",
            value=f"{percentual:.1f}%",
            delta=f"{(percentual - 50):+.1f} pp"
        )


# =============================
# TABELA
# =============================
with st.expander("📄 Dados filtrados"):
    st.dataframe(df_filtrado)
