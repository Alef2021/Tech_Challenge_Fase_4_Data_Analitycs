import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
import math





try:
    model = joblib.load(f'src/modelos/modelo_obesidade_xgb_model_3_class.pkl')



except FileNotFoundError:
    st.error("Erro: Arquivo do modelo 'src/modelos/modelo_obesidade_xgb_model_3_class.pkl' não encontrado. Certifique-se de ter salvo o modelo no diretório correto.")
    st.stop()

COLUNAS_FEATURES = ['Gênero', 'Idade', 'Histórico_Familiar_Obesidade', 
                        'Frequencia_Consumo_Alimento_Calorico', 'Frequencia_Consumo_Vegetais', 
                        'Numero_Refeicoes_Principais', 'Consumo_Alimento_Entre_Refeicoes', 
                        'Fumante', 'Consumo_Agua', 'Monitoramento_Calorico', 
                        'Frequencia_Atividade_Fisica', 'Tempo_Uso_Tecnologia', 
                        'Consumo_Alcool','Meio_Transporte'] 

def coletar_dados_paciente():
    st.sidebar.header('Dados do Paciente')
##################################################################################################

    # Gênero: 0=Feminino, 1=Masculino
    genero_map = {'Feminino': 0, 'Masculino': 1}
    genero = st.sidebar.selectbox('Gênero', options=list(genero_map.keys()))

#################################################################################################

    # Idade, Peso, Altura (uso simples no Streamlit)
    idade = st.sidebar.slider('Idade (anos)', min_value=10, max_value=80, value=30, step=1)

################################################################################################# 

    # colunas bbinarias (0/1)
    sim_nao_map = {'Não': 0, 'Sim': 1}
    hist_familiar = st.sidebar.selectbox('Tem histórico Familiar de Obesidade?', options=list(sim_nao_map.keys()))
    favc = st.sidebar.selectbox('Consome Fast food frequentemente?', options=list(sim_nao_map.keys()))
    fumante = st.sidebar.selectbox('É Fumante?', options=list(sim_nao_map.keys()))
    scc = st.sidebar.selectbox('Você monitora as calorias que ingere diariamente?', options=list(sim_nao_map.keys()))

#################################################################################################

    # Colunas de 0 a 3 (Frequência/Quantidade)
    # Consumo_Alimento_Entre_Refeicoes, Consumo_Alcool (no=0, Sometimes=1, Frequently=2, Always=3)
    frequencia_map = {'Nunca ': 0, 'Às vezes ': 1, 'Frequentemente ': 2, 'Sempre ': 3}
    caec = st.sidebar.selectbox('Come alimentos caloricos entre as refeições ?', options=list(frequencia_map.keys()))
    calc = st.sidebar.selectbox('Consome Álcool ?', options=list(frequencia_map.keys()))
    fcvc = st.sidebar.selectbox('Consome Vegetais ?', options=list(frequencia_map.keys()))
    tue = st.sidebar.selectbox('Passa muito tempo no celular?', options=list(frequencia_map.keys()))
    faf = st.sidebar.selectbox('Frequência Atividade Física ?', options=list(frequencia_map.keys()))

################################################################################################# 

    # Consumo_Agua Numero_Refeicoes_Principais (0=Baixa, 1=Média, 2=Alta intensidade)
    litro_map = {'Até 1 litro': 0, 'Até 2 litros': 1, 'Até 3 litros': 2,'Até 4 litros': 3}
    ch2o = st.sidebar.selectbox('Consome quanto Litros de Água por dia ? ', options=list(litro_map.keys()))

    refeicao_map = {'1': 0, '2': 1, '3': 2,'4': 3}
    ncp = st.sidebar.selectbox('Nº de Refeições Principais no dia ?', options=list(refeicao_map.keys()))


    # meio_transportes (0=Baixa, 1=Média, 2=Alta intensidade)
    transporte_map = {'Automóvel/Moto': 0, 'Transporte Público': 1, 'Caminhar/Bike': 2}
    mtrans = st.sidebar.selectbox('Qual meio de transporte você costuma usar?', options=list(transporte_map.keys()))

#################################################################################################

    # Estruturars os dados no formato que o modelo espera
    dados_entrada = {
        'Gênero': genero_map[genero],
        'Idade': idade,
        'Histórico_Familiar_Obesidade': sim_nao_map[hist_familiar],
        'Frequencia_Consumo_Alimento_Calorico': sim_nao_map[favc],
        'Frequencia_Consumo_Vegetais': frequencia_map[fcvc],
        'Numero_Refeicoes_Principais': refeicao_map[ncp],
        'Consumo_Alimento_Entre_Refeicoes': frequencia_map[caec],
        'Fumante': sim_nao_map[fumante],
        'Consumo_Agua': litro_map[ch2o],
        'Monitoramento_Calorico': sim_nao_map[scc],
        'Frequencia_Atividade_Fisica': frequencia_map[faf],
        'Tempo_Uso_Tecnologia': frequencia_map[tue],
        'Consumo_Alcool': frequencia_map[calc],
        'Meio_Transporte': transporte_map[mtrans], 
    }
    
    # Retornar como um DataFrame com a ordem de colunas correta
    features = pd.DataFrame(dados_entrada, index=[0])
    return features[COLUNAS_FEATURES]    

#################################################################################################

# Título e descrição do aplicativo
st.title('Sistema Preditivo de Risco de Obesidade')
st.markdown("""
Este aplicativo utiliza um modelo de Machine Learning (XGBoost) para prever o risco de obesidade 
de um paciente com base em fatores de estilo de vida.
""")
#################################################################################################

# Chamada da função para coletar os dados do paciente
input_df = coletar_dados_paciente()

st.subheader('Preencha os dados a esquerda e clique em Fazer Previsão de Risco')

#################################################################################################

#gif animado de fundo

gif_path = r"gif_fundo/giphy.gif"

if os.path.exists(gif_path):
    with open(gif_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")

    # HTML sem fundo preto, apenas o GIF centralizado
    html = f"""
    <div style="padding:20px; border-radius:10px; display:flex; justify-content:center;">
        <img src="data:image/gif;base64,{b64}" alt="GIF animado" style="width:400px; border-radius:10px;" />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
else:
    st.error(f"GIF não encontrado em: {gif_path}")

#################################################################################################

# Botão para fazer a previsão
if st.button('Fazer Previsão de Risco'):
    # Fazer a previsão usando o modelo carregado
    prediction = model.predict(input_df)
    
    # Fazer a previsão da probabilidade (para dar mais contexto)
    prediction_proba = model.predict_proba(input_df)

    # NOVO Mapeamento para 3 classes: 0, 1 e 2
    resultado_map = {
        0: 'PESO NORMAL / BAIXO RISCO', 
        1: 'SOBREPESO / RISCO MÉDIO', 
        2: 'OBESIDADE / ALTO RISCO'
    }
    
    # NOVAS Cores para 3 classes: Verde, Amarelo/Laranja, Vermelho
    cor_map = {
        0: '#34D399',  # Verde
        1: '#FBBF24',  # Laranja
        2: '#EF4444'   # Vermelho
    }
    
    # O valor predito é a primeira (e única) previsão no array
    classe_predita = prediction[0]
    status_risco = resultado_map[classe_predita]
    
    # Pega a probabilidade da classe que foi prevista
    prob_risco = prediction_proba[0][classe_predita] * 100
    
    st.markdown("---")
    st.subheader('Resultado da Previsão para a Equipe Médica')
    
    # Apresentação do resultado com cor e classe (CSS inline)
    st.markdown(f"""
    <div style='background-color: {cor_map[classe_predita]}; padding: 15px; border-radius: 10px;'>
        <h3 style='color: white; text-align: center;'>Status de Risco: {status_risco}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.success(f"O modelo prevê que o paciente está na categoria **{status_risco}** com **{prob_risco:.2f}%** de confiança.")
    
    # Lógica de recomendação ajustada para 3 classes
    if classe_predita == 2:
        st.error("Risco ALTO: Recomenda-se acompanhamento e intervenção imediatos para tratamento da obesidade.")
    elif classe_predita == 1:
        st.warning("Risco MÉDIO: Recomenda-se monitoramento e ajuste de hábitos para evitar progressão para obesidade.")
    else: # classe_predita == 0
        st.info("Risco BAIXO: Recomenda-se a manutenção dos hábitos atuais e monitoramento periódico.")

#################################################################################################

    ########GRAFICO 1###############
    st.markdown("---")
    st.subheader("🛡️ Principais Hábitos que Mais Impactaram Negativamente")

    # Todas as respostas do usuário (escala 0–3)
    habitos_usuario = {
        'Atividade Física': input_df['Frequencia_Atividade_Fisica'][0],
        'Consumo de Vegetais': input_df['Frequencia_Consumo_Vegetais'][0],
        'Consumo de Água': input_df['Consumo_Agua'][0],
        'Número de Refeições': input_df['Numero_Refeicoes_Principais'][0],
        'Alimentos Entre Refeições': input_df['Consumo_Alimento_Entre_Refeicoes'][0],
        'Consumo de Álcool': input_df['Consumo_Alcool'][0],
        'Uso de Tecnologia': input_df['Tempo_Uso_Tecnologia'][0],
        'Alimentos Calóricos': input_df['Frequencia_Consumo_Alimento_Calorico'][0],
        'Monitoramento Calórico': input_df['Monitoramento_Calorico'][0],
        'Tabagismo': input_df['Fumante'][0],
    }

    # Impacto negativo = distância do ideal (3)
    impacto_negativo = {
        k: abs(v - 3) for k, v in habitos_usuario.items()
    }

    df_impacto = (
        pd.DataFrame.from_dict(
            impacto_negativo,
            orient='index',
            columns=['Impacto Negativo']
        )
        .sort_values('Impacto Negativo', ascending=False)
    )

    # Remove hábitos adequados (impacto zero)
    df_impacto = df_impacto[df_impacto['Impacto Negativo'] > 0]

    # Top 3 hábitos que mais influenciaram negativamente
    df_pizza_usuario = df_impacto.head(3)

    # Cores neutras
    cores_pb = ['black', 'gray', 'lightgray']

    fig1, ax1 = plt.subplots()
    ax1.pie(
        df_pizza_usuario['Impacto Negativo'],
        labels=df_pizza_usuario.index,
        autopct='%1.0f%%',
        startangle=90,
        colors=cores_pb,
        wedgeprops={'edgecolor': 'black'}
    )

    ax1.set_title('Top 3 Hábitos que Mais Impactaram Negativamente Seu Resultado')
    st.pyplot(fig1)


    ########GRAFICO 2###############
    st.markdown("---")
    st.subheader("📊 Perfil Comportamental Geral")

    labels = [
        'Atividade Física',
        'Vegetais',
        'Água',
        'Álcool',
        'Tecnologia',
        'Alimentos Calóricos'
    ]

    valores = [
        input_df['Frequencia_Atividade_Fisica'][0],
        input_df['Frequencia_Consumo_Vegetais'][0],
        input_df['Consumo_Agua'][0],
        3 - input_df['Consumo_Alcool'][0],
        3 - input_df['Tempo_Uso_Tecnologia'][0],
        3 - input_df['Frequencia_Consumo_Alimento_Calorico'][0]
    ]

    valores += valores[:1]

    angles = [n / float(len(labels)) * 2 * math.pi for n in range(len(labels))]
    angles += angles[:1]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, polar=True)

    ax2.plot(angles, valores, color='black')
    ax2.fill(angles, valores, alpha=0.15, color='black')

    ax2.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax2.set_ylim(0, 3)

    st.pyplot(fig2)


    ### GRAFICO 3###############

    st.markdown("---")
    st.subheader("🛡️ Prioridades de Melhoria para Prevenção da Obesidade")

    # Valores atuais do usuário (0 a 3)
    habitos_usuario = {
        'Atividade Física': input_df['Frequencia_Atividade_Fisica'][0],
        'Consumo de Vegetais': input_df['Frequencia_Consumo_Vegetais'][0],
        'Ingestão de Água': input_df['Consumo_Agua'][0],
        'Controle do Álcool': 3 - input_df['Consumo_Alcool'][0],
        'Tempo de Tela': 3 - input_df['Tempo_Uso_Tecnologia'][0],
        'Alimentos Calóricos': 3 - input_df['Frequencia_Consumo_Alimento_Calorico'][0],
    }

    # GAP para o comportamento ideal
    gap_preventivo = {k: 3 - v for k, v in habitos_usuario.items()}

    df_gap = (
        pd.DataFrame.from_dict(
            gap_preventivo,
            orient='index',
            columns=['Distância do Ideal']
        )
        .sort_values('Distância do Ideal', ascending=False)
    )

    # Remove hábitos já adequados
    df_gap = df_gap[df_gap['Distância do Ideal'] > 0]

    # Seleciona os 3 principais
    df_pizza_3 = df_gap.head(3)

    # Cores preto, cinza e branco
    cores_pb = ['black', 'gray', 'lightgray']

    fig3, ax3 = plt.subplots()
    ax3.pie(
        df_pizza_3['Distância do Ideal'],
        labels=df_pizza_3.index,
        autopct='%1.0f%%',
        startangle=90,
        colors=cores_pb,
        wedgeprops={'edgecolor': 'black'}
    )

    ax3.set_title('Top 3 Fatores Onde Pequenas Mudanças Mais Ajudam')
    st.pyplot(fig3)