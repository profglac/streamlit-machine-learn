import streamlit as st
import pandas as pd

from trainmodels import trainModels
from printChart import printChart

st.title('Modelos de Machine Learning')

#Barra lateral de navegação
st.sidebar.title('Navegação')
options = ['Início', 'Carregar Dados', 'Selecionar Features', 'Treinar Modelo', 'Resultado' ]
selection = st.sidebar.radio("Escolha a página:", options)

if selection == 'Início':
    st.write(""" 
        ### Bem-vindo à aplicação de Modelos de Machine Learn!
        Esta aplicação permite:
        - Carregar dados de um arquivo CSV
        - Selecionar as Características (Features) e variáveis alvo (Target variables)
        - Escolher e treinar os modelos de Machine Learn
        - Avaliar e visualizar a performance de cada modelo
    """)

if selection == 'Carregar Dados':
    st.header('Carregue seus dados')
    uploaded_file = st.file_uploader('Faça upload de um arquivo CSV', type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write('### Pre-visualização dos dados')
        st.dataframe(df.head(10))

        st.session_state['df'] = df

if selection == 'Selecionar Features':
    st.header('Selecionar as Features e Target variables')
    
    if 'df' not in st.session_state:
        st.warning('Favor carregar os dados primeiro!')
    else:
        df = st.session_state['df']

        # Só Colunas que não são categóricas ou object, apenas numéricas
        all_columns = df.select_dtypes(exclude=['category', 'object']).columns.tolist()

        # Todas as opções de all_columns
        target = st.selectbox("Target variable (Variáveis alvo)", all_columns)
        
        # todas as colunas de all_columns, exceto a variável target.
        features = st.multiselect("Features (características)", [col for col in all_columns if col!= target])

        if st.button("Confirme a seleção"):
            if not features:
                st.error("Favor selecionar ao menos uma Feature")
            else:
                st.session_state['features'] = features
                st.session_state['target'] = target
                st.success("Features e Target variables selecionadas com sucesso")

if selection == 'Treinar Modelo':
    st.header("Train ML models")

    if 'df' not in st.session_state or 'features' not in st.session_state or 'target' not in st.session_state:
        st.warning('Favor carregar e selecionar features primeiro')
    else:
        trainModels()

if selection == 'Resultado':
    st.header('Resultado da performance dos modelos')

    if 'trained_models' not in st.session_state:
        st.warning('Favor treinar os modelos primeiro')
    else:
        trained_models = st.session_state['trained_models']
        x_test = st.session_state['x_test']
        y_test = st.session_state['y_test']

    for model_name, model in trained_models.items():
        printChart(model_name, model, x_test, y_test)