import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def trainModels():
    df = st.session_state['df']
    features = st.session_state['features']
    target = st.session_state['target']

    X = df[features]
    y = df[target]

    test_size = st.slider("Tamanho do Teste (%)", 10, 50, 20)
    random_state = st.number_input("Ramdom State", value=42, step=1)

    models = {
        'LinearRegression': LinearRegression(),
        'RamdomForestRegressor': RandomForestRegressor(),
        'Support Vector Regressor': SVR(),
        'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    }
    
    selected_models = st.multiselect('Selecione modelos para o trem', list(models.keys()), default=list(models.keys()))

    if st.button("Modelos do trem"):
        if not selected_models:
            st.error("Favor selecionar ao menos um modelo para o trem")
        else:
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state) 

        # O standardScaler coloca as variáveis na mesma escala
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        st.session_state['x_train'] = x_train
        st.session_state['x_test'] = x_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['scaler'] = scaler

        st.success("Os dados foram divididos em treino e teste com sucesso")

        trained_models = {}
        for model_name in selected_models:
            model = models[model_name]
            model.fit(x_train, y_train)
            trained_models[model_name] = model

        st.session_state["trained_models"] = trained_models
        st.success("Todos os modelos foram treinados")
