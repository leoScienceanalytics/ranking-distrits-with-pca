#Importando bibliotecas
import pandas as pd
import numpy as np
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity   
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#Conectando dataset
dados = pd.read_csv('distritossp.csv')
print(dados)
print(dados.info())

#Tratando dados
dados_number = dados.drop(['cod_ibge', 'distritos'], axis=1)
corr = dados_number.corr()

sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns, cmap='YlGnBu')
plt.show()
print(corr)

#KMO
kmo_v, kmo_g = calculate_kmo(dados_number)
print('Var KMO:', kmo_v)
print('Global KMO:', kmo_g)

#Bartlett
qui_quadrado, p_value = calculate_bartlett_sphericity(dados_number)
print('Qui Quadrado:', qui_quadrado)
print('P-Value:', p_value) #Benchmark para taxa de insignificância--> 0.05
#P-Value é menor que 0.05, portanto deve ser considerada a Hipótese Alternativa


x = np.matrix(dados_number)
print(x)

s = np.cov(np.transpose(x))
print(s)
print('')

#Padronização da base de dados
number_columns = dados_number.columns
standard = StandardScaler()
dados_number = standard.fit_transform(dados_number)
df_dados_number = pd.DataFrame(dados_number, columns=number_columns)

#Construindo o PCA com todas as variavéis numéricas
n_fatores = df_dados_number.shape[1]
pca = PCA(n_components=n_fatores)
pca.fit(df_dados_number) #PCA feito, agora, Análise de fatores


components = pca.components_
print('COEFICIENTES DA COMBINAÇÃO LINEAR:')
print(components)
print('')


x = pca.components_[0,:]
x2 = x **2
soma = x2.sum()
print(soma)


#Entrega a porcentagem de variância explicada por cada um dos fatores gerados pela PCA
explaned_variance_ratio = pca.explained_variance_ratio_
print('Autovetores:',explaned_variance_ratio) 
print('')
#Definindo nome para cada um dos fatores
fatores = [f'F{i+1}' for i in range(n_fatores)]



fig = plt.figure(figsize= (10, 5))
plt.plot(explaned_variance_ratio, 'ro-', linewidth=3)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eingevalue (Autovalor)')
plt.show()

df_components = pd.DataFrame(components, columns=number_columns, index = [f'Autovetor {i+1}' for i in range(n_fatores)])
df_components


variancia_acumulada = [sum(explaned_variance_ratio[0:i+1]) for i in range(n_fatores)]
variancia_acumulada = np.round(variancia_acumulada, 2)
print('Variância Acumulada:', variancia_acumulada) #Autovalores acumulados


fig = plt.figure(figsize= (10, 5))
plt.plot(variancia_acumulada, 'ro-', linewidth=3)
plt.title('Porcentagem da variancia acumulada')
plt.xlabel('Principal Component')
plt.ylabel('Eingevalue (Autovalor) Acumulada')
plt.show()


#pca.explanded_variance_ --> Representa a quantidade de variáveis por cada um dos fatores.
#O valor mais exato pode ser explicado, multplicando: pca.explaned_variance_ratio pela quantidade de fatores.
autovalores = pca.explained_variance_ratio_ * n_fatores
print('proximação da Quantidade de Variáveis por cada um dos fatores:', autovalores)
#Se realizar a soma, o segundo método é oq mais se aproxima de 9.

fatores_selecionados = ['Fator selecionado' if autovalor > 1 else 'Fator não selecionado' for autovalor in autovalores]

fig = plt.figure(figsize= (10, 5))
plt.plot(autovalores, 'ro-', linewidth=3)
plt.title('Scree Plot - Autovalores multplicados por 9')
plt.xlabel('Componentes')
plt.ylabel('Autovalor')
plt.show()



raiz_autovalores = np.sqrt(autovalores)
print('Raiz Autovalores:',raiz_autovalores)

cargas_fatoriais = pd.DataFrame(components.T * raiz_autovalores, columns=fatores, index = number_columns)
print('Cargas Fatoriais:',cargas_fatoriais)

fig = plt.figure(figsize=(10,5))
plt.scatter(x=cargas_fatoriais['F1'], y=cargas_fatoriais['F2'])
plt.xlabel('F1')
plt.ylabel('F2')
plt.show()

#Pode-se observar que as variáveis que tem maior carga fatoriais e são relevantes para o rankeamento são as: Renda, Quota, Escolaridade e Idade.
#Todas elas apresenta

#Reduzindo a dimensionalidade para 4 variáveis
#Método Aula professora USP

pca = PCA(n_components=4)
pca.fit(df_dados_number)
components_principais = pca.components_
print(components_principais)


components_scores = []
for i in range(4):
  scores = pca.transform(df_dados_number)[:,i]
  components_scores.append(scores)

components_scores = pd.DataFrame(components_scores).T
print(components_scores)

dados['scoresCP1'] = components_scores[0]
dados['scoresCP2'] = components_scores[1]
dados['scoresCP3'] = components_scores[2]
dados['scoresCP4'] = components_scores[3]
#Scores --> indicam os valores de relação de uma variável com a componente principal em questão


dados['Ranking'] = dados['scoresCP1'] * explaned_variance_ratio[0] + dados['scoresCP2'] * explaned_variance_ratio[1]
print(dados)

filtro_scorecp1 = dados.sort_values(by='scoresCP1', ascending=False)
filtro_scorecp1 = filtro_scorecp1.drop(['scoresCP3', 'scoresCP4'], axis=1)
filtro_scorecp1

#Reduzindo dimensionalidade
#Método professor Alura


f1 = np.zeros(df_dados_number.shape[0])
for indice, variavel in enumerate(pca.feature_names_in_):
    f1 += pca.components_[0][indice]*df_dados_number[variavel]
print(f1)

scores = np.zeros(df_dados_number.shape)
for i in range(4):
    scores[i] = pca.components_[i]/raiz_autovalores[i]
print(scores)


f1 = np.zeros(df_dados_number.shape[0])
for indice, variavel in enumerate(pca.feature_names_in_):
    f1 += scores[0][indice]*df_dados_number[variavel]
print(f1)
f2 = np.zeros(df_dados_number.shape[0])
for indice, variavel in enumerate(pca.feature_names_in_):
    f2 += scores[1][indice]*df_dados_number[variavel]
print(f2)
f3 = np.zeros(df_dados_number.shape[0])
for indice, variavel in enumerate(pca.feature_names_in_):
    f3 += scores[2][indice]*df_dados_number[variavel]
print(f3)
f4 = np.zeros(df_dados_number.shape[0])
for indice, variavel in enumerate(pca.feature_names_in_):
    f4 += scores[3][indice]*df_dados_number[variavel]
print(f4)


dados['F1'] = f1
dados['F2'] = f2
dados['F3'] = f3
dados['F4'] = f4

colunas = ['scoresCP1', 'scoresCP2', 'scoresCP3', 'scoresCP4', 'F1', 'F2', 'F3', 'F4']
dataframe_scores = dados[colunas]
dataframe_scores

dados['Rankings'] = dados['F1'] * explaned_variance_ratio[0] + dados['F2'] * explaned_variance_ratio[1]
dados

filtro_f1_f2 = dados.sort_values(by='Rankings', ascending=False)
colunas = ['scoresCP1', 'scoresCP2', 'scoresCP3', 'scoresCP4', 'Ranking', 'F3', 'F4']
filtro_f1_f2 = filtro_f1_f2.drop(colunas, axis=1)
print(filtro_f1_f2)


print(filtro_scorecp1)




