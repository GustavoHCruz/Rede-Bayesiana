# Adicionar as bibliotecas necessárias
# !pip install pandas
# !pip install pybbn

# Importando tudo que será usado
# Manipulação dos dados CSV
import pandas as pd 

# Para criar a rede neural
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController

# Pré processamento dos dados
# Importando a base de dados que será usada
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Discretizando colunas
df['age'] = df['age'].apply(lambda x: 'young' if x <= 18 else 'adult' if 18 < x <= 60 else 'aged')
df['avg_glucose_level'] = df['avg_glucose_level'].apply(lambda x: 'normal' if x < 162 else 'high')
df['bmi'] = df['bmi'].apply(lambda x: 'undefined' if type(x) == str else 'ok' if x < 30 else 'high')
df['smoking_status'] = df['smoking_status'].apply(lambda x: 'yes' if x == 'formerly smoked' or x == 'smokes' else 'no')

# Visualizar o *dataframe* do Pandas
print(df)

# Processando e gerando dados dos nós individuais
total = len(df)

genero = [(df['gender'] == 'Male').sum() / total, (df['gender'] == 'Female').sum() / total, (df['gender'] == 'Other').sum() / total]

idade = [(df['age'] == 'young').sum() / total, (df['age'] == 'adult').sum() / total, (df['age'] == 'aged').sum() / total]

imc = [(df['bmi'] == 'ok').sum() / total, (df['bmi'] == 'high').sum() / total, (df['bmi'] == 'undefined').sum() / total]

residencia = [(df['Residence_type'] == 'Urban').sum() / total, (df['Residence_type'] == 'Rural').sum() / total]

# Definição dos nós sem país
Genero = BbnNode(Variable(0, 'Genero', ['Homem', 'Mulher', 'Outro']), genero)
Idade = BbnNode(Variable(1, 'Idade', ['Jovem', 'Adulto', 'Idoso']), idade)
IMC = BbnNode(Variable(2, 'IMC', ['Ok', 'Alto', 'Indefinido']), imc)
Residencia = BbnNode(Variable(3, 'Tipo Residência', ['Urbana', 'Rural']), residencia)

# Processando nós com dependência
fumante_tabela = [df[(df['smoking_status'] == 'yes') & (df['gender'] == 'Male')]['smoking_status'].count() / total,
                  df[(df['smoking_status'] == 'no') & (df['gender'] == 'Male')]['smoking_status'].count() / total,
                  df[(df['smoking_status'] == 'yes') & (df['gender'] == 'Female')]['smoking_status'].count() / total,
                  df[(df['smoking_status'] == 'no') & (df['gender'] == 'Female')]['smoking_status'].count() / total,
                  df[(df['smoking_status'] == 'yes') & (df['gender'] == 'Other')]['smoking_status'].count() / total,
                  df[(df['smoking_status'] == 'no') & (df['gender'] == 'Other')]['smoking_status'].count() / total]

Fumante = BbnNode(Variable(4, 'Fumante', ['Sim', 'Não']), fumante_tabela)

glicose_tabela = [df[(df['avg_glucose_level'] == 'normal') & (df['gender'] == 'Male')]['avg_glucose_level'].count() / total,
                  df[(df['avg_glucose_level'] == 'high') & (df['gender'] == 'Male')]['avg_glucose_level'].count() / total,
                  df[(df['avg_glucose_level'] == 'normal') & (df['gender'] == 'Female')]['avg_glucose_level'].count() / total,
                  df[(df['avg_glucose_level'] == 'high') & (df['gender'] == 'Female')]['avg_glucose_level'].count() / total,
                  df[(df['avg_glucose_level'] == 'normal') & (df['gender'] == 'Other')]['avg_glucose_level'].count() / total,
                  df[(df['avg_glucose_level'] == 'high') & (df['gender'] == 'Other')]['avg_glucose_level'].count() / total]

Glicose = BbnNode(Variable(5, 'Glicose', ['Normal', 'Alta']), glicose_tabela)

hipertensao_tabela = [df[(df['hypertension'] == 1) & (df['smoking_status'] == 'yes') & (df['avg_glucose_level'] == 'normal')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 0) & (df['smoking_status'] == 'yes') & (df['avg_glucose_level'] == 'normal')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 1) & (df['smoking_status'] == 'yes') & (df['avg_glucose_level'] == 'high')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 0) & (df['smoking_status'] == 'yes') & (df['avg_glucose_level'] == 'high')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 1) & (df['smoking_status'] == 'no') & (df['avg_glucose_level'] == 'normal')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 0) & (df['smoking_status'] == 'no') & (df['avg_glucose_level'] == 'normal')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 1) & (df['smoking_status'] == 'no') & (df['avg_glucose_level'] == 'high')]['hypertension'].count() / total,
                      df[(df['hypertension'] == 0) & (df['smoking_status'] == 'no') & (df['avg_glucose_level'] == 'high')]['hypertension'].count() / total]

Hipertensao = BbnNode(Variable(6, 'Hipertensão', ['Sim', 'Não']), hipertensao_tabela)

casamento_tabela = [df[(df['ever_married'] == 'Yes') & (df['age'] == 'young')]['ever_married'].count() / total,
                    df[(df['ever_married'] == 'No') & (df['age'] == 'adult')]['ever_married'].count() / total,
                    df[(df['ever_married'] == 'Yes') & (df['age'] == 'aged')]['ever_married'].count() / total,
                    df[(df['ever_married'] == 'No') & (df['age'] == 'young')]['ever_married'].count() / total,
                    df[(df['ever_married'] == 'Yes') & (df['age'] == 'adult')]['ever_married'].count() / total,
                    df[(df['ever_married'] == 'No') & (df['age'] == 'aged')]['ever_married'].count() / total]

Casamento = BbnNode(Variable(7, 'Casou-se', ['Sim', 'Não']), casamento_tabela)

coracao_tabela = [df[(df['heart_disease'] == 1) & (df['age'] == 'young') & (df['ever_married'] == 'Yes')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 0) & (df['age'] == 'young') & (df['ever_married'] == 'Yes')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 1) & (df['age'] == 'adult') & (df['ever_married'] == 'Yes')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 0) & (df['age'] == 'adult') & (df['ever_married'] == 'Yes')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 1) & (df['age'] == 'aged') & (df['ever_married'] == 'Yes')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 0) & (df['age'] == 'aged') & (df['ever_married'] == 'Yes')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 1) & (df['age'] == 'young') & (df['ever_married'] == 'No')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 0) & (df['age'] == 'young') & (df['ever_married'] == 'No')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 1) & (df['age'] == 'adult') & (df['ever_married'] == 'No')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 0) & (df['age'] == 'adult') & (df['ever_married'] == 'No')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 1) & (df['age'] == 'aged') & (df['ever_married'] == 'No')]['heart_disease'].count() / total,
                  df[(df['heart_disease'] == 0) & (df['age'] == 'aged') & (df['ever_married'] == 'No')]['heart_disease'].count() / total]

Coracao = BbnNode(Variable(8, 'Doença no Coração', ['Sim', 'Não']), coracao_tabela)

derrame_tabela = [df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'ok')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'high')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Urban') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 1) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'Yes') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 1) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total,
                  df[(df['stroke'] == 0) & (df['ever_married'] == 'No') & (df['heart_disease'] == 0) & (df['Residence_type'] == 'Rural') & (df['bmi'] == 'undefined')]['stroke'].count() / total]

Derrame = BbnNode(Variable(9, 'Derrame', ['Sim', 'Não']), derrame_tabela)

# Construindo estrutura da rede
bbn = Bbn() \
      .add_node(Genero) \
      .add_node(Idade) \
      .add_node(IMC) \
      .add_node(Residencia) \
      .add_node(Fumante) \
      .add_node(Glicose) \
      .add_node(Hipertensao) \
      .add_node(Casamento) \
      .add_node(Coracao) \
      .add_node(Derrame) \
      .add_edge(Edge(Genero, Fumante, EdgeType.DIRECTED)) \
      .add_edge(Edge(Genero, Glicose, EdgeType.DIRECTED)) \
      .add_edge(Edge(Fumante, Hipertensao, EdgeType.DIRECTED)) \
      .add_edge(Edge(Glicose, Hipertensao, EdgeType.DIRECTED)) \
      .add_edge(Edge(Idade, Casamento, EdgeType.DIRECTED)) \
      .add_edge(Edge(Idade, Coracao, EdgeType.DIRECTED)) \
      .add_edge(Edge(Hipertensao, Coracao, EdgeType.DIRECTED)) \
      .add_edge(Edge(Casamento, Derrame, EdgeType.DIRECTED)) \
      .add_edge(Edge(Coracao, Derrame, EdgeType.DIRECTED)) \
      .add_edge(Edge(IMC, Derrame, EdgeType.DIRECTED)) \
      .add_edge(Edge(Residencia, Derrame, EdgeType.DIRECTED))

# Relacionar e criar inferências
tree = InferenceController.apply(bbn)

# Funções auxiliares para impressão e adição de evidências
def imprimir_probabilidades():
  for node in tree.get_bbn_nodes():
    valores = tree.get_bbn_potential(node)
    print("Nó:", node)
    print("Valores:")
    print(valores)
    print('----------------------')

def imprimir_probabilidade_derrame():
  node = tree.get_bbn_node_by_name('Derrame')
  valores = tree.get_bbn_potential(node)
  print("Probabilidades do Derrame:")
  print(valores)

def add_evidencia(ev, node, cat, val):
    ev = EvidenceBuilder() \
    .with_node(tree.get_bbn_node_by_name(node)) \
    .with_evidence(cat, val) \
    .build()
    tree.set_observation(ev)

# Cenário 1
add_evidencia('ev1', 'Genero', 'Homem', 1.0)
add_evidencia('ev2', 'Idade', 'Idoso', 1.0)
add_evidencia('ev3', 'Doença no Coração', 'Sim', 1.0)
add_evidencia('ev4', 'Casou-se', 'Sim', 1.0)
add_evidencia('ev5', 'IMC', 'Alto', 1.0)
add_evidencia('ev6', 'Tipo Residência', 'Urbana', 1.0)

imprimir_probabilidades()

# Cenário 2
add_evidencia('ev1', 'Genero', 'Mulher', 1.0)
add_evidencia('ev2', 'Idade', 'Jovem', 1.0)
add_evidencia('ev3', 'Doença no Coração', 'Não', 1.0)
add_evidencia('ev4', 'IMC', 'Ok', 1.0)
add_evidencia('ev5', 'Tipo Residência', 'Rural', 1.0)

imprimir_probabilidades()

# Cenário 3
add_evidencia('ev1', 'Genero', 'Homem', 1.0)
add_evidencia('ev2', 'Idade', 'Adulto', 1.0)
add_evidencia('ev3', 'Tipo Residência', 'Urbana', 1.0)
add_evidencia('ev4', 'Casou-se', 'Não', 1.0)

imprimir_probabilidades()