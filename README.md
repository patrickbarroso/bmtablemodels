# BMTABLEMODELS
https://arxiv.org/abs/2410.17725
## Definição

BMTABLEMODELS é um software direcionado para avaliar a qualidade e precisão de modelos deep learning pré-treinados. Tais modelos são responsáveis pela execução de 3 tarefas: (i) detecção de tabelas (TR – Table Recognition); (ii) detecção da estrutura das tabelas (TSR – Table Structure Recognition) e; (iii) extração do conteúdo das células das tabelas (TE – Table Extraction). A relação de datasets e modelos pré-treinados avaliados constam abaixo: 

| Dataset                                                                                                                 | Modelo                      | Operações      | Autor                         |   Ano   | Qtd Tabelas |  Qtd Imagens | 
|--------------------|------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------|-------------|-------------|--------------|
| [COCO](https://cocodataset.org/#download)                       | DEtection TRansformer (DETR) |  TR             | [Carion, 2020](https://arxiv.org/abs/2108.07732)              |  2020        |      118 k |       118k   |     
| [Marmot](https://www.icst.pku.edu.cn/szwdclyjs/sjzy/index.htm)       | Tablenet                     |  TR, TSR, TE    | [Paliwal, 2020](https://arxiv.org/pdf/2001.01469.pdf)         |  2020        |      2000 |     2000   |      
| [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m)  | Table Transformer (TATR)     |  TR, TSR, TE    | [Smock, 2022](https://ieeexplore.ieee.org/document/9879666)   |  2022        |      1 M+ |       1 M+    |          
| [COCO](https://cocodataset.org/#download)   | YOLOv11 |  TR   | [Khanam, 2024](https://arxiv.org/abs/2410.17725) |  2024  |   118 k  |   118 k    | 

## Avaliação dos Modelos

Para avaliar os modelos pré-treinados mencionados, o software utilizou o método de predição (ou avaliação) de cada modelo, que recebem como parâmetro de entrada uma imagem de uma pagina PDF (contendo uma ou mais tabelas), e como resultado as seguintes informações:

•	<u>Para operações de TR</u>: as coordenadas em pixels do entorno da tabela no formato [x1, y1, x2, y2]. Sendo x1, y1 o canto superior esquerdo da tabela e o x2,y2 o canto inferior direito da tabela. <br>
•	Para operações de TSR: as coordenadas em pixels de cada célula (bounding box ou bbox) no formato [x1, y1, x2, y2]. Sendo x1, y1 o canto superior esquerdo da célula e o x2,y2 o canto inferior direito da célula. <br>
•	Para operações de TE: uma estrutura de dados com o conteúdo textual contido em cada célula , que pode ser um conjunto/lista de dataframes ou um conjunto/lista de listas. <br>

As imagens utilizadas na avaliação dos modelos foram oriundas de certificados de calibração (em torno de 400 arquivos no formato PDF), que totalizou em torno de 1130 imagens de tabelas. Os certificados não foram disponibilizados neste repositório devido a acordos de confidencialidade firmados com os laboratórios.

## Caminho das pastas de configuração

| Variável                | Caminho                                          | Definição                                                                                    | 
|-------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------------|
|   CERT_PATH             | "/MyDrive/DataSets/Certificados/In/"             | Caminho de entrada das pastas dos certificados dos laboratórios                              |    
|   DIR_OUT/MODEL         | "/MyDrive/DataSets/Certificados/Out/<MODEL-NAME>"| Caminho de saída para geração das pastas por laboratório e respectivos arquivos de metadados |  
|   GT_PATH             | "/MyDrive/DataSets/Certificados/GT/In/"             | Diretório dos arquivos do GT - Ground Truth (referência para comparação das tabelas)         | 

## Arquivos Python (.ipynb)

Cada arquivo .ipynb disponibilizado na pasta /Evaluation é referente a um modelo e todos seguem a estrutura:

1.	Funções Pré-processamento para MAIN: instalação e importação de bibliotecas e definição das funções de suporte para execução da função principal MAIN responsável por gerar os arquivos de anotação (.INFO, .METADADOS, .BBOX, .HTML) <br>
2.	Código MAIN para Rodar em Lote: função principal que varre os arquivos da pasta de entrada dos documentos PDF (CERT_PATH) para executar as operações de TR, TSR e TE e refletir nos arquivos de anotação. <br>
3.	TEDS – Cálculo Para O Modelo: instalação e importação de bibliotecas e funções para cálculo do TED e demais estatísticas necessárias. <br>
4.	Função MAIN para gerar estatísticas: geração dos arquivos dos arquivos de estatísticas de cada tabela .STATS e a geração do relatório de sumário  <br>

## Arquivos Python (.py)

Os arquivos em Python (Py_files) são versões mais recentes do código em relação aos arquivos .ipynb da pasta /Evaluation, caso queira rodar em ambiente IDE.

## Processo de anotação das tabelas

As informações de coordenadas da tabela, das células e o conteúdo das tabelas foram anotadas com os seguintes formatos:

<b>Arquivo .INFO</b> = Arquivo com informações resumidas da tabela extraída para análise. 

Exemplo: ABC1|001|3_INFO.INFO <br>
```
{'LAB':'ABC',
'FILE':'001',
'PAGE':3,
'TABLEID': 'ABC1',
'DIMENSION':'8X7',
'BBOX':[52, 816, 1252, 1136],
'HEAD':['Faixa Range ( V )', 'Ref ( V )', 'Instrumento UUT ( V )', 'Erro Error ( V )', 'Incerteza Uncertainty ( V )', 'νeff', 'k'],
'FIRST_LINE':['600 m', '570,000 m', '570,00 m', '0,000 m', '0,064 m', '∞', '2,00']
}
```

•	LAB = Descrição do laboratório <br>
•	FILE = Nome do arquivo <br>
•	PAGE = Número da página <br>
•	TABLEID = Identificação unívoca da tabela <br>
•	DIMENSION = Dimensão da tabela <br>
•	BBOX = Coordenadas do contorno externo da tabela (x1,y1,x2,y2) <br>
•	HEAD = Cabeçalho da tabela <br>
•	FIRST_LINE = Primeira linha da tabela <br>

<b>Arquivo .METADADOS</b> = Arquivo de metadados da tabela.

```
{
   'filename': str,  #nome do arquivo
   'split': str,
   'imgid': int,      #id da tabela
   'html': {
     'structure': {'tokens': [str]},  #estrutura html da tabela
     'cell': [
       {
         'tokens': [str],   #tokens, dados das tabelas       
         'bbox': [x0, y0, x1, y1]  # coordenadas de cada célula
       }
     ]
   }
}
```
<b>Arquivo .BBOX</b> = Arquivo das coordenadas de cada célula da tabela.

Exemplo:<br>
```
...
{'tokens': ['k'], 'bbox': [1032, 0, 1200, 96]}, 
{'tokens': ['6', '0', '0', ' ', 'm'], 'bbox': [0, 96, 192, 128]}, 
{'tokens': ['5', '7', '0', ',', '0', '0', ' ', 'm'], 'bbox': [192, 96, 360, 128]}, 
{'tokens': ['5', '7', '0', ',', '2', '0', ' ', 'm'], 'bbox': [360, 96, 528, 128]}, 
{'tokens': ['0', ',', '2', '0', ' ', 'm'], 'bbox': [528, 96, 696, 128]},
...
```

<br>Arquivo .HTML </b> = Arquivo da estrutura HTML da tabela (sem o conteúdo)

## Comparação das anotações com a referência ground truth (GT)

A avaliação das 3 tarefas utilizam uma base de referência (denominado ground truth, ou simplesmente GT) oriundo de um dataset de 400 certificados de calibração de 12 laboratórios, totalizando em torno de 700 tabelas de medições de grandezas elétricas (tensão, corrente e resistência) de multímetros (em sua grande maioria), termômetros e resistores.  

## Geração das estatísticas

Para cada tabela de análise, é gerado um arquivo .STATS para comparação entre a tabela de análise e a tabela de referência (GT). Os dados do arquivo são gerados no seguinte formato:

Exemplo:
```
{"LAB": "ABC",
 "FILE": "003",
 "PAGE": "4",
 "TABLEID": "ABC1",
 "DIMENSION": "7X9",
 "QTDCELLS": 63,
 "QTDACERTOSCELLS": 2,
 "PERCACERTOSCELLS": 0.03,
 "PERCACERTOSBBOXINFO": 0.2,
 "PERCACERTOSBBOX": 0.0,
 "QTDNAOLIDOSBBOX": 63,
 "PERCNAOLIDOSBBOX": 1.0,
 "QTDNAOLIDOSTOKEN": 49,
 "PERCNAOLIDOSTOKEN": 0.78,
 "TEDS": 0.149413
}
```
Sendo:

Dados administrativos 

LAB = Pasta do laboratório <br>
FILE = Nome do arquivo <br>
PAGE = Número da página <br>
TABLEID = Identificação unívoca da tabela de análise <br>

Estatísticas para TR – Table Recognition:

DIMENSION = Dimensão da tabela (quantidade de linhas x quantidade de colunas) <br>
QTDCELLS = Quantidade de células (resultado da multiplicação de quantidade de linhas por quantidade de colunas) <br>
PERCACERTOSBBOXINFO = Percentual de acerto das coordenadas da tabelas em relação ao GT. Para possibilitar este cálculo deve ser considerado calcular a similaridade de cada número das coordenadas com sua referência (GT) e depois calcular a média total. O cálculo de similaridade é feita através da métrica Intersection Over Union (IoU). A área de overlap mapeia a área entre a bbox prevista e a bbox do GT. A área de união abrange tanto a bbox prevista quanto a bbox da GT.  A IoU mede o grau de sobreposição entre as áreas previstas e reais, fornecendo uma pontuação simples e eficaz para o desempenho da localização. 

<i>Exemplo: se uma coordenada de uma célula de análise é [112, 1205, 1188, 1492] e a GT é [114, 1206, 1189, 1492], a similaridade de cada numero respectivamente seria: 33,33%,  50%, 50% e 100%, sendo o resultado final a média = 58,33%. </i>

Estatísticas para TSR – Table Structure Recognition 

QTDACERTOSCELLS = Quantidade de acertos do conteúdo da célula em relação ao GT (acerto significa similaridade de 100% entre as strings). <br>
PERCACERTOSCELLS = Percentual de acerto se baseando no cálculo <i> QTDACERTOSCELLS / QTDCELLS *100 </i> <br>
PERCACERTOSBBOX = Percentual de acerto das coordenadas da célula da tabela em relação ao GT. O calculo de similaridade é o mesmo informado em PERCACERTOSBBOXINFO. <br>
QTDNAOLIDOSBBOX = quantidade de coordenadas das células com dados nulos impossibilitando a comparação. <br>
PERCNAOLIDOSBBOX: percentual de coordenadas das células com dados nulos impossibilitando a comparação, se baseia no cálculo <i>QTDNAOLIDOSBBOX / QTDCELLS * 100</i> <br>
QTDNAOLIDOSTOKEN = quantidade de células com dados nulos impossibilitando a comparação.  <br>
PERCNAOLIDOSTOKEN = percentual de células com dados nulos impossibilitando a comparação, se baseia no cálculo <i>QTDNAOLIDOSTOKEN / QTDCELLS * 100</i> <br>
TEDS = mede a similaridade da estrutura em árvore html das tabelas. É necessário apresentar as tabelas como uma estrutura de árvore no formato HTML. onde Ta e Tb são a estrutura em árvore das tabelas nos formatos HTML. EditDist representa a distância de edição da árvore e |T| é o número de nós em T. <br>

![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/60c33585-eb0a-4804-83b7-99719a8077f6)

Após geração dos arquivos .STATS, o seu conteúdo é refletido na planilha <i>Summary.xlsx</i>.








 
















