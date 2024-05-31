# BMTABLEMODELS

## Definição

BMTABLEMODELS é um software direcionado para avaliar a qualidade e precisão de modelos deep learning pré-treinados. Tais modelos são responsáveis pela execução de 3 tarefas: (i) detecção de tabelas (TR – Table Recognition); (ii) detecção da estrutura das tabelas (TSR – Table Structure Recognition) e; (iii) extração do conteúdo das células das tabelas (TE – Table Extraction). A relação de datasets e modelos pré-treinados avaliados constam abaixo: 

| Dataset                                                                                                                 | Modelo                      | Operações      | Autor                         |   Ano   | Qtd Tabelas |  Qtd Imagens | 
|--------------------|------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------|-------------|-------------|--------------|
| [COCO 2017](https://cocodataset.org/#download)                       | DEtection TRansformer (DETR) |  TR             | [Carion, 2020](https://arxiv.org/abs/2108.07732)              |  2020        |      118 k |       118k   |     
| [Marmot](https://www.icst.pku.edu.cn/szwdclyjs/sjzy/index.htm)       | Tablenet                     |  TR, TSR, TE    | [Paliwal, 2020](https://arxiv.org/pdf/2001.01469.pdf)         |  2020        |      2000 |     2000   |      
| [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m)  | Table Transformer (TATR)     |  TR, TSR, TE    | [Smock, 2022](https://ieeexplore.ieee.org/document/9879666)   |  2022        |      1 M+ |       1 M+    |          
| [Roboflow](https://universe.roboflow.com/mohamed-traore-2ekkp/table-extraction-pdf/dataset/2?ref=roboflow2huggingface) | YOLO |  TR   | [Zhang, 2022](https://link.springer.com/article/10.1007/s10032-022-00400-z) |  2022  |   238 |   238   | 

## Avaliação dos Modelos

Para avaliar os modelos pré-treinados mencionados, o software utilizou o método de predição (ou avaliação) de cada modelo, que recebem como parâmetro de entrada uma imagem de uma pagina PDF (contendo uma ou mais tabelas), e como resultado as seguintes informações:

•	<u>Para operações de TR</u>: as coordenadas em pixels do entorno da tabela no formato [x1, y1, x2, y2]. Sendo x1, y1 o canto superior esquerdo da tabela e o x2,y2 o canto inferior direito da tabela. <br>
•	Para operações de TSR: as coordenadas em pixels de cada célula (bounding box ou bbox) no formato [x1, y1, x2, y2]. Sendo x1, y1 o canto superior esquerdo da célula e o x2,y2 o canto inferior direito da célula. <br>
•	Para operações de TE: uma estrutura de dados com o conteúdo textual contido em cada célula , que pode ser um conjunto/lista de dataframes ou um conjunto/lista de listas. <br>

As imagens utilizadas na avaliação dos modelos foram oriundas de certificados de calibração (em torno de 400 arquivos no formato PDF), que totalizou em torno de 700 imagens de tabelas. Os certificados não foram disponibilizados neste repositório devido a acordos de confidencialidade firmados com os laboratórios.

## Caminho das pastas de configuração

| Variável                | Caminho                                          | Definição                                                                                    | 
|-------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------------|
|   CERT_PATH             | "/MyDrive/DataSets/Certificados/In/"             | Caminho de entrada das pastas dos certificados dos laboratórios                              |    
|   DIROUT_<MODEL NAME>   | "/MyDrive/DataSets/Certificados/Out/<MODEL-NAME>"| Caminho de saída para geração das pastas por laboratório e respectivos arquivos de metadados |  
|   CERT_PATH             | "/MyDrive/DataSets/Certificados/In/"             | Diretório dos arquivos do GT - Ground Truth (referência para comparação das tabelas)         | 

## Arquivos Python (.ipynb)

Cada arquivo .ipynb disponibilizado na pasta /evaluation é referente a um modelo e todos seguem a estrutura:

1.	Funções Pré-processamento para MAIN: instalação e importação de bibliotecas e definição das funções de suporte para execução da função principal MAIN responsável por gerar os arquivos de anotação (.INFO, .METADADOS, .BBOX, .HTML) <br>
2.	Código MAIN para Rodar em Lote: função principal que varre os arquivos da pasta de entrada dos documentos PDF (CERT_PATH) para executar as operações de TR, TSR e TE e refletir nos arquivos de anotação. <br>
3.	TEDS – Cálculo Para O Modelo: instalação e importação de bibliotecas e funções para cálculo do TED e demais estatísticas necessárias. <br>
4.	Função MAIN para gerar estatísticas: geração dos arquivos dos arquivos de estatísticas de cada tabela .STATS e a geração do relatório de sumário  <br>





