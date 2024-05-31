![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/20feb7d8-1693-4156-ab3d-2074503a9b58)![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/041b4939-7280-48c7-af32-0985aab4daae)![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/3d5183e2-cbc9-4ad1-af08-03c273e87ebd)![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/0c32be5d-eca0-4387-866a-83be959d8467)![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/18c76250-4055-4c49-9c39-8bbae5271fa4)![image](https://github.com/patrickbarroso/bmtablemodels/assets/37444862/14e665c3-d3f7-47e9-b3ec-2c70bb4b42ff)# BMTABLEMODELS

# Definição

BMTABLEMODELS é um software direcionado para avaliar a qualidade e precisão de modelos deep learning pré-treinados. Tais modelos são responsáveis pela execução de 3 tarefas: (i) detecção de tabelas (TR – Table Recognition); (ii) detecção da estrutura das tabelas (TSR – Table Structure Recognition) e; (iii) extração do conteúdo das células das tabelas (TE – Table Extraction). A relação de datasets e modelos pré-treinados avaliados constam abaixo: 

# Avaliação dos Modelos

Para avaliar os modelos pré-treinados mencionados, o software utilizou o método de predição (ou avaliação) de cada modelo, que recebem como parâmetro de entrada uma imagem de uma pagina PDF (contendo uma ou mais tabelas), e como resultado as seguintes informações:

•	<u>Para operações de TR</u>: as coordenadas em pixels do entorno da tabela no formato [x1, y1, x2, y2]. Sendo x1, y1 o canto superior esquerdo da tabela e o x2,y2 o canto inferior direito da tabela. <br>
•	Para operações de TSR: as coordenadas em pixels de cada célula (bounding box ou bbox) no formato [x1, y1, x2, y2]. Sendo x1, y1 o canto superior esquerdo da célula e o x2,y2 o canto inferior direito da célula. <br>
•	Para operações de TE: uma estrutura de dados com o conteúdo textual contido em cada célula , que pode ser um conjunto/lista de dataframes ou um conjunto/lista de listas. <br>

As imagens utilizadas na avaliação dos modelos foram oriundas de certificados de calibração (em torno de 400 arquivos no formato PDF), que totalizou em torno de 700 imagens de tabelas. Os certificados não foram disponibilizados neste repositório devido a acordos de confidencialidade firmados com os laboratórios.

| Dataset            | Modelo                                         | Operações                                               | Autor                                                       |   Ano       | Qtd Tabelas |  Qtd Imagens | 
|--------------------|------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------------|-------------|-------------|--------------|
| COCO 2017          | DEtection TRansformer                          |  TR  | [Carion, 2020](https://arxiv.org/abs/2108.07732)             |  2020        |      118 k |       118k   |     
| Marmot             | Tablenet                          |  TR, TSR, TE    | [Paliwal, 2020](https://arxiv.org/pdf/2001.01469.pdf)             |  2020        |      2000 |     2000   |          
| [PubTables-1M](https://huggingface.co/datasets/bsmock/pubtables-1m)  | Table Transformer (TATR)    |  TR, TSR, TE  | [Smock, 2022](https://ieeexplore.ieee.org/document/9879666)             |  2022        |      1 M+ |       1 M+    |          
| [Roboflow](https://universe.roboflow.com/mohamed-traore-2ekkp/table-extraction-pdf/dataset/2?ref=roboflow2huggingface) |  TR   | [Zhang, 2022](https://link.springer.com/article/10.1007/s10032-022-00400-z) |  2020    |      118 k |       118k   | 
