import torch
from ultralyticsplus import YOLO  # Se estiver usando ultralytics YOLO
from PIL import Image
import cv2
import torchvision.transforms as transforms
from funcoes import *
import sys

pytesseract.pytesseract.tesseract_cmd = r'/home/bin/tesseract'
 
#path_modelo = "foduucom/table-detection-and-extraction" #original
path_modelo = "/home/YOLO_LORA/Model/checkpoints/YOLO_LORA.pt" #caminho YOLO ajustado com LORA
 
#caminho dos certificados para analise
CERT_PATH = "/home/Certificados/In/"
#caminho de saida para geração das anotações das tabelas
OUT_PATH = "/home/Certificados/Out/"
#diretorio dos arquivos do GT - Ground Truth (REFERENCIA para comparacao das tabelas)
GT_PATH = "/home/Out/GT/"
#diretorio de escrita de arquivos
DIROUT_YOLO = "/home/Certificados/Out/Files/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO(path_modelo) #versão ajustada
model = model.to(device)
print("Usando modelo :", model)

LAB_PATHS = ['LAB_01', 'LAB_02', 'LAB_03', 'LAB_04', 'LAB_05', 'LAB_06', 'LAB_07', 'LAB_08',
'LAB_09', 'LAB_10', 'LAB_11', 'LAB_12']

################################### FUNCAO PRINCIPAL - GERAR ARQUIVOS DE METADADOS #######################

# Definir o fuso horário de SP
fuso_horario_brasilia = pytz.timezone('America/Sao_Paulo')

for LAB_PATH in LAB_PATHS:

  lstDF = None
  df = None
  dfAux = None
  dfArq = listFiles(CERT_PATH, LAB_PATH) # carregar a lista de arquivos no DATAFRAME

  pathLab = DIROUT_YOLO + "/" + LAB_PATH + "/"

  if(os.path.exists(pathLab)):
    print('Removendo arquivos gerados anteriormente.., caminho:', pathLab)
    deleteFiles2(pathLab) #deletando os arquivos anteriores

  for filepath, pages in zip(dfArq["PATH"], dfArq["QTDPAGES"]):

    arrcurfile = filepath.split("/")
    curFile = arrcurfile[len(arrcurfile)-1]
    labName = arrcurfile[len(arrcurfile)-2]

    #inicializando variaveis GT
    listFilesGT = []
    listTablesInfoGT = []

    #coletando dados das tabelas GT para comparacao e gerar o ID da imagem correto
    GT_LAB_OUT = GT_PATH + labName + "/"
    print("GT_LAB_OUT", GT_LAB_OUT)

    #varrendo cada pagina do arquivo
    for i in range(int(pages)):

      page = i+1
      #verificando a quantidade de tabelas por pagina

      #verificando se a pasta do laboratorio existe, caso negativo, cria
      labDirOut = DIROUT_YOLO + "/" + labName + "/"
      if not os.path.exists(labDirOut):
        os.makedirs(labDirOut)

      #definindo variaveis para gravacao do arquivo de saida
      noExtension = curFile.replace(".pdf","")
      #arquivo de PDF de leitura
      path_pdf_in = filepath
      #caminho para arquivo BMP convertido
      path_bmp_noextension =  labDirOut + "/" + noExtension

      #nova pagina, carrega a lista de referencia (GT) da pagina em questao
      listFilesGT = getListFilesGTInfo(noExtension, page, GT_LAB_OUT)
      listTablesInfoGT = getListTablesInfo(GT_LAB_OUT,listFilesGT)
      qtdTabelasGT = len(listTablesInfoGT)

      print("Arquivo: [", path_pdf_in , "] / qtd de tabelas para ler da pagina[" + str(page) + "]: ", len(listTablesInfoGT))

      print("listTablesInfoGT =", listTablesInfoGT)
      #sys.exit()
      #temos tabelas para processar....
      if(qtdTabelasGT >0):

        ####1 - converter pdf para png  ####
        #convert_pdf_to_bmp(path_pdf_in, path_bmp_noextension, page)

        output_png = path_bmp_noextension + "_" + str(page) + ".png"

        pdf_page_to_png(path_pdf_in, page, output_png)

        # perform inference (predição)
        results = model.predict(output_png)
        boxes = results[0].boxes

        #carregar os dataframes pra uma lista de dataframes lstDF
        # Iterar sobre as bounding boxes e imprimir as coordenadas
        lstDF = []
        lstDFRaw = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]  # Extrair coordenadas da bounding box
            #print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            img = np.array(Image.open(output_png))

            #x1, y1, x2, y2 = 999999
            #x1, y1, x2, y2, _, _ = tuple(int(item) for item in box.data.numpy()[0]) #primeiro resultado
            x1, y1, x2, y2, _, _ = tuple(int(item) for item in box.cpu().data.numpy()[0])

            cropped_image = img[y1:y2, x1:x2]
            cropped_image = Image.fromarray(cropped_image)

            table_content = pytesseract.image_to_string(cropped_image)

            #coletar o conteúdo de uma imagem para string
            table_content = pytesseract.image_to_string(cropped_image)

            # Dividir o texto em linhas
            lines = table_content.split('\n')

            # Dividir cada linha em palavras
            data = [line.split() for line in lines if line.strip() != '']

            # Criar um DataFrame
            df = pd.DataFrame(data)

            dfAux = copy.deepcopy(df)
            lstDFRaw.append(dfAux) #dados nao normalizados

            #normalizar dataFrame
            df = normalizeDataframe2(df)

            lstDF.append(df)

        #print("lstDF = ",lstDF)
        #break

        qtdlinhasLstDF = 0
        qtdColunasLstDF = 0
        if len(lstDF) > 0:
          qtdlinhasLstDF = len(lstDF)
          qtdColunasLstDF = len(lstDF[0].columns)

        qtdlinhasGT = 0
        qtdColunasGT = 0
        if len(listTablesInfoGT) > 0:
          qtdlinhasGT = len(listTablesInfoGT)
          qtdColunasGT = len(listTablesInfoGT[0])

        print("=============>Tabela pagina {0} caminho: {1} ".format(page,path_pdf_in))
        print("Dimensão lstDF {0} x {1} ".format(qtdlinhasLstDF, qtdColunasLstDF))
        print("Dimensão listTablesInfoGT {0} x {1} ".format(qtdlinhasGT, qtdColunasGT))

        i = 0
        print("len(lstDF) e len(listTablesInfoGT) com tamanho > 1 - CASO COMPLEXO ")
        #para cada tabela (dataframe) da lista de dataframes
        for df in lstDF:

          #print ("df original = ", df)
          qtdlinhasDF = 0
          qtdColunasDF = 0
          if len(lstDF) > 0:
            qtdlinhasDF = len(df)
            qtdColunasDF = len(df.columns)

          dfOriginal = copy.deepcopy(df) #guarda o DF original sem correcoes para gravar no HTML (calculo do TED)
          #verificando possivel ajuste de dimensao do dataframe com GT para comparação

          qtdLinhasGT = 0
          qtdColsGT = 0
          if listTablesInfoGT is not None and len(listTablesInfoGT) > 0:

            #ajustar o tamanho do dataframe, caso necessário
            qtdLinhasGT = int(listTablesInfoGT[0]["DIMENSION"].split("X")[0])
            qtdColsGT = int(listTablesInfoGT[0]["DIMENSION"].split("X")[1])
            if (qtdColunasDF != qtdColsGT or qtdlinhasDF != qtdLinhasGT):
              print("Ajustando dataframe para dimensão, dimensão atual DF = ", qtdlinhasDF,"x",qtdColunasDF)
              print("Ajustando dataframe para dimensão do GT, nova dimensão = ", qtdLinhasGT,"x",qtdColsGT)
              df = ajustDataframe(df, qtdLinhasGT, qtdColsGT)

          #carregando o objeto dicionário INFO da tabela de análise
          bbox = [x1, y1, x2, y2]
          lstHead = [] if len(df) == 0 else df.iloc[0].tolist()
          lstFirst = [] if len(df) == 0 or len(df) == 1 else df.iloc[1].tolist()
          print("lstHead = ", lstHead)
          print("lstFirst = ", lstFirst)
          dicTable = getDicTableInfo(labName, noExtension, page, qtdLinhasGT, qtdColsGT, bbox, lstHead, lstFirst)
          print("dicTable FIRST_LINE 1", dicTable["FIRST_LINE"])

          #verificando qual tabela de maior similaridade
          #apenas uma tabela GT existente na pagina (df recebe o TABLEID existente)
          if len(listTablesInfoGT) ==1:
            print("Apenas uma tabela GT na página, logo, TABLEID a ser utilizado para dicTable = ",listTablesInfoGT[0]["TABLEID"])
            dicTable["TABLEID"] = listTablesInfoGT[0]["TABLEID"]
          else:
            print("listTablesInfoGT > 1, verificar similaridade...")
            dicTable["TABLEID"] = getMaiorSimilaridade(dicTable, listTablesInfoGT)

          listFileGT = getGTInfo(dicTable["TABLEID"], noExtension, page, GT_LAB_OUT)
          listTableInfoGT = getListTablesInfo(GT_LAB_OUT,listFileGT)
          print("TABLEID de maior similaridade = ", dicTable["TABLEID"])

          print("dicTable HEAD", dicTable["HEAD"])
          print("dicTable FIRST_LINE 2", dicTable["FIRST_LINE"])

          #gravar o arquivo INFO da tabela para comparacao com GT
          #encontrou tabela identica ou similar
          if dicTable["TABLEID"] != "TBD" and dicTable["TABLEID"] != "":
            filePath = labDirOut + dicTable["TABLEID"] + "|" + noExtension + "|" + str(page) + "_INFO.info"
            print("gerando arquivo INFO de resultado - SUCESSO: ", filePath)
            SaveDicTableInfo(filePath, dicTable)
          else:
            #no caso de nao ter encontrado tabela similar ao GT para comparação, registrar arquivo de erro
            filePath = labDirOut + noExtension + "|" + str(page) + "|" + "_INFO_ERRO.error"
            dicTable["OBS"] = "ERRO - TABLEID NÃO ENCONTRADO"
            print("gerando arquivo INFO de resultado - ERRO: ", filePath)
            SaveDicTableInfo(filePath, dicTable)

          #modelo do dicmetada
          dicMetaData = {
          "filename": 0,
          "split": "train",
          "imgid": "",
          "html": {
            "cells": 0,
            "structure": 0
                  }
          }

          #inicializando a lista
          dicMetaData["filename"] = path_pdf_in
          dicMetaData["imgid"] = dicTable["TABLEID"]

          #gravando no arquivo as informacoes de METADADOS, BBOX E HTML
          lstCells = []
          lstStructure = []

          #print("df = ", df)
          #print("df shape ", df.shape)
          #print("df.at[0,0] ", df.at[0,0])
          #carrega e concatena a lista de tokens das celulas e bbox calculando como referencia as coordenadas da tabela principal
          lstCells.extend(noteListTokensBbox(None, df, None)) # no futuro alterar para inserir bbox
          #lstStructure.extend(noteTokensHTML(dfOriginal)) #carrega e concatena a lista de tokens html (df original sem ajustes)
          lstStructure.extend(noteTokensHTML(lstDFRaw[i]))
          dicMetaData["html"]["cells"] = lstCells
          dicMetaData["html"]["structure"] = lstStructure

          print("Gravando Metadados: ", DIROUT_YOLO)
          saveAnnotationFile(dicMetaData, DIROUT_YOLO , page)
          saveElementMetadata(dicMetaData, "BBOX", DIROUT_YOLO, page)
          saveElementMetadata(dicMetaData, "HTML", DIROUT_YOLO, page)
          saveElementMetadata(dicMetaData, "HTML_PRETTY", DIROUT_YOLO, page)
          i = i + 1

          #break #end for lstDF

        #end if temos paginas para processar

      #break #end of pages

    #break # end for files

#removendo arquivos BMP
print("Removendo arquivos png temporários...")
deleteFiles(labDirOut, "png")

# Obter a data e hora corrente
data_e_hora_corrente = datetime.now(fuso_horario_brasilia)
# Formatar a data e hora corrente para o formato desejado
data_e_hora_formatadas = data_e_hora_corrente.strftime("%d/%m/%Y %H:%M:%S")
#MAXFILES = 1 #apenas para testes, delimitar a quantidade de certificados a ler
print('FIM DO PROCESSAMENTO ', data_e_hora_formatadas)