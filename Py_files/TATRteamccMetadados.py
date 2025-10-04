import torch
from ultralyticsplus import YOLO  # Se estiver usando ultralytics YOLO
from PIL import Image
import cv2
import torchvision.transforms as transforms
from funcoes import *
import sys
from transformers import AutoImageProcessor, AutoModelForObjectDetection, TableTransformerForObjectDetection
from peft import PeftModel

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ("Using device ===> ", device)

FINE_TUNING = "LORA"

#caminho dos certificados para analise
CERT_PATH = "/home/Certificados/In/"
#caminho de saida para geração das anotações das tabelas
OUT_PATH = "/home/Certificados/Out/"
#diretorio dos arquivos do GT - Ground Truth (REFERENCIA para comparacao das tabelas)
GT_PATH = "/home/Certificados/Out/GT/"
#diretorio de escrita de arquivos
DIROUT_TATR = "/home/Out/TATR/" 
#PASTA DE IMAGENS DO GT
PASTA_IMAGENS_GT = "/home/GT_IMG"

############ MODELO 1 - DETEÇÃO DE TABELA ------- ##############

#MODELO DE DETECCAO DE TABELA
model_path = "microsoft/table-transformer-detection"

#modelo para detectar as tabelas
model = AutoModelForObjectDetection.from_pretrained(model_path, revision="no_timm")
model.to(device)

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

############ MODELO 2 - ESTRUTURA DE TABELA ------- ##############

# Carregar processador de imagem
processor_path = "microsoft/table-transformer-structure-recognition"
image_processor = AutoImageProcessor.from_pretrained(processor_path)

#MODELO DETECCAO DE ESTRUTURA (caminho modelo TATR a utilizar)
structure_model_path = "/home/checkpoint-210"

# Carregar modelo base
structure_model = TableTransformerForObjectDetection.from_pretrained(structure_model_path).to(device)

# Aplicar o LoRA no modelo base (se for aplica LORA)
#structure_model = PeftModel.from_pretrained(base_model, structure_model_path).to(device)
print ("structure_model type = ", type(structure_model))

structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

LAB_PATHS = ['LAB_01', 'LAB_02', 'LAB_03', 'LAB_04', 'LAB_05', 'LAB_06', 'LAB_07', 'LAB_08',
'LAB_09', 'LAB_10', 'LAB_11', 'LAB_12']

################################### FUNCAO PRINCIPAL - GERAR ARQUIVOS DE METADADOS #######################

cropped_table = {}

# Definir o fuso horário de SP
fuso_horario_brasilia = pytz.timezone('America/Sao_Paulo')

for LAB_PATH in LAB_PATHS:

  pathLab = DIROUT_TATR + "/" + LAB_PATH + "/"

  dfArq = listFiles(CERT_PATH, LAB_PATH) # carregar a lista de arquivos no DATAFRAME

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
    GT_LAB_OUT = GT_PATH + "/" + labName + "/"

    #varrendo cada pagina do arquivo
    for i in range(int(pages)):

      page = i+1
      #verificando a quantidade de tabelas por pagina

      #verificando se a pasta do laboratorio existe, caso negativo, cria
      labDirOut = DIROUT_TATR + "/" + labName + "/"
      if not os.path.exists(labDirOut):
        os.makedirs(labDirOut)

      #definindo variaveis para gravacao do arquivo de saida
      noExtension = curFile.replace(".pdf","")
      #arquivo de PDF de leitura
      #path_pdf_in = dfArq["PATH"][0]
      path_pdf_in = filepath
      #caminho para arquivo PNG convertido
      path_png_out =  labDirOut + labName + "|" + noExtension + "|" + str(page) + ".png"
      #caminho para arquivo PNG convertido e com melhoria de contraste e brilho
      path_png_out_plus =  labDirOut + labName + "|" + noExtension + "|" + str(page) + "_plus.png"

      ####1 - converter pdf para imagem  ####
      #pdf_page_to_png(path_pdf_in, page, path_png_out)
 
      ####2 - melhorar a qualidade melhorar a qualidade da imagem#####
      #img_plus = aumentar_qualidade_e_contraste(path_png_out, 1, 5.5)
      #cv2.imwrite(path_png_out_plus, img_plus)

      ####3 Carregar imagem para o modelo e tratar contraste
      #image = Image.open(path_png_out).convert("RGB")

      ####4 - PREPARAR IMAGEM PARA MODELO 1 (DETECÇÃO DE TABELA)
      #pixel_values = detection_transform(image).unsqueeze(0)
      #pixel_values = pixel_values.to(device)

      #model.to(device)
      #outputs = None
      #with torch.no_grad():
        #outputs = model(pixel_values)

      #print("model.config.id2label ", model.config.id2label)
      #print("pixel_values.shape ", pixel_values.shape)
      #print("outputs.logits.shape ", outputs.logits.shape)

      # update id2label to include "no object"
      #id2label = model.config.id2label
      #id2label[len(model.config.id2label)] = "no object"
      
      onlylab = labName.split("_")[2]
      #coletando coordenadas da tabela (bbox)
      #objects = outputs_to_objects(outputs, image.size, id2label)
      #objects = carregar_bboxes_info(GT_LAB_OUT, onlylab, noExtension, page)
      INFO_GT = carregar_bboxes_info_GT(GT_LAB_OUT, onlylab, noExtension, page)

      PASTA_LAB = CERT_PATH + "/" + labName
      objects = carregar_bboxes_info_analise(INFO_GT, PASTA_IMAGENS_GT, PASTA_LAB, detection_transform, model, device, onlylab)

      print("objects - ", objects)
      print("objects len ", len(objects))
      #sys.exit()
      
      tokens = []
      detection_class_thresholds = {
          "table": 0.5,
          "table rotated": 0.5,
          "no object": 10
      }
      crop_padding = 10

      #tabelas coletadas do modelo TSR
      #tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=0)
      #print("tables_crops 1", tables_crops)
      
      print("onlylab", onlylab)
      print("noExtension", noExtension)
      print("page", page)
      tables_crops = carregar_tables_crops_flex(PASTA_IMAGENS_GT, onlylab, noExtension, page)
      print("tables_crops ", tables_crops)
      print("tables_crops len", len(tables_crops))

      print("Arquivo: [", path_pdf_in , "] / qtd de tabelas para ler da pagina[" + str(page) + "]: ", len(tables_crops))

      #if (len(objects)> 0):
      #  sys.exit()

      # FOR PARA CADA TABLE DA IMAGEM COLETADA
      for j in range(len(tables_crops)):
        
        print("j = ", j)
        #nova pagina, carrega a lista de referencia (GT) da pagina em questao
        if j==0:
          print("Novo page scan, carregando listTablesInfoGT....")
          listFilesGT = getListFilesGTInfo(noExtension, page, GT_LAB_OUT)
          listTablesInfoGT = getListTablesInfo(GT_LAB_OUT,listFilesGT)

        pathTable =  labDirOut + labName + "|" + noExtension + "|" + str(page) + "|tab" + str(j+1)

        cropped_table = tables_crops[j]['image'].convert("RGB")
        #cropped_table.save(labDirOut + labName + f"_cropped_table_{noExtension}.jpg")

        ####5 - PREPARAR IMAGEM PARA MODELO 2 (ESTRUTURA DE TABELA)
        pixel_values2 = structure_transform(cropped_table).unsqueeze(0)
        pixel_values2 = pixel_values2.to(device)

        # forward pass
        outputs2 = None
        with torch.no_grad():
          outputs2 = structure_model(pixel_values2)

        #==== 5 - GRAVANDO AS INFORMAÇÕES PARA O ARQUIVO INFO ======
        # update id2label to include "no object"
        structure_id2label = structure_model.config.id2label
        structure_id2label[len(structure_id2label)] = "no object"

        cells = outputs_to_objects(outputs2, cropped_table.size, structure_id2label)
        cell_coordinates = get_cell_coordinates_by_row(cells)

        #print("cell_coordinates = ", cell_coordinates)

        qtdlinhas = 0
        qtdcolunas = 0
        if cell_coordinates is not None and len(cell_coordinates) >0:
          qtdlinhas = len(cell_coordinates)
          qtdcolunas = len(cell_coordinates[0]["cells"])

        print("=============>Tabela pagina {0} caminho: {1} ".format(page,pathTable))
        print("Dimensão cell_cordinates {0} x {1} ".format(qtdlinhas, qtdcolunas))

        #CARREGANDO DADOS DA TABELA LIDA DO MODELO
        data = apply_ocr(cell_coordinates, cropped_table)

        #CARREGANDO E CORRIGINDO OS DADOS (CASO NECESSÁRIO)
        lstData = []
        for row, row_data in data.items():
          lstData.append(row_data)

        #REMOVER CARACTERES ESPECIAIS QUE PODEM DAR PROBLEMA NOS DICIONÁRIOS AO GERAR NO ARQUIVO
        lstData = clean_list_of_lists(lstData)
        
        #print("cell_coordinates", cell_coordinates)
        #print("lstData", lstData)

        listRawData = lstData #dados crus sem tratamento

        qtdLinhasData = 0
        qtdColunasData = 0
        if(lstData is not None and len(lstData) >0):
          qtdLinhasData = len(lstData)
          qtdColunasData = len(lstData[0])

        print("Dimensão lstData {0} x {1} ".format(qtdLinhasData, qtdColunasData))

        #coletando o head e primeira linha da tabela para geração da tabela INFO
        l = 0
        lstHead = []
        lstFirst = []
        for row, row_data in data.items():
          if l==0:
            lstHead = list(row_data)
          elif l==1:
            lstFirst = list(row_data)
          elif l==2 and labName == "LAB_12_PRECISOTEC": #CASO ESPECIFICO PARA PRECISOTEC
            if numTimes(lstHead, "") > 2 or temRepeticoes(lstHead):
              print("Pulando 2 casas (PRECISOTEC).... ==> ",lstHead)
              lstHead = lstFirst
              lstFirst = list(row_data)
              break 
          elif l==2 and labName != "LAB_12_PRECISOTEC": #(CASO NORMAL)
            if numTimes(lstHead, "") > 2 or temRepeticoes(lstHead):
              print("Possivel ocorrencia de cabeçalho, pulando 1.... ==> ",lstHead)
              lstHead = copy.deepcopy(lstFirst)
              lstFirst = list(row_data)
            else:
              pass
          l += 1

         #REMOVER CARACTERES ESPECIAIS QUE PODEM DAR PROBLEMA NOS DICIONÁRIOS AO GERAR NO ARQUIVO
        lstHead = clean_list(lstHead)
        lstFirst = clean_list(lstFirst)

        #print("dicTable HEAD antes", dicTable["HEAD"])
        #print("dicTable FIRST_LINE antes", dicTable["FIRST_LINE"])

        #gerando o dicionario para comparacao com GT
        #esse bbox é da TABELA em si
        print("objects line 285 =>", objects)
        
        bbox = [0,0,0,0]  
        if len(objects) > 0:
          bbox = [round(objects[j]["bbox"][0]), round(objects[j]["bbox"][1]), round(objects[j]["bbox"][2]), round(objects[j]["bbox"][3])]
        dicTable = getDicTableInfo(labName, noExtension, page, qtdlinhas, qtdcolunas, bbox, lstHead, lstFirst)

        #sys.exit()

        #CASO 1 - VERIFICANDO SE TABELAS SAO IDENTICAS  (DE MESMA DIMENSAO)
        #comparando o dicionario dos dados da tabela com a lista da GT para verificar qual ID será gerado
        equalTables = False
        for dicTableGT in listTablesInfoGT:
          if checkDimensionINFO(dicTableGT, dicTable) and getPercSimTablesINFO(dicTableGT, dicTable , 75):
            dicTable["TABLEID"] = dicTableGT["TABLEID"]
            dicTable["DIMENSION"] = dicTableGT["DIMENSION"]
            dicTable["OBS"] = "CASO 1 - TABELAS IDENTICAS"
            print("CASO 1 - TABELAS IDENTICAS, tableID ", dicTableGT["TABLEID"])
            equalTables = True
            #remove da lista de dicionario o valor do tableid encontrado
            listTablesInfoGT[:] = [dic for dic in listTablesInfoGT if dic.get("TABLEID") != dicTable["TABLEID"]]
            break

        #CASO 2 - BUG DO INFINITO (TABELA TRUNCADA DEVIDO A NAO RECONHECER SIMBOLO DO INFINITO
        isBugInfinito = False
        posInf = -1
        if not equalTables:

          for dicTableGT in listTablesInfoGT:
            #print("CASO 2 - Comparacao, dicTableGT: ", dicTableGT)
            #print("CASO 2 - Comparacao, dicTable antes: ", dicTable)
 
            isBugInf, posInf, char = isBUGInfinito(dicTable, dicTableGT)
            if isBugInf:
              print("CASO 2 - BUG do infinto confirmado [", dicTableGT["TABLEID"], "] posInf ", posInf)
              dicTable["TABLEID"] = dicTableGT["TABLEID"]
              dicTable["DIMENSION"] = dicTableGT["DIMENSION"]
              dicTable["HEAD"].insert(posInf, "NAN")
              dicTable["FIRST_LINE"].insert(posInf, char)
              dicTable["OBS"] = "CASO 2 - TABELAS SIMILARES (BUG INFINITO)"
              #print("CASO 2 - Comparacao, dicTable depois: ", dicTable)
              isBugInfinito = True
              #remove da lista de dicionario o valor do tableid encontrado
              listTablesInfoGT[:] = [dic for dic in listTablesInfoGT if dic.get("TABLEID") != dicTable["TABLEID"]]
              break

        #CASO 3 - Coletar maior similaridade entre as tabelas da mesma página (de mesma dimensao)
        if not equalTables and not isBugInfinito:
          print("CASO 3 - Coletar maior similaridade entre as tabelas da mesma página")
          dicTable["TABLEID"] = getMaiorSimilaridade(dicTable, listTablesInfoGT)
          listFileGT = getGTInfo(dicTable["TABLEID"], noExtension, page, GT_LAB_OUT)
          listTableInfoGT = getListTablesInfo(GT_LAB_OUT,listFileGT)
          if(len(listTableInfoGT) > 0):
            print("entrou getListTablesInfo ", listTableInfoGT[0]["DIMENSION"])
            dicTable["DIMENSION"] = listTableInfoGT[0]["DIMENSION"]
          dicTable["OBS"] = "CASO 3 - TABELA DE MAIOR SIMILARIDADE"
          print("TABLEID de maior similaridade = ", dicTable["TABLEID"])
          #remove da lista de dicionario o valor do tableid encontrado
          listTablesInfoGT[:] = [dic for dic in listTablesInfoGT if dic.get("TABLEID") != dicTable["TABLEID"]]

        print("dicTable HEAD", dicTable["HEAD"])
        print("dicTable FIRST_LINE", dicTable["FIRST_LINE"])

        if dicTable["DIMENSION"] != 0:
          qtdLinhasGT = int(dicTable["DIMENSION"].split("X")[0])
          qtdColunasGT = int(dicTable["DIMENSION"].split("X")[1])
          print("Dimensão dicTableGT {0} x {1} ".format(qtdLinhasGT, qtdColunasGT))

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

        #==== 5 - GRAVANDO AS INFORMAÇÕES DAS TABELAS (METADADOS, BBOX E HTML) ======

        #encontrou o table ID para comparar com o GT
        if dicTable["TABLEID"] != "TBD" and dicTable["TABLEID"] != "":

          #AJUSTANDO ARRAYS (SE NECESSÁRIO) lsData e cell_coordinates
          if qtdLinhasData != qtdLinhasGT or qtdColunasData != qtdColunasGT: #corrigir lsData
            print("Ajustando lstData  - qtdLinhasData =", qtdLinhasData, " x qtdLinhasGT =", qtdLinhasGT)
            print("Ajustando lstData  - qtdColunasData =", qtdColunasData, " x qtdColunasGT =", qtdColunasGT)
            lstData = ajustList(lstData, qtdLinhasGT, qtdColunasGT)

          if qtdlinhas != qtdLinhasGT or qtdcolunas != qtdColunasGT: #corrigir cell_coordinates
            print("Ajustando cell_coordinates  - qtdlinhas =", qtdlinhas, " x qtdLinhasGT =", qtdLinhasGT)
            print("Ajustando cell_coordinates  - qtdcolunas =", qtdcolunas, " x qtdColunasGT =", qtdColunasGT)
            cell_coordinates = ajustCellCord(cell_coordinates, qtdLinhasGT, qtdColunasGT)

          #print("TABLEID válido, iniciando a coleta de dados (METADADOS E BBOX)...., TABLEID [",dicTable["TABLEID"],"]")
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

          qtdRowBbox = 0
          qtdColBbox = 0
          #dimensão do BBOX
          if cell_coordinates is not None and len(cell_coordinates) > 0:
            qtdRowBbox = len(cell_coordinates)
            qtdColBbox = len(cell_coordinates[0]["cells"])

          #dimensao dos dados
          qtdRowData = 0
          qtdColData = 0
          if lstData is not None and len(lstData) > 0:
            qtdRowData = len(lstData)
            qtdColData = len(lstData[0])

          print("TABLEID diferente de vazio (APOS TRATAMENTO DAS ESTRUTURAS)...")
          print("qtdRowBbox", qtdRowBbox, " x qtdColBbox", qtdColBbox)
          print("qtdRowData", qtdRowBbox, " x qtdColData", qtdColData)


          #se a dimensão do tokens da celula e do BBOX forem identicos, gravar no arquivo
          if (qtdRowBbox == qtdRowData and qtdColBbox == qtdColData):

            lstCells = []
            lstStructure = []

            #carrega e concatena a lista de tokens das celulas e bbox calculando como referencia as coordenadas da tabela principal
            lstCells.extend(noteListTokensBboxTATR(cell_coordinates, lstData, None))
            dfDados = pd.DataFrame(listRawData[0:])
            lstStructure.extend(noteTokensHTML(dfDados)) #carrega e concatena a lista de tokens html
            dicMetaData["html"]["cells"] = lstCells
            dicMetaData["html"]["structure"] = lstStructure

            print("Gravando Metadados: ", DIROUT_TATR)
            saveAnnotationFile(dicMetaData, DIROUT_TATR , page)
            saveElementMetadata(dicMetaData, "BBOX", DIROUT_TATR, page)
            saveElementMetadata(dicMetaData, "HTML", DIROUT_TATR, page)
            #saveElementMetadata(dicMetaData, "HTML_PRETTY", DIROUT_PUBTABLES, page)
            
            deleteFiles(labDirOut, "png")
            #sys.exit()
          else:
            print("Dimensão diferente entre as tabelas")

        #break #quebra de uma determinada tabela
      #break #quebra de uma determinada pagina
      #removendo arquivos BMP
      print("Removendo arquivos png temporários...")
      deleteFiles(labDirOut, "png") 
  
    #sys.exit() #finalizar para um determinado arquivo

# Obter a data e hora corrente
data_e_hora_corrente = datetime.now(fuso_horario_brasilia)
# Formatar a data e hora corrente para o formato desejado
data_e_hora_formatadas = data_e_hora_corrente.strftime("%d/%m/%Y %H:%M:%S")
#MAXFILES = 1 #apenas para testes, delimitar a quantidade de certificados a ler
print('FIM DO PROCESSAMENTO ', data_e_hora_formatadas)