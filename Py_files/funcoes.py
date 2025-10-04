from PIL import ImageDraw
from huggingface_hub import hf_hub_download
import fitz
import os
import pandas as pd
from pdf2image import convert_from_path
from torchvision import transforms
import torch
import numpy as np
import csv
from tqdm.auto import tqdm
import cv2
from bs4 import BeautifulSoup as bs
import torch
import numpy as np
from ultralyticsplus import YOLO, render_result
from PIL import Image
import requests
from pytesseract import Output
import pytesseract
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import pytz
import random
from datetime import datetime
from PIL import Image #esse problema de import da classe Image deve ser considerado para demais modelos
import easyocr
import re
import sys

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

#funcao para varrer os arquivos PDF dos laboratórios nas pastas e retorna um dataframe
#! o método está bem lento (deve verificar o motivo posteriormente)
def InfoPDF(pathPDF):

  lenPages = 0
  doc = fitz.open(pathPDF)

  lenPages = len(doc)
  for pagina in doc:
    isText = bool(pagina.get_text())
    break

  typeArq = ["TEXT" if isText else "IMAGE"]

  return typeArq, lenPages

def listFiles_OLD(CERT_PATH):

  dfArq = pd.DataFrame()
  dirs = [nome for nome in os.listdir(CERT_PATH) if os.path.isdir(os.path.join(CERT_PATH, nome))]

  i = 0
  for dir in dirs:

    #coletando dados do diretorio (id, laboratorio)
    lstDir = dir.split("_")

    if(len(lstDir)==3):

      for file in os.listdir(os.path.join(CERT_PATH, dir)):

        #é arquivo PDF
        if file.lower().endswith(".pdf"):

          pathFile = CERT_PATH + dir + "/" + file
          dfArq.at[i,"LAB"] = lstDir[2]
          dfArq.at[i,'PATH'] = pathFile

          typeArq, qtdPages = InfoPDF(pathFile)
          dfArq.at[i,'TYPE'] = typeArq
          dfArq.at[i,'QTDPAGES'] = qtdPages

          i = i + 1

  return dfArq

def listFiles(CERT_PATH, LAB_PATH):

  dfArq = pd.DataFrame()

  if LAB_PATH is None:
    dirs = [nome for nome in os.listdir(CERT_PATH) if os.path.isdir(os.path.join(CERT_PATH, nome))]
  else:
    dirs = [LAB_PATH]

  i = 0
  for dir in dirs:

    #coletando dados do diretorio (id, laboratorio)
    lstDir = dir.split("_")
    print("lstDir ", lstDir);

    if(len(lstDir)==3):

      for file in os.listdir(os.path.join(CERT_PATH, dir)):

        #é arquivo PDF
        if file.lower().endswith(".pdf"):

          pathFile = CERT_PATH + dir + "/" + file
          print("pathFile ", pathFile);
          dfArq.at[i,"LAB"] = lstDir[2]
          dfArq.at[i,'PATH'] = pathFile

          typeArq, qtdPages = InfoPDF(pathFile)
          dfArq.at[i,'TYPE'] = typeArq
          dfArq.at[i,'QTDPAGES'] = qtdPages

          i = i + 1

  return dfArq

#dfArq = listFiles(CERT_PATH)

def deleteFiles2(dirpath):
  # Obtém a lista de arquivos no diretório
  files = os.listdir(dirpath)

  # Itera sobre os arquivos e os remove
  for file in files:
    filepath = os.path.join(dirpath, file)
    if os.path.isfile(filepath):
      os.remove(filepath)

def is_image_by_extension(file_path):
  image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']  # Adicione outras extensões se necessário
  file_extension = file_path.lower().split('.')[-1]
  return file_extension in image_extensions

def is_image(file_path):
    try:
        # Tenta abrir o arquivo como uma imagem
        Image(file_path)
        return True
    except IOError:
        # Se não for possível abrir como uma imagem, retorna False
        return False

def pdf_page_to_png(pdf_path, page_number, output_path):
  # Convertendo a página do PDF para uma lista de imagens
  images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)

  # Salvando a imagem como PNG
  images[0].save(output_path, 'PNG')

class MaxResize(object):
  def __init__(self, max_size=800):
      self.max_size = max_size

  def __call__(self, image):
      width, height = image.size
      current_max_size = max(width, height)
      scale = self.max_size / current_max_size
      resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

      return resized_image

def get_cell_coordinates_by_row(table_data):
  # Extract rows and columns
  rows = [entry for entry in table_data if entry['label'] == 'table row']
  columns = [entry for entry in table_data if entry['label'] == 'table column']

  # Sort rows and columns by their Y and X coordinates, respectively
  rows.sort(key=lambda x: x['bbox'][1])
  columns.sort(key=lambda x: x['bbox'][0])

  # Function to find cell coordinates
  def find_cell_coordinates(row, column):
      cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
      return cell_bbox

  # Generate cell coordinates and count cells in each row
  cell_coordinates = []

  for row in rows:
      row_cells = []
      for column in columns:
          cell_bbox = find_cell_coordinates(row, column)
          row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

      # Sort cells in the row by X coordinate
      row_cells.sort(key=lambda x: x['column'][0])

      # Append row information to cell_coordinates
      cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

  # Sort rows from top to bottom
  cell_coordinates.sort(key=lambda x: x['row'][1])

  return cell_coordinates

def aumentar_qualidade_e_contraste(imagem_path, fator_contraste, fator_brilho):
  # Carregar a imagem
  imagem = cv2.imread(imagem_path)

  # Converter a imagem para o espaço de cores LAB (Luminância, Azul, Vermelho)
  lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

  # Separar os canais L, A, B
  l, a, b = cv2.split(lab)

  # Aplicar o aumento de contraste na imagem L (luminância)
  l = cv2.add(l, fator_brilho)
  l = cv2.multiply(l, fator_contraste)

  # Mesclar novamente os canais LAB
  lab = cv2.merge((l, a, b))

  # Converter a imagem de volta para o espaço de cores BGR
  imagem_contraste = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

  return imagem_contraste

#FUNÇÕES DE MANIPULACAO DE IMAGENS E ARQUIVOS

#ler dados da celula (se aplica apenas para o modelo TATR - que possui o objeto cropped_table)
def apply_ocr(cell_coordinates, cropped_table):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
      row_text = []
      for cell in row["cells"]:
        # crop cell out of image
        #(se aplica apenas para o modelo TATR - que possui o objeto cropped_table)
        cell_image = np.array(cropped_table.crop(cell["cell"]))
        # apply OCR
        result = reader.readtext(np.array(cell_image))
        if len(result) > 0:
          # print([x[1] for x in list(result)])
          text = " ".join([x[1] for x in result])
          row_text.append(text)

      if len(row_text) > max_num_columns:
          max_num_columns = len(row_text)

      data[idx] = row_text

    #print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
          row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

#similaridade de strings conhecido como "Distância de Levenshtein"
def calcPercSimStrings(str1, str2):

  #retirando quebra de linhas da string
  str1 = str1.replace("\n", " ")
  str2 = str2.replace("\n", " ")

  tamanho_str1 = len(str1)
  tamanho_str2 = len(str2)

  matriz = [[0] * (tamanho_str2 + 1) for _ in range(tamanho_str1 + 1)]

  for i in range(tamanho_str1 + 1):
    matriz[i][0] = i

  for j in range(tamanho_str2 + 1):
    matriz[0][j] = j

  for i in range(1, tamanho_str1 + 1):
    for j in range(1, tamanho_str2 + 1):
        if str1[i - 1] == str2[j - 1]:
            custo_substituicao = 0
        else:
            custo_substituicao = 1
        matriz[i][j] = min(matriz[i - 1][j] + 1,       # Deletar
                            matriz[i][j - 1] + 1,       # Inserir
                              matriz[i - 1][j - 1] + custo_substituicao)  # Substituir

  distancia = matriz[tamanho_str1][tamanho_str2]
  maximo_tamanho = max(tamanho_str1, tamanho_str2)

  similaridade = 0
  if maximo_tamanho > 0:
    similaridade = (maximo_tamanho - distancia) / maximo_tamanho
  #print (" similaridade entre {0} e {1}: {2}".format(str1, str2, similaridade * 100))
  return similaridade * 100

#funcao que compara o valor de duas listas e calcula a media do percentual de similaridade entre eles
#(para calcular o valor do bbox das tabelas e células das tabelas)

def calcPercSimValueLists(lista1, lista2):
  if len(lista1) != len(lista2):
      print("calcPercSimValueLists, listas de tamanhos diferentes, lista1=",lista1,"/ lista2 = ",lista2)
      raise ValueError("As listas devem ter o mesmo comprimento.")

  percSim = [ (1 / (1 + (abs(num1 - num2)))) * 100 for num1, num2 in zip(lista1, lista2)]


  return sum(percSim) / len (percSim)

#(para calcular o percentual de similaridade entre dois números
def calcPercSimValueNums(num1, num2):

  percSim = (1 / (1 + (abs(num1 - num2)))) * 100
  #print (" similaridade entre os numeros {0} e {1}: {2}".format(num1, num2, percSim))
  return percSim

#coletar os arquivos de acordo com premissas (prefixo e sufixo)
def getFilesByPrefix(path, prefix, sufix):
  lstFiles = []
  for fileName in os.listdir(path):
      if prefix in fileName and fileName.endswith(sufix):
          lstFiles.append(fileName)
  return lstFiles

#coletar no arquivo do GT dados das tabelas de uma determinada pagina
def getListFilesGTInfo(curfile, page, path):

  prefix = curfile + "|" + str(page)
  sufix = "_INFO.info"
  lstTables = getFilesByPrefix (path, prefix, sufix)

  return lstTables

def getGTInfo(tableID, curfile, page, path):

  prefix = tableID + "|" + curfile + "|" + str(page)
  sufix = "_INFO.info"
  lstTables = getFilesByPrefix (path, prefix, sufix)

  return lstTables

#coletar as informacoes da tabela do arquivo _INFO e retornar para uma lista
def getListTablesInfo(path, listFiles):

  listTablesInfo = []
  for fileName in listFiles:
    with open(path + fileName, 'r') as file:
      conteudo = file.read()
      listTablesInfo.append(eval(conteudo))

  return listTablesInfo

# verificar se a estrutura dos dois dicionários INFO são similares
def checkDimensionINFO(dicTableGT, dicTable):

  msgErro = "0 - SUCESSO"
  isIdentical = True

  #se o tamanho das chaves dos dicionários são diferentes
  if(dicTableGT.keys() != dicTable.keys()):
    print("Dicionários não possuem o mesmo indice")
    msgErro = "1 - Dicionários não possuem o mesmo indice"
    isIdentical = False

  #se não tiver a mesma dimensao já descarta
  if dicTableGT["DIMENSION"] != dicTable["DIMENSION"]:
    msgErro = "2 - Dicionários não possuem a mesma dimensão"
    isIdentical = False

  #se o tamanho das colunas HEAD e FIRST_LINE não batem
  if(len(dicTableGT["HEAD"]) != len(dicTable["HEAD"])):
    print("Dicionários não possuem o mesmo tamanho da chave HEAD")
    msgErro = "3 - Dicionários não possuem o mesmo tamanho da chave HEAD"
    isIdentical = False

  if(len(dicTableGT["FIRST_LINE"]) != len(dicTable["FIRST_LINE"])):
    msgErro = "4 - Dicionários não possuem o mesmo tamanho da chave FIRST_LINE"
    isIdentical = False

  #return msgErro, isSimilar
  return isIdentical

#calcular similaridade da string das células dos dicionários INFO de mesma dimensão
#(para determinar se dois dicionarios INFO sao similares)
def getPercSimTablesINFO(dicTableGT, dicTable , percTolerancia):

  i = 0
  for textTableGT in dicTableGT["HEAD"]:
    textTable = dicTable["HEAD"][i]
    if (calcPercSimStrings(textTableGT, textTable) < percTolerancia):
      print("Tolerancia entre string ", textTableGT, "e ", textTable,  " menor que ", percTolerancia)
      return False
    i+=1

  i = 0
  for textTableGT in dicTableGT["FIRST_LINE"]:
    textTable = dicTable["FIRST_LINE"][i]
    if (calcPercSimStrings(textTableGT, textTable) < percTolerancia):
      print("Tolerancia entre string ", textTableGT, "e ", textTable,  " menor que ", percTolerancia)
      return False
    i+=1

  return True


#verifica se a dimensão dos dicionários INFO possuem tamanhos parecidos (maximo 1 de diferença)
def isAlmostSimilarINFO(dicTableGT, dicTable):

  #devem possuir a mesma quantidade de linhas e quantidade semelhante de colunas (maximo no modulo 1)

  #colunas
  qtdColGT = len(dicTableGT["HEAD"])
  qtdColTab = len(dicTable["HEAD"])

  #linhas
  qtdLinhasGT = dicTableGT["DIMENSION"].split("X")[0]
  qtdLinhas = dicTable["DIMENSION"].split("X")[0]

  if abs(qtdColGT-qtdColTab) <=1 and qtdLinhasGT == qtdLinhas:
    return True

  return False

#verificar maior valor na lista
def maiorValor(lista):

  maiorValor = 0
  for item in lista:
    if item > maiorValor:
        maiorValor = item

  return maiorValor

#verificar possivel similaridade no cabeçalho dos dicionários INFO
def checkAVGSimilaritiesINFO(dicTableGT, dicTable, percTolerancia):

  qtdColGT = len(dicTableGT["HEAD"])
  qtdColTab = len(dicTable["HEAD"])

  arrSim = []
  arrSummary = []
  limit = 4

  #verificar similaridade nas 3 primeiras colunas
  for i in range(qtdColTab):

    for j in range(qtdColGT):
      strTab = dicTable["HEAD"][i]
      strGT = dicTableGT["HEAD"][j]
      arrSim.append(calcPercSimStrings(strTab, strGT))

    arrSummary.append(maiorValor(arrSim))
    arrSim = []

    if(i>= limit):
      break

  #print(arrSummary)
  avgPerc = sum(arrSummary) / len(arrSummary)
  return avgPerc >= percTolerancia


#verificar possivel similaridade no cabeçalho dos dicionários FIRST_LINE
def checkAVGSimilarities2INFO(dicTableGT, dicTable, percTolerancia):

  qtdColGT = len(dicTableGT["FIRST_LINE"])
  qtdColTab = len(dicTable["FIRST_LINE"])

  arrSim = []
  arrSummary = []
  limit = 3

  #verificar similaridade nas 3 primeiras colunas
  for i in range(qtdColTab):

    for j in range(qtdColGT):
      strTab = dicTable["FIRST_LINE"][i]
      strGT = dicTableGT["FIRST_LINE"][j]
      arrSim.append(calcPercSimStrings(strTab, strGT))

    arrSummary.append(maiorValor(arrSim))
    arrSim = []

    if(i>= limit):
      break

  #print(arrSummary)
  avgPerc = sum(arrSummary) / len(arrSummary)
  return avgPerc >= percTolerancia

def getDicTableInfo(labName, curfile, page, qtdlinhas, qtdcolunas, bbox, head, firstLine):
  dicTableInfo = {}

  dicTableInfo["LAB"] = labName
  dicTableInfo["FILE"] = curfile
  dicTableInfo["PAGE"] = page
  dicTableInfo["TABLEID"] = "TBD"
  dicTableInfo["DIMENSION"] = str(qtdlinhas) + "X" + str(qtdcolunas)
  dicTableInfo["BBOX"] = bbox
  dicTableInfo["HEAD"] = head
  dicTableInfo["FIRST_LINE"] = firstLine

  return dicTableInfo

def SaveDicTableInfo (filePath, dicTableInfo):

  strFile = "{"
  lenDic = len(dicTableInfo.items())
  #print(lenDic)
  i = 0
  for chave, valor in dicTableInfo.items():
    vir = "," if i < lenDic-1 else ""
    if type(valor) == str:
      strFile += "'" + str(chave)+ "':'" + str(valor) + "'" + vir + "\n"
    else:
      strFile += "'" + str(chave) + "':" + str(valor) + vir + "\n"
    i = i + 1

  strFile += "}"
  with open(filePath, 'w') as arquivo:
    arquivo.write(strFile)

#numero de ocorrencias de um numero em uma lista
def numTimes(list, num):

  cont = 0
  for valor in list:
    if valor == num:
      cont+=1

  return cont

def numDecimals(list):

  cont = 0
  for valor in list:
    if isDecimal(valor):
      cont+=1

  return cont

def numNotDecimals(list):

  cont = 0
  for valor in list:
    if not isDecimal(valor):
      cont+=1

  return cont

def isDecimal(valor):
  try:

    if valor == "NAN" or valor == "nan" or valor is None:
      return False
    else:
      valor = valor.replace(",",".")
      float(valor)
      return True
  except ValueError:
      return False

def noteTokensHTML(df):

  lsthead = []
  lsttd = []
  hasheader = False
  hasdata = False
  dict_tokens_html = {}
  lsthead

  #percorre o dataframe para construir a estrutura html de colunas
  for i in range(len(df)):
    primeiraColuna = True
    j = 0
    for column in df.columns:

        value = "NAN"
        #print("noteTokensHTML i, j ", i," ",j)
        if not df.empty and not pd.isna(df.at[i, column]):
          # Value exists, access it
          value = str(df.at[i, column])
          value = value.replace("'","")
          value = value.replace("\\","")

        #primeira linha sao os cabeçalhos
        if i == 0:
          hasheader = True
          lsthead.append("<td>")
          #lsthead.append(value) #apenas para teste, comentar depois
          lsthead.append("</td>")
        else:
          hasdata = True
          if(primeiraColuna):
            primeiraColuna = False
            #a partir da 3a linha fecha a linha anterior </tr>
            if(j==0 and i > 1):
              lsttd.append("</tr>")
            lsttd.append("<tr>")
          lsttd.append("<td>")
          #lsttd.append(value) #apenas para teste, comentar depois
          lsttd.append("</td>")

        primeiraColuna = False
        j = j + 1

  if(hasheader):
    lsthead.insert(0,"<thead>")
    lsthead.insert(1,"<tr>")
    lsthead.append("</tr>")
    lsthead.append("</thead>")

  if(hasdata):
    lsttd.insert(0,"<tbody>")
    lsttd.append("</tbody>")

  #se a estrutura tiver completa, adiciona no dicionario tokens
  if(hasheader and hasdata):
    lsthead.extend(lsttd)
  else:
    dict_tokens_html = {"tokens": "vazio"}
    lsthead.extend(dict_tokens_html)

  lsthead.insert(0,"<table>")
  lsthead.extend("</table>")
  return lsthead


def noteListTokensBboxTATR(cell_coordinates, lstData, lstTableRef):
  list_tokens_bbox = []

  #dimensao dos dados
  qtdRowData = len(lstData)
  qtdColData = len(lstData[0])
  print("dimensao Data {0} x {1} ".format(qtdRowData, qtdColData))

  #carregando BBOX de cada celula por linha para uma lista
  qtdlinhasBbox = len(cell_coordinates)
  qtdcolunasBbox = len(cell_coordinates[0]["cells"])
  print("dimensao Bbox {0} x {1} ".format(qtdlinhasBbox, qtdcolunasBbox))


  i = 0
  for row in cell_coordinates:
    j = 0
    for bbox in row["cells"]:
      x1 = round(bbox["cell"][0])
      y1 = round(bbox["cell"][1])
      x2 = round(bbox["cell"][2])
      y2 = round(bbox["cell"][3])

      #print(f"noteListTokensBboxTATR - bbox: {[x1, y1, x2, y2]}")

      if(lstTableRef is None):
        dict_tokens_bbox = {'tokens': list(lstData[i][j]), 'bbox': [x1, y1, x2, y2]}
      else:
        dict_tokens_bbox = {'tokens': list(lstData[i][j]), 'bbox': [x1 - lstTableRef[0], y1 - lstTableRef[1], x2 - lstTableRef[0], y2 - lstTableRef[1]]}
      list_tokens_bbox.append(dict_tokens_bbox)
      j+=1
    i+=1

  return list_tokens_bbox

def noteListTokensBbox(lstBbox, df, lstTableRef):
  list_tokens_bbox = []

  #dimensao dos dados
  qtdRowDf = 0 if df.empty else df.shape[0]
  qtdColDf = 0 if df.empty else df.shape[1]
  print("noteListTokensBbox - dimensao df {0} x {1} ".format(qtdRowDf, qtdColDf))

  #carregando BBOX de cada celula por linha para uma lista
  #qtdlinhasBbox = 0 if len(lstBbox) == 0 else len(lstBbox)
  #qtdcolunasBbox = 0 if len(lstBbox) == 0 and len(lstBbox[0]) else len(lstBbox)

  for i in range(qtdRowDf):
    for j in range(qtdColDf):
      #bbox = lstBbox[i][j]
      #x1 = round(bbox[0])
      #y1 = round(bbox[1])
      #x2 = round(bbox[2])
      #y2 = round(bbox[3])
      x1 = np.nan
      y1 = np.nan
      x2 = np.nan
      y2 = np.nan

      value = "NAN"
      if not df.empty and not pd.isna(df.at[i, j]):
        # Value exists, access it
        value = str(df.at[i, j])

      if(lstTableRef is None):
        dict_tokens_bbox = {'tokens': list(value), 'bbox': [x1, y1, x2, y2]}
      else:
        dict_tokens_bbox = {'tokens': list(value), 'bbox': [x1, y1, x2, y2]}
        #dict_tokens_bbox = {'tokens': list(df.at[i,j]), 'bbox': [x1 - lstTableRef[0], y1 - lstTableRef[1], x2 - lstTableRef[0], y2 - lstTableRef[1]]}
      list_tokens_bbox.append(dict_tokens_bbox)

  return list_tokens_bbox

def printMetaDados(dicMetaData):

  strout = []
  strout.append("{ \n")

  if "filename" in dicMetaData:
    strout.append("filename: '" + str(dicMetaData["filename"]) + "',\n")
  else:
    strout[0] = "chave filename não existe na estrutura"
    return strout

  if "split" in dicMetaData:
    strout.append("split: '" + str(dicMetaData["split"]) + "',\n")
  else:
    strout[0] = "chave split não existe na estrutura"
    return strout

  if "imgid" in dicMetaData:
    strout.append("'imgid': " + str(dicMetaData["imgid"]) + ",\n")
  else:
    strout[0] = "chave imgid não existe na estrutura"
    return strout

  if "html" in dicMetaData:

    strout.append("--INICIO HTML \n")
    strout.append("'html': \n {")

    if "cells" in dicMetaData["html"] and "structure" in dicMetaData["html"]:

      if isinstance(dicMetaData["html"]["cells"], list ) and isinstance(dicMetaData["html"]["structure"], list ):

        #varrendo o conteudo da lista dicMetaData["html"]["cells"]
        #que contem as duas sublistas tokens e bbox
        strout.append("--INICIO CELLS \n")
        strout.append("'cells': [\n")
        i = 0
        for arrcells in dicMetaData["html"]["cells"]:

           #print da estrutura dos dicionarios tokens e bbox
           tokens =  arrcells["tokens"]
           bbox =  arrcells["bbox"]
           comma = ","
           if i == len(dicMetaData["html"]["cells"]) -1:
            comma = ""
           else:
            comma = ","

           strout.append("      {'tokens': " + str(tokens) + ", 'bbox': " + str(bbox) + "}" + comma + " \n")
           i = i + 1

        strout.append("] --FIM CELLS\n")

        #print da estrutura do dicionario structure
        if(dicMetaData["html"] is not None and dicMetaData["html"]["structure"] is not None):
          #strout.append("      'structure': [" + str("','".join(dicMetaData["html"]["structure"])) + "' \n")
          strout.append("      'structure': ['" + str("','".join([x for x in dicMetaData["html"]["structure"] if x is not None])) + "' \n")

        else:
          strout.append("      'structure': ['None'] \n")


        strout.append("] --FIM STRUCTURE \n")

      else:
        strout[0] = "chave html/cells ou structure não existe na estrutura"
        return strout

    strout.append("} --FIM HTML\n")
  else:
    strout[0] = "chave html não existe na estrutura"
    return strout

  strout.append("} \n")
  return strout

def saveAnnotationFile(dicMetaData, dirout , numPage):

  vecArq = dicMetaData["filename"].split("/")

  nomeArq = ""
  if(len(vecArq)>0):

    tableID = dicMetaData["imgid"].replace("'","")
    filename = vecArq[len(vecArq)-1].split(".")[0]
    labname = vecArq[len(vecArq)-2].split(".")[0]

    nomeArq = tableID + "|" + filename + "|" + str(numPage) + "_METADADOS.mtd"
    print("gravando arquivo ", nomeArq)

    strout = printMetaDados(dicMetaData)

    labDirOut = dirout + "/" + labname + "/"
    if not os.path.exists(labDirOut):
      os.makedirs(labDirOut)

    with open(labDirOut + nomeArq, 'w') as arquivo:
      for linha in strout:
            arquivo.write(linha)

def getPos(lst, key):

  for k, item in enumerate(lst):
    if item == key:
      return k

  return -1

#FUNCAO DE VERIFICA SE EXISTE O BUG DE TRUNCAR O VALOR ∞
def isBUGInfinito(dicTable, dicTableGT):

  char = ""
  posInf = getPos(dicTableGT["FIRST_LINE"], "∞")
  posInfV = getPos(dicTableGT["FIRST_LINE"], "V")

  #possui valor ∞ na tabela? segue análise
  if posInf >-1:
    char = "∞"
    print("Possui valor ∞ na tabela, posInf", posInf)
    #2 - possuem o mesmo valor de dimensao em DIMENSION
    if dicTable["DIMENSION"] == dicTableGT["DIMENSION"]:
      print("Valor DIMENSION iguais")
      #4 primeiras colunas das duas tabelas possuem o mesmo valor?
      print("checkAVGSimilarities2INFO >=60 perc? ",checkAVGSimilarities2INFO(dicTableGT, dicTable, 80))
      if checkAVGSimilarities2INFO(dicTableGT, dicTable, 60):
        print("Quatro primeiras colunas similares")
        #4 - chave HEAD tem o tamanho um a menos que GT
        if len(dicTable["HEAD"]) == len(dicTableGT["HEAD"])-1 and len(dicTable["FIRST_LINE"]) == len(dicTableGT["FIRST_LINE"])-1:
          #dicTable["FIRST_LINE"].insert(posInf, "∞")
          return True, posInf, "∞"

  #possui valor ∞ na tabela? segue análise
  if posInfV >-1:
    char = "V"
    print("Possui valor V na tabela, posInf", posInf)
    #2 - possuem o mesmo valor de dimensao em DIMENSION
    if dicTable["DIMENSION"] == dicTableGT["DIMENSION"]:
      print("Valor DIMENSION iguais")
      #4 primeiras colunas das duas tabelas possuem o mesmo valor?
      print("checkAVGSimilarities2INFO >=60 perc? ",checkAVGSimilarities2INFO(dicTableGT, dicTable, 80))
      if checkAVGSimilarities2INFO(dicTableGT, dicTable, 60):
        print("Quatro primeiras colunas similares")
        #4 - chave HEAD tem o tamanho um a menos que GT
        if len(dicTable["HEAD"]) == len(dicTableGT["HEAD"])-1 and len(dicTable["FIRST_LINE"]) == len(dicTableGT["FIRST_LINE"])-1:
          #dicTable["FIRST_LINE"].insert(posInf, "∞")
          return True, posInfV, "V"

  return False, -1, ""

import glob

def deleteFiles(dir, ext):
  # Obter todos os arquivos com a extensão especificada
  files = glob.glob(os.path.join(dir, f'*.{ext}'))

  # Remover cada arquivo encontrado
  for file in files:
      try:
          os.remove(file)
          print(f"Arquivo {file} removido com sucesso.")
      except OSError as e:
          print(f"Erro ao remover o arquivo {file}: {e}")

def printHTML2(lst, tipo): #com TAB
  # tipo: RAW (cru) ou PRETTY (html com identações)

  strout = ""
  #strres = ''.join(lst)
  strres = ''.join([str(x) for x in lst])
  if tipo == "PRETTY":
    soup = bs(strres, 'html.parser')
    strout = soup.prettify()
    # Substituir espaços por TAB
    strout = strout.replace("  ", "\t")
  else:
    strout = strres

  return strout

def printElementMetaData(dicMetaData, elem):

  strout = []
  strout.append("[")

  if not "filename" in dicMetaData:
    strout[0] = "chave filename não existe na estrutura"
    return strout

  if not "split" in dicMetaData:
    strout[0] = "chave split não existe na estrutura"
    return strout

  if not "imgid" in dicMetaData:
    strout[0] = "chave imgid não existe na estrutura"
    return strout

  if "html" in dicMetaData:
    if "cells" in dicMetaData["html"] and "structure" in dicMetaData["html"]:
      if isinstance(dicMetaData["html"]["cells"], list ) and isinstance(dicMetaData["html"]["structure"], list ):

        i = 0

        if(elem == "BBOX"):
          #strout.append("{")
          for arrcells in dicMetaData["html"]["cells"]:

            tokens =  arrcells["tokens"]
            bbox =  arrcells["bbox"]
            comma = ","
            if i == len(dicMetaData["html"]["cells"]) -1:
              comma = ""
            else:
              comma = ","

            strout.append("{'tokens': " + str(tokens) + ", 'bbox': " + str(bbox) + "}" + comma + " \n")
            i = i + 1

        elif(elem == "HTML_PRETTY"):
          strout.append(printHTML2(dicMetaData["html"]["structure"], "PRETTY"))

        else:
          html = "'"
          html = html + "','".join([element for element in dicMetaData["html"]["structure"] if element]) + "'"
          strout.append(html)

      else:
        strout[0] = "chave html/cells ou structure não existe na estrutura"
        return strout

  else:
    strout[0] = "chave html não existe na estrutura"
    return strout

  strout.append("]")
  return strout

def saveElementMetadata(dicMetaData, elem, dirout, numPage):

  vecArq = dicMetaData["filename"].split("/")
  #print(vecArq)
  nomeArq = ""
  if(len(vecArq)>0):

    tableID = str(dicMetaData["imgid"]).replace("'","")
    filename = vecArq[len(vecArq)-1].split(".")[0]
    labname = vecArq[len(vecArq)-2].split(".")[0]

    nomeArq = tableID + "|" + filename + "|" + str(numPage) + "_" + elem + "." + elem.lower()
    strout = printElementMetaData(dicMetaData, elem)

    labDirOut = dirout + "/" + labname + "/"

    print("Arquivo de anotação ", elem, " gerado = ",  nomeArq)
    with open(labDirOut + nomeArq, 'w') as arquivo:
      for linha in strout:
          arquivo.write(linha)

#retorna o tableID de maior similiaridade entre os dicionarios GT de comparacao
def getMaiorSimilaridade(dic, listasGT):

  lstDicRes = []

  firstLineCopy = copy.deepcopy(dic["FIRST_LINE"])
  firstLineAux = firstLineCopy

  for dicGT in listasGT:

    lstRes = []

    #ajusta caso necessário a dimensao entre as tabelas
    #print("a tratar...", dic["FIRST_LINE"])
    #print(type(dic["FIRST_LINE"]))

    print("getMaiorSimilaridade, FIRST_LINE GT", dicGT["FIRST_LINE"])
    print("getMaiorSimilaridade, FIRST_LINE ANALISE", firstLineAux)
    for i in range(len(dicGT["FIRST_LINE"])):

      lenDicGT = len(dicGT["FIRST_LINE"])

      firstLineAux = ajustColList(firstLineAux, len(dicGT["FIRST_LINE"]))
      lenDic = len(firstLineAux)

      if lenDic != lenDicGT: #tamanhos diferentes, retornar vazio
        return ""

      str1 = str(firstLineAux[i])
      str2 = str(dicGT["FIRST_LINE"][i])
      strDec1 = str1.replace(",", ".").strip()
      strDec2 = str2.replace(",", ".").strip()

      #se os valores forem numeros converter para float para calcular similiaridade com maior exatidao
      if (isDecimal(strDec1) and isDecimal(strDec2)):
        #print(" Similaridade entre dois numeros ", strDec1, " e ", strDec2, " = ", round(calcPercSimValueNums(float(strDec1), float(strDec2)), 2))
        lstRes.append(round(calcPercSimValueNums(float(strDec1), float(strDec2)), 2))
      #no caso de string
      else:
        #print(" Similaridade entre duas strings ", strDec1, " e ", strDec2, " = ", round(calcPercSimStrings(str1, str2),4))
        lstRes.append(round(calcPercSimStrings(str1, str2),2))

      firstLineAux = firstLineCopy

    lstDicRes.append({"TABLEID": dicGT["TABLEID"], "RESULT":lstRes})

  #verificando maior media
  tableId = ""
  maiorMedia = 0
  for dicRes in lstDicRes:
    avg = sum(dicRes["RESULT"]) / len(dicRes["RESULT"])
    print("Media ", avg)
    if avg > maiorMedia:
      tableId = dicRes["TABLEID"]
      maiorMedia = avg

  return tableId

def ajustColList(lst1, qtdColsRef):

  # Calcula o número de colunas de cada lista
  #num_cols_lstRef = len(lstRef)
  num_cols_lst1 = len(lst1) if lst1 else 0

  # Se lst1 tiver menos colunas que lstRef, preenche com NaN
  if num_cols_lst1 < qtdColsRef:
    # Calcula o número de colunas a serem adicionadas
    num_cols_adicionais = qtdColsRef - num_cols_lst1
    # Preenche lst2 com NaN nas novas colunas

    #print("num_cols_adicionais ", num_cols_adicionais)
    for i in range(num_cols_adicionais):
      lst1.append("NAN")

  #se lst1 tiver mais coluna que lstRef, remove as colunas adicionais de lst1
  elif num_cols_lst1 > qtdColsRef:
    # Calcula o número de colunas a serem removidas
    num_cols_adicionais = num_cols_lst1 - qtdColsRef
    for i in range(num_cols_adicionais):
      if (len(lst1)>0):
        del(lst1[len(lst1)-1])

  return lst1

#funcao para ajustar uma lista de acordo com a quantidade de linhas e colunas de referencia
# se tiver a mais linhas ou colunas, adiciona, se tiver menos, remove
def ajustList(lst1, qtdRowsRef, qtdColsRef):

  qtdRows = len(lst1)

  #lsteste = eval(strdata)
  lst1_ajust = []

  #1 - ajustando as colunas
  for lstRow in lst1:
    lst1_ajust.append(ajustColList(lstRow, qtdColsRef))

  #1 - ajustando as linhas
  #se precisar adicionar linhas
  if(qtdRowsRef > qtdRows):
    qtdLinhasAdicionais = qtdRowsRef - qtdRows
    for i in range(qtdLinhasAdicionais):
      if len(lst1_ajust) >0 and len(lst1_ajust[0]) >0:
        lst1_ajust.insert(len(lst1_ajust), ["NAN" for _ in range(len(lst1_ajust[0]))])
  #se precisar remover linhas adicionais
  elif(qtdRows > qtdRowsRef):
    qtdLinhasAdicionais = qtdRows - qtdRowsRef
    for i in range(qtdLinhasAdicionais):
      del(lst1_ajust[len(lst1_ajust)-1])

  return lst1_ajust

#funcao para ajustar a estrutura cell_coordinates em relacao a referencia para possibilitar a comparacao e geracao de estatisticas
def ajustCellCord (cellCord, qtdRowsRef, qtdColsRef):

  lstCoord = [999999, 999999, 999999, 999999] #nova lista de coordenadas
  dicNewCol = {'column': lstCoord, 'cell': lstCoord} #uma nova coluna (celula)
  #nova linha da tabela
  newLine =  "{'row': [99999, 99999, 99999, 99999], \
              'cells': [], \
              'cell_count': 0}"
  dicNewLine = eval(newLine)

  #verificando dimensao da estrutura atual
  qtdRows = 0
  qtdCols = 0
  if cellCord is not None and len(cellCord) >0:
    qtdRows = len(cellCord)
    qtdCols = len(cellCord[0]["cells"])
  #qtdRows = len(cellCord)
  #qtdCols = len(cellCord[0]["cells"])

  #adicionando estrutura inicial para cada quantidade de colunas de referencia
  for i in range(qtdColsRef):
    dicNewLine["cells"].insert(i,dicNewCol)

  #adiciona para cada coluna adicional necessária
  if qtdCols < qtdColsRef:
    qtdColAdicionais = qtdColsRef - qtdCols

    print("Adicionando coluna, qtd = ", qtdColAdicionais)
    for i in range(qtdColAdicionais):
      for row in cellCord:
        row['cells'].append(dicNewCol)

  #removendo uma coluna para cada adicional
  elif qtdCols > qtdColsRef:
    qtdColAdicionais = qtdCols - qtdColsRef

    print("Removendo coluna, qtd = ", qtdColAdicionais)
    for i in range(qtdColAdicionais):
      for row in cellCord:
        row['cells'] = row['cells'][:-1]

  #adiciona linha para cada linha adicional necessária
  if qtdRows < qtdRowsRef:
    qtdRowAdicionais = qtdRowsRef - qtdRows

    print("Adicionando linha, qtd = ", qtdRowAdicionais)
    for i in range(qtdRowAdicionais):
      cellCord.append(dicNewLine)

  #removendo linha para cada linha adicional necessária
  elif qtdRows > qtdRowsRef:
    qtdRowAdicionais = qtdRows - qtdRowsRef

    print("Removendo linha, qtd = ", qtdRowAdicionais)
    #removendo linha para cada adicional
    for i in range(qtdRowAdicionais):
      del(cellCord[len(cellCord)-1])

  return cellCord

def temRepeticoes(lista):
    return len(lista) != len(set(lista))

#ajustar df2 para incrementar mais linhas ou colunas em relacao a referencia (df1)
def ajustDataframe(df1, df2):

  # Verifica se o número de colunas de df2 é menor que o de df1
  if df2.shape[1] < df1.shape[1]:
      # Calcula quantas colunas precisam ser adicionadas
      num_cols_adicionais = df1.shape[1] - df2.shape[1]
      # Adiciona as colunas extras em df2 preenchidas com NaN
      for i in range(num_cols_adicionais):
          df2[f'C{i+1}'] = np.nan
  #neste caso remove as colunas adicionais
  elif df2.shape[1] > df1.shape[1]:
    # Calcula quantas colunas precisam ser removidas
    num_cols_remover = df2.shape[1] - df1.shape[1]
    # Remove as colunas extras em df2 da direita para a esquerda
    df2 = df2.iloc[:, :-num_cols_remover]

  # Verifica se o número de linhas de df2 é menor que o de df1
  if df2.shape[0] < df1.shape[0]:
      # Calcula quantas linhas precisam ser adicionadas
      num_linhas_adicionais = df1.shape[0] - df2.shape[0]
      # Adiciona as linhas extras em df2 preenchidas com NaN
      linhas_extras = pd.DataFrame(index=[f'L{i+1}' for i in range(num_linhas_adicionais)],
                                    columns=df2.columns)
      df2 = pd.concat([df2, linhas_extras])
  #neste caso remove as linhas adicionais
  elif df2.shape[0] > df1.shape[0]:
    # Calcula quantas linhas precisam ser removidas
    num_linhas_remover = df2.shape[0] - df1.shape[0]
    # Remove as linhas extras em df2 de baixo para cima
    df2 = df2.iloc[:-num_linhas_remover, :]

  return df2

import copy

#ajustar df2 para incrementar mais linhas ou colunas em relacao a referencia (df1)
def ajustDataframe(df1, qtdLinhasRef, qtdColunasRef):

  dfAux = copy.deepcopy(df1)
  # Verifica se o número de colunas de df1 é menor que qtdColunasRef
  if df1.shape[1] < qtdColunasRef:
      # Calcula quantas colunas precisam ser adicionadas
      num_cols_adicionais = qtdColunasRef - df1.shape[1]
      # Adiciona as colunas extras em df1 preenchidas com NaN
      #print("adicionando qtd colunas", num_cols_adicionais)
      ultIndice = len(dfAux.columns) - 1
      for i in range(num_cols_adicionais):
          if (i==0):
            novoIndice = ultIndice + 1
          else:
            novoIndice = ultIndice + (i+1)
          #dfAux[f'C{i+1}'] = np.nan novoIndice
          dfAux[novoIndice] = np.nan
  #neste caso remove as colunas adicionais
  elif qtdColunasRef < df1.shape[1]:
    # Calcula quantas colunas precisam ser removidas
    num_cols_remover = df1.shape[1] - qtdColunasRef
    # Remove as colunas extras em df1 da direita para a esquerda
    dfAux = dfAux.iloc[:, :-num_cols_remover]

  # Verifica se o número de linhas de df1 é menor que qtdLinhasRef
  if df1.shape[0] < qtdLinhasRef:
      # Calcula quantas linhas precisam ser adicionadas
      num_linhas_adicionais = qtdLinhasRef - df1.shape[0]
      # Adiciona as linhas extras em df1 preenchidas com NaN
      #linhas_extras = pd.DataFrame(index=[f'L{i+1}' for i in range(num_linhas_adicionais)],
                                    #columns=df1.columns)
      #print("adicionando qtd linhas", num_linhas_adicionais)
      ultIndice = len(dfAux.columns) - 1
      linhas_extras = pd.DataFrame(index=[f'{ultIndice + i + 1}' for i in range(num_linhas_adicionais)],
                             columns=dfAux.columns)
      dfAux = pd.concat([dfAux, linhas_extras])
  #neste caso remove as linhas adicionais
  elif qtdLinhasRef < df1.shape[0]:
    # Calcula quantas linhas precisam ser removidas
    num_linhas_remover = df1.shape[0] - qtdLinhasRef
    # Remove as linhas extras em df1 de baixo para cima
    dfAux = dfAux.iloc[:-num_linhas_remover, :]

  #normalizando indices
  dfAux = dfAux.reset_index(drop=True)

  return dfAux

#funcao para converter PDF para png
def pdf_page_to_png(pdf_path, page_number, output_path):
  # Convertendo a página do PDF para uma lista de imagens
  images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)

  # Salvando a imagem como PNG
  images[0].save(output_path, 'PNG')


#funcao para retirar cabeçalhos adicionais
def normalizeDataframe(dfTable):

  dfAux = copy.deepcopy(dfTable)

  for i in dfTable.index.tolist():

    line = dfTable.iloc[i].tolist()
    #print("i = ", i, " line = ", line)

    #se for entre a primeira e quinta linha (possiveis cabeçalhos)
    if (i>=0 and i<=5):

      #possui valores nulos, vazios ou não numéricos (cabeçalho)
      if( numTimes(line, None) > 3 or numDecimals(line)<=1 ):
        print("i = ",i, " É CABEÇALHO" )
        print("line do cabeçalho = ",line )
        #verificar se na proxima linha possui valores numéricos (neste caso não deve remover - possivel cabeçalho)
        prox = i + 1
        if(prox < len(dfTable)):
          proxLine = dfTable.iloc[prox].tolist()
          print("i = ",i, "prox = ", prox, "proxLine = ", proxLine)
          if(numDecimals(proxLine)>2):
            #print("NÃO REMOVER CABEÇALHO DO, ", i, " , pois PROX possui valores numericos , qtd decimals(prox) =", numDecimals(proxLine))
            #dfAux.loc[len(dfAux)] = line
            continue
          else:
            #print("REMOVER LINHA ", i, "pois PROX não possui valores numericos , qtd decimals(prox) = ", numDecimals(proxLine))
            dfAux = dfAux.drop(i, inplace=False)
    else:
      break

  #normalizando indices
  dfAux = dfAux.reset_index(drop=True)

  return dfAux

#funcao para retirar cabeçalhos adicionais
def normalizeDataframe2(dfTable):

  dfAux = copy.deepcopy(dfTable)

  j = 0

  idLastHead = 0
  for i in dfTable.index.tolist():

    line = dfTable.iloc[i].tolist()
    #print("i = ", i, " line = ", line)

    #possui valores numericos (remover todos cabeçalhos pra tras)
    if(numDecimals(line)> 4 and i >0):
      #print("i = ",i, " É INICIO PÓS CABEÇALHO, REMOVER TUDO PRA TRÁS" )
      #print("LINE ", line )
      idLastHead = i
      break

  if idLastHead > 0:
    for i in range(idLastHead -1):
      #print("removendo linha ", i)
      dfAux = dfAux.drop(i, inplace=False)

  #normalizando indices
  dfAux = dfAux.reset_index(drop=True)

  return dfAux

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-padding, bbox[2]+padding, bbox[3]+padding]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops

def clean_list_of_lists(lstData):
    cleaned_data = []
    
    for sublist in lstData:
        cleaned_sublist = []
        for item in sublist:
            # Remover colchetes e aspas simples
            item = item.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")

            # Substituir vírgula decimal por ponto se for um número
            item = re.sub(r'(\d+),(\d+)', r'\1.\2', item)

            cleaned_sublist.append(item)
        cleaned_data.append(cleaned_sublist)
    
    return cleaned_data

def clean_list(lstData):
    cleaned_list = []
    
    for item in lstData:
        # Remover colchetes e aspas simples
        item = item.replace("[", "").replace("]", "").replace("'", "").replace("\"", "")

        # Substituir vírgula decimal por ponto se for um número
        item = re.sub(r'(\d+),(\d+)', r'\1.\2', item)

        cleaned_list.append(item)
    
    return cleaned_list

import os
from PIL import Image

def carregar_tables_crops_flex(pasta_imagens, NOMELAB, NOMEARQ, PAGE):
  """
  Lê imagens .jpg da pasta onde:
  - Primeiro índice contenha NOMELAB
  - Segundo índice seja igual a NOMEARQ
  Retorna a lista 'tables_crops' ordenada alfabeticamente.
  """
  tables_crops = []

  for nome_arquivo in os.listdir(pasta_imagens):
      if nome_arquivo.lower().endswith(".jpg"):
          partes = nome_arquivo.split("_")
          if len(partes) >= 3:
              lab = partes[0]
              arq = partes[1]
              pag = partes[2].split(".")[0]  # Remove extensão .jpg
              if NOMELAB in lab and arq == NOMEARQ and pag == str(PAGE):
                  caminho_completo = os.path.join(pasta_imagens, nome_arquivo)
                  imagem = Image.open(caminho_completo).convert("RGB")
                  tables_crops.append({
                      'arq': nome_arquivo,
                      'image': imagem,
                      'tokens': []
                  })

  # Ordenar pela chave 'arq'
  if len(tables_crops) > 0:
    tables_crops.sort(key=lambda x: x['arq'])
    
  return tables_crops

import ast
def carregar_bboxes_info_GT(pasta_info, NOMELAB, NOMEARQ, PAGE):
    """
    Busca arquivos .info na pasta_info onde:
    - A primeira parte do nome (antes do primeiro |) contém NOMELAB
    - A segunda parte é NOMEARQ
    - A terceira parte é PAGE
    - O nome termina com _INFO.info

    Retorna uma lista de dicionários com arq, label, score e bbox extraído.
    """
    objects = []

    arquivos_info = [
        arq for arq in os.listdir(pasta_info)
        if arq.endswith("_INFO.info")
    ]

    arquivos_info.sort()

    for nome_arquivo in arquivos_info:
        partes = nome_arquivo.replace("_INFO.info", "").split("|")

        if len(partes) != 3:
            continue

        lab_part, file_part, page_part = partes

        # Verifica as condições desejadas
        if NOMELAB in lab_part and file_part == NOMEARQ and page_part == str(PAGE):
            caminho_completo = os.path.join(pasta_info, nome_arquivo)
            try:
                with open(caminho_completo, "r", encoding="utf-8") as f:
                    conteudo = f.read()
                    dados = ast.literal_eval(conteudo)
                    bbox = dados.get("BBOX")
                    if bbox:
                        objects.append({
                            'arq': nome_arquivo,
                            'label': 'table',
                            'score': 0.9,
                            'bbox': bbox
                        })
            except Exception as e:
                print(f"Erro ao processar {nome_arquivo}: {e}")

    return objects

def carregar_bboxes_info_analise(GT_LAB_OUT, IMG_GT, IMG_LAB, detection_transform, model, device, onlylab):
    # 3. Loop pelos elementos e gerar nova estrutura com bbox atualizado

    GT_LAB_ANALISE = []
    i = 0

    for item in GT_LAB_OUT:
      arq = item['arq']
      
      # 2.1 - Separar partes
      parts = arq.split('|')
      tableid = parts[0]
      arquivo = parts[1]
      pagina = parts[2].split('_')[0]

      print("carregar_bboxes_info_analise - arquivo: ", arquivo)

      #quantas tabelas possui naquela pagina
      listaArqTab = listar_arquivos_info(onlylab, IMG_GT, arquivo, pagina)
      i = 0
      posGT = 0
      print("carregar_bboxes_info_analise, listaArqTab = ", listaArqTab)
      print("carregar_bboxes_info_analise, tableid = ", tableid)
      for tableidGT in listaArqTab:
        if tableid in tableidGT:
          posGT = i
          break
        i = i + 1
      #verifica posicao

      # 2.2 - Construir caminho e carregar imagem
      img_path = f"{IMG_LAB}/{arquivo}_{pagina}.png"

      print("carregar_bboxes_info_analise - img_path: ", img_path)
      image = Image.open(img_path).convert("RGB")

      pixel_values = detection_transform(image).unsqueeze(0)
      pixel_values = pixel_values.to(device)

      model.to(device)
      outputs = None
      with torch.no_grad():
          outputs = model(pixel_values)
      
      id2label = model.config.id2label
      id2label[len(model.config.id2label)] = "no object"

      objects = outputs_to_objects(outputs, image.size, id2label)

      # se quantidade de objetos detectado for menor do que de tabelas de GT pega a primeira
      if len(objects) != len(listaArqTab):
        print(f"Quantidade de objetos detectados {len(objects)} diferente do GT {len(listaArqTab)} / posGT = 0")
        posGT = 0
      elif len(objects) == len(listaArqTab):
        print(f"Quantidade de objetos detectado {len(objects)} e do GT {len(listaArqTab)} IGUAIS! / posGT = {posGT}")

      if len(objects) >0: 
          print("carregar_bboxes_info_analise - objects[posGT]: ", objects[posGT])
          GT_LAB_ANALISE.append(objects[posGT])  

    return GT_LAB_ANALISE

def listar_arquivos_info(lab, caminho, arquivo, pagina):
  padrao_inicial = f"{lab}"
  padrao_final = f"{arquivo}_{pagina}.jpg"
  arquivos_filtrados = []

  #print("listar_arquivos_info - caminho: ", caminho)
  #print("listar_arquivos_info - padrao_inicial: ", padrao_inicial)
  #print("listar_arquivos_info - padrao_final: ", padrao_final)

  # Percorre todos os arquivos no diretório fornecido
  for nome in os.listdir(caminho):
      if nome.startswith(padrao_inicial) and nome.endswith(padrao_final):
          arquivos_filtrados.append(nome)
  
  # Ordena por nome
  arquivos_filtrados.sort()

  return arquivos_filtrados