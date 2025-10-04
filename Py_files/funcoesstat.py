import os
import glob
import json
import ast
import pandas as pd
import sys

#calcula IOU entre dois bounding box
def calculate_iou(bbox1, bbox2):
    """Calcula o IoU (Intersection over Union) entre dois bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox1[2] * bbox1[3]
    area2 = bbox2[2] * bbox2[3]

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

#coletar as informacoes da tabela do arquivo _INFO e retornar para uma lista
def getListTablesInfo(path, listFiles):

  listTablesInfo = []
  for fileName in listFiles:
    with open(path + fileName, 'r') as file:
      conteudo = file.read()
      listTablesInfo.append(eval(conteudo))

  return listTablesInfo

def sortList(lst, len):
    def personList(item):
        # Extrai o número após 'CTM' e converte para inteiro
        return int(item[len:])

    return sorted(lst, key=personList)

#coletar arquivo de acordo com premissas (prefixo e sufixo)
def getFileByPrefix(path, prefix, sufix):

  for fileName in os.listdir(path):
      if prefix in fileName and fileName.endswith(sufix):
          return fileName
  return ""

def getFiles(folderDir, ext):
  # Construir o padrão de busca usando a extensão fornecida
  pattern = os.path.join(folderDir, f"*.{ext}")

  files = []
  arrPath = folderDir.split("/")

  #print("arrPath ", arrPath)
  if len(arrPath) >0 and len(arrPath[len(arrPath)-1].split("_")) >0:
    #print(folderDir)
    lab = arrPath[len(arrPath)-1].split("_")[2]
    # Usar a função glob para encontrar os arquivos correspondentes ao padrão
    files = glob.glob(pattern)

  # Retornar a lista de arquivos encontrados
  return sorted(files)

def getInfoFiles(list):

  lstInfoFiles = []
  for path in list:

    arrPath = path.split("/")
    file = arrPath[len(arrPath)-1]
    tableId = file.split("|")[0]
    fileName = file.split("|")[1]
    page = file.split("|")[2].split("_")[0]

    lstInfoFiles.append({"TABLEID":tableId, "TABLEID":tableId, "FILE":fileName, "PAGE":page})

  return lstInfoFiles

#files = getFiles("/content/drive/MyDrive/DataSets/Certificados/Out/img2table/LAB_01_CTM", "info")
#lstInfoFiles = getInfoFiles (files)

#funcao que compara o valor de duas listas e calcula a media do percentual de similaridade entre eles
#(para calcular o valor do bbox das tabelas e células das tabelas)
def calcPercSimValueLists(lista1, lista2):
  if len(lista1) != len(lista2):
      print("calcPercSimValueLists, listas de tamanhos diferentes, lista1=",lista1,"/ lista2 = ",lista2)
      raise ValueError("As listas devem ter o mesmo comprimento.")
  
  percSim = [ (1 / (1 + (abs(num1 - num2)))) * 100 for num1, num2 in zip(lista1, lista2)]

  #print("calcPercSimValueLists, percSim = ", percSim)
  #print("result percSim ", sum(percSim) / len (percSim))
  return sum(percSim) / len (percSim)

#(para calcular o percentual de similaridade entre dois números
def calcPercSimValueNums(num1, num2):

  percSim = (1 / (1 + (abs(num1 - num2)))) * 100
  #print (" similaridade entre os numeros {0} e {1}: {2}".format(num1, num2, percSim))
  return percSim

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

#numero de ocorrencias de um numero em uma lista
def numTimes(list, num):

  cont = 0
  for valor in list:
    if valor == num:
      cont+=1

  return cont

#numero de ocorrencias de um numero ser maior ou igual que um numero
def numTimesMoreThen(list, num):

  cont = 0
  for valor in list:
    if valor >= num and valor <100:
      cont+=1

  return cont

def calStatsIouBbox(lstTable1, lstTable2):

  lstResBbox = []

  if(len(lstTable1) > 0 and len(lstTable2) > 0):
    print("calStatsIouBbox - listaBbox1 = ", lstTable1[0]["bbox"])
    print("calStatsIouBbox - listaBbox2 = ", lstTable2[0]["bbox"])

  for item1, item2 in zip(lstTable1, lstTable2):

    listaBbox1 = item1["bbox"]
    listaBbox2 = item2["bbox"]

    if len(listaBbox1) >0 and len(listaBbox2) >0:

      lstResBbox.append(calculate_iou(listaBbox1, listaBbox2))
    else:
      lstResBbox.append(0)
    
  return lstResBbox

#calcular similaridades em valores das celulas bbox e tokens de duas listas de tabelas
def calStatsTablesValues(lstTable1, lstTable2):

  lstResTokens = []
  lstResBbox = []

  if(len(lstTable1) > 0 and len(lstTable2) > 0):
    print("calStatsTablesValues - listaBbox1 = ", lstTable1[0]["bbox"])
    print("calStatsTablesValues - listaBbox2 = ", lstTable2[0]["bbox"])

  for item1, item2 in zip(lstTable1, lstTable2):

    listaBbox1 = item1["bbox"]
    listaBbox2 = item2["bbox"]
    str1 = "".join(item1["tokens"])
    str2 = "".join(item2["tokens"])

    if len(listaBbox1) >0 and len(listaBbox2) >0:

      lstResBbox.append(round(calcPercSimValueLists(listaBbox1, listaBbox2),2))
    else:
      lstResBbox.append(0)

    strDec1 = str1.replace(",", ".").strip()
    strDec2 = str2.replace(",", ".").strip()
    #se os valores forem numeros converter para float para calcular similiaridade com maior exatidao
    if (isDecimal(strDec1) and isDecimal(strDec2)):
      lstResTokens.append( round( calcPercSimValueNums(float(strDec1), float(strDec2) ),2) )
    #no caso de string
    else:
      lstResTokens.append(round(calcPercSimStrings(str1, str2),2))
    
    

  return lstResTokens, lstResBbox

#calcular similaridades em valores das celulas bbox das tabelas detectadas (arquivo INFO)
def calStatsBboxInfo(lstTable1, lstTable2):

  lstBboxInfo = []
  listaBbox1 = [] if len(lstTable1) ==0 else lstTable1["BBOX"]
  listaBbox2 = [] if len(lstTable2) ==0 else lstTable2["BBOX"]
  #print("calStatsBboxInfo = listaBbox1 = ", listaBbox1)
  #print("calStatsBboxInfo = listaBbox2 = ", listaBbox2)

  if len(listaBbox1) >0 and len(listaBbox2) >0:
    lstBboxInfo.append(round(calcPercSimValueLists(listaBbox1, listaBbox2),2))
  else:
    lstBboxInfo.append(0)

  return lstBboxInfo

#calcular quantidade de valores não lidos pelo modelo (NAN ou 999999)
def calStatsNAN(lstTable):

  qtdNANToken = 0
  qtdNANBbox = 0

  for item in lstTable:
    listaBbox = item["bbox"]
    token = "".join(item["tokens"])

    if numTimes(listaBbox, 999999) ==4:
      qtdNANBbox+=1

    if token == "NAN":
      qtdNANToken+=1

  return qtdNANToken, qtdNANBbox

def removeSpecialChars(strTexto):
  # Remover os caracteres especiais
  speChars = ["\\", ]

  strTexto = strTexto.replace(speChars, "")

  return strTexto

def readFile(filePath):
  try:
    with open(filePath, 'r') as arquivo:
        conteudo = arquivo.read()
        conteudo = conteudo.replace("[nan, nan, nan, nan]", "[999999,999999,999999,999999]")
        #conteudo = conteudo.replace("\'", "")
        conteudo = conteudo.replace("\n", " ")
    return conteudo
  except FileNotFoundError:
    print(f'O arquivo "{filePath}" não foi encontrado.')
    return None
  except Exception as e:
    print(f'Ocorreu um erro ao ler o arquivo: {e}')
    return None

def SaveFileStats (filePath, dicTableStats):

  strFile = json.dumps(dicTableStats)
  strFile = strFile.replace(",",",\n")

  print("Salvando arquivo de estatísticas: ",filePath)
  with open(filePath, 'w') as arquivo:
    arquivo.write(strFile)

def strInDic(dicionario, string):

  for chave, valor in dicionario.items():
      if isinstance(valor, str) and string in valor:
          return True
  return False

def strInList(lstDic, string):

  for dic in lstDic:
    for chave, valor in dic.items():
        if isinstance(valor, str) and string == valor:
            return True
  return False

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

def ExportCSVSummary(ToolPath, GTPath):
  #coletando os diretorios dos laboratorios
  dirLabs = [nome for nome in os.listdir(ToolPath)]

  listStats = []
  filesSTATS = []

  lstTableIdGT = []
  #colentando os arquivos de statisticas dos laboratorios
  for dirLab in dirLabs:

    labPath =  ToolPath + dirLab
    filesSTATS = getFiles(labPath, "stats")

    #print("filesSTATS ", filesSTATS)
    #coletando arquivos de estatisticas
    for fileSTATS in filesSTATS:
      dicStats = ast.literal_eval(readFile(fileSTATS))
      listStats.append(dicStats)

    #coletando os tablesID do GT para comparacao
    filesGTInfo = GTPath + dirLab
    filesInfo = getFiles(filesGTInfo, "info")

    #print("filesInfo ", filesInfo)

    for fileInfo in filesInfo:
      arrFile = fileInfo.split("/")
      tableId = arrFile[len(arrFile)-1].split("|")[0]
      lstTableIdGT.append(dirLab+"|"+tableId)

  lstErros = []

  for item in lstTableIdGT:

    #print(item)
    lab = item.split("|")[0]
    tableId = item.split("|")[1]

    if not strInList(listStats, tableId): #não encontrou, adicionar ao erro

      print("Não encontrou TABLEID ", tableId, ", adicionando....")
      #print("GTPath ", GTPath)
      #print("lab ", lab)
      #print("tableId ", tableId)
      #print("fileInfo ", fileInfo)

      #coletando informacoes do statsInfo
      #print("parametros a carregar na funcao getFileByPrefix", GTPath + lab, tableId+"|", "info")
      fileInfo = getFileByPrefix(GTPath + lab, tableId+"|", "info")
      #pathInfo = GTPath + dirLab + fileInfo
      lstFile = [fileInfo]
      #print("parametros a carregar na funcao getListTablesInfo", GTPath + lab + "/", lstFile)
      lstInfo = getListTablesInfo(GTPath + lab + "/", lstFile)

      dicTableStats = {}
      dicTableStats["LAB"] = lstInfo[0]["LAB"]
      dicTableStats["FILE"] = lstInfo[0]["FILE"]
      dicTableStats["PAGE"] = lstInfo[0]["PAGE"]
      dicTableStats["TABLEID"] = lstInfo[0]["TABLEID"]
      dicTableStats["DIMENSION"] = lstInfo[0]["DIMENSION"]
      dicTableStats["QTDCELLS"] = 0
      dicTableStats["QTDACERTOSCELLS"] = 0
      dicTableStats["PERCACERTOSCELLS"] = 0
      dicTableStats["PERCACERTOSBBOXINFO"] = 0
      dicTableStats["PERCACERTOSBBOX"] = 0
      dicTableStats["QTDNAOLIDOSBBOX"] = 0
      dicTableStats["PERCNAOLIDOSBBOX"] = 0
      dicTableStats["QTDNAOLIDOSTOKEN"] = 0
      dicTableStats["PERCNAOLIDOSTOKEN"] = 0
      dicTableStats["TEDS"] = 0

      lstErros.append(dicTableStats)

  lstTotal = listStats + lstErros

  dtStats = pd.DataFrame(lstTotal)
  dtStatsORD = dtStats.sort_values(by=['LAB','FILE', 'PAGE'])
  #print(dtStatsORD)
  print("Arquivo de sumário gerado ", ToolPath + "Summary.xlsx")
  dtStatsORD.to_excel(ToolPath + "Summary.xlsx", index=False)  # index=False para não incluir o índice do DataFrame
  return dtStatsORD