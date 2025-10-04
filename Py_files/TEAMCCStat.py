from funcoesstat import *
from teds import *

#gerando estatísticas

import os
import ast
import pandas as pd

#diretorio do modelo a gerar a estatistica
modelFolder = "/home/Out/modelo/" 
#diretorio de referencia GT
GTDir = "/home//Out/GT/"

#coletando os diretorios dos laboratorios

dirLabs = ['LAB_01', 'LAB_02', 'LAB_03', 'LAB_04', 'LAB_05', 'LAB_06', 'LAB_07', 'LAB_08',
'LAB_09', 'LAB_10', 'LAB_11', 'LAB_12']

for dirLab in dirLabs:

  if not os.path.isfile(dirLab):
    labPath =  modelFolder + dirLab
    labPathGT =  GTDir + dirLab
    #coleta os arquivos info para analise dos tablesID
    filesINFO = getFiles(labPath, "info")
    #coleta os tablesID para comparacao
    lstInfoFiles = getInfoFiles(filesINFO)

    #para cada tableID, gerar estatísticas
    for dicInfoFile in lstInfoFiles:

      print("dicInfoFile", dicInfoFile)

      fileInfo = dicInfoFile["TABLEID"] + "|" + dicInfoFile["FILE"] + "|" + dicInfoFile["PAGE"] + "_INFO.info"
      fileTokenBbox = dicInfoFile["TABLEID"] + "|" + dicInfoFile["FILE"] + "|" + dicInfoFile["PAGE"] + "_BBOX.bbox"
      fileHTML = dicInfoFile["TABLEID"] + "|" + dicInfoFile["FILE"] + "|" + dicInfoFile["PAGE"] + "_HTML.html"

      pathFileInfo = labPath + "/" + fileInfo
      pathFileTokenBbox = labPath + "/" + fileTokenBbox
      pathFileHTML = labPath + "/" + fileHTML

      pathFileInfoGT = labPathGT + "/" + fileInfo
      pathFileTokenBboxGT = labPathGT + "/" + fileTokenBbox
      pathFileHTMLGT = labPathGT + "/" + fileHTML
      print("pathFileInfo =", pathFileInfo)

      #gerando estatisticas do token e bbox
      if os.path.exists(pathFileHTMLGT) and os.path.exists(pathFileTokenBbox) and os.path.exists(pathFileTokenBboxGT):

        #print("pathFileHTML", readFile(pathFileHTML).replace("\n", " "))
        #print("pathFileHTMLGT", readFile(pathFileHTMLGT).replace("\n", " "))

        import re
        #carrega lista de tokens/bbox e HTMLs para comparacao do arquivo corrente com GT
        def substituir_aspas(input_string):
          # Expressão regular para encontrar padrões que começam com "" e terminam com "
          pattern = r'""(.*?)"'
          
          # Substitui as aspas duplas iniciais por aspas simples
          output_string = re.sub(pattern, r'"\1', input_string)
          
          return output_string

        def adicionar_aspas(input_string):
          # Expressão regular para encontrar padrões que começam com uma aspa e terminam sem outra aspa no final
          pattern = r'(^|(?<=\s))"([^"]+)(?=\s|$)'

          # Substitui as palavras que têm apenas uma aspa no início, colocando aspas no final também
          output_string = re.sub(pattern, r'"\2"', input_string)
          
          return output_string

        #arquivo info analise (.info)
        stringFileInfo = readFile(pathFileInfo)
        #stringFileInfo = stringFileInfo.replace("'", '"')
        stringFileInfo = stringFileInfo.replace("null", "None")
        stringFileInfo = stringFileInfo.replace("nan", "'NAN'")
        stringFileInfo = stringFileInfo.replace("np.float64('NAN')", '"NaN"')
        #stringFileInfo = substituir_aspas(stringFileInfo)
        #stringFileInfo = adicionar_aspas(stringFileInfo)

        #print("stringFileInfo = ", stringFileInfo)
        try:
          lstInfo = ast.literal_eval(stringFileInfo) 
        except Exception as e:
          print(f"Erro ao carregar literal_eval: {e}")
          lstInfo = stringFileInfo
          sys.exit()
          continue
        
        
        #lstInfo = json.loads(stringFileInfo)
        
        #arquivo info GT (.info)
        lstInfoGT = ast.literal_eval(readFile(pathFileInfoGT))
        #arquivo bbox (.bbox)
        #print("pathFileTokenBbox", pathFileTokenBbox)
        lstTkBox = ast.literal_eval(readFile(pathFileTokenBbox))
        #arquivo bbox GT (.bbox)
        lstTkBoxGT = ast.literal_eval(readFile(pathFileTokenBboxGT))
        #arquivo html (.html)
        #print("pathFileHTML ", pathFileHTML)
        strHTML = "".join(ast.literal_eval(readFile(pathFileHTML).replace("\n", " ")))
        #arquivo html GT(.html)
        strHTMLGT = "".join(ast.literal_eval(readFile(pathFileHTMLGT).replace("\n", " ")))
        #break


        #TEDS apenas funciona se tiver na estrutura html as tags html e body
        if "<body>" not in strHTML:
          strHTML = "<body>" + strHTML + "</body>"
        if "<html>" not in strHTML:
          strHTML = "<html>" + strHTML + "</html>"
        if "<body>" not in strHTMLGT:
          strHTMLGT = "<body>" + strHTMLGT + "</body>"
        if "<html>" not in strHTMLGT:
          strHTMLGT = "<html>" + strHTMLGT + "</html>"

        qtdLinhas = int(lstInfoGT["DIMENSION"].split("X")[0])
        qtdColunas = int(lstInfoGT["DIMENSION"].split("X")[1])
        print("qtdLinhas ", qtdLinhas)
        print("qtdColunas ", qtdColunas)
        qtdCells = qtdLinhas * qtdColunas
        print("qtdCells ", qtdCells)

        #print("lstInfo" , lstInfo)
        #print("lstInfoGT" , lstInfoGT)
        #print("lstTkBox" , lstTkBox) #analisar depois os tokens
        #print("lstTkBoxGT" , lstTkBox) #analisar depois os tokens

        print("bboxInfo" , lstInfo["BBOX"])
        print("bboxInfoGT" , lstInfoGT["BBOX"])

        lstBboxInfo = calStatsBboxInfo(lstInfo, lstInfoGT)
        print("lstBboxInfo Stats" , lstBboxInfo)
        print("lstBbox Iou" , calculate_iou(lstInfo["BBOX"], lstInfoGT["BBOX"]))

        iouAcertosBboxInfo = round(calculate_iou(lstInfo["BBOX"], lstInfoGT["BBOX"])*100, 2)

        percAcertosBboxInfo = round((sum(lstBboxInfo) / len(lstBboxInfo))/100, 2)


        print("lstTkBox =",lstTkBox[0])
        print("lstTkBoxGT",lstTkBoxGT[0])
        lstStatsTokens, lstStatsBbox = calStatsTablesValues(lstTkBox, lstTkBoxGT)
        print("lstStatsBbox",lstStatsBbox[0])
        print("lstStatsTokens",lstStatsBbox[0])

        lstStatsIouBbox = calStatsIouBbox(lstTkBox, lstTkBoxGT)
        #print("lstStatsIouBbox ", lstStatsIouBbox)

        qtdNANToken, qtdNANBbox = calStatsNAN(lstTkBox)

        #calcula qtd de acertos (com similaridade 100% entre os valores dos tokens e tabelas)
        qtdAcertosTokens = numTimes(lstStatsTokens, 100.0)
        #qtdAcertosBbox = numTimes(lstStatsBbox, 100.0)
        percAcertosBbox = round((sum(lstStatsBbox) / len(lstStatsBbox))/100, 2)
        print("percAcertosBbox ", percAcertosBbox)

        iouAcertosBbox = round((sum(lstStatsIouBbox) / len(lstStatsBbox))/100, 2)

        #calcula similidade maior que 95%
        qtdTokensMaior95 = numTimesMoreThen(lstStatsTokens, 95)
        qtdBboxMaior95 = numTimesMoreThen(lstStatsBbox, 95)

        #calcula TEDS entre as estruturas HTMLs (img2table VS GT)
        teds = TEDS()

        scoreTEDS = round(teds.evaluate(strHTML, strHTMLGT), 6)

        #salvando a categoria de GAP da tabela
        catGAP = ', '.join(str(item) for item in lstInfoGT["CAT"]) if "CAT" in lstInfoGT else ""

        #salvado arquivo de estatística
        fileStats = dicInfoFile["TABLEID"] + "|" + dicInfoFile["FILE"] + "|" + dicInfoFile["PAGE"] + "_STATS.stats"
        pathFileStats = labPath + "/" + fileStats

        percAcertosTokens = round((qtdAcertosTokens / qtdCells), 2)
        percNaoLidosBbox = round((qtdNANBbox / qtdCells), 2)
        percNaoLidosToken = round((qtdNANToken / qtdCells), 2)

        dicTableStats = {}
        dicTableStats["LAB"] = lstInfo["LAB"]
        dicTableStats["FILE"] = dicInfoFile["FILE"]
        dicTableStats["PAGE"] = dicInfoFile["PAGE"]
        dicTableStats["TABLEID"] = dicInfoFile["TABLEID"]
        dicTableStats["DIMENSION"] = lstInfo["DIMENSION"]
        dicTableStats["QTDCELLS"] = qtdCells
        dicTableStats["QTDACERTOSCELLS"] = qtdAcertosTokens
        dicTableStats["PERCACERTOSCELLS"] = percAcertosTokens
        dicTableStats["PERCACERTOSBBOXINFO"] = percAcertosBboxInfo
        dicTableStats["IOUACERTOSBBOXINFO"] = iouAcertosBboxInfo
        dicTableStats["PERCACERTOSBBOX"] = percAcertosBbox
        dicTableStats["IOUACERTOSIOUBBOX"] = iouAcertosBbox
        dicTableStats["QTDNAOLIDOSBBOX"] = qtdNANBbox
        dicTableStats["PERCNAOLIDOSBBOX"] = percNaoLidosBbox
        dicTableStats["QTDNAOLIDOSTOKEN"] = qtdNANToken
        dicTableStats["PERCNAOLIDOSTOKEN"] = percNaoLidosToken
        #dicTableStats["QTDCELLSMAIOR95"] = qtdTokensMaior95
        #dicTableStats["QTDABBOXMAIOR95"] = qtdBboxMaior95
        dicTableStats["TEDS"] = scoreTEDS
        dicTableStats["CAT"] = catGAP

        SaveFileStats (pathFileStats, dicTableStats)

        print("Dimensão da tabela ", lstInfo["DIMENSION"], ", total ", str(qtdCells), " células" )
        print("qtdCells:",qtdCells)
        print("qtdAcertosTokens:",qtdAcertosTokens)
        print("percAcertosTokens:",(percAcertosTokens*100),"%")
        print("iouAcertosBboxInfo:",iouAcertosBboxInfo,"%")
        print("percAcertosBboxInfo:",percAcertosBboxInfo,"/",(percAcertosBboxInfo*100),"%")
        print("percAcertosBbox:",(percAcertosBbox*100),"%")
        print("iouAcertosBbox:",(iouAcertosBbox*100),"%")
        print("qtdNaoLidosBbox:",qtdNANBbox)
        print("percNaoLidosBbox:",(percNaoLidosBbox*100),"%")
        print("qtdNaoLidosTokens:",qtdNANToken)
        print("percNaoLidosToken:",(percNaoLidosToken*100),"%")
        #print("qtdAcertosTokens>95:",qtdTokensMaior95)
        #print("qtdAcertosBbox>95:",qtdBboxMaior95)
        print('TEDS score:', scoreTEDS,"/",round((scoreTEDS*100),2),"%")
        #break # fim primeiro for
        
        
  #break # fim segundo for
#depois de gerar arquivos de estatística, montar sumário no EXCEL
#ExportCSVSummary(pubTables, GTDir)