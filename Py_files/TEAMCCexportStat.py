from funcoesstat import *
from teds import *

#diretorio do YOLO ou TATR a gerar a estatistica (DIR MODELO = TATR OU YOLO)
model = "/home/Out/MODELO/"
#diretorio de referencia GT
GTDir = "/home/Certificados/Out/GT/"

ExportCSVSummary(model, GTDir)