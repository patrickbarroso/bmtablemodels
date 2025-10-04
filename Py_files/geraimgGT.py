import os
import ast
from PIL import Image
import glob

def processa_info_e_salva_tabela(PASTA_LABORATORIO, PASTA_SAIDA, INFO_PATH):
    """
    Processa um arquivo .info contendo informações de uma tabela e salva a imagem recortada da tabela.

    Parâmetros:
    - PASTA_LABORATORIO (str): Caminho para a pasta onde estão as imagens.
    - PASTA_SAIDA (str): Caminho para a pasta onde a imagem recortada será salva.
    - INFO_PATH (str): Caminho para o arquivo .info com as informações da tabela.
    """
    # Lê o conteúdo do arquivo .info
    with open(INFO_PATH, 'r', encoding='utf-8') as f:
        conteudo = f.read()

    # Converte o conteúdo de string para dicionário
    info_dict = ast.literal_eval(conteudo)

    # Extrai os parâmetros necessários
    file_id = info_dict['FILE']
    page_num = info_dict['PAGE']
    table_id = info_dict['TABLEID']
    bbox = info_dict['BBOX']  # [x1, y1, x2, y2]

    # Constrói o nome do arquivo de imagem
    nome_imagem = f"{file_id}_{page_num}.png"
    caminho_imagem = os.path.join(PASTA_LABORATORIO, nome_imagem)

    # Verifica se a imagem existe
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    # Abre a imagem e recorta com base na BBOX
    imagem = Image.open(caminho_imagem)
    x1, y1, x2, y2 = bbox
    imagem_crop = imagem.crop((x1, y1, x2, y2))

    # Garante que a pasta de saída exista
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    # Define o nome do arquivo de saída
    nome_saida = f"{table_id}_{file_id}_{page_num}.jpg"
    caminho_saida = os.path.join(PASTA_SAIDA, nome_saida)

    # Salva a imagem recortada
    imagem_crop.save(caminho_saida)
    print(f"Imagem recortada salva em: {caminho_saida}")


PASTA_LABORATORIO = "/home/Certificados/In/Labs"
PASTA_SAIDA = "/home/GT_IMG"
PASTA_INFOS = "/home/Out/Labs/info"

# Pega todos os arquivos .info da pasta
arquivos_info = glob.glob(os.path.join(PASTA_INFOS, "*.info"))

# Loop para processar cada arquivo .info
for INFO_PATH in arquivos_info:
    try:
        processa_info_e_salva_tabela(PASTA_LABORATORIO, PASTA_SAIDA, INFO_PATH)
    except Exception as e:
        print(f"[ERRO] Falha ao processar {INFO_PATH}: {e}")

processa_info_e_salva_tabela(PASTA_LABORATORIO, PASTA_SAIDA, INFO_PATH)