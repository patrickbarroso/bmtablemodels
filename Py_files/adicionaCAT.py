
import os
import json

def processar_arquivos_info(pasta):
    # Lista todos os arquivos com extensão .INFO na pasta
    arquivos_info = [f for f in os.listdir(pasta) if f.endswith('.info')]
    
    for arquivo in arquivos_info:
        caminho_arquivo = os.path.join(pasta, arquivo)
        
        try:
            # Ler o conteúdo do arquivo como string e corrigir formato inválido se necessário
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                conteudo_str = f.read()
                
            # Substituir aspas simples por aspas duplas, se necessário, para validar como JSON
            conteudo_str = conteudo_str.replace("'", '"')
            
            # Tentar carregar o JSON corrigido
            conteudo = json.loads(conteudo_str)
            
            # Adicionar a nova chave 'CAT'
            conteudo['CAT'] = ['SH']

            # Converter de volta para string com aspas simples
            conteudo_formatado = json.dumps(conteudo, indent=4)
            conteudo_formatado = conteudo_formatado.replace('\"', "'")
            
            # Salvar de volta no mesmo formato
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                json.dump(conteudo, f, indent=4)
            
            print(f"Processado: {arquivo}")
        except json.JSONDecodeError as e:
            print(f"Erro ao processar {arquivo}: JSON inválido - {e}")
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

# Caminho da pasta onde estão os arquivos INFO
pasta_info = "/home/labs/info/"

processar_arquivos_info(pasta_info)
