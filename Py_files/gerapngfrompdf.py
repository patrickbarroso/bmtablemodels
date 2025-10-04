import os
from pdf2image import convert_from_path

def converter_pdfs_para_pngs(PASTA_PDFS, PASTA_SAIDA, dpi=300):
    """
    Converte todas as páginas de todos os PDFs em uma pasta para imagens PNG.
    
    Parâmetros:
    - PASTA_PDFS (str): Pasta onde estão os arquivos PDF.
    - PASTA_SAIDA (str): Pasta onde os PNGs serão salvos.
    - dpi (int): Resolução das imagens de saída.
    """
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    for nome_arquivo in os.listdir(PASTA_PDFS):
        if nome_arquivo.lower().endswith(".pdf"):
            caminho_pdf = os.path.join(PASTA_PDFS, nome_arquivo)
            nome_base = os.path.splitext(nome_arquivo)[0]

            print(f"Convertendo: {nome_arquivo}")
            try:
                paginas = convert_from_path(caminho_pdf, dpi=dpi)
                for i, pagina in enumerate(paginas, start=1):
                    nome_saida = f"{nome_base}_{i}.png"
                    caminho_saida = os.path.join(PASTA_SAIDA, nome_saida)
                    pagina.save(caminho_saida, "PNG")
                    print(f"  Página {i} salva como {nome_saida}")
            except Exception as e:
                print(f"[ERRO] Falha ao converter {nome_arquivo}: {e}")

converter_pdfs_para_pngs(
    PASTA_PDFS="/HOME/PDF",
    PASTA_SAIDA="/HOME/PNG"
)
