import re

def clean_list(lstData):
    cleaned_data = []
    
    for sublist in lstData:
        cleaned_sublist = []
        for item in sublist:
            # Remover colchetes e aspas simples
            item = item.replace("[", "").replace("]", "").replace("'", "")

            # Substituir vírgula decimal por ponto se for um número
            item = re.sub(r'(\d+),(\d+)', r'\1.\2', item)

            cleaned_sublist.append(item)
        cleaned_data.append(cleaned_sublist)
    
    return cleaned_data

# Exemplo de uso
lstData = [
    ['Procedime\'nto utilizado:', 'IT 48 0 Metodo cons[iste na calibracao por comparacao direta com um multimetro digital La Catric? To (Cc Ja+ric? Gto Itina Iica'],
    ['Condicoes ambientais:', '(lelisao elellica) 6 um Multinielio alCale (coneme eieliica)- temperatura: 15,1*C * 0,6PC umidade:56% + 6%'],
    ['Data da calibracao:', '05/09/23'],
    ['Tamanho:', "10,5"],
    ['Data da emissao;', '14/09/23']
]

cleaned_lstData = clean_list(lstData)
for row in cleaned_lstData:
    print(row)