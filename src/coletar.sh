#!/bin/bash

# Extensões que serão procuradas
EXTENSOES=("cu" "cpp" "h" "hpp")

# Separador solicitado
SEPARADOR="---------------------"

echo "Iniciando a varredura..."

# Loop para cada tipo de extensão
for ext in "${EXTENSOES[@]}"; do
    # Define o nome do arquivo de saída (ex: saida_py.txt)
    ARQUIVO_SAIDA="saida_${ext}.txt"
    
    # Limpa (ou cria) o arquivo de saída para garantir que esteja vazio no início
    > "$ARQUIVO_SAIDA"
    
    echo "Processando arquivos .$ext -> $ARQUIVO_SAIDA"

    # Comando find para buscar recursivamente
    # -print0 e read -d '' são usados para lidar com espaços em nomes de arquivos
    find . -type f -name "*.$ext" -print0 | while IFS= read -r -d '' arquivo; do
        
        # 1. Escreve o caminho do arquivo no topo
        echo "CAMINHO DO ARQUIVO: $arquivo" >> "$ARQUIVO_SAIDA"
        echo "" >> "$ARQUIVO_SAIDA" # Linha em branco por estética
        
        # 2. Despeja o conteúdo do script
        cat "$arquivo" >> "$ARQUIVO_SAIDA"
        
        # 3. Adiciona uma quebra de linha e o separador ao final
        echo -e "\n$SEPARADOR\n" >> "$ARQUIVO_SAIDA"
        
    done
done

echo "Concluído! Verifique os arquivos 'saida_EXTENSAO.txt' no diretório atual."
