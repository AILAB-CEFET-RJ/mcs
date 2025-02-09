# Iterative binary dilation

import numpy as np

def iterative_dilation(array, max_value):
    rows, cols = array.shape
    output_array = np.copy(array)

    # Executando várias iterações, uma para cada valor
    for current_value in range(1, max_value):
        # Criar uma cópia para evitar modificação durante a iteração
        temp_array = np.copy(output_array)
        
        for x in range(rows):
            for y in range(cols):
                if output_array[x, y] == current_value:
                    # Lista de posições adjacentes para atualizar
                    neighbors = [
                        (x-1, y),   # Acima
                        (x, y-1),   # Esquerda
                        (x-1, y-1), # Diagonal superior esquerda
                        (x+1, y-1), # Diagonal inferior esquerda
                        (x+1, y),   # Abaixo
                        (x, y+1),   # Direita
                        (x-1, y+1), # Diagonal superior direita
                        (x+1, y+1)  # Diagonal inferior direita
                    ]
                    
                    # Aplicar atualizações se as posições são válidas e o valor atual é 0
                    for nx, ny in neighbors:
                        if 0 <= nx < rows and 0 <= ny < cols and temp_array[nx, ny] == 0:
                            temp_array[nx, ny] = current_value + 1
        
        # Atualizar o output_array com as mudanças da iteração atual
        output_array = temp_array

    return output_array

# Exemplo de array 2D
example_array = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

# Número máximo de valores a serem propagados
max_value = 5

# Aplicar a função de dilatação iterativa
dilated_array = iterative_dilation(example_array, max_value)

print("Array Original:")
print(example_array)
print("\nArray após Dilatação Iterativa:")
print(dilated_array)
