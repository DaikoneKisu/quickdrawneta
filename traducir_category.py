import random
from googletrans import Translator

# Obetner arreglo de categorias en el archivo categories.txt
def get_categories():
    # Abre el archivo category.txt en modo lectura
    with open('categories.txt', 'r') as file:
        # Lee todas las líneas del archivo y elimina los saltos de línea
        categories = [line.strip() for line in file]
    
    return categories

def traduction(random_category):
    # Crear una instancia del traductor
    translator = Translator()

    # Traducir el elemento aleatorio al español
    translation = translator.translate(random_category, src='en', dest='es')
    
    return translation

# Main
categories = get_categories()

# Selecciona un elemento aleatorio de la lista
random_category = random.choice(categories)

# Traducir el elemento aleatorio al español
translation = traduction(random_category)

# Imprime el elemento aleatorio seleccionado
print("Ingles: ", random_category)
print("Español: ", translation.text)