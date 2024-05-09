
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import pandas as pd

class_to_index = {
    0: 'Alder Flycatcher',
    1: 'American Avocet',
    2: 'American Bittern',
    3: 'American Crow',
    4: 'American Goldfinch',
    5: 'American Kestrel',
    6: 'Buff-bellied Pipit',
    7: 'American Redstart',
    8: 'American Robin',
    9: 'American Wigeon',
    10: 'American Woodcock',
    11: 'American Tree Sparrow',
    12: "Anna's Hummingbird",
    13: 'Ash-throated Flycatcher',
    14: "Baird's Sandpiper",
    15: 'Bald Eagle',
    16: 'Baltimore Oriole',
    17: 'Sand Martin',
    18: 'Barn Swallow',
    19: 'Black-and-white Warbler',
    20: 'Belted Kingfisher',
    21: "Bell's Sparrow",
    22: "Bewick's Wren",
    23: 'Black-billed Cuckoo',
    24: 'Black-billed Magpie',
    25: 'Blackburnian Warbler',
    26: 'Black-capped Chickadee',
    27: 'Black-chinned Hummingbird',
    28: 'Black-headed Grosbeak',
    29: 'Blackpoll Warbler',
    30: 'Black-throated Sparrow',
    31: 'Black Phoebe',
    32: 'Blue Grosbeak',
    33: 'Blue Jay',
    34: 'Brown-headed Cowbird',
    35: 'Bobolink',
    36: "Bonaparte's Gull",
    37: 'Barred Owl',
    38: "Brewer's Blackbird",
    39: "Brewer's Sparrow",
    40: 'Brown Creeper',
    41: 'Brown Thrasher',
    42: 'Broad-tailed Hummingbird',
    43: 'Broad-winged Hawk',
    44: 'Black-throated Blue Warbler',
    45: 'Black-throated Green Warbler',
    46: 'Black-throated Grey Warbler',
    47: 'Bufflehead',
    48: 'Blue-grey Gnatcatcher',
    49: 'Blue-headed Vireo',
    50: "Bullock's Oriole",
    51: 'American Bushtit',
    52: 'Blue-winged Warbler',
    53: 'Cactus Wren',
    54: 'California Gull',
    55: 'California Quail',
    56: 'Cape May Warbler',
    57: 'Canada Goose',
    58: 'Canada Warbler',
    59: 'Canyon Wren',
    60: 'Carolina Wren',
    61: "Cassin's Finch",
    62: 'Caspian Tern',
    63: "Cassin's Vireo",
    64: 'Cedar Waxwing',
    65: 'Chipping Sparrow',
    66: 'Chimney Swift',
    67: 'Chestnut-sided Warbler',
    68: 'Chukar Partridge',
    69: "Clark's Nutcracker",
    70: 'American Cliff Swallow',
    71: 'Common Goldeneye',
    72: 'Common Grackle',
    73: 'Common Loon',
    74: 'Common Merganser',
    75: 'Common Nighthawk',
    76: 'Northern Raven',
    77: 'Common Redpoll',
    78: 'Common Tern',
    79: 'Common Yellowthroat',
    80: "Cooper's Hawk",
    81: "Costa's Hummingbird",
    82: 'California Scrub Jay',
    83: 'Dark-eyed Junco',
    84: 'Downy Woodpecker',
    85: 'American Dusky Flycatcher',
    86: 'Black-necked Grebe',
    87: 'Eastern Bluebird',
    88: 'Eastern Kingbird',
    89: 'Eastern Meadowlark',
    90: 'Eastern Phoebe',
    91: 'Eastern Towhee',
    92: 'Eastern Wood Pewee',
    93: 'Eurasian Collared Dove',
    94: 'Common Starling',
    95: 'Evening Grosbeak',
    96: 'Field Sparrow',
    97: 'Fish Crow',
    98: 'Red Fox Sparrow',
    99: 'Gadwall',
    100: 'Green-tailed Towhee',
    101: 'Eurasian Teal',
    102: 'Golden-crowned Kinglet',
    103: 'Golden-crowned Sparrow',
    104: 'Golden Eagle',
    105: 'Great Blue Heron',
    106: 'Great Crested Flycatcher',
    107: 'Great Egret',
    108: 'Greater Roadrunner',
    109: 'Greater Yellowlegs',
    110: 'Great Horned Owl',
    111: 'Green Heron',
    112: 'Great-tailed Grackle',
    113: 'Grey Catbird',
    114: 'American Grey Flycatcher',
    115: 'Hairy Woodpecker',
    116: "Hammond's Flycatcher",
    117: 'European Herring Gull',
    118: 'Hermit Thrush',
    119: 'Hooded Merganser',
    120: 'Hooded Warbler',
    121: 'Horned Lark',
    122: 'House Finch',
    123: 'House Sparrow',
    124: 'House Wren',
    125: 'Indigo Bunting',
    126: 'Juniper Titmouse',
    127: 'Killdeer',
    128: 'Ladder-backed Woodpecker',
    129: 'Lark Sparrow',
    130: 'Lazuli Bunting',
    131: 'Least Bittern',
    132: 'Least Flycatcher',
    133: 'Least Sandpiper',
    134: "LeConte's Thrasher",
    135: 'Lesser Goldfinch',
    136: 'Lesser Nighthawk',
    137: 'Lesser Yellowlegs',
    138: "Lewis's Woodpecker",
    139: "Lincoln's Sparrow",
    140: 'Long-billed Curlew',
    141: 'Long-billed Dowitcher',
    142: 'Loggerhead Shrike',
    143: 'Long-tailed Duck',
    144: 'Louisiana Waterthrush',
    145: "MacGillivray's Warbler",
    146: 'Magnolia Warbler',
    147: 'Mallard',
    148: 'Marsh Wren',
    149: 'Merlin',
    150: 'Mountain Bluebird',
    151: 'Mountain Chickadee',
    152: 'Mourning Dove',
    153: 'Northern Cardinal',
    154: 'Northern Flicker',
    155: 'Northern Harrier',
    156: 'Northern Mockingbird',
    157: 'Northern Parula',
    158: 'Northern Pintail',
    159: 'Northern Shoveler',
    160: 'Northern Waterthrush',
    161: 'Northern Rough-winged Swallow',
    162: "Nuttall's Woodpecker",
    163: 'Olive-sided Flycatcher',
    164: 'Orange-crowned Warbler',
    165: 'Western Osprey',
    166: 'Ovenbird',
    167: 'Palm Warbler',
    168: 'Pacific-slope Flycatcher',
    169: 'Pectoral Sandpiper',
    170: 'Peregrine Falcon',
    171: 'Phainopepla',
    172: 'Pied-billed Grebe',
    173: 'Pileated Woodpecker',
    174: 'Pine Grosbeak',
    175: 'Pinyon Jay',
    176: 'Pine Siskin',
    177: 'Pine Warbler',
    178: 'Plumbeous Vireo',
    179: 'Prairie Warbler',
    180: 'Purple Finch',
    181: 'Pygmy Nuthatch',
    182: 'Red-breasted Merganser',
    183: 'Red-breasted Nuthatch',
    184: 'Red-breasted Sapsucker',
    185: 'Red-bellied Woodpecker',
    186: 'Red Crossbill',
    187: 'Red-eyed Vireo',
    188: 'Red-necked Phalarope',
    189: 'Red-shouldered Hawk',
    190: 'Red-tailed Hawk',
    191: 'Red-winged Blackbird',
    192: 'Ring-billed Gull',
    193: 'Ring-necked Duck',
    194: 'Rose-breasted Grosbeak',
    195: 'Rock Dove',
    196: 'Rock Wren',
    197: 'Ruby-throated Hummingbird',
    198: 'Ruby-crowned Kinglet',
    199: 'Ruddy Duck',
    200: 'Ruffed Grouse',
    201: 'Rufous Hummingbird',
    202: 'Rusty Blackbird',
    203: 'Sagebrush Sparrow',
    204: 'Sage Thrasher',
    205: 'Savannah Sparrow',
    206: "Say's Phoebe",
    207: 'Scarlet Tanager',
    208: "Scott's Oriole",
    209: 'Semipalmated Plover',
    210: 'Semipalmated Sandpiper',
    211: 'Short-eared Owl',
    212: 'Sharp-shinned Hawk',
    213: 'Snow Bunting',
    214: 'Snow Goose',
    215: 'Solitary Sandpiper',
    216: 'Song Sparrow',
    217: 'Sora',
    218: 'Spotted Sandpiper',
    219: 'Spotted Towhee',
    220: "Steller's Jay",
    221: "Swainson's Hawk",
    222: 'Swamp Sparrow',
    223: "Swainson's Thrush",
    224: 'Tree Swallow',
    225: 'Trumpeter Swan',
    226: 'Tufted Titmouse',
    227: 'Tundra Swan',
    228: 'Veery',
    229: 'Vesper Sparrow',
    230: 'Violet-green Swallow',
    231: 'Warbling Vireo',
    232: 'Western Bluebird',
    233: 'Western Grebe',
    234: 'Western Kingbird',
    235: 'Western Meadowlark',
    236: 'Western Sandpiper',
    237: 'Western Tanager',
    238: 'Western Wood Pewee',
    239: 'White-breasted Nuthatch',
    240: 'White-crowned Sparrow',
    241: 'White-faced Ibis',
    242: 'White-throated Sparrow',
    243: 'White-throated Swift',
    244: 'Willow Flycatcher',
    245: "Wilson's Snipe",
    246: 'Wild Turkey',
    247: 'Winter Wren',
    248: "Wilson's Warbler",
    249: 'Wood Duck',
    250: "Woodhouse's Scrub Jay",
    251: 'Wood Thrush',
    252: 'American Coot',
    253: 'Yellow-bellied Flycatcher',
    254: 'Yellow-bellied Sapsucker',
    255: 'Yellow-headed Blackbird',
    256: 'Mangrove Warbler',
    257: 'Myrtle Warbler',
    258: 'Yellow-throated Vireo'
}

img_size = 224

columnas_cargar = ["spectral_centroid","zero_crossing_rate","spectral_rolloff","tempo","spectral_bandwidth"]
df = pd.read_csv(os.path.join(os.getcwd(),'dataframeTotal.csv'))
n_classes = len(class_to_index.values())

class SimpleCNN(nn.Module):
    def __init__(self,base_model_spectrogram,n_layers_spectrogram,n_classes,unfreezed_layers_spectrogram,list_dropouts_spectrogram,list_neuronas_salida_spectrogram,n_layers_dense,in_features_dense,list_dropouts_dense,list_neuronas_salida_dense,n_layers_combinacion,list_dropouts_combinacion,list_neuronas_salida_combinacion):
        super(SimpleCNN, self).__init__()
        
        self.base_model = base_model_spectrogram
        self.num_classes = n_classes

        self.n_layers_spectrogram = n_layers_spectrogram
        self.n_layers_dense = n_layers_dense
        self.n_layers_combinacion = n_layers_combinacion
        self.in_features = self.base_model.classifier[1].in_features

        self.in_features_dense = in_features_dense


        # Freeze convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of layers
        for name, child in self.base_model.features.named_children():
            if int(name) >= unfreezed_layers_spectrogram:  # Descongelar capas a partir del bloque 14
                for param in child.parameters():
                    param.requires_grad = True

        new_classifier = nn.Sequential()

        for i in range(1, self.n_layers_spectrogram-1):
            if i== 1:
                new_classifier.add_module(f'dp{i}',nn.Dropout(list_dropouts_spectrogram[i]))
                new_classifier.add_module(f'fc{i}', nn.Linear(self.in_features, list_neuronas_salida_spectrogram[i]))
                new_classifier.add_module(f'relu{i}',nn.ReLU())
                
            else:

                new_classifier.add_module(f'dp{i}',nn.Dropout(list_dropouts_spectrogram[i]))
                new_classifier.add_module(f'fc{i}', nn.Linear(list_neuronas_salida_spectrogram[i-1], list_neuronas_salida_spectrogram[i]))
                new_classifier.add_module(f'relu{i}',nn.ReLU())
        
        self.base_model.classifier = new_classifier

        self.dense_network = nn.Sequential()

        for i in range(1, self.n_layers_dense):
            if i== 1:
                self.dense_network.add_module(f'dp_dense{i}',nn.Dropout(list_dropouts_dense[i]))
                self.dense_network.add_module(f'fc_dense{i}', nn.Linear(self.in_features_dense, list_neuronas_salida_dense[i]))
                self.dense_network.add_module(f'relu_dense{i}',nn.ReLU())
                
            else:

                self.dense_network.add_module(f'dp_dense{i}',nn.Dropout(list_dropouts_dense[i]))
                self.dense_network.add_module(f'fc_dense{i}', nn.Linear(list_neuronas_salida_dense[i-1], list_neuronas_salida_dense[i]))
                self.dense_network.add_module(f'relu_dense{i}',nn.ReLU())
        
        self.in_features_combinacion =list_neuronas_salida_dense[-1]+list_neuronas_salida_spectrogram[-2]

        self.combinacion = nn.Sequential()

        for i in range(1, self.n_layers_combinacion-1):
            if i== 1:
                self.combinacion.add_module(f'dp_combinacion{i}',nn.Dropout(list_dropouts_combinacion[i]))
                self.combinacion.add_module(f'fc_combinacion{i}', nn.Linear(self.in_features_combinacion, list_neuronas_salida_combinacion[i]))
                self.combinacion.add_module(f'relu_combinacion{i}',nn.ReLU())
                
            else:

                self.combinacion.add_module(f'dp_combinacion{i}',nn.Dropout(list_dropouts_combinacion[i]))
                self.combinacion.add_module(f'fc_combinacion{i}', nn.Linear(list_neuronas_salida_combinacion[i-1], list_neuronas_salida_combinacion[i]))
                self.combinacion.add_module(f'relu_combinacion{i}',nn.ReLU())
                j = i

        
        # Add a new softmax output layer
        self.combinacion.add_module(f'dp_combinacion{j+1}',nn.Dropout(list_dropouts_combinacion[-1]))
        self.combinacion.add_module(f'fc_combinacion{j+1}', nn.Linear(list_neuronas_salida_combinacion[13], self.num_classes))
        self.combinacion.add_module(f'relu_combinacion{j+1}',nn.Softmax(dim=1))
    def forward(self, spectrogram,audio_features_in):

        spectrogram_features = self.base_model(spectrogram)
        audio_features = self.dense_network(audio_features_in)


        # Concatenación y fusión de características
        combined_features = torch.cat((spectrogram_features, audio_features), dim=1)

        x = self.combinacion(combined_features)

        return x

    
def load_model():
    #dict = torch.load("DeepLearning/model_48.pth",map_location=torch.device('cpu'))
    dict = torch.load("modeloFinalBirdClassfication.pth")

    unfreezed_layers_spectrogram = 5

    lr = 0.00041397640987495454
    optimizer_name = "RMSprop"

    list_dropouts_spectrogram = [0.4927614309704043, 0.16383169172809497, 0.04835096268498523, 0.3696410979867431, 0.03594983161132542, 0.44547142188464167, 0.03491330008924742, 0.18818815493924806, 0.49157365273819453, 0.33987397792315616, 0.189718251216213, 0.3743916220212043, 0.05577064700785128, 0.34762489149241116, 0.4976949281650799, 0.15413035206557485, 0.46049424758739643, 0.4504914779260535, 0.1727669849123345, 0.23468901656428215, 0.2835237189505787, 0.4300072288816658, 0.26572102549870774, 0.3871428665929948, 0.39750843644041495, 0.4201932021229188, 0.02892716018175634, 0.13149471753806802, 0.319456708744585, 0.0962759165636099, 0.474666551900224, 0.30818792373115084]
    list_neuronas_salida_spectrogram = [2, 4, 28, 50, 19, 46, 27, 11, 22, 43, 13, 2, 24, 28, 33, 9, 30, 7, 14, 3, 19, 33, 33, 23, 44, 11, 5, 16, 48, 1, 38, 19]

    list_dropouts_dense = [0.39179467520490324, 0.1432688345587423, 0.3228565811631759, 0.44514878164166366, 0.44232779044097364, 0.14875601601402577, 0.4309319465132682, 0.10662763792065688, 0.2542994350803, 0.1234975703010815, 0.37604310517678813, 0.47081094455884487, 0.12018858531954046, 0.16355871670704136, 0.06737823633636253, 0.46726299266583765, 0.05314026862372795, 0.3269493511438719, 0.05308113609364573, 0.2832621894613903, 0.29860810583093494, 0.08935903441691356, 0.48586433631142667, 0.14187773060700426, 0.43591945631100126]
    list_neuronas_salida_dense  = [9, 2, 6, 31, 40, 16, 7, 36, 1, 44, 5, 22, 37, 29, 7, 26, 19, 29, 30, 30, 18, 7, 34, 9, 22]

    list_dropouts_combinacion = [0.17708806667073623, 0.12019200276274344, 0.18910439828503617, 0.19173048633733564, 0.3218408262987587, 0.4257133895579787, 0.41220503386082935, 0.21979205092063958, 0.3340481025592274, 0.3263597946685228, 0.06528125512230426, 0.2997174769724678, 0.31183190620751444, 0.45154091269027946, 0.183914631430]
    list_neuronas_salida_combinacion  = [18, 36, 48, 29, 47, 19, 8, 32, 1, 18, 26, 3, 3, 27, 2, 26, 12, 26, 41]

    base_model_spectrogram = torchvision.models.mobilenet_v2(weights='DEFAULT')

    n_layers_spectrogram = 32
    n_layers_dense =25
    n_layers_combinacion = 15
    in_features_dense = len(columnas_cargar)

    model = SimpleCNN(base_model_spectrogram,n_layers_spectrogram,n_classes,unfreezed_layers_spectrogram,list_dropouts_spectrogram,list_neuronas_salida_spectrogram,n_layers_dense,in_features_dense,list_dropouts_dense,list_neuronas_salida_dense,n_layers_combinacion,list_dropouts_combinacion,list_neuronas_salida_combinacion)
    model = model

 
    dict = torch.load("modeloFinalBirdClassfication.pth")
    model.load_state_dict(dict,strict =False)
    model.eval()
    return model


def getFiles(files):
    files_finales = []
    for file in files:
        class_name = file.split('.')[0]
        class_name = class_name+".mp3"
        if df['filename'].isin([class_name]).any():

            files_finales.append(file)
        else:
            pass
    
    print("Total files: {}".format(len(files_finales)))
    return files_finales

def cargarArchivo(directory,filename):


    class_name = filename.split('.')[0]
    class_name = getClass(class_name)
    # Convertir la etiqueta de clase a un índice

    #class_idx = class_to_index[class_name]

    image = Image.open(os.path.join(directory,filename))
    convertir_a_tensor =  transforms.Compose([
transforms.Grayscale(num_output_channels=3), # Convertir a escala de grises
transforms.Resize(size=(224,224)),
transforms.ToTensor() # Convertir a tensor
])

    image = convertir_a_tensor(image)
    features = list(load_features(filename.split('.')[0]))
    features = torch.tensor(features)
    return [image,features],class_name


    

def load_features(filename):
    filename = filename+".mp3"
    # Filtrar el DataFrame para obtener solo la fila con el nombre de fichero especificado
    features_row = df[df['filename'] == filename]
    if not features_row.empty:
        # Retornar la fila sin la columna de 'filename'
        return features_row[columnas_cargar].iloc[0].to_dict().values()
    else:
        return "No se encontró el fichero especificado."

def getClass(clase):
    filename = clase+".mp3"
    features_row = df[df['filename'] == filename]
    if not features_row.empty:
        # Retornar la fila sin la columna de 'filename'
        return features_row["species"].array[0]
    else:
        return "No se encontró el fichero especificado."
        

model = load_model()



errors_by_class = {class_name: 0 for class_name in class_to_index.values()}
elements_by_class = {class_name: 0 for class_name in class_to_index.values()}


# Configura tu dataset de validación y DataLoader aquí
# Por ejemplo:
transformaciones = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor() # Convertir a tensor
    ])

    
carpeta_origen = "validation"
#carpeta_origen = "DeepLearning/datasetcompleto"


# Evalúa el modelo
# Cargar las imágenes utilizando ImageFolder
archivos = [f for f in os.listdir(carpeta_origen) if os.path.isfile(os.path.join(carpeta_origen, f))]

archivos = getFiles(archivos)
errores =0 
# Iterar sobre el dataset transformado y guardar las imágenes
for i,archivo in enumerate(archivos):
    datos,class_name = cargarArchivo(carpeta_origen,archivo)

    image = datos[0]
    features = datos[1]
    
    # Aplicar transformaciones de nuevo a la misma imagen
    
    input_tensor = image.unsqueeze(0)
    features = features.unsqueeze(0)

    outputs = model(input_tensor,features)
    salida = class_to_index[outputs.argmax(1).item()]
    elements_by_class[class_name] = elements_by_class[class_name]+1
    print("La clase de entrada era: {}, la predicha es: {}".format(class_name,salida))
    if salida== class_name:
        # Se actualiza el diccionario de errores y se pone un error 
        pass
    else:
        errors_by_class[class_name] = errors_by_class[class_name]+1
        errores = errores+1
result_dict = {key: (elements_by_class[key] -errors_by_class[key])*100 / elements_by_class[key] for key in errors_by_class if key in elements_by_class}
print("La precisión final es de: {}".format((len(archivos)-errores)/len(archivos)))

plt.figure(figsize=(10, 6))
plt.bar(result_dict.keys(), result_dict.values(), color='skyblue')
plt.xlabel('Clase')
plt.ylabel('Valor')
plt.title('Valor por Clase')
plt.xticks(rotation=45)
plt.savefig('distribucionError.png')  # Guardar el gráfico antes de mostrarlo
