import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import optuna
import numpy as np
import scipy
import os
from tqdm import tqdm
import time
from PIL import Image
import torchvision
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directorioTrain = "training"
directorioTest = "validation"


class_to_index = {'Alder Flycatcher': 0,
 'American Avocet': 1,
 'American Bittern': 2,
 'American Crow': 3,
 'American Goldfinch': 4,
 'American Kestrel': 5,
 'Buff-bellied Pipit': 6,
 'American Redstart': 7,
 'American Robin': 8,
 'American Wigeon': 9,
 'American Woodcock': 10,
 'American Tree Sparrow': 11,
 "Anna's Hummingbird": 12,
 'Ash-throated Flycatcher': 13,
 "Baird's Sandpiper": 14,
 'Bald Eagle': 15,
 'Baltimore Oriole': 16,
 'Sand Martin': 17,
 'Barn Swallow': 18,
 'Black-and-white Warbler': 19,
 'Belted Kingfisher': 20,
 "Bell's Sparrow": 21,
 "Bewick's Wren": 22,
 'Black-billed Cuckoo': 23,
 'Black-billed Magpie': 24,
 'Blackburnian Warbler': 25,
 'Black-capped Chickadee': 26,
 'Black-chinned Hummingbird': 27,
 'Black-headed Grosbeak': 28,
 'Blackpoll Warbler': 29,
 'Black-throated Sparrow': 30,
 'Black Phoebe': 31,
 'Blue Grosbeak': 32,
 'Blue Jay': 33,
 'Brown-headed Cowbird': 34,
 'Bobolink': 35,
 "Bonaparte's Gull": 36,
 'Barred Owl': 37,
 "Brewer's Blackbird": 38,
 "Brewer's Sparrow": 39,
 'Brown Creeper': 40,
 'Brown Thrasher': 41,
 'Broad-tailed Hummingbird': 42,
 'Broad-winged Hawk': 43,
 'Black-throated Blue Warbler': 44,
 'Black-throated Green Warbler': 45,
 'Black-throated Grey Warbler': 46,
 'Bufflehead': 47,
 'Blue-grey Gnatcatcher': 48,
 'Blue-headed Vireo': 49,
 "Bullock's Oriole": 50,
 'American Bushtit': 51,
 'Blue-winged Warbler': 52,
 'Cactus Wren': 53,
 'California Gull': 54,
 'California Quail': 55,
 'Cape May Warbler': 56,
 'Canada Goose': 57,
 'Canada Warbler': 58,
 'Canyon Wren': 59,
 'Carolina Wren': 60,
 "Cassin's Finch": 61,
 'Caspian Tern': 62,
 "Cassin's Vireo": 63,
 'Cedar Waxwing': 64,
 'Chipping Sparrow': 65,
 'Chimney Swift': 66,
 'Chestnut-sided Warbler': 67,
 'Chukar Partridge': 68,
 "Clark's Nutcracker": 69,
 'American Cliff Swallow': 70,
 'Common Goldeneye': 71,
 'Common Grackle': 72,
 'Common Loon': 73,
 'Common Merganser': 74,
 'Common Nighthawk': 75,
 'Northern Raven': 76,
 'Common Redpoll': 77,
 'Common Tern': 78,
 'Common Yellowthroat': 79,
 "Cooper's Hawk": 80,
 "Costa's Hummingbird": 81,
 'California Scrub Jay': 82,
 'Dark-eyed Junco': 83,
 'Downy Woodpecker': 84,
 'American Dusky Flycatcher': 85,
 'Black-necked Grebe': 86,
 'Eastern Bluebird': 87,
 'Eastern Kingbird': 88,
 'Eastern Meadowlark': 89,
 'Eastern Phoebe': 90,
 'Eastern Towhee': 91,
 'Eastern Wood Pewee': 92,
 'Eurasian Collared Dove': 93,
 'Common Starling': 94,
 'Evening Grosbeak': 95,
 'Field Sparrow': 96,
 'Fish Crow': 97,
 'Red Fox Sparrow': 98,
 'Gadwall': 99,
 'Green-tailed Towhee': 100,
 'Eurasian Teal': 101,
 'Golden-crowned Kinglet': 102,
 'Golden-crowned Sparrow': 103,
 'Golden Eagle': 104,
 'Great Blue Heron': 105,
 'Great Crested Flycatcher': 106,
 'Great Egret': 107,
 'Greater Roadrunner': 108,
 'Greater Yellowlegs': 109,
 'Great Horned Owl': 110,
 'Green Heron': 111,
 'Great-tailed Grackle': 112,
 'Grey Catbird': 113,
 'American Grey Flycatcher': 114,
 'Hairy Woodpecker': 115,
 "Hammond's Flycatcher": 116,
 'European Herring Gull': 117,
 'Hermit Thrush': 118,
 'Hooded Merganser': 119,
 'Hooded Warbler': 120,
 'Horned Lark': 121,
 'House Finch': 122,
 'House Sparrow': 123,
 'House Wren': 124,
 'Indigo Bunting': 125,
 'Juniper Titmouse': 126,
 'Killdeer': 127,
 'Ladder-backed Woodpecker': 128,
 'Lark Sparrow': 129,
 'Lazuli Bunting': 130,
 'Least Bittern': 131,
 'Least Flycatcher': 132,
 'Least Sandpiper': 133,
 "LeConte's Thrasher": 134,
 'Lesser Goldfinch': 135,
 'Lesser Nighthawk': 136,
 'Lesser Yellowlegs': 137,
 "Lewis's Woodpecker": 138,
 "Lincoln's Sparrow": 139,
 'Long-billed Curlew': 140,
 'Long-billed Dowitcher': 141,
 'Loggerhead Shrike': 142,
 'Long-tailed Duck': 143,
 'Louisiana Waterthrush': 144,
 "MacGillivray's Warbler": 145,
 'Magnolia Warbler': 146,
 'Mallard': 147,
 'Marsh Wren': 148,
 'Merlin': 149,
 'Mountain Bluebird': 150,
 'Mountain Chickadee': 151,
 'Mourning Dove': 152,
 'Northern Cardinal': 153,
 'Northern Flicker': 154,
 'Northern Harrier': 155,
 'Northern Mockingbird': 156,
 'Northern Parula': 157,
 'Northern Pintail': 158,
 'Northern Shoveler': 159,
 'Northern Waterthrush': 160,
 'Northern Rough-winged Swallow': 161,
 "Nuttall's Woodpecker": 162,
 'Olive-sided Flycatcher': 163,
 'Orange-crowned Warbler': 164,
 'Western Osprey': 165,
 'Ovenbird': 166,
 'Palm Warbler': 167,
 'Pacific-slope Flycatcher': 168,
 'Pectoral Sandpiper': 169,
 'Peregrine Falcon': 170,
 'Phainopepla': 171,
 'Pied-billed Grebe': 172,
 'Pileated Woodpecker': 173,
 'Pine Grosbeak': 174,
 'Pinyon Jay': 175,
 'Pine Siskin': 176,
 'Pine Warbler': 177,
 'Plumbeous Vireo': 178,
 'Prairie Warbler': 179,
 'Purple Finch': 180,
 'Pygmy Nuthatch': 181,
 'Red-breasted Merganser': 182,
 'Red-breasted Nuthatch': 183,
 'Red-breasted Sapsucker': 184,
 'Red-bellied Woodpecker': 185,
 'Red Crossbill': 186,
 'Red-eyed Vireo': 187,
 'Red-necked Phalarope': 188,
 'Red-shouldered Hawk': 189,
 'Red-tailed Hawk': 190,
 'Red-winged Blackbird': 191,
 'Ring-billed Gull': 192,
 'Ring-necked Duck': 193,
 'Rose-breasted Grosbeak': 194,
 'Rock Dove': 195,
 'Rock Wren': 196,
 'Ruby-throated Hummingbird': 197,
 'Ruby-crowned Kinglet': 198,
 'Ruddy Duck': 199,
 'Ruffed Grouse': 200,
 'Rufous Hummingbird': 201,
 'Rusty Blackbird': 202,
 'Sagebrush Sparrow': 203,
 'Sage Thrasher': 204,
 'Savannah Sparrow': 205,
 "Say's Phoebe": 206,
 'Scarlet Tanager': 207,
 "Scott's Oriole": 208,
 'Semipalmated Plover': 209,
 'Semipalmated Sandpiper': 210,
 'Short-eared Owl': 211,
 'Sharp-shinned Hawk': 212,
 'Snow Bunting': 213,
 'Snow Goose': 214,
 'Solitary Sandpiper': 215,
 'Song Sparrow': 216,
 'Sora': 217,
 'Spotted Sandpiper': 218,
 'Spotted Towhee': 219,
 "Steller's Jay": 220,
 "Swainson's Hawk": 221,
 'Swamp Sparrow': 222,
 "Swainson's Thrush": 223,
 'Tree Swallow': 224,
 'Trumpeter Swan': 225,
 'Tufted Titmouse': 226,
 'Tundra Swan': 227,
 'Veery': 228,
 'Vesper Sparrow': 229,
 'Violet-green Swallow': 230,
 'Warbling Vireo': 231,
 'Western Bluebird': 232,
 'Western Grebe': 233,
 'Western Kingbird': 234,
 'Western Meadowlark': 235,
 'Western Sandpiper': 236,
 'Western Tanager': 237,
 'Western Wood Pewee': 238,
 'White-breasted Nuthatch': 239,
 'White-crowned Sparrow': 240,
 'White-faced Ibis': 241,
 'White-throated Sparrow': 242,
 'White-throated Swift': 243,
 'Willow Flycatcher': 244,
 "Wilson's Snipe": 245,
 'Wild Turkey': 246,
 'Winter Wren': 247,
 "Wilson's Warbler": 248,
 'Wood Duck': 249,
 "Woodhouse's Scrub Jay": 250,
 'Wood Thrush': 251,
 'American Coot': 252,
 'Yellow-bellied Flycatcher': 253,
 'Yellow-bellied Sapsucker': 254,
 'Yellow-headed Blackbird': 255,
 'Mangrove Warbler': 256,
 'Myrtle Warbler': 257,
 'Yellow-throated Vireo': 258}


n_classes = len(class_to_index.values())

columnas_cargar = ["spectral_centroid","zero_crossing_rate","spectral_rolloff","tempo","spectral_bandwidth"]
class CustomDataset(Dataset):
    """Dataset para cargar archivos .jpg."""
    
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directorio con todos los archivos .mat.
            transform (callable, optional): Opcional transformación a ser aplicada
                en una muestra.
        """
        self.directory = directory
        self.transform = transform
        self.files = [f for f in os.listdir(directory) if f.endswith('.png')]
        self.filenames = os.listdir(directory)
        self.class_to_index = class_to_index
        self.classes = n_classes
        self.df = pd.read_csv('dataFrameTotal.csv')
        
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        filename = self.filenames[idx]

        class_name = filename.split('.')[0]
        class_name = self.getClass(class_name)
        # Convertir la etiqueta de clase a un índice
        class_idx = self.class_to_index[class_name]
       
        label_one_hot = torch.zeros(self.classes)
        label_one_hot[class_idx] = 1
        image = Image.open(os.path.join(self.directory,filename))
        convertir_a_tensor =  transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convertir a escala de grises
    transforms.Resize(size=(224,224)),
    transforms.ToTensor() # Convertir a tensor
])

        image = convertir_a_tensor(image)
        features = list(self.load_features(filename.split('.')[0]))
        features = torch.tensor(features)
        return [image,features],label_one_hot
    
    def load_features(self,filename):
        filename = filename+".mp3"
        # Filtrar el DataFrame para obtener solo la fila con el nombre de fichero especificado
        features_row = self.df[self.df['filename'] == filename]
        if not features_row.empty:
            # Retornar la fila sin la columna de 'filename'
            return features_row[columnas_cargar].iloc[0].to_dict().values()
        else:
            return "No se encontró el fichero especificado."

    def getClass(self,clase):
        filename = clase+".mp3"
        features_row = self.df[self.df['filename'] == filename]
        if not features_row.empty:
            # Retornar la fila sin la columna de 'filename'
            return features_row["species"].array[0]
        else:
            return "No se encontró el fichero especificado."

class EarlyStopper:
    def __init__(self, patience=30, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        # TODO: La epoch anterior menos la actual tien que ser menor que el delta 0.0001 paciencia 0
        return False
    
class SimpleCNN(nn.Module):
    def __init__(self,base_model_spectrogram,n_layers_spectrogram,n_classes,unfreezed_layers_spectrogram,list_dropouts_spectrogram,list_neuronas_salida_spectrogram,n_layers_dense,in_features_dense,list_dropouts_dense,list_neuronas_salida_dense,n_layers_combinacion,list_dropouts_combinacion,list_neuronas_salida_combinacion):
        super(SimpleCNN, self).__init__()
        self.earlyStopper = EarlyStopper()
        
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
        
        self.in_features_combinacion =list_neuronas_salida_dense[-1]+list_neuronas_salida_spectrogram[-1]

        self.combinacion = nn.Sequential()

        for i in range(1, self.n_layers_combinacion-1):
            if i== 1:
                self.dense_network.add_module(f'dp_combinacion{i}',nn.Dropout(list_dropouts_combinacion[i]))
                self.dense_network.add_module(f'fc_combinacion{i}', nn.Linear(self.in_features_combinacion, list_neuronas_salida_combinacion[i]))
                self.dense_network.add_module(f'relu_combinacion{i}',nn.ReLU())
                
            else:

                self.dense_network.add_module(f'dp_combinacion{i}',nn.Dropout(list_dropouts_combinacion[i]))
                self.dense_network.add_module(f'fc_combinacion{i}', nn.Linear(list_neuronas_salida_combinacion[i-1], list_neuronas_salida_combinacion[i]))
                self.dense_network.add_module(f'relu_combinacion{i}',nn.ReLU())

        
        # Add a new softmax output layer
        self.combinacion.add_module(f'dp{i}',nn.Dropout(list_dropouts_combinacion[-1]))
        self.combinacion.add_module(f'fc{i}', nn.Linear(list_neuronas_salida_combinacion[-2], self.num_classes))
        self.combinacion.add_module(f'relu{i}',nn.Softmax(dim=1))

        

    

    def forward(self, spectrogram,audio_features):

        spectrogram_features = self.base_model(spectrogram)
        audio_features = self.dense_network(audio_features)


        # Concatenación y fusión de características
        combined_features = torch.cat((spectrogram_features, audio_features), dim=1)
        fused_features = torch.relu(self.fusion_layer(combined_features))    

        x = self.combinacion(fused_features)

        return x
    
    
def train_and_validate(model, device, train_loader, val_loader, epochs, optimizer,trial):
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_accuracy': [], 'valid_loss': [], 'valid_accuracy': []}
    print("Empieza el entrenamiento")

    creado = False
    for epoch in tqdm(range(epochs)):
        model.train()
        training_losses = []
        tiempo_comienzo = time.time()
        correct_training = 0
        train_loss = 0.0
        train_accuracy = 0.0

        tiempo_inicial = time.time()
        for data, labels in train_loader:
            images = data[0]
            features = data[1]

            tiempo_actual = time.time()
            tiempo_inicial =tiempo_actual
            images = images.to(device)
            features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            try:
                outputs = model(images,features)
            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                print("Combinación incorrecta")
                raise optuna.exceptions.TrialPruned
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()


        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_accuracy = train_accuracy*100
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Train Loss: {train_loss:.4f}, '
                f'Train Accuracy: {train_accuracy:.4f}')

        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            valid_accuracy += (outputs.argmax(1) == labels.argmax(1)).sum().item()

        valid_loss /= len(val_loader)
        valid_accuracy /= len(val_loader.dataset)
        valid_accuracy = valid_accuracy*100
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)

        print(f'Epoch {epoch + 1}/{epochs} - '
                f'Validation Loss: {valid_loss:.4f}, '
                f'Validation Accuracy: {valid_accuracy:.4f}')


        trial.report(valid_accuracy,step = epoch)
        if trial.should_prune():
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        if model.earlyStopper.early_stop(valid_loss):
            print("Se ha hecho early stopping")
            model_file = f"model_{trial.number}.pth"
            torch.save(model.state_dict(), model_file)
            #wandb.save(model_file)
            return valid_accuracy
            #validation_accuracy = input("Dime la precisión")
            #trial.report(validation_accuracy,epoch+1)
            #raise optuna.exceptions.TrialPruned()
        
    # UNa vez que ha terminado de entrenar, lo guarda
    model_file = f"model_{trial.number}.pth"
    torch.save(model.state_dict(), model_file)
    #wandb.save(model_file)

    return valid_accuracy

def objective(trial):



    # Hiperparámetros a optimizar
    n_layers_spectrogram = trial.suggest_int("n_layers_spectrogram",5,50)
    n_layers_dense = trial.suggest_int("n_layers_dense",5,50)
    n_layers_combinacion = trial.suggest_int("n_layers_combinacion",5,50)

    lr = trial.suggest_float("lr", 1e-5, 1e-1,log = True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    unfreezed_layers_spectrogram = trial.suggest_int("unfreezed_layers_spectrogram",1,5)
    # Eligiendo el número de capas internas
    list_neuronas_salida_spectrogram = []
    list_dropouts_spectrogram = []

    in_features_dense = len(columnas_cargar)
    # Eligiendo los parámetros de la dense
    list_dropouts_dense = []
    list_neuronas_salida_dense = []

    # Parámetros de la combinacion
    list_dropouts_combinacion = []
    list_neuronas_salida_combinacion = []

    base_model_spectrogram = torchvision.models.mobilenet_v2(weights='DEFAULT')

    for i in range(n_layers_spectrogram):
        # Dropout
        dropout = trial.suggest_float("dropout_spectrogram"+str(i),0,0.5)
        list_dropouts_spectrogram.append(dropout)
        #
        neuronas_spectrogram = trial.suggest_int("neuronas_spectrogram"+str(i),0,50)
        list_neuronas_salida_spectrogram.append(neuronas_spectrogram)

    for i in range(n_layers_dense):
        # Dropout
        dropout_dense = trial.suggest_float("dropout_dense"+str(i),0,0.5)
        list_dropouts_dense.append(dropout_dense)
        #
        neuronas_dense = trial.suggest_int("neuronas_dense"+str(i),0,50)
        list_neuronas_salida_dense.append(neuronas_dense)
    
    for i in range(n_layers_combinacion):
        # Dropout
        dropout_combinacion = trial.suggest_float("dropout_combinacion"+str(i),0,0.5)
        list_dropouts_combinacion.append(dropout_combinacion)
        #
        neuronas_combinacion = trial.suggest_int("neuronas_combinacion"+str(i),0,50)
        list_neuronas_salida_combinacion.append(neuronas_combinacion)
        
    try:
        model = SimpleCNN(base_model_spectrogram,n_layers_spectrogram,n_classes,unfreezed_layers_spectrogram,list_dropouts_spectrogram,list_neuronas_salida_spectrogram,n_layers_dense,in_features_dense,list_dropouts_dense,list_neuronas_salida_dense,n_layers_combinacion,list_dropouts_combinacion,list_neuronas_salida_combinacion).to(device)
        model = model.to(device)
    except Exception as e:
        print(e)

        print("Demasiado grande")
        torch.cuda.empty_cache()
        raise optuna.exceptions.TrialPruned()
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    """
    DATA AUGMENTATION
    """

    
    train_dataset = CustomDataset(directory=directorioTrain)
    test_dataset = CustomDataset(directory=directorioTest)

    # Datos de entrenamiento y validación
    batch_size = 64 #TODO Aumentar batch size 128 meter en optuna
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size, shuffle=True,num_workers=2)

    val_loader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=True,num_workers=2)

    epochs = 1000  # Puedes ajustar esto según sea necesario
   
    accuracy = train_and_validate(model, device, train_loader, val_loader, epochs, optimizer,trial)

    return accuracy


                                  
if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")# good solution !!!!

    study = optuna.create_study(direction="maximize")
    #study = optuna.load_study(study_name="DatasetImagenesV2", storage="mysql://root:T3mp0r4l@192.168.253.159/optimizacion")
        
    #study.optimize(objective, n_trials=100)

    study.optimize(objective, n_trials=10000)  # Ajusta el número de trials según sea necesario

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")









