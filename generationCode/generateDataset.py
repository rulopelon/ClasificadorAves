import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import matplotlib as plt
from matplotlib import cm
import scipy



def spectrogram(x,Fs=1.0,PLOT=False,L = None,low_band: bool = False):
        print('Starting __spectrogram__')
        numRowsPNG = 224
        L = 3
        nfft = 1024

        # Spectrogram estimation:
        Nx = x.size
        nsc = int(np.floor(Nx/(numRowsPNG/2)/L))  # segment samples
        x = x[0:int(nsc*numRowsPNG*L/2)]
        Nx = x.size
        nov = int(np.floor(nsc/2))   # overlaping samples
        hamming_wnd = np.hamming(nsc)
        sqrt_nfft = np.sqrt(nfft)

        S = np.zeros((numRowsPNG*L,nfft)) #prealocation
        iter = 0
        segment_range = range(0, Nx, nsc-nov)
        for k in segment_range:

            #X = np.np.fft.fftshift(np.fft.fft(x[k:k+nsc], n=self.nfft))/self.sqrt_nfft
            
            X = np.np.fft.fftshift(np.fft.fft(x[k:k+nsc], n=nfft*2))/sqrt_nfft/np.sqrt(2)
            #X=X[self.nfft:self.nfft*2]
            if low_band:
                X=X[0:nfft]
            else:   
                X=X[nfft:nfft*2]
            
            
            Pxx = 10*np.log10(np.real(X*np.conj(X)))
            #S.append(Pxx)
            S[iter,:] = Pxx
            iter = iter + 1
        
        print('Spectrogram computed')

        # Frequencies:
        f = np.np.np.fft.fftshift(np.fft.fftfreq(nfft, d=1/Fs))
        #f = f[0,:]
        t = np.array(segment_range)/Fs
        #t = t[0,:]


        if PLOT:
            # Spectrogram rendering:
            #plt.imshow(S.T, origin='lower')
            fig, ax = plt.subplots()
            #ax.pcolormesh(f, t, S, shading='flat', vmin=S.min(), vmax=S.max())
            ax.pcolormesh(f*1e-6, t,  S[:-1, :-1], shading='flat', vmin=S.min(), vmax=S.max())
            plt.xlabel('Freq. [MHz]')



            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            F,T = np.meshgrid(f, t)
            # Plot the surface.
            surf = ax.plot_surface(T,F*1e-6,S, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)
            plt.xlabel('Freq. [MHz]')
            plt.show()

        
        return S,t,f
def normalizeSpectrogramMatrix(S,refLevel=None,nBits = 8, PLOT:bool = False):

        Sv = np.reshape(S[:,:,:],(S.shape[0]*S.shape[1]*S.shape[2],1),order = 'F')
        n_bins = 30
        ind = (Sv!=np.inf) & (Sv!=-np.inf)
        print('Computing histogram')
        hist, bins = np.histogram(Sv[ind],bins = n_bins)
        noiseFloor = bins[np.where(hist == np.max(hist))[0]]
        maxi = np.max(Sv)

        if PLOT:
            #plot histogram hist bins
            plt.figure()
            plt.hist(Sv[ind],bins = n_bins)
            plt.show()
            #include a vertical line at noiseFloor
            plt.axvline(x=noiseFloor, color='r', linestyle='--')
            plt.show()


        print('Reference Level is ' + str(refLevel) + '; NoiseFloor is ' + str(noiseFloor) +'; Maximum value is ' + str(maxi))
        refLevel = maxi + 15.0
        maxi_norm = maxi*1.05
        mini_norm = noiseFloor + 5
        #mini_norm = noiseFloor -5


        maxis_block = np.amax(S,(0,1))  # get the maximum for each block
        S_norm = np.zeros(S.shape)  #preallocation
        S_norm = S.copy()
        
        
        S_norm[S_norm<mini_norm] = mini_norm  # clipping lower powers to mini
        S_norm = S_norm - mini_norm # moves the minimum to 0


#        S_norm = S_norm / np.max(S_norm)*((2**nBits)-1) # streches the maximum to the number of possible values
        maxis_block_norm = np.amax(S_norm,(0,1))  # get the maximum for each block
        S_norm = S_norm / maxis_block_norm*((2**nBits)-1) # streches the maximum to the number of possible values  TODO: esto creoq eu tamcpoo está bien porque habría que estirarlo respecto al nuevo máximo!(ya que has movido para arriba el mínmo)


        return S_norm, Sv,noiseFloor

def saveImage(C,nombre,wider=False):
    im = Image.fromarray(C)
    
    #dt = dt.strftime('%A %d-%m-%Y_%H-%M-%S')
    dt = dt.strftime('%A %d-%m-%Y_%H-%M-%S') + '.{0:02d}'.format(int(dt.microsecond / 10000))

 
    print('Saving image to {}'.format(nombre) )
    im.save(nombre) # save it removing the .tiq extension


# Definir la ruta de la carpeta de origen y la carpeta destino
carpeta_origen = 'datasetTotal'
carpeta_destino_train = 'training'
carpeta_destino_test = 'validation'

# Asegurarse de que el directorio destino existe, si no, crearlo
os.makedirs(carpeta_destino_test, exist_ok=True)
os.makedirs(carpeta_destino_train, exist_ok=True)

img_size = 224
num_imagenes_por_original = 20

# Definir las transformaciones para data augmentation
transformaciones = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(img_size),
    transforms.Grayscale()
    #transforms.Normalize(mean=[0.485], std=[0.229])
    # Añade aquí más transformaciones según sea necesario
])

# Cargar las imágenes utilizando ImageFolder
archivos = [f for f in os.listdir(carpeta_origen) if os.path.isfile(os.path.join(carpeta_origen, f))]

# Iterar sobre el dataset transformado y guardar las imágenes
for i,archivo in enumerate(archivos):
    mat = scipy.io.loadmat(os.path.join(carpeta_origen, archivo))
    signal =signal["signal"]

    print("Se ha cargado {}".format(os.path.join(carpeta_origen, archivo)))

    # Aplicar transformaciones de nuevo a la misma imagen
    #imagen_transformada = transformaciones(imagen)
    # Definir la ruta de archivo donde se guardará la imagen

    s = spectrogram(signal,Fs=50e6,PLOT=False,L =3)[0]

    s_norm = normalizeSpectrogramMatrix(s)[0]

    es_train = np.random.rand() < 0.8
    carpeta_destino = carpeta_destino_train if es_train else carpeta_destino_test
    
    # Definir la ruta de archivo para guardar la imagen transformada
    ruta_archivo = os.path.join(carpeta_destino, f'{os.path.splitext(archivo)[0]}-transformada_{i}-{os.path.splitext(archivo)[1]}')
    
    # Guardar la imagen transformada
    saveImage(s_norm, ruta_archivo)

print(f'Imágenes transformadas guardadas en {carpeta_destino}')