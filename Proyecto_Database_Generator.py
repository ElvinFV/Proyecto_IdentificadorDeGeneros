
import json
import os
import math
import librosa

DATASET_PATH = "BaseDatos" #Ubicación de la base de datos
JSON_PATH = "database_10s.json" #Ubicación y nombre el archivo json
SAMPLE_RATE = 22050
TRACK_DURATION = 30 #En segundos
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
Segments=3 #30/Segments 

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=Segments):

    # Organización diccionario
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Cambio entre sub-folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Asegurandonos de no estar en la carpeta base
        if dirpath is not dataset_path:

            #Guardar los nombres de las carpetas en el mapeo
            folder_name = dirpath.split("/")[-1]
            data["mapping"].append(folder_name)
            print("\nProcesando: {}".format(folder_name))

            # Procesar todos los archivos dentro de la carpeta
            for f in filenames:

		#Archivo de audio
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # Dividir el archivo en pequeños segmentos de si mismo
                for d in range(num_segments):

                    # Calculo de los segmentos de un mismo archivo
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    #MFCC 
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                   
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
save_mfcc(DATASET_PATH, JSON_PATH, num_segments=Segments)