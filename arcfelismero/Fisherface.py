import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog

image_dir = 'C:\\Users\\renat\\Desktop\\Allamvizsaga\\Allamvizsga\\arcfelismero\\images'
test_image_dir ='C:\\Users\\renat\\Desktop\\Allamvizsaga\\Allamvizsga\\arcfelismero\\testImages'

haarcascade_frontalface = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

def detect_face(input_img):
    # gép szürkités
    image= cv2.cvtColor(input_img,  cv2.COLOR_BGR2GRAY)
    face_cascade = haarcascade_frontalface
    face= face_cascade.detectMultiScale(image,scaleFactor=1.2, minNeighbors=5)
    if(len(face)==0):
        return-1,-1

    
    # koordinatak beállitása a kapott arcok alapján
    (x,y,w,h)=face[0]

    return image[y:y+w,x:x+h], face[0]

label_map = dict() 
def setLabelNames():
    training_images_dirs= os.listdir(image_dir)

        # fajlok bejárása
    for i, dir_name in enumerate(training_images_dirs):
        # nevek bealitasa
        label= dir_name
        label_map[label]=i
setLabelNames()

detected_faces=[]
faces_labels=[]
FisherFace_recognizer= cv2.face.FisherFaceRecognizer_create()

def train():
    def create_training_data(training_data_folder_path):
        training_images_dirs= os.listdir(training_data_folder_path)

        # fájlbejérás
        for dir_name in training_images_dirs:
            # cimkek beálitása
            label= dir_name
            # fájl elérés elmentés
            training_image_path=training_data_folder_path+ "/"+ dir_name

            # tanito nevek beolvasasa 
            training_images_names= os.listdir(training_image_path)

            for image_name in training_images_names:
                    
                #minden kép eleresenek beallitasa 
                image_path=training_image_path+"/"+image_name

                #képek beolvasasa
                image=cv2.imread(image_path)
                    
                # arcok detektalasa
                face,rect=detect_face(image)

                # kép atmeretezes
                resized_face= cv2.resize(face, (300,341), interpolation= cv2.INTER_AREA) 

                # #arcok és hoza tartozo cimkek elmentése

                detected_faces.append(resized_face)
                faces_labels.append(int(label_map[label]))
    create_training_data(image_dir)


    def recognizer_trainer(recognizer):
        recognizer.train(detected_faces, np.array(faces_labels))

    recognizer_trainer(FisherFace_recognizer)
    print("Fisherface tanitas befejezodott")


def fisherFace():
    print("Fisherface Test Started")

    def draw_rectangle(test_image, rect):
            # kocka koordinatak
            x,y,w,h=rect

           #arcok és hoza tartozo cimkek elmentése
            cv2.rectangle(test_image, (x,y), (x+w, y+h), (65,65,65), 2)

    #cimkézés 
    def write_text(test_image, text_label,x,y):
        key=[k for k, v in label_map.items() if v == text_label]
        
        print("Felismert arc", key)
        cv2.putText(test_image, str(key), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,	(255,0,0),2)


    # becslés
    def predict(test_image):
        #arc detektálás
        detected_face,rect=detect_face(test_image)

        # átméretezés
        resized_test_image= cv2.resize(detected_face, (300,341), interpolation= cv2.INTER_AREA) 

        # becslés
        label= FisherFace_recognizer.predict(resized_test_image)

        #kocka kirajzolás
        draw_rectangle(test_image,rect)

        # cimke rátevés
        write_text(test_image,label[0], rect[0],rect[1]-5)
        return test_image, label[0]

    def fileDialog():
        filename = filedialog.askopenfilename(
            initialdir =  test_image_dir, title = "Select A File", filetype =(("jpeg files","*.jpg"),("all files","*.*")))
        return filename

    test_img=cv2.imread(fileDialog())

    pred_img, pred_label= predict(test_img)

    cv2.imshow('Fisherface teszt ablak', pred_img)
    cv2.waitKey(0)