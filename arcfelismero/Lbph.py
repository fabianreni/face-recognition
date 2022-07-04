import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import argparse
import imutils

from skimage import feature

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
LBPHFace_recognizer= cv2.face.LBPHFaceRecognizer_create()

def train():
    def create_training_data(training_data_folder_path):
        # tanito kepek elerese
        training_images_dirs= os.listdir(training_data_folder_path)
        # fajlok bejárása
        for dir_name in training_images_dirs:
            # nevek bealitasa
            label= dir_name
            # eleres bealitasa 
            training_image_path=training_data_folder_path+ "/"+ dir_name
            # tanitasi nevek beolvasasa 
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
                #arcok és hoza tartozo cimkek elmentése
                detected_faces.append(resized_face)
                faces_labels.append(int(label_map[label]))
    create_training_data(image_dir)


    def recognizer_trainer(recognizer):
        recognizer.train(detected_faces, np.array(faces_labels))

    recognizer_trainer(LBPHFace_recognizer)
    print("Lbph tanitas befejezodott")

def lbph():
    print("Lbph Teszt elindult")

    def draw_rectangle(test_image, rect):
            # kocka koordináták
            x,y,w,h=rect

            #kocka rajzolás
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
        label= LBPHFace_recognizer.predict(resized_test_image)

        # kocka kirajzolás
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

    cv2.imshow('Local binary pettern histogram teszt ablak', pred_img)
    cv2.waitKey(0)

# construct the figure
plt.style.use("ggplot")
(fig, ax) = plt.subplots()
fig.suptitle("Local Binary Patterns")
plt.ylabel("% of Pixels")
plt.xlabel("LBP pixel bucket")

# plot a histogram of the LBP features and show it
def resizeImage(image):
    (h, w) = image.shape[:2]

    width = 360  #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized

# 1 load the image
imagepath = "C:\\Users\\renat\\Desktop\\Allamvizsaga\\Allamvizsga\\arcfelismero\\images\\Reni\\1.jpg"
# , double it in size, and grab the cell size
image = cv2.imread(imagepath)
#image = imutils.resize(image, width=image.shape[1] * 2, inter=cv2.INTER_CUBIC)

# 2 resize the image
image = resizeImage(image)
(h, w) = image.shape[:2]
#cellSize = 16 * 2
cellSize = h/10

# 3 convert the image to grayscale and show it
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", gray)
cv2.waitKey(0)

# displaying default to make cool image
features = feature.local_binary_pattern(gray, 10, 5, method="default") # method="uniform")
cv2.imshow("LBP", features.astype("uint8"))
cv2.waitKey(0)

# Save figure of lbp_image
cv2.imwrite("C:\\Users\\renat\\Desktop\\Allamvizsaga\\Allamvizsga\\arcfelismero\\images\\1.jpg", features.astype("uint8"))

ax.hist(features.ravel(), density=True, bins=20, range=(0, 256))
ax.set_xlim([0, 256])
ax.set_ylim([0, 0.030])
# save figure
fig.savefig('C:\\Users\\renat\\Desktop\\Allamvizsaga\\Allamvizsga\\arcfelismero\\images\\test1')   # save the figure to file
plt.show()

cv2.destroyAllWindows()


# create the 3D grayscale image --> so that I can make color squares for figures to the thesis
# This does not change the histograms created. 
stacked = np.dstack([gray]* 3)

# Divide the image into 100 pieces
(h, w) = stacked.shape[:2]
cellSizeYdir = h / 10
cellSizeXdir = w / 10

# Draw the box around area
# loop over the x-axis of the image
for x in xrange(0, w, cellSizeXdir):
    # draw a line from the current x-coordinate to the bottom of
    # the image

    cv2.line(stacked, (x, 0), (x, h), (0, 255, 0), 1)
    #   
# loop over the y-axis of the image
for y in xrange(0, h, cellSizeYdir):
    # draw a line from the current y-coordinate to the right of
    # the image
    cv2.line(stacked, (0, y), (w, y), (0, 255, 0), 1)

# draw a line at the bottom and far-right of the image
cv2.line(stacked, (0, h - 1), (w, h - 1), (0, 255, 0), 1)
cv2.line(stacked, (w - 1, 0), (w - 1, h - 1), (0, 255, 0), 1)