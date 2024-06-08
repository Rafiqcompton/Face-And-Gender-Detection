
'''
Assigment 1 Detect Faces from static Images Predict their Gender and the Age of the Person/s in the Photo 
'''


import cv2



face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


img = cv2.imread('./Images/img1.jpg', 1)  # Load in color mode (1)


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


GENDER_MODEL = 'Weights/deploy_gender.prototxt'  # Path to gender model text file
GENDER_PROTO = 'Weights/gender_net.caffemodel'  # Path to gender model weights
AGE_MODEL = 'weights/deploy_age.prototxt'  # Path to age model text file
AGE_PROTO = 'weights/age_net.caffemodel'  # Path to age model weights


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


faces = face_classifier.detectMultiScale(gray_image, 1.3, 2)


for (x, y, w, h) in faces:

    # Draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x + w, y + h), (100, 25, 70), 5, cv2.LINE_AA)

    # Gender and Age Prediction 
    if GENDER_MODEL and GENDER_PROTO and AGE_MODEL and AGE_PROTO:
        # Preprocess the face region for prediction
        face_blob = cv2.dnn.blobFromImage(img[y:y + h, x:x + w], 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Load and make predictions using gender and age models 
        face_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)
        age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

        face_net.setInput(face_blob)
        gender_preds = face_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        age_net.setInput(face_blob)
        age_preds = age_net.forward()
        age = AGE_INTERVALS[age_preds[0].argmax()]

        # Display predicted gender and age above the face
        cv2.putText(img, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)
        cv2.putText(img, age, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2, cv2.LINE_AA)

# Display the final image with detected faces (and predicted gender/age if applicable)
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
