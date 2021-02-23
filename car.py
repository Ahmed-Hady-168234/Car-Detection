
import cv2
#n = 0
# capture frames from a video
cap = cv2.VideoCapture('cars.mp4')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 3, 1)


    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 255), 2)
        #cv2.putText(frames, str(y), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        #cv2.PutText(img, text, org, font, color)

    cv2.imshow('sKSama', frames )
    if cv2.waitKey(33) == 13:
        break


cap.release()
cv2.destroyAllWindows()
