import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('faces/face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, minNeighbors=20, minSize=(30, 30), maxSize=(300, 300))
eye = eye_cascade.detectMultiScale(gray, minNeighbors=20, minSize=(10, 10), maxSize=(90, 90))

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

for (x, y, w, h) in eye:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('face', img)

# ESC - sair
# S - salvar

k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('face_detection.png', img)
    cv2.destroyAllWindows()
