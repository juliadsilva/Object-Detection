import cv2

# Imagens
imagem1 = cv2.imread('object/images/xadrez05.jpg')
imagem2 = cv2.imread('object/images/xadrez05.jpg')
imagem3 = cv2.imread('object/images/xadrez05.jpg')
imagem4 = cv2.imread('object/images/xadrez05.jpg')

# Imagens em cinza
imagemcinza1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
imagemcinza2 = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)
imagemcinza3 = cv2.cvtColor(imagem3, cv2.COLOR_BGR2GRAY)
imagemcinza4 = cv2.cvtColor(imagem4, cv2.COLOR_BGR2GRAY)

# Classificadores
classificador1 = cv2.CascadeClassifier('object/cascade_xadrez_01.xml')
classificador2 = cv2.CascadeClassifier('object/cascade_xadrez_02.xml')
classificador3 = cv2.CascadeClassifier('object/cascade_xadrez_03.xml')
classificador4 = cv2.CascadeClassifier('object/cascade_xadrez_04.xml')

# Detecção
deteccoes1 = classificador1.detectMultiScale(imagemcinza1, minNeighbors=40, minSize=(20, 20), maxSize=(150, 150))
deteccoes2 = classificador2.detectMultiScale(imagemcinza2, minNeighbors=40, minSize=(20, 20), maxSize=(150, 150))
deteccoes3 = classificador3.detectMultiScale(imagemcinza3, minNeighbors=40, minSize=(20, 20), maxSize=(150, 150))
deteccoes4 = classificador4.detectMultiScale(imagemcinza4, minNeighbors=40, minSize=(50, 50), maxSize=(150, 150))

for (x, y, l, a) in deteccoes1:
    cv2.rectangle(imagem1, (x, y), (x + l, y + a), (0, 255, 0), 2)

for (x, y, l, a) in deteccoes2:
    cv2.rectangle(imagem2, (x, y), (x + l, y + a), (255, 0, 0), 2)

for (x, y, l, a) in deteccoes3:
    cv2.rectangle(imagem3, (x, y), (x + l, y + a), (0, 0, 255), 2)

for (x, y, l, a) in deteccoes4:
    cv2.rectangle(imagem4, (x, y), (x + l, y + a), (0, 0, 0), 2)

# Mostrar imagem
cv2.imshow('1', imagem1)
cv2.imshow('2', imagem2)
cv2.imshow('3', imagem3)
cv2.imshow('4', imagem4)

# ESC - sair
# S - salvar
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('object_detection.png', imagem4)
    cv2.destroyAllWindows()
