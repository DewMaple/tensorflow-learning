import cv2

faceCascade = cv2.CascadeClassifier(cv2.haarcascades + "haarcascade_frontalface_default.xml")


def extract(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.1,
    #     minNeighbors=5,
    #     minSize=(30, 30)
    # )
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('image', image)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    print("Found {} faces of {} ".format(len(faces), image_path))
    if len(faces) < 1:
        return cv2.resize(image, (224, 224))
    (x, y, w, h) = faces[0]
    roi = image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (224, 224))
    return roi

#
# target = extract('Aamir_Khan.jpg')
# #
# # im = cv2.imread('ak.png')
# im = extract('ak.png')
# cv2.imshow('dd', im)
# cv2.waitKey()
# distance = calculate_distance(im, target)
# if cv2.waitKey(0) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()
