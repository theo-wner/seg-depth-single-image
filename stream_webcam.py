'''Script that streams webcam feed'''

import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(4) # 0 is Laptop webcam, 4 is external webcam
    while True:
        ret, frame = cap.read()
        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()