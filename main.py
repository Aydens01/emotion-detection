import cv2
from src.emotion_recognition import EmotionRecognition

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    CNN_PATH = "./models/network_v2.pth"

    er = EmotionRecognition(CNN_PATH)

    while True:

        ret, frame = cap.read()

        response, frame = er.main(frame)

        cv2.imshow("frame", frame)

        if response:
            print("Emotion:", response)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
