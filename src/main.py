import cv2
import torch
from lib.face_detection import FaceDetection
from lib.network import Net

class EmotionRecognition():

    def __init__(self, nn_pth):

        self.spikes = {'emotion': None, 'count': 0}

        self.face_detection = FaceDetection()

        self.net = Net()
        self.net.load_state_dict(torch.load(nn_pth))
        self.net = self.net.double()

        self.emotions = ['neutral', 
                         'happiness', 
                         'surprise', 
                         'sadness', 
                         'anger', 
                         'disgust',
                         'fear',
                         'contempt']
    
    def recognize(self, face):
        outputs = self.net(face.double())
        _, predicted = torch.max(outputs.data, 1)
        
        return(self.emotions[predicted])
    
    def fire(self, face):

        fire = False
        emotion = self.recognize(face)

        if self.spikes['emotion']:
            if self.spikes['emotion'] == emotion:
                self.spikes['count']+=1
            else:
                self.spikes['count']=0
                self.spikes['emotion']=None
        else:
            self.spikes['emotion']=emotion
            self.spikes['count']=1

        if self.spikes['count'] >= 10:
            fire = True
            self.spikes['count']=0
        
        return(self.spikes['emotion'] if fire else None)
    
    def main(self, frame):

        frame, faces = self.face_detection.main(frame)

        if len(faces)>0:
            response = self.fire(faces[0].unsqueeze(0))
        else:
            response = None
        
        cv2.imshow('frame', frame)

        return(response)


if __name__ == "__main__":

    CNN_PATH = './models/network_v2.pth'

    cap = cv2.VideoCapture(0)

    er = EmotionRecognition(CNN_PATH)

    while(True):

        ret, frame = cap.read()

        response = er.main(frame)

        if response:
            print('Emotion:', response)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()