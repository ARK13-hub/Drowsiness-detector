import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import pygame
import logging
from logging.handlers import RotatingFileHandler

# Logging setup
logger = logging.getLogger('drowsiness')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('drowsiness.log', maxBytes=1_000_000, backupCount=3, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Initialize pygame mixer
try:
    pygame.mixer.init()
    pygame.mixer.music.load("alarm.wav")
    logger.info('Loaded alarm.wav and initialized pygame mixer')
except Exception as e:
    logger.warning(f'Could not initialize pygame mixer or load alarm: {e}')

# ------------- ALARM CONTROL FUNCTIONS -------------
def play_alarm():
    try:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play(-1)   # loop continuously
    except Exception as e:
        logger.warning(f"Failed to play alarm: {e}")

def stop_alarm():
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass
# ----------------------------------------------------

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    D = dist.euclidean(mouth[0], mouth[6])
    return (A + B + C) / (3.0 * D)

# Thresholds
eye_thresh = 0.25
mouth_thresh = 0.65
frame_limit = 20
counter = 0

# Dlib Models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
(mStart, mEnd) = (48, 61)

# Video Capture
cap = cv2.VideoCapture(0)

def safe_release():
    try:
        cap.release()
    except:
        pass
    try:
        stop_alarm()
        cv2.destroyAllWindows()
    except:
        pass

logger.info('Starting drowsiness detection loop')

# ---------------- MAIN LOOP ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error('Frame grab failed')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            EAR = (leftEAR + rightEAR) / 2

            MAR = mouth_aspect_ratio(mouth)

            # ------- DROWSINESS LOGIC -------
            if EAR < eye_thresh or MAR > mouth_thresh:
                counter += 1
                logger.info(f"Drowsiness check: EAR={EAR:.3f}, MAR={MAR:.3f}, counter={counter}")

                if counter >= frame_limit:
                    play_alarm()
                    cv2.putText(frame, "DROWSINESS ALERT!!!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                counter = 0
                stop_alarm()
            # --------------------------------

            cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) == 27:  # ESC
            logger.info("ESC pressed — exiting")
            break

except KeyboardInterrupt:
    logger.info("KeyboardInterrupt — exiting")

except Exception as e:
    logger.exception(f"Unhandled exception: {e}")

finally:
    safe_release()
    logger.info("Shutdown complete")
