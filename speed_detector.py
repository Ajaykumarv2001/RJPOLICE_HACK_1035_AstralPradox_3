import cv2
import time
import math

# Classifier File
carCascade = cv2.CascadeClassifier("C:\\Users\\herow\\Robotic_hand\\vehicle-speed-detection-using-opencv-python\\vech.xml")

# Video file capture
video = cv2.VideoCapture("C:\\Users\\herow\\Robotic_hand\\vehicle-speed-detection-using-opencv-python\\carsVideo.mp4")

# Constant Declaration
WIDTH = 1280
HEIGHT = 720
SLOW_DOWN_FACTOR = 3  # Adjust this factor to control the slow-motion effect

# Estimate speed function
def estimateSpeed(location1, location2, fps):
    d_pixels = math.sqrt((location2[0] - location1[0]) ** 2 + (location2[1] - location1[1]) ** 2)
    ppm = 8.8
    d_meters = d_pixels / ppm
    speed = d_meters * fps * 3.6
    return speed

# Tracking multiple objects using MIL tracker
def trackMultipleObjects():
    rectangleColor = (0, 255, 255)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outTraffic_slowmotion.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 3, (WIDTH, HEIGHT))  # Reduced frame rate

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            tracker, initialLocation = carTracker[carID]
            trackingQuality, trackedPosition = tracker.update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print("Removing carID " + str(carID) + ' from the list of trackers. ')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    _, initialLocation = carTracker[carID]
                    t_x, t_y, t_w, t_h = initialLocation

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if (
                        (t_x <= x_bar <= (t_x + t_w))
                        and (t_y <= y_bar <= (t_y + t_h))
                        and (x <= t_x_bar <= (x + w))
                        and (y <= t_y_bar <= (y + h))
                    ):
                        matchCarID = carID

                if matchCarID is None:
                    print("Creating new tracker" + str(currentCarID))

                    tracker = cv2.TrackerMIL_create()
                    tracker.init(image, (x, y, w, h))

                    carTracker[currentCarID] = (tracker, (x, y, w, h))
                    carLocation1[currentCarID] = (x, y, w, h)

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            tracker, _ = carTracker[carID]
            trackingQuality, trackedPosition = tracker.update(image)

            t_x = int(trackedPosition[0])
            t_y = int(trackedPosition[1])
            t_w = int(trackedPosition[2])
            t_h = int(trackedPosition[3])

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = (t_x, t_y, t_w, t_h)

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                (x1, y1, w1, h1) = carLocation1[i]
                (x2, y2, w2, h2) = carLocation2[i]

                carLocation1[i] = (x2, y2, w2, h2)

                if (x1, y1, w1, h1) != (x2, y2, w2, h2):
                    if (speed[i] is None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed((x1, y1), (x1, y2), fps)

                    if speed[i] is not None and y1 >= 180:
                        cv2.putText(
                            resultImage,
                            str(int(speed[i])) + "km/h",
                            (int(x1 + w1 / 2), int(y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 100),
                            2,
                        )

        cv2.imshow("result", resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    trackMultipleObjects()