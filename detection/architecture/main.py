import ApplicationCore
import FrameCapture
import ObjectDetector
import ObjectSelector
import ObjectTracker
import os
from dotenv import load_dotenv
load_dotenv("C:\\Users\\User\\Documents\\python\\mapping\\detection\\.env")

videoPath = os.getenv("VIDEO_PATH")

frameCapture = FrameCapture.OpenCVFrameCapture(videoPath)
objectDetector = ObjectDetector.ObjectDetector()
objectSelector = ObjectSelector.ObjectSelector()
objectTracker = ObjectTracker.ObjectTracker()

core = ApplicationCore.ApplicationCore(frameCapture, objectDetector, objectSelector, ObjectTracker)
core.run()