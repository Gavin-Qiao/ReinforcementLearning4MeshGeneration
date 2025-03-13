import numpy as np
import cv2
from tkinter import *

# ============================================================================

CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (200, 200, 200)
WORKING_LINE_COLOR = (127, 127, 127)


# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = "Mesh Generation UI" # Name for our window

        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon


    def on_mouse(self, event, x, y, buttons, user_param):
        # Mouse callback that gets called for every mouse event (i.e. moving, clicking, etc.)

        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click means we're done
            print("Completing polygon with %d points." % len(self.points))
            self.done = True


    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, np.full(CANVAS_SIZE, 255, dtype=np.uint8))
        # cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))

        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
            # This is our drawing loop, we just continuously draw new images
            # and show them in the named window
            # canvas = np.zeros(CANVAS_SIZE, np.uint8)
            canvas = np.full(CANVAS_SIZE, 255, dtype=np.uint8)

            if (len(self.points) > 0):
                # Draw all the current polygon segments
                cv2.polylines(canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                # And  also show what the current segment would look like
                # cv2.circle(canvas, self.points[-1], 63, (0, 0, 255), -1)
                [cv2.circle(canvas, p, 2, (0, 0, 255), -1) for p in self.points]
                cv2.line(canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
            # Update the window
            cv2.imshow(self.window_name, canvas)
            # And wait 50ms before next iteration (this will pump window messages meanwhile)
            if cv2.waitKey(50) == 27: # ESC hit
                self.done = True

        # User finised entering the polygon points, so let's make the final drawing
        # canvas = np.zeros(CANVAS_SIZE, np.uint8)
        canvas = np.full(CANVAS_SIZE, 255, dtype=np.uint8)
        # of a filled polygon
        if (len(self.points) > 0):
            cv2.fillPoly(canvas, np.array([self.points]), FINAL_LINE_COLOR)
            [cv2.circle(canvas, p, 2, (0, 0, 255), -1) for p in self.points]
        # And show it

        cv2.imshow(self.window_name, canvas)
        # Waiting for the user to press any key
        cv2.waitKey()

        cv2.destroyWindow(self.window_name)
        return canvas

# ============================================================================

if __name__ == "__main__":
    pd = PolygonDrawer("Polygon")
    image = pd.run()
    cv2.imwrite("polygon.png", image)
    print("Polygon = %s" % pd.points)