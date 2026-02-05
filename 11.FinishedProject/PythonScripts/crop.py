import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os


class QuadCropper:
    def __init__(self, image_path, output_path=""):

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if output_path == "" :
            self.output_path = image_path
        else:
            self.output_path = output_path

        # Load and prepare image
        self.original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width = self.original_image.shape[:2]

        # Setup window
        self.root = tk.Tk()
        self.root.title("Quad Cropper")

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.display_width = min(self.img_width, screen_width - 100)
        self.display_height = min(self.img_height, screen_height - 100)

        # Scaling (if image is larger than screen)
        self.scale = min(self.display_width / self.img_width, self.display_height / self.img_height)
        self.display_width = int(self.img_width * self.scale)
        self.display_height = int(self.img_height * self.scale)

        self.canvas = tk.Canvas(self.root, width=self.display_width, height=self.display_height)
        self.canvas.pack()

        resized_image = cv2.resize(self.original_image, (self.display_width, self.display_height))
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(resized_image))
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Initial quad corners
        margin = 50
        self.quad = [
            [margin, margin],
            [self.display_width - margin, margin],
            [self.display_width - margin, self.display_height - margin],
            [margin, self.display_height - margin]
        ]

        self.points = []
        self.lines = []
        self.drag_data = {"index": None}
        self.point_radius = 6

        self.draw_quad()
        self.bind_events()

        self.btn = tk.Button(self.root, text="Crop & Save", command=self.crop_and_save)
        self.btn.pack()

        self.root.mainloop()

    def draw_quad(self):
        for p in self.points:
            self.canvas.delete(p)
        for l in self.lines:
            self.canvas.delete(l)
        self.points.clear()
        self.lines.clear()

        for x, y in self.quad:
            point = self.canvas.create_oval(
                x - self.point_radius, y - self.point_radius,
                x + self.point_radius, y + self.point_radius,
                fill="red", outline="black"
            )
            self.points.append(point)

        for i in range(4):
            x1, y1 = self.quad[i]
            x2, y2 = self.quad[(i + 1) % 4]
            line = self.canvas.create_line(x1, y1, x2, y2, fill="cyan", width=2)
            self.lines.append(line)

    def bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_down(self, event):
        for i, (x, y) in enumerate(self.quad):
            if abs(event.x - x) < 10 and abs(event.y - y) < 10:
                self.drag_data["index"] = i
                return

    def on_mouse_move(self, event):
        i = self.drag_data["index"]
        if i is not None:
            self.quad[i] = [event.x, event.y]
            self.draw_quad()

    def on_mouse_up(self, event):
        self.drag_data["index"] = None

    def crop_and_save(self):
        src = np.array(self.quad, dtype="float32") / self.scale  # Scale back to original image coords

        width_top = np.linalg.norm(src[0] - src[1])
        width_bottom = np.linalg.norm(src[2] - src[3])
        width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(src[0] - src[3])
        height_right = np.linalg.norm(src[1] - src[2])
        height = int(max(height_left, height_right))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.original_image, matrix, (width, height))

        # Save image
        cv2.imwrite(self.output_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        print(f"Saved cropped image to: {self.output_path}")
        cv2.destroyAllWindows()

        # Show result
        # cv2.imshow("Cropped & Flattened", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    QuadCropper("C:/Users/up107/Downloads/profile.jpg", "output_cropped.jpg")
