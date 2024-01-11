import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageDraw
import numpy as np
import cv2

class DigitRecGUI:
    def __init__(self, window, model, predict_callback,train_model_callback):
        #Create Window
        self.window = window
        self.window.title('Digit Recognition')
        self.window.attributes("-fullscreen", True)

        #Create Theme
        style = ttk.Style()
        style.theme_use('clam')

        #Var
        self.model = model
        self.color = 'black'
        self.points = []
        self.pen_width = 10
        self.image = None
        self.last_pressed_index = 1
        self.line_points = []

        #Create UI
        self.create_mainUI()
        self.predict_callback = predict_callback
        self.current_label = None

        self.train_button = tk.Button(self.bottom_frame, width=25, bg="#DC7561", text="Train Model", command=train_model_callback)
        self.train_button.pack(side='bottom', pady=5)
        self.accuracy_label = tk.Label(self.bottom_frame, bg="#707070", text="Accuracy: ", font=('Helvetica', 15))
        self.accuracy_label.pack(side='bottom', pady=5)
        self.prediction_label = tk.Label(self.bottom_frame, bg='#707070' ,text="", font=('Helvetica', 15))
        self.prediction_label.pack(side='bottom')
        self.prediction_label.config(text="Predicted digit: None")
        #Create custom data
        try:
            loaded_data = np.load('data.npz')
            self.trainImages = loaded_data['images'].tolist()
            self.trainLabels = loaded_data['labels'].tolist()
        except: 
            self.trainImages = []
            self.trainLabels = []

    def paint(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))

        # Check if the mouse button is still pressed
        if self.drawing and len(self.points) > 1:
            # Draw lines between consecutive points if distance is greater than pen size
            x1, y1 = self.points[-2]
            x2, y2 = self.points[-1]

            distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            if distance > self.pen_width:
                self.canvas.create_line(x1, y1, x2, y2, fill=self.color, width=self.pen_width * 2, smooth=True)

        # Draw oval at the current point
        x1, y1 = (x - self.pen_width), (y - self.pen_width)
        x2, y2 = (x + self.pen_width), (y + self.pen_width)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.color)

        self.predict()

    def press(self, event):
        self.drawing = True
        x, y = event.x, event.y
        self.points.append((x, y))
        self.last_pressed_index = len(self.points)

    def release(self, event):
        self.drawing = False
        x, y = event.x, event.y
        self.line_points.append((x, y))

    def erase(self, event):
        if self.drawing:
            x, y = event.x, event.y
            erase_radius = self.pen_width / 2.0  # Adjust the radius as needed

            # Clear the canvas
            self.canvas.delete('all')

            # Draw lines between consecutive points and color with white over the erased points
            for i in range(1, len(self.points)):
                x1, y1 = self.points[i - 1]
                x2, y2 = self.points[i]
                distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                if distance > self.pen_width:
                    self.canvas.create_line(x1, y1, x2, y2, fill='white', width=self.pen_width * 2, smooth=True)

            # Delete the black points that are covered by white
            self.points = [point for point in self.points if not (
                point[0] >= x - erase_radius and point[0] <= x + erase_radius and
                point[1] >= y - erase_radius and point[1] <= y + erase_radius
            )]

            # Redraw the remaining points
            for point in self.points:
                x, y = point
                x1, y1 = (x - self.pen_width), (y - self.pen_width)
                x2, y2 = (x + self.pen_width), (y + self.pen_width)
                self.canvas.create_oval(x1, y1, x2, y2, fill='white')

            self.predict()

    def create_mainUI(self):
        canvas_frame = tk.Frame(self.window, background='#353535')
        canvas_frame.pack(fill='both')

        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, background='#505050', highlightthickness=0)
        self.canvas.pack(anchor='center', pady=25)
         
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind('<B1-Motion>', self.paint)
        self.drawing = False
        self.lines = []

        digit_buttons = []
        self.percentage_labels = []
        middle_frame = tk.Frame(self.window, background='black', height=2)
        middle_frame.pack(fill='x')
        self.bottom_frame = tk.Frame(self.window, background='#707070')
        self.bottom_frame.pack(expand=True, fill='both')
        button_frame = tk.Frame(self.bottom_frame, background='#707070')
        button_frame.pack(anchor='n', padx=22, pady=22)

        for digit in range(10):
            button = tk.Button(button_frame, text=str(digit), width=2, height=1, command=lambda d=digit: self.button_click(d),
                               bg='#DC7561', fg='black', font=('Helvetica', 15))
            button.grid(row=0, column=digit, padx=3, pady=3)
            digit_buttons.append(button)

            label = tk.Label(button_frame, text="--%", font=('Helvetica', 12), anchor='w', justify='left', bg='#707070', fg='black')
            label.grid(row=1, column=digit, padx=3, pady=10)
            self.percentage_labels.append(label)

            self.window.bind(str(digit), lambda event, d=digit: self.key_press(event, str(d)))

        self.window.bind('c', lambda event: self.key_press(event, 'c'))
        self.window.bind('<space>', lambda event: self.key_press(event, 'space'))
        self.digit_buttons = digit_buttons
        
        canvas_bframe = tk.Frame(canvas_frame, background='#353535')
        canvas_bframe.pack(anchor='s', pady=12)
        add_button = tk.Button(canvas_bframe, text="Add", command=self.add_to_data, bg='#DC7561', fg='black', font=('Helvetica', 15))
        add_button.grid(row=0, column=0, padx=2)

        clear_button = tk.Button(canvas_bframe, text="Clear", command=self.clear_canvas, bg='#DC7561', fg='black', font=('Helvetica', 15))
        clear_button.grid(row=0, column=1, padx=2)

        self.canvas.bind('<Button-3>', self.erase)

    def key_press(self, event, key):
        if key.isdigit():
            self.button_click(int(key))
        elif key == 'c':
            self.clear_canvas()
        elif key == 'space':
            self.add_to_data()

    def predict_wrapper(self):
        self.predict()
        self.predict_callback(self.image, self.current_label)  
    def predict(self):
        if self.drawing and len(self.points) > 1:
            image = Image.new("L", (280, 280), 'white')
            draw = ImageDraw.Draw(image)

            # Draw lines between consecutive points
            for i in range(1, len(self.points)):
                x1, y1 = self.points[i - 1]
                x2, y2 = self.points[i]
                distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                if distance > self.pen_width and self.points[i - 1] not in self.line_points:
                    draw.line([x1, y1, x2, y2], fill='black', width=self.pen_width * 2)

            # Draw ovals at the points
            for point in self.points:
                draw.ellipse([point[0] - self.pen_width, point[1] - self.pen_width,
                              point[0] + self.pen_width, point[1] + self.pen_width], fill='black')

            # Resize to 28x28
            image = image.resize((28, 28))

            # Convert to NumPy array
            image = np.invert(np.array(image))

            # Thresholding and reshaping
            img = image
            img = img.reshape(28, 28)

            # Save the image
            cv2.imwrite('TestImage.png', img)

            # Normalize and reshape for prediction
            image = image / 255.0
            image = image.reshape(-1)

            self.image = image
            prediction = self.model.predict(self.image)
            self.display_prediction(prediction)

    def add_to_data(self):
        try:
            loaded_data = np.load('data.npz')
            self.trainImages = loaded_data['images'].tolist()
            self.trainLabels = loaded_data['labels'].tolist()
            print(self.trainLabels)
        except:
            print("Failed")
            
        if self.current_label is not None:
            self.trainImages.append(self.image)
            self.trainLabels.append(int(self.current_label))
            np.savez('data.npz',images = self.trainImages,labels = self.trainLabels)
            self.clear_canvas()
    def clear_canvas(self):
        self.canvas.delete('all')
        self.points = []
    def button_click(self, digit):
        for btn in self.digit_buttons:
            if btn != self.digit_buttons[digit]:
                btn['state'] = tk.NORMAL
                btn['fg'] = 'black'

        if self.digit_buttons[digit]['fg'] == 'black':
            self.digit_buttons[digit]['fg'] = '#83f28f'
            self.current_label = digit
        elif self.digit_buttons[digit]['fg'] == '#83f28f':
            self.digit_buttons[digit]['fg'] = 'black'
            self.current_label = None
            
    def display_prediction(self, prediction):
        self.prediction_label.config(text=f"Predicted digit: {prediction}")

        softmax_activation = self.model.activations[-1]

        probabilities = softmax_activation.output
        max_percentage_index = np.argmax(probabilities, axis=1)

        for digit, label in zip(range(10), self.percentage_labels):
            percentage = np.round(100 * probabilities[:, digit].max(), 2)
            formatted_percentage = f"{percentage:.0f}%".zfill(3)

            if digit == max_percentage_index[0]:  # Highlight the highest percentage in green 
                label.config(text=f"{formatted_percentage}", fg='#83f28f')
            else:
                label.config(text=f"{formatted_percentage}", fg='black')
        if self.model._accuracy is not None:
            self.accuracy_label.config(text=f"Accuracy: {self.model._accuracy * 100}%")




