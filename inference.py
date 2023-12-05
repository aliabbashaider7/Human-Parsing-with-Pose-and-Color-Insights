import tkinter
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, ImageSequence
import time
from collections import OrderedDict
import torchvision.transforms as transforms
import networks
import pandas as pd
from src.pose_utils import *
from src.extraction_utils import *
from threading import Thread

num_classes = 18
input_size = [512, 512]
labels = ['Background', 'Hat', 'Hair', 'Sunglasses', 'Shirt', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']

weights_path = 'weights/parsing_model.pth'

model_parsing = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)

state_dict = torch.load(weights_path, map_location='cpu')['state_dict']

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
model_parsing.load_state_dict(new_state_dict)
model_parsing.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
])
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv('weights/color_values.csv', names=index, header=None)

body_estimation = Body('weights/body_pose_model.pth')
pose_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
              [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [0, 14], [14, 16], [0, 15], [15, 17]]
pose_colors = []
for i in range(len(pose_pairs)):
    color = np.random.randint(0, 255, size=(3,))
    color = (int(color[0]), int(color[1]), int(color[2]))
    pose_colors.append(color)


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.screen_width = self.window.winfo_screenwidth()
        self.screen_height = self.window.winfo_screenheight()
        self.picture_width = int(self.screen_width/4)
        self.picture_height = int(self.picture_width*1.333333)
        self.window.title(window_title)
        self.window.geometry(f"{self.screen_width}x{self.screen_height}")
        self.window.attributes('-fullscreen', True)

        self.main_canvas = tkinter.Canvas(self.window, width=self.screen_width,
                 height=int(self.screen_height/1.6))
        self.main_canvas.place(x=0, y=0)

        self.main_image = Image.open('modules/src/utils/pngs/title.png')
        self.main_image = self.main_image.resize((int(self.screen_width), int(self.screen_height/1.6)), Image.ANTIALIAS)
        self.main_image = ImageTk.PhotoImage(self.main_image)
        self.main_canvas.create_image(0, 0, image=self.main_image, anchor=tkinter.NW)

        self.canvas_1 = tkinter.Canvas(self.window, width=self.picture_width, height=self.picture_height, highlightthickness=0.5, highlightbackground="gray")
        self.canvas_2 = tkinter.Canvas(self.window, width=self.picture_width, height=self.picture_height, highlightthickness=0.5, highlightbackground="gray")
        self.canvas_1.place(x=int(self.screen_width/2 - self.picture_width - 0.1 * self.picture_width), y=int(self.screen_height/9))
        self.canvas_2.place(x=int(self.screen_width/2 + 0.1 * self.picture_width), y=int(self.screen_height/9))

        self.cl1 = ttk.Label(self.window, text='Input Image', background='black', foreground='white')
        self.cl2 = ttk.Label(self.window, text='Output Image', background='black', foreground='white')
        self.cl1.place(x=int(self.screen_width/2 - self.picture_width - 0.1 * self.picture_width), y=int(self.screen_height/10 - 0.2*self.screen_height/15))
        self.cl2.place(x=int(self.screen_width/2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width/13.3)), y=int(self.screen_height/10 - 0.2*self.screen_height/15))

        pose_image = Image.open('modules/src/utils/pngs/pose.png')
        pose_image = pose_image.resize((int(self.screen_width/40), int(self.screen_height/18)), Image.ANTIALIAS)
        pose_image = ImageTk.PhotoImage(pose_image)
        self.pose_mark = tkinter.Label(image=pose_image)
        self.pose_mark.image = pose_image
        self.pose_mark.place(x=self.screen_width/30, y=int(self.screen_height/9 + self.picture_height + self.screen_height/90))
        self.pose_heading = tkinter.Label(text='Pose Insights:', font=("Times 20 italic bold", int(self.screen_width/35)), foreground='dark cyan')
        self.pose_heading.place(x=int(self.screen_width/30+self.screen_width/40+0.4*self.screen_width/40), y=int(self.screen_height/9 + self.picture_height + self.screen_height/90))

        parsing_image = Image.open('modules/src/utils/pngs/parsing.png')
        parsing_image = parsing_image.resize((int(self.screen_width / 40), int(self.screen_height / 18)), Image.ANTIALIAS)
        parsing_image = ImageTk.PhotoImage(parsing_image)
        self.parsing_mark = tkinter.Label(image=parsing_image)
        self.parsing_mark.image = parsing_image
        self.parsing_mark.place(x=int(self.screen_width/2 + 0.1 * self.picture_width),
                             y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90))
        self.parsing_heading = tkinter.Label(text='Parsing Results:',
                                          font=("Times 20 italic bold", int(self.screen_width / 35)),
                                          foreground='red')
        self.parsing_heading.place(x=int(int(self.screen_width/2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
                                y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90))

        bg_image = Image.open('modules/src/utils/pngs/parsing_items/1_background.png')
        bg_image = bg_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                             Image.ANTIALIAS)
        bg_image = ImageTk.PhotoImage(bg_image)
        self.bg_mark = tkinter.Label(image=bg_image)
        self.bg_mark.image = bg_image
        self.bg_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
                                y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.1 * int(self.screen_height / 18))
        self.bg_heading = tkinter.Label(text='Background:',
                                             font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.bg_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(1.1 * self.screen_width / 40),
                                   y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(self.screen_height / 18))
        self.bg_text = tkinter.Text(self.window, height=int(self.screen_height / (400)), width=int(self.screen_width / (50)))
        self.bg_text.place(x=int(self.screen_width/2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width/15)) + int(int(self.screen_width / 40)) + int(self.screen_width / 55),
                                   y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(self.screen_height / 18))
        self.can_bg = tkinter.Canvas(self.window, width=int(self.screen_width / 55), height=int(self.screen_width / 55), highlightthickness=0.5, highlightbackground="gray")
        self.can_bg.place(x=int(self.screen_width/2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width/15)),
                                   y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(self.screen_height / 18))

        hair_image = Image.open('modules/src/utils/pngs/parsing_items/2_hair.png')
        hair_image = hair_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                   Image.ANTIALIAS)
        hair_image = ImageTk.PhotoImage(hair_image)
        self.hair_mark = tkinter.Label(image=hair_image)
        self.hair_mark.image = hair_image
        self.hair_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.135 * int(
                self.screen_height / 18) + int(self.screen_height / 35) + 0.2*int(self.screen_height / 35))
        self.hair_heading = tkinter.Label(text='Hair:',
                                        font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.hair_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                    self.screen_height / 18) + int(self.screen_height / 35) + 0.1 * int(
                                    self.screen_height / 35))
        self.can_hair = tkinter.Canvas(self.window, width=int(self.screen_width / 55), height=int(self.screen_width / 55),
                                     highlightthickness=0.5, highlightbackground="gray")
        self.can_hair.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.135 * int(
                self.screen_height / 18) + int(self.screen_height / 35) + 0.2 * int(self.screen_height / 35))
        self.hair_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                    width=int(self.screen_width / (50)))
        self.hair_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                           y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.135 * int(
                               self.screen_height / 18) + int(self.screen_height / 35) + 0.2 * int(
                               self.screen_height / 35))

        skin_image = Image.open('modules/src/utils/pngs/parsing_items/3_skin.png')
        skin_image = skin_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                       Image.ANTIALIAS)
        skin_image = ImageTk.PhotoImage(skin_image)
        self.skin_mark = tkinter.Label(image=skin_image)
        self.skin_mark.image = skin_image
        self.skin_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.skin_heading = tkinter.Label(text='Skin:',
                                          font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.skin_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                    self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.1 * int(
                                    self.screen_height / 35))
        self.can_skin = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_skin.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))

        self.skin_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                      width=int(self.screen_width / (50)))
        self.skin_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                             y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                 self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.15 * int(
                                 self.screen_height / 35))

        shirt_image = Image.open('modules/src/utils/pngs/parsing_items/4_shirt.png')
        shirt_image = shirt_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                       Image.ANTIALIAS)
        shirt_image = ImageTk.PhotoImage(shirt_image)
        self.shirt_mark = tkinter.Label(image=shirt_image)
        self.shirt_mark.image = shirt_image
        self.shirt_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.shirt_heading = tkinter.Label(text='Shirt:',
                                          font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.shirt_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                    self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.1 * int(
                                    self.screen_height / 35))
        self.can_shirt = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_shirt.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.shirt_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                      width=int(self.screen_width / (50)))
        self.shirt_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                             y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                 self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(
                                 self.screen_height / 35))

        pants_image = Image.open('modules/src/utils/pngs/parsing_items/5_pants.png')
        pants_image = pants_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                         Image.ANTIALIAS)
        pants_image = ImageTk.PhotoImage(pants_image)
        self.pants_mark = tkinter.Label(image=pants_image)
        self.pants_mark.image = pants_image
        self.pants_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(4 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.pants_heading = tkinter.Label(text='Pants:',
                                           font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.pants_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                 y=int(
                                     self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                     self.screen_height / 18) + int(4 * self.screen_height / 35) + 0.1 * int(
                                     self.screen_height / 35))
        self.can_pants = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_pants.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(4 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.pants_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.pants_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(4 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        dress_image = Image.open('modules/src/utils/pngs/parsing_items/6_dress.png')
        dress_image = dress_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                         Image.ANTIALIAS)
        dress_image = ImageTk.PhotoImage(dress_image)
        self.dress_mark = tkinter.Label(image=dress_image)
        self.dress_mark.image = dress_image
        self.dress_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.dress_heading = tkinter.Label(text='Dress:',
                                           font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.dress_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                 y=int(
                                     self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                     self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.1 * int(
                                     self.screen_height / 35))
        self.can_dress = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_dress.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.dress_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.dress_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        skirt_image = Image.open('modules/src/utils/pngs/parsing_items/7_skirt.png')
        skirt_image = skirt_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                         Image.ANTIALIAS)
        skirt_image = ImageTk.PhotoImage(skirt_image)
        self.skirt_mark = tkinter.Label(image=skirt_image)
        self.skirt_mark.image = skirt_image
        self.skirt_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(6 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.skirt_heading = tkinter.Label(text='Skirt:',
                                           font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.skirt_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                 y=int(
                                     self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                     self.screen_height / 18) + int(6 * self.screen_height / 35) + 0.1 * int(
                                     self.screen_height / 35))
        self.can_skirt = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_skirt.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(6 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.skirt_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.skirt_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(6 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        shoes_image = Image.open('modules/src/utils/pngs/parsing_items/8_shoes.png')
        shoes_image = shoes_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                         Image.ANTIALIAS)
        shoes_image = ImageTk.PhotoImage(shoes_image)
        self.shoes_mark = tkinter.Label(image=shoes_image)
        self.shoes_mark.image = shoes_image
        self.shoes_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.shoes_heading = tkinter.Label(text='Shoes:',
                                           font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.shoes_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                 y=int(
                                     self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                     self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.1 * int(
                                     self.screen_height / 35))
        self.can_shoes = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_shoes.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.shoes_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.shoes_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        hat_image = Image.open('modules/src/utils/pngs/parsing_items/9_hat.png')
        hat_image = hat_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                         Image.ANTIALIAS)
        hat_image = ImageTk.PhotoImage(hat_image)
        self.hat_mark = tkinter.Label(image=hat_image)
        self.hat_mark.image = hat_image
        self.hat_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.225 * int(
                self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.hat_heading = tkinter.Label(text='Hat:',
                                           font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.hat_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                 y=int(
                                     self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                     self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.1 * int(
                                     self.screen_height / 35))
        self.can_hat = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_hat.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.hat_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.hat_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        sunglasses_image = Image.open('modules/src/utils/pngs/parsing_items/10_sunglasses.png')
        sunglasses_image = sunglasses_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                     Image.ANTIALIAS)
        sunglasses_image = ImageTk.PhotoImage(sunglasses_image)
        self.sunglasses_mark = tkinter.Label(image=sunglasses_image)
        self.sunglasses_mark.image = sunglasses_image
        self.sunglasses_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.sunglasses_heading = tkinter.Label(text='Sunglasses:',
                                         font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.sunglasses_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                   self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.1 * int(
                                   self.screen_height / 35))
        self.can_sunglasses = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                      height=int(self.screen_width / 55),
                                      highlightthickness=0.5, highlightbackground="gray")
        self.can_sunglasses.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.sunglasses_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                     width=int(self.screen_width / (50)))
        self.sunglasses_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                                   y=int(
                                       self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                       self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(
                                       self.screen_height / 35))

        bag_image = Image.open('modules/src/utils/pngs/parsing_items/11_bag.png')
        bag_image = bag_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                                   Image.ANTIALIAS)
        bag_image = ImageTk.PhotoImage(bag_image)
        self.bag_mark = tkinter.Label(image=bag_image)
        self.bag_mark.image = bag_image
        self.bag_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.bag_heading = tkinter.Label(text='Bag:',
                                                font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.bag_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                      y=int(
                                          self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                          self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.1 * int(
                                          self.screen_height / 35))
        self.can_bag = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                             height=int(self.screen_width / 55),
                                             highlightthickness=0.5, highlightbackground="gray")
        self.can_bag.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.bag_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                            width=int(self.screen_width / (50)))
        self.bag_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(
                                self.screen_height / 35))

        belt_image = Image.open('modules/src/utils/pngs/parsing_items/12_belt.png')
        belt_image = belt_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                     Image.ANTIALIAS)
        belt_image = ImageTk.PhotoImage(belt_image)
        self.belt_mark = tkinter.Label(image=belt_image)
        self.belt_mark.image = belt_image
        self.belt_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(11 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.belt_heading = tkinter.Label(text='Belt:',
                                         font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.belt_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                   self.screen_height / 18) + int(11 * self.screen_height / 35) + 0.1 * int(
                                   self.screen_height / 35))
        self.can_belt = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                      height=int(self.screen_width / 55),
                                      highlightthickness=0.5, highlightbackground="gray")
        self.can_belt.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(11 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.belt_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                     width=int(self.screen_width / (50)))
        self.belt_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                             y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                 self.screen_height / 18) + int(11 * self.screen_height / 35) + 0.15 * int(
                                 self.screen_height / 35))

        scarf_image = Image.open('modules/src/utils/pngs/parsing_items/13_scarf.png')
        scarf_image = scarf_image.resize((int(self.screen_width / 40), int(self.screen_height / 35)),
                                       Image.ANTIALIAS)
        scarf_image = ImageTk.PhotoImage(scarf_image)
        self.scarf_mark = tkinter.Label(image=scarf_image)
        self.scarf_mark.image = scarf_image
        self.scarf_mark.place(x=int(
            int(self.screen_width / 2 + 0.1 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.05 * int(
                self.screen_height / 18) + int(12 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.scarf_heading = tkinter.Label(text='Scarf:',
                                          font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.scarf_heading.place(x=int(
            int(self.screen_width / 2 + 0.12 * self.picture_width) + self.screen_width / 40 + 0.4 * self.screen_width / 40) + int(
            1.1 * self.screen_width / 40),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.13 * int(
                                    self.screen_height / 18) + int(12 * self.screen_height / 35) + 0.1 * int(
                                    self.screen_height / 35))
        self.can_scarf = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_scarf.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(12 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.scarf_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                      width=int(self.screen_width / (50)))
        self.scarf_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(12 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        self.pose_1_heading = tkinter.Label(text='Main Pose:',
                                            font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.pose_1_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40),
                                  y=int(
                                      self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                      self.screen_height / 18) + int(self.screen_height / 35) + 0.1 * int(
                                      self.screen_height / 35))

        self.pose_2_heading = tkinter.Label(text='Estimated Position:',
                                            font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.pose_2_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40),
                                  y=int(
                                      self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                      self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.1 * int(
                                      self.screen_height / 35))

        self.pose_3_heading = tkinter.Label(text='Detailed Posture:',
                                            font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.pose_3_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40),
                                  y=int(
                                      self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                      self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(
                                      self.screen_height / 35))

        self.pose_1_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (35)))
        self.pose_1_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                   self.screen_height / 18) + int(self.screen_height / 35) + 0.1 * int(
                                   self.screen_height / 35))

        self.pose_2_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                        width=int(self.screen_width / (35)))
        self.pose_2_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                   self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.1 * int(
                                   self.screen_height / 35))

        self.pose_3_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                        width=int(self.screen_width / (35)))
        self.pose_3_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                   self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(
                                   self.screen_height / 35))

        limbs_image = Image.open('modules/src/utils/pngs/limbs.png')
        limbs_image = limbs_image.resize((int(self.screen_width / 25), int(self.screen_height / 25)),
                                         Image.ANTIALIAS)
        limbs_image = ImageTk.PhotoImage(limbs_image)
        self.limbs_mark = tkinter.Label(image=limbs_image)
        self.limbs_mark.image = limbs_image
        self.limbs_mark.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40),
                                  y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))

        self.limbs_heading = tkinter.Label(text='Limbs Details:',
                                            font=("Times 20 italic bold", int(self.screen_width / 55)), fg='#969600')
        self.limbs_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40)
                                  +int((self.screen_width / 25) + 0.2*int(self.screen_width / 25)),
                                  y=int(
                                      self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                      self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(
                                      self.screen_height / 35))

        self.limbs_1_heading = tkinter.Label(text='Right Arm:',
                                            font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.limbs_1_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40)
                                  + int((self.screen_width / 25) + 0.2*int(self.screen_width / 25)),
                                   y=int(
                                       self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                       self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(
                                       self.screen_height / 35))
        self.limbs_2_heading = tkinter.Label(text='Left Arm:',
                                             font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.limbs_2_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40)
                                     + int((self.screen_width / 25) + 0.2 * int(self.screen_width / 25)),
                                   y=int(
                                       self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                       self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(
                                       self.screen_height / 35))
        self.limbs_3_heading = tkinter.Label(text='Right Leg:',
                                             font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.limbs_3_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40)
                                     + int((self.screen_width / 25) + 0.2 * int(self.screen_width / 25)),
                                   y=int(
                                       self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                       self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(
                                       self.screen_height / 35))
        self.limbs_4_heading = tkinter.Label(text='Left Leg:',
                                             font=("Times 20 italic bold", int(self.screen_width / 75)))
        self.limbs_4_heading.place(x=int(self.screen_width / 30 + self.screen_width / 40 + 0.4 * self.screen_width / 40)
                                     + int((self.screen_width / 25) + 0.2 * int(self.screen_width / 25)),
                                   y=int(
                                       self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                       self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(
                                       self.screen_height / 35))

        self.limbs_1_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                        width=int(self.screen_width / (35)))
        self.limbs_1_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))
        self.limbs_2_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_2_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))
        self.limbs_3_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_3_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))
        self.limbs_4_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_4_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))

        self.style = ttk.Style()

        self.create_widgets()

        self.window.mainloop()

    def create_widgets(self):
        self.style.theme_use('alt')
        self.style.configure('TButton', font=('American typewriter', int(self.screen_width/100)), background='#232323', foreground='white')
        self.style.map('TButton',
                       background=[('active', 'red'), ('disabled', 'gray')], click=[('active', 'blue'), ('disabled', 'gray')],
                       )
        self.button1 = ttk.Button(self.window, text="Load Image", command=self.process)
        self.button1.place(x=int(self.screen_width/2 - self.screen_width/20), y=int(self.screen_height/10 - 1.2*self.screen_height/15), height=int(self.screen_height/15), width=int(self.screen_width/10))
        self.stop_button = tkinter.PhotoImage(file=r'modules/src/utils/pngs/exit.png').subsample(int(self.screen_width/80), int(self.screen_width/80))
        self.restart_button = tkinter.PhotoImage(file=r'modules/src/utils/pngs/reload.png').subsample(int(self.screen_width/80), int(self.screen_width/80))
        self.button2 = ttk.Button(self.window, image=self.stop_button, command=self.window.destroy)
        self.button3 = ttk.Button(self.window, image=self.restart_button, command=self.reload)
        self.button2.place(x=int(self.screen_width - self.screen_width/20), y=0, height=int(self.screen_width/20), width=int(self.screen_width/20))
        self.button3.place(x=0, y=0, height=int(self.screen_width/20), width=int(self.screen_width/20))

    def process(self):
        self.reload()
        image_path = askopenfilename()
        image = cv2.imread(image_path)
        img_copy = image.copy()
        final_result, uniques = main_process(image, input_size, model_parsing, transform, labels)
        final_mask, detections = post_proces(image, final_result, uniques, csv, labels)
        colors_data = get_color_palette(detections)

        if len(colors_data['bg']) != 0:
            bg_color = rgb_to_hex(tuple(colors_data['bg'][1]))
            bg_text = colors_data['bg'][0]
            self.can_bg.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=bg_color)
            self.bg_text.insert(tkinter.END, bg_text)
        else:
            self.bg_text.insert(tkinter.END, 'Background not Detected')

        if len(colors_data['hair']) != 0:
            hair_color = rgb_to_hex(tuple(colors_data['hair'][1]))
            hair_text = colors_data['hair'][0]
            self.can_hair.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=hair_color)
            self.hair_text.insert(tkinter.END, hair_text)
        else:
            self.hair_text.insert(tkinter.END, 'Hair not Detected')

        if len(colors_data['skin']) != 0:
            skin_color = rgb_to_hex(tuple(colors_data['skin'][1]))
            skin_text = colors_data['skin'][0]
            self.can_skin.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=skin_color)
            self.skin_text.insert(tkinter.END, skin_text)
        else:
            self.skin_text.insert(tkinter.END, 'Skin not Detected')

        if len(colors_data['shirt']) != 0:
            shirt_color = rgb_to_hex(tuple(colors_data['shirt'][1]))
            shirt_text = colors_data['shirt'][0]
            self.can_shirt.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=shirt_color)
            self.shirt_text.insert(tkinter.END, shirt_text)
        else:
            self.shirt_text.insert(tkinter.END, 'Shirt not Detected')

        if len(colors_data['pants']) != 0:
            pants_color = rgb_to_hex(tuple(colors_data['pants'][1]))
            pants_text = colors_data['pants'][0]
            self.can_pants.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=pants_color)
            self.pants_text.insert(tkinter.END, pants_text)
        else:
            self.pants_text.insert(tkinter.END, 'Pants not Detected')

        if len(colors_data['dress']) != 0:
            dress_color = rgb_to_hex(tuple(colors_data['dress'][1]))
            dress_text = colors_data['dress'][0]

            self.can_dress.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=dress_color)
            self.dress_text.insert(tkinter.END, dress_text)
        else:
            self.dress_text.insert(tkinter.END, 'Dress not Detected')

        if len(colors_data['skirt']) != 0:
            skirt_color = rgb_to_hex(tuple(colors_data['skirt'][1]))
            skirt_text = colors_data['skirt'][0]
            self.skirt_text.insert(tkinter.END, skirt_text)
            self.can_skirt.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=skirt_color)
        else:
            self.skirt_text.insert(tkinter.END, 'Skirt not Detected')

        if len(colors_data['shoes']) != 0:
            shoes_color = rgb_to_hex(tuple(colors_data['shoes'][1]))
            shoes_text = colors_data['shoes'][0]
            self.shoes_text.insert(tkinter.END, shoes_text)
            self.can_shoes.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=shoes_color)
        else:
            self.shoes_text.insert(tkinter.END, 'Shoes not Detected')

        if len(colors_data['hat']) != 0:
            hat_color = rgb_to_hex(tuple(colors_data['hat'][1]))
            hat_text = colors_data['hat'][0]
            self.hat_text.insert(tkinter.END, hat_text)
            self.can_hat.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=hat_color)
        else:
            self.hat_text.insert(tkinter.END, 'Hat not Detected')

        if len(colors_data['sunglasses']) != 0:
            sunglasses_color = rgb_to_hex(tuple(colors_data['sunglasses'][1]))
            sunglasses_text = colors_data['sunglasses'][0]
            self.sunglasses_text.insert(tkinter.END, sunglasses_text)
            self.can_sunglasses.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=sunglasses_color)
        else:
            self.sunglasses_text.insert(tkinter.END, 'Glasses not Detected')

        if len(colors_data['bag']) != 0:
            bag_color = rgb_to_hex(tuple(colors_data['bag'][1]))
            bag_text = colors_data['bag'][0]
            self.bag_text.insert(tkinter.END, bag_text)
            self.can_bag.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=bag_color)
        else:
            self.bag_text.insert(tkinter.END, 'Bag not Detected')

        if len(colors_data['belt']) != 0:
            belt_color = rgb_to_hex(tuple(colors_data['belt'][1]))
            belt_text = colors_data['belt'][0]
            self.belt_text.insert(tkinter.END, belt_text)
            self.can_belt.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=belt_color)
        else:
            self.belt_text.insert(tkinter.END, 'Belt not Detected')

        if len(colors_data['scarf']) != 0:
            scarf_color = rgb_to_hex(tuple(colors_data['scarf'][1]))
            scarf_text = colors_data['scarf'][0]
            self.scarf_text.insert(tkinter.END, scarf_text)

            self.can_bg.create_rectangle(0, 0, int(self.screen_width / 40),
                                int(self.screen_width / 40), fill=scarf_color)
        else:
            self.scarf_text.insert(tkinter.END, 'Scarf not Detected')

        points = detect(img_copy, body_estimation)
        insights_data = get_insights(points)

        self.pose_1_text.insert(tkinter.END, insights_data['Main Pose'])
        self.pose_2_text.insert(tkinter.END, insights_data['Position'])
        self.pose_3_text.insert(tkinter.END, insights_data['Detailed Posture'])

        self.limbs_1_text.insert(tkinter.END, insights_data['Right Arm Posture'])
        self.limbs_2_text.insert(tkinter.END, insights_data['Left Arm Posture'])
        self.limbs_3_text.insert(tkinter.END, insights_data['Right Knee Posture'])
        self.limbs_4_text.insert(tkinter.END, insights_data['Left Knee Posture'])

        out_img = draw_pose(final_mask, points, pose_pairs, pose_colors)
        frame_1 = cv2.cvtColor(cv2.resize(image, (self.picture_width, self.picture_height)), cv2.COLOR_BGR2RGB)
        frame_2 = cv2.cvtColor(cv2.resize(out_img, (self.picture_width, self.picture_height)), cv2.COLOR_BGR2RGB)
        self.photo_1 = ImageTk.PhotoImage(image=Image.fromarray(frame_1))
        self.photo_2 = ImageTk.PhotoImage(image=Image.fromarray(frame_2))
        self.canvas_1.create_image(0, 0, image=self.photo_1, anchor=tkinter.NW)
        self.canvas_2.create_image(0, 0, image=self.photo_2, anchor=tkinter.NW)

    def reload(self):
        self.canvas_1 = tkinter.Canvas(self.window, width=self.picture_width, height=self.picture_height,
                                       highlightthickness=0.5, highlightbackground="gray")
        self.canvas_2 = tkinter.Canvas(self.window, width=self.picture_width, height=self.picture_height,
                                       highlightthickness=0.5, highlightbackground="gray")
        self.canvas_1.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                            y=int(self.screen_height / 9))
        self.canvas_2.place(x=int(self.screen_width / 2 + 0.1 * self.picture_width), y=int(self.screen_height / 9))

        self.can_bg = tkinter.Canvas(self.window, width=int(self.screen_width / 55), height=int(self.screen_width / 55),
                                     highlightthickness=0.5, highlightbackground="gray")
        self.can_bg.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                self.screen_height / 18))
        self.can_hair = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_hair.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.135 * int(
                self.screen_height / 18) + int(self.screen_height / 35) + 0.2 * int(self.screen_height / 35))
        self.can_skin = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_skin.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))

        self.can_shirt = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_shirt.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_pants = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_pants.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(4 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_dress = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_dress.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_skirt = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_skirt.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(6 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_shoes = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_shoes.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_hat = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                      height=int(self.screen_width / 55),
                                      highlightthickness=0.5, highlightbackground="gray")
        self.can_hat.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_sunglasses = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                             height=int(self.screen_width / 55),
                                             highlightthickness=0.5, highlightbackground="gray")
        self.can_sunglasses.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_bag = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                      height=int(self.screen_width / 55),
                                      highlightthickness=0.5, highlightbackground="gray")
        self.can_bag.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_belt = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                       height=int(self.screen_width / 55),
                                       highlightthickness=0.5, highlightbackground="gray")
        self.can_belt.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(11 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))
        self.can_scarf = tkinter.Canvas(self.window, width=int(self.screen_width / 55),
                                        height=int(self.screen_width / 55),
                                        highlightthickness=0.5, highlightbackground="gray")
        self.can_scarf.place(
            x=int(self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)),
            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                self.screen_height / 18) + int(12 * self.screen_height / 35) + 0.15 * int(self.screen_height / 35))

        self.bg_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                    width=int(self.screen_width / (50)))
        self.bg_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                           y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                               self.screen_height / 18))
        self.hair_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                      width=int(self.screen_width / (50)))
        self.hair_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                             y=int(
                                 self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.135 * int(
                                 self.screen_height / 18) + int(self.screen_height / 35) + 0.2 * int(
                                 self.screen_height / 35))
        self.skin_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                      width=int(self.screen_width / (50)))
        self.skin_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                             y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                 self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.15 * int(
                                 self.screen_height / 35))
        self.shirt_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.shirt_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))
        self.pants_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.pants_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(4 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))
        self.dress_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.dress_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(5 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))
        self.skirt_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.skirt_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(6 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))
        self.shoes_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.shoes_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))
        self.hat_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                     width=int(self.screen_width / (50)))
        self.hat_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                            y=int(
                                self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(
                                self.screen_height / 35))
        self.sunglasses_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                            width=int(self.screen_width / (50)))
        self.sunglasses_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                                   y=int(
                                       self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                       self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(
                                       self.screen_height / 35))
        self.bag_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                     width=int(self.screen_width / (50)))
        self.bag_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                            y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(
                                self.screen_height / 35))
        self.belt_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                      width=int(self.screen_width / (50)))
        self.belt_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                             y=int(self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                 self.screen_height / 18) + int(11 * self.screen_height / 35) + 0.15 * int(
                                 self.screen_height / 35))
        self.scarf_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                       width=int(self.screen_width / (50)))
        self.scarf_text.place(x=int(
            self.screen_width / 2 + 0.1 * self.picture_width + self.picture_width - int(self.screen_width / 15)) + int(
            int(self.screen_width / 40)) + int(self.screen_width / 55),
                              y=int(
                                  self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                  self.screen_height / 18) + int(12 * self.screen_height / 35) + 0.15 * int(
                                  self.screen_height / 35))

        self.pose_1_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                        width=int(self.screen_width / (35)))
        self.pose_1_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                   self.screen_height / 18) + int(self.screen_height / 35) + 0.1 * int(
                                   self.screen_height / 35))

        self.pose_2_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                        width=int(self.screen_width / (35)))
        self.pose_2_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.2 * int(
                                   self.screen_height / 18) + int(2 * self.screen_height / 35) + 0.1 * int(
                                   self.screen_height / 35))

        self.pose_3_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                        width=int(self.screen_width / (35)))
        self.pose_3_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                               y=int(
                                   self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                   self.screen_height / 18) + int(3 * self.screen_height / 35) + 0.15 * int(
                                   self.screen_height / 35))

        self.limbs_1_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_1_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(7 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))
        self.limbs_2_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_2_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(8 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))
        self.limbs_3_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_3_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(9 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))
        self.limbs_4_text = tkinter.Text(self.window, height=int(self.screen_height / (400)),
                                         width=int(self.screen_width / (35)))
        self.limbs_4_text.place(x=int(self.screen_width / 2 - self.picture_width - 0.1 * self.picture_width),
                                y=int(
                                    self.screen_height / 9 + self.picture_height + self.screen_height / 90) + 1.18 * int(
                                    self.screen_height / 18) + int(10 * self.screen_height / 35) + 0.15 * int(
                                    self.screen_height / 35))


if __name__ == "__main__":
    App(tkinter.Tk(), "Fashion Plus GUI")
