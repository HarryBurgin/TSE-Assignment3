import tkinter as tk
import csv

import NeuralNetLib as NN
 
class SingleReviewWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Sentiment Analysis for Movie Reviews")
        self.verdict1 = tk.StringVar()
        self.verdict2 = tk.StringVar()
 
        self.frame = tk.Frame(self.master, width = 1000, height = 600) ## Window
        self.frame.pack()

        
 
        self.label = tk.Label(self.frame, text = "Sentiment Analysis for Movie Reviews", font=("Arial",25)) ## Title
        self.label.place( x = 12, y = 20)

        self.label2 = tk.Label(self.frame, text = "By Alex, Ben, George, Harry, Joe, and Will", font=("Arial",15)) ## Credits Subtitle
        self.label2.place( x = 12, y = 70)

        self.label3 = tk.Label(self.frame, text = "Single Review: Type Review Below", font=("Arial",12)) ## Subtitle for Single Review
        self.label3.place( x = 12, y = 135)

        self.entry = tk.Entry(self.frame, width =161) ## Input for Single Review
        self.entry.place( x = 12, y = 160)
        
        self.button = tk.Button(self.frame, text = "Analyse Review ", command=self.runMLsingle, font=("Arial",20)) ## Confirm button for Single Review
        self.button.place( x = 12, y = 200)

        self.label4 = tk.Label(self.frame, textvariable = self.verdict1, font=("Arial",15)) ## Result for Single Review
        self.label4.place( x = 250, y = 215)

        

        self.label5 = tk.Label(self.frame, text = "Multiple Reviews: Type File Location (Must be .csv)", font=("Arial",12)) ## Subtitle for Multi Review
        self.label5.place( x = 12, y = 305)

        self.entry2 = tk.Entry(self.frame, width =161) ## Input for Multi Review
        self.entry2.place( x = 12, y = 330)

        self.button2 = tk.Button(self.frame, text = "Analyse Reviews", command=self.runMLmulti, font=("Arial",20)) ## Confirm button for Single Review
        self.button2.place( x = 12, y = 370)

        self.label6 = tk.Label(self.frame, textvariable = self.verdict2, font=("Arial",10)) ## Result for Multi Review
        self.label6.place( x = 250, y = 365)



        self.button.bind('<Enter>', lambda event: self.color_change(self.button, "green"))
        self.button.bind('<Leave>', lambda event: self.color_change(self.button, "black"))

        self.button2.bind('<Enter>', lambda event: self.color_change(self.button2, "green"))
        self.button2.bind('<Leave>', lambda event: self.color_change(self.button2, "black"))

        master.mainloop()
 
    def color_change(self, widget, color):
        widget.config(foreground = color)

    def runMLsingle(self):
        if self.entry.get() != "":
            result = NN.predictReview([self.entry.get()])
            confidence = round(((abs(0.5 - result[0][0])* 2) * 100), 2)
            if result[0][0] < 0.5:
                text = str("Negative : ") + str(confidence) + str("% Confidence")
                self.verdict1.set(text)
            else:
                text = str("Positive : ") + str(confidence) + str("% Confidence")
                self.verdict1.set(text)

    def runMLmulti(self):
        path = self.entry2.get()
        try:
            fileName = open(path, 'r', encoding='utf-8-sig')
        except:
            print("File Not Found")
            return

        try:
            file = csv.DictReader(fileName)
        except:
            print("File Not Valid")
            return

        reviews = []
        for col in file:
            reviews.append(col['review'])

        results = NN.predictReview(reviews)
        output = ""
        for result in range(0,len(results)):
            confidence = round(((abs(0.5 - results[result][0])* 2) * 100), 2)
            if results[result][0] < 0.5:
                output += (str("Negative : ") + str(confidence) + str("% Confidence") + str("\n"))
            else:
                output += (str("Positive : ") + str(confidence) + str("% Confidence") + str("\n"))

        self.verdict2.set(output)

        

window = SingleReviewWindow(tk.Tk())

