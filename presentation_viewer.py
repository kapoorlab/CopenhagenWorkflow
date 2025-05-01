import fitz  # PyMuPDF
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class PDFWindowViewer:
    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)
        self.total_pages = len(self.doc)
        self.current_page = 0

        self.root = tk.Tk()
        self.root.title("Movable PDF Viewer")
        self.root.geometry("1024x768")  # Start size, can resize manually
        self.root.configure(bg='black')

        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.tk_img = None
        self.img_size = None

        self.canvas.bind("<Configure>", self.render_page)
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Right>", self.next_slide)
        self.root.bind("<Left>", self.prev_slide)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.root.mainloop()

    def render_page(self, event=None):
        page = self.doc[self.current_page]
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        scale = min(canvas_width / pix.width, canvas_height / pix.height)
        new_size = (int(pix.width * scale), int(pix.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        self.img_size = new_size

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        x = (canvas_width - new_size[0]) // 2
        y = (canvas_height - new_size[1]) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_img)
        self.root.title(f"Slide {self.current_page + 1} / {self.total_pages}")

    def next_slide(self, event=None):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.render_page()

    def prev_slide(self, event=None):
        if self.current_page > 0:
            self.current_page -= 1
            self.render_page()

    def on_click(self, event=None):
        links = self.doc[self.current_page].get_links()
        for link in links:
            uri = link.get("uri") or link.get("file")
            if uri:
                print(f"▶️ Opening: {uri}")
                try:
                    os.startfile(uri)
                except Exception as e:
                    print(f"❌ Failed to open: {uri} | {e}")
                return
        print("⚠️ No link found on this slide.")

# Select PDF
pdf_file = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF files", "*.pdf")])
if pdf_file:
    PDFWindowViewer(pdf_file)
