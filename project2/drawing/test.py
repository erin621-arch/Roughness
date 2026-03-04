import tkinter as tk

def simple_test():
    root = tk.Tk()
    root.title("テスト")
    root.geometry("400x300")
    
    label = tk.Label(root, text="これはテストです")
    label.pack(pady=50)
    
    print("GUIを表示します...")
    root.mainloop()
    print("mainloopが終了しました")

if __name__ == "__main__":
    simple_test()