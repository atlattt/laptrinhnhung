import os
import socket
import wave
import numpy as np
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Thông tin kết nối đến ESP32
HOST = '192.168.137.186'
PORT = 8888

# Cấu hình âm thanh
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 4  
CHANNELS = 1
RECORD_TIME = 10  

# Thư mục lưu file
NORMAL_DIR = "normal"
HAPPY_DIR = "happy"
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(HAPPY_DIR, exist_ok=True)

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder & Classifier")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')

        # Frame chính
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Tiêu đề
        self.label = ttk.Label(main_frame, text="ESP32 Audio Recorder", 
                            font=("Helvetica", 18, "bold"),
                            foreground="#2c3e50")
        self.label.pack(pady=(0, 20))

        # Biểu đồ sóng
        self.canvas = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax = self.canvas.add_subplot(111)
        self.ax.set_title("Real-time Waveform")
        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], color='#3498db')
        plt.style.use('seaborn-v0_8')

        self.graph = FigureCanvasTkAgg(self.canvas, master=main_frame)
        self.graph.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

        # Frame cho các nút
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        # Nút ghi âm
        self.record_button = ttk.Button(button_frame, 
                                    text="Start Recording", 
                                    command=self.start_recording,
                                    style='Record.TButton')
        self.record_button.pack(pady=(0, 15))

        # Frame cho các nút lưu
        save_frame = ttk.Frame(button_frame)
        save_frame.pack()

        # Nút lưu
        self.save_normal_button = ttk.Button(save_frame, 
                                        text="Save as Normal", 
                                        command=lambda: self.save_audio(NORMAL_DIR),
                                        style='Save.TButton',
                                        state=tk.DISABLED)
        self.save_normal_button.pack(side=tk.LEFT, padx=10)

        self.save_happy_button = ttk.Button(save_frame, 
                                        text="Save as Happy", 
                                        command=lambda: self.save_audio(HAPPY_DIR),
                                        style='Save.TButton',
                                        state=tk.DISABLED)
        self.save_happy_button.pack(side=tk.RIGHT, padx=10)

        # Thiết lập style
        style = ttk.Style()
        style.configure('Record.TButton', 
                    font=('Helvetica', 12, 'bold'), 
                    foreground='black', 
                    background='#3498db')
        style.configure('Save.TButton', 
                    font=('Helvetica', 11), 
                    foreground='black', 
                    background='#2ecc71')

        self.audio_data_np = None
        self.filename = None
        self.is_recording = False

    def start_recording(self):
        """Bắt đầu quá trình ghi âm và cập nhật biểu đồ thời gian thực."""
        self.record_button.config(state=tk.DISABLED)
        self.is_recording = True
        self.audio_samples_list = []
        self.total_received_samples = 0
        self.expected_samples = SAMPLE_RATE * RECORD_TIME

        # Khởi tạo socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.client_socket.connect((HOST, PORT))
            print(f"Đã kết nối đến ESP32: {HOST}:{PORT}")
            print("Bắt đầu nhận stream...")
            self.root.after(100, self.receive_audio_stream)  # Bắt đầu vòng lặp nhận dữ liệu
        except socket.error as e:
            print(f"Lỗi socket: {e}")
            self.stop_recording()

    def receive_audio_stream(self):
        """Nhận stream âm thanh và cập nhật biểu đồ thời gian thực."""
        if not self.is_recording:
            return

        try:
            chunk = self.client_socket.recv(4096)
            if not chunk:
                print("ESP32 ngắt kết nối.")
                self.stop_recording()
                return

            audio_chunk_np = np.frombuffer(chunk, dtype=np.int32)
            self.audio_samples_list.extend(audio_chunk_np)
            self.total_received_samples += len(audio_chunk_np)

            # Cập nhật biểu đồ thời gian thực
            self.update_waveform(self.audio_samples_list)

            # Kiểm tra nếu đã đủ thời gian ghi âm
            if self.total_received_samples >= self.expected_samples:
                print("Hoàn thành nhận stream.")
                self.audio_data_np = np.array(self.audio_samples_list)
                self.filename = f"{int(time.time())}.wav"
                self.save_wav(self.filename, self.audio_data_np)
                self.save_normal_button.config(state=tk.NORMAL)
                self.save_happy_button.config(state=tk.NORMAL)
                self.stop_recording()
            else:
                # Tiếp tục nhận dữ liệu sau 10ms
                self.root.after(10, self.receive_audio_stream)

        except socket.error as e:
            print(f"Lỗi socket: {e}")
            self.stop_recording()

    def stop_recording(self):
        """Dừng quá trình ghi âm và đóng socket."""
        self.is_recording = False
        self.client_socket.close()
        self.record_button.config(state=tk.NORMAL)

    def update_waveform(self, data):
        """Cập nhật biểu đồ sóng thời gian thực."""
        self.ax.clear()
        self.ax.set_title("Real-time Waveform", fontsize=12)
        self.ax.set_xlabel("Samples", fontsize=10)
        self.ax.set_ylabel("Amplitude", fontsize=10)
        self.ax.grid(True)

        ydata = np.array(data[-512:])  # Lấy 512 mẫu cuối cùng
        xdata = np.arange(len(ydata))
        self.ax.plot(xdata, ydata, color='#3498db')
        self.graph.draw()

    def save_wav(self, filename, data):
        """Lưu dữ liệu âm thanh thành file WAV."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(data.tobytes())
        print(f"Đã lưu file WAV: {filename}")

    def save_audio(self, folder):
        """Lưu file âm thanh vào thư mục được chọn."""
        if self.filename and os.path.exists(self.filename):
            new_path = os.path.join(folder, os.path.basename(self.filename))
            os.rename(self.filename, new_path)
            messagebox.showinfo("Success", f"File saved to {new_path}")
            self.save_normal_button.config(state=tk.DISABLED)
            self.save_happy_button.config(state=tk.DISABLED)
        else:
            messagebox.showerror("Error", "No recorded file found")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()