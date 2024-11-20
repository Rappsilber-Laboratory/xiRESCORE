import os
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import logging
import threading
from importlib.resources import files

import xirescore

threads = []
xi_proc = None

class GuiLoggingHandler(logging.Handler):
    """
    This class allows logging messages to be sent to a Tkinter Text widget.
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.config(state=tk.DISABLED)
        self.text_widget.see(tk.END)  # Auto-scroll to the end


def log_subprocess_output(pipe, logger, level=logging.INFO):
    for line in iter(pipe.readline, b''):  # Read line by line
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if line.endswith(os.linesep):
            lines = line.split(os.linesep)
            line = os.linesep.join(lines[:-1])
        if line == '':
            continue
        logger.log(level, f'{line}')  # Log each line
    pipe.close()


def run_xirescore(input_path, config_path, output_path, logger):
    global xi_proc
    # Run xiRescore
    opt_config = []
    if config_path.get() != '':
        opt_config = ['-c', config_path.get()]
    xi_proc = subprocess.Popen(
        [
            sys.executable,
            "-m", "xirescore",
            "-i", f"{input_path.get()}",
            "-o", f"{output_path.get()}",
        ]+opt_config,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )

    stdout_thread = threading.Thread(target=log_subprocess_output, args=(xi_proc.stdout, logger))
    stderr_thread = threading.Thread(target=log_subprocess_output, args=(xi_proc.stderr, logger))

    stdout_thread.start()
    stderr_thread.start()

    threads.append(stdout_thread)
    threads.append(stderr_thread)


def _open_file_selector(filepath_var):
    """Open a file dialog to select a file."""
    filepath = filedialog.askopenfilename(title="Select a file")
    filepath_var.set(filepath)


def _save_file_selector(filepath_var):
    """Open a file dialog to specify a file to save."""
    filepath = filedialog.asksaveasfilename(
        title="Specify a file to save",
        defaultextension=".csv",
        filetypes=[
            ("Comma Separated Values", "*.csv"),
            ("Tab Separated Values", "*.tsv"),
            ("Zipped Comma Separated Values", "*.csv.gz"),
            ("Zipped Tab Separated Values", "*.tsv.gz"),
            ("Apache Parquet", "*.parquet"),
        ]
    )
    filepath_var.set(filepath)


def info_box(root, title, text):
    popup = tk.Toplevel(root)
    popup.title(title)
    label = tk.Label(popup, text=text)
    label.pack(pady=10)
    close_button = tk.Button(popup, text="Close", command=popup.destroy)
    close_button.pack(pady=5)


def check_finished(root):
    global xi_proc
    if xi_proc is not None:
        exit_code = xi_proc.poll()
        if exit_code is not None:
            if exit_code == 0:
                info_box(root, "Info", "Rescoring finished")
            else:
                info_box(
                    root,
                    "ERROR",
                    f"Rescoring failed with exit code {exit_code}!"
                )
            xi_proc = None
    root.after(1000, lambda: check_finished(root))

# Create the GUI
def create_gui():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    root = tk.Tk()
    icon = tk.PhotoImage(file=files("xirescore.assets").joinpath("xirescore_logo.png"))
    root.iconphoto(False, icon)
    root.title(f"xiRescore {xirescore.__version__}")
    root.rowconfigure(4, weight=1)
    root.columnconfigure(1, weight=1)

    # Filepath variables
    filepath_input = tk.StringVar()
    filepath_config = tk.StringVar()
    filepath_output = tk.StringVar()

    # First File Selection
    label1 = tk.Label(root, text="Input:")
    label1.grid(row=0, column=0, padx=5, pady=5)
    entry1 = tk.Entry(root, textvariable=filepath_input, width=40, state='readonly')
    entry1.grid(row=0, column=1, padx=5, pady=5, sticky="new")
    select_button1 = tk.Button(root, text="Select File", command=lambda: _open_file_selector(filepath_input))
    select_button1.grid(row=0, column=2, padx=5, pady=5)

    # Second File Selection (No functionality yet)
    label2 = tk.Label(root, text="Config:")
    label2.grid(row=1, column=0, padx=5, pady=5)
    entry2 = tk.Entry(root, textvariable=filepath_config, width=40, state='readonly')
    entry2.grid(row=1, column=1, padx=5, pady=5, sticky="new")
    select_button2 = tk.Button(root, text="Select File", command=lambda: _open_file_selector(filepath_config))
    select_button2.grid(row=1, column=2, padx=5, pady=5)

    # Third File Selection (Save As functionality)
    label3 = tk.Label(root, text="Output:")
    label3.grid(row=2, column=0, padx=5, pady=5)
    entry3 = tk.Entry(root, textvariable=filepath_output, width=40, state='readonly')
    entry3.grid(row=2, column=1, padx=5, pady=5, sticky="new")
    select_button3 = tk.Button(root, text="Save As", command=lambda: _save_file_selector(filepath_output))
    select_button3.grid(row=2, column=2, padx=5, pady=5)

    # Go Button
    go_button = tk.Button(
        root, text="Go"
    )

    def on_go_button():
        entry1.config(state=tk.DISABLED)
        entry2.config(state=tk.DISABLED)
        entry3.config(state=tk.DISABLED)
        select_button1.config(state=tk.DISABLED)
        select_button2.config(state=tk.DISABLED)
        select_button3.config(state=tk.DISABLED)
        go_button.config(state=tk.DISABLED)
        # Run xiRescore
        run_xirescore(filepath_input, filepath_config, filepath_output, logger)

    go_button.config(command=on_go_button)
    go_button.grid(row=3, column=2, padx=5, pady=5)

    # Textbox for displaying logs
    label4 = tk.Label(root, text="Log:")
    label4.grid(row=4, column=0, padx=5, pady=5)
    textbox = scrolledtext.ScrolledText(root, height=15, width=60, wrap="none")
    textbox.config(state=tk.DISABLED)
    textbox.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="nsew")

    textbox_handler = GuiLoggingHandler(textbox)
    formatter = logging.Formatter('%(message)s')
    textbox_handler.setFormatter(formatter)
    logger.addHandler(textbox_handler)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(1000, lambda: check_finished(root))
    root.mainloop()


def on_close():
    os._exit(os.EX_OK)
