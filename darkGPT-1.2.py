# REVISED code as of 19/2/25 (my code still sucks lmao)
# this code crashed my computer (9300h gtx 1650  (i shouldnt be flexing)) so run at ur own risk.
# made using a tutorial from youtube. 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext
import threading

model_name = "EleutherAI/gpt-neo-1.3B" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

chat_history_ids = None

# yapping
def generate_response(user_input):
    global chat_history_ids

    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        new_user_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    bot_output = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    bot_response = tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    chat_history_ids = bot_output

    return bot_response

# da gui
def on_message(event=None):
    user_input = message_box.get()
    if user_input.strip():
        response_text.configure(state='normal')
        response_text.insert(tk.END, f"You: {user_input}\n")
        response_text.configure(state='disabled')
        response_text.yview(tk.END)

        def respond():
            response = generate_response(user_input)
            response_text.configure(state='normal')
            response_text.insert(tk.END, f"AI: {response}\n\n")
            response_text.configure(state='disabled')
            response_text.yview(tk.END)

        threading.Thread(target=respond).start()
    message_box.delete(0, tk.END)

# da gui
def start_chatbot():
    global message_box, response_text

    app = tk.Tk()
    app.title("darkGPT-1.2")  # cringe name lol
    app.configure(bg="#121212")

    message_box = tk.Entry(app, width=50, bg="#1E1E1E", fg="white", insertbackground="white", bd=0, highlightthickness=1, highlightbackground="#333333")
    message_box.pack(pady=5, padx=10)
    message_box.bind('<Return>', on_message)

    response_text = scrolledtext.ScrolledText(app, width=60, height=20, bg="#1E1E1E", fg="white", insertbackground="white", bd=0, highlightthickness=1, highlightbackground="#333333", wrap=tk.WORD)
    response_text.pack(pady=5, padx=10)
    response_text.configure(state='disabled')

    # send
    message_button = tk.Button(app, text="Send", command=on_message, bg="#333333", fg="white", activebackground="#555555", activeforeground="white", bd=0, highlightthickness=1, highlightbackground="#555555")
    message_button.pack(pady=5)

    app.mainloop()

if __name__ == '__main__':
    start_chatbot()
