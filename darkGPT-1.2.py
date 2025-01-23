from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tkinter as tk
from tkinter import scrolledtext
import threading

# Load the pre-trained GPT-Neo 1.3B model and tokenizer from Hugging Face
model_name = "EleutherAI/gpt-neo-1.3B"  # GPT-Neo 1.3B, under 5GB

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize conversation history
chat_history_ids = None

# Function to generate a response using the chatbot
def generate_response(user_input):
    global chat_history_ids

    # Tokenize the user input and append EOS token to signal the end of input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # If there is previous conversation, add it to the input for context
    if chat_history_ids is not None:
        new_user_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate a response from the model
    bot_output = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the output to get the response
    bot_response = tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update the chat history to include the current conversation
    chat_history_ids = bot_output

    return bot_response

# GUI Setup
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

# Create the Tkinter GUI for the chatbot
def start_chatbot():
    global message_box, response_text

    app = tk.Tk()
    app.title("darkGPT-1.2")  # Updated app title
    app.configure(bg="#121212")

    # Create a message box (entry widget)
    message_box = tk.Entry(app, width=50, bg="#1E1E1E", fg="white", insertbackground="white", bd=0, highlightthickness=1, highlightbackground="#333333")
    message_box.pack(pady=5, padx=10)
    message_box.bind('<Return>', on_message)

    # Create the response text area (scrolledtext widget)
    response_text = scrolledtext.ScrolledText(app, width=60, height=20, bg="#1E1E1E", fg="white", insertbackground="white", bd=0, highlightthickness=1, highlightbackground="#333333", wrap=tk.WORD)
    response_text.pack(pady=5, padx=10)
    response_text.configure(state='disabled')

    # Create the send button
    message_button = tk.Button(app, text="Send", command=on_message, bg="#333333", fg="white", activebackground="#555555", activeforeground="white", bd=0, highlightthickness=1, highlightbackground="#555555")
    message_button.pack(pady=5)

    app.mainloop()

if __name__ == '__main__':
    start_chatbot()
