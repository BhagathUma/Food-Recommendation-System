import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

# Load dataset
food_data = pd.read_csv("food-dataset-extended-2.csv", encoding="utf-8")

rating_columns = ["spice level", "sweet level", "sour level", "salt level", "bitter level"]
food_data[rating_columns] = food_data[rating_columns] / 10.0

ratings = food_data[rating_columns].mean(axis=1).values
dishes = np.arange(len(food_data)) 

X_train, X_test, y_train, y_test = train_test_split(dishes, ratings, test_size=0.2, random_state=42)

dish_input = Input(shape=(1,), name="dish_input")
dish_embedding = Embedding(input_dim=len(food_data), output_dim=16, name="dish_embedding")(dish_input)
dish_embedding = Flatten()(dish_embedding)

x = Dense(64, activation='relu')(dish_embedding)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)

output = Dense(1, activation='sigmoid', name="output_layer")(x)

ncf_model = Model(inputs=dish_input, outputs=output)
ncf_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
ncf_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

def recommend_dishes():
    cuisine = cuisine_var.get().lower()
    taste = taste_var.get().lower()
    temperature = temperature_var.get().lower()
    spice_level = spice_var.get() / 10.0
    sweet_level = sweet_var.get() / 10.0
    salt_level = salt_var.get() / 10.0
    sour_level = sour_var.get() / 10.0
    bitter_level = bitter_var.get() / 10.0
    
    user_prefs = {
        "cuisine": cuisine,
        "taste": taste,
        "temperature": temperature,
        "ratings": np.array([spice_level, sweet_level, salt_level, sour_level, bitter_level]).mean()
    }
    
    filtered_data = food_data[(food_data['Cuisine'].str.lower() == user_prefs['cuisine']) &
                              (food_data['taste'].str.lower() == user_prefs['taste']) &
                              (food_data['temperature'].str.lower() == user_prefs['temperature'])]
    
    if filtered_data.empty:
        result_label.config(text="No exact matches found. Showing top-rated dishes instead.")
        filtered_data = food_data
    
    dish_indices = np.arange(len(filtered_data))
    predicted_scores = ncf_model.predict(dish_indices)
    top_n_indices = np.argsort(predicted_scores.flatten())[::-1][:5]
    
    recommended_dishes = "\n".join(filtered_data.iloc[top_n_indices]['Dish name'].tolist())
    result_label.config(text=f"Top 5 Recommended Dishes:\n{recommended_dishes}")

# GUI Setup
root = tk.Tk()
root.title("Food Recommendation System")

cuisine_var = tk.StringVar()
taste_var = tk.StringVar()
temperature_var = tk.StringVar()
spice_var = tk.IntVar()
sweet_var = tk.IntVar()
salt_var = tk.IntVar()
sour_var = tk.IntVar()
bitter_var = tk.IntVar()

cuisines = ["Indian", "Chinese", "American"]
tastes = ["Sweet", "Sour", "Bitter","Spicy","Salty"]
temperatures = ["Hot", "Cold"]

tk.Label(root, text="Cuisine:").pack()
ttk.Combobox(root, textvariable=cuisine_var, values=cuisines).pack()

tk.Label(root, text="Taste:").pack()
ttk.Combobox(root, textvariable=taste_var, values=tastes).pack()

tk.Label(root, text="Temperature:").pack()
ttk.Combobox(root, textvariable=temperature_var, values=temperatures).pack()

tk.Label(root, text="Spice Level (0-10):").pack()
tk.Scale(root, from_=0, to=10, orient=tk.HORIZONTAL, variable=spice_var).pack()

tk.Label(root, text="Sweet Level (0-10):").pack()
tk.Scale(root, from_=0, to=10, orient=tk.HORIZONTAL, variable=sweet_var).pack()

tk.Label(root, text="Salt Level (0-10):").pack()
tk.Scale(root, from_=0, to=10, orient=tk.HORIZONTAL, variable=salt_var).pack()

tk.Label(root, text="Sour Level (0-10):").pack()
tk.Scale(root, from_=0, to=10, orient=tk.HORIZONTAL, variable=sour_var).pack()

tk.Label(root, text="Bitter Level (0-10):").pack()
tk.Scale(root, from_=0, to=10, orient=tk.HORIZONTAL, variable=bitter_var).pack()

tk.Button(root, text="Get Recommendations", command=recommend_dishes).pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
