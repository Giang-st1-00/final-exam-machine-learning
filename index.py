import csv
import tkinter as tk
from tkinter import Button, Label, filedialog, messagebox, ttk

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

csv_columns = []


def load_csv_file():
    global csv_columns
    global csv_dt
    
    # Chọn tập tin CSV từ hộp thoại
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    
    if file_path:
        # Đọc dữ liệu từ tập tin CSV và chỉ chọn các cột quan trọng
        csv_dt = pd.read_csv(file_path, usecols=["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross margin percentage", "gross income", "Rating"])
        
        # Lấy tên các cột
        csv_columns = csv_dt.columns.tolist()
        
        # Làm sạch dữ liệu
        clean_data()
        
        # Cập nhật giá trị của combobox và listbox
        target_combobox["values"] = csv_columns
        update_input_listbox()


def clean_data():
    global csv_dt
    
    # Thay thế các ô chứa NaN bằng giá trị trung vị của cột tương ứng
    csv_dt.fillna(csv_dt.median(), inplace=True)
    median = csv_dt.median()
    # Hiển thị thông báo (nếu muốn)
    messagebox.showinfo("Clean Data Result", f"NaN values replaced with column medians. \n The median of each column will be: \n {median}")


def update_input_listbox():
    input_listbox.delete(0, tk.END)
    for column in csv_columns:
        if column != target_combobox.get():
            input_listbox.insert(tk.END, column)


def add_variable():
    selected_indices = input_listbox.curselection()
    for index in selected_indices:
        selected_variable = input_listbox.get(index)
        selected_listbox.insert(tk.END, selected_variable)
        input_listbox.delete(index)
    input_listbox.selection_clear(0, tk.END)


def remove_variable():
    selected_indices = selected_listbox.curselection()
    for index in selected_indices:
        removed_variable = selected_listbox.get(index)
        input_listbox.insert(tk.END, removed_variable)
        selected_listbox.delete(index)
    selected_listbox.selection_clear(0, tk.END)


def execute_model():
    target_variable = target_combobox.get()
    input_variables = selected_listbox.get(0, tk.END)
    le = LabelEncoder()

    # if input variable is categorical convert to numerical
    for column in input_variables:
        if csv_dt[column].dtype == "object":
            csv_dt[column] = le.fit_transform(csv_dt[column])
    if csv_dt[target_variable].dtype == "object":
        csv_dt[target_variable] = le.fit_transform(csv_dt[target_variable])
    # convert to list
    input_variables = list(input_variables)
    # delete empty items in the list
    input_variables = [x for x in input_variables if x != ""]

    X = csv_dt[input_variables]
    y = csv_dt[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = model_combobox.get()

    global model_train

    print("Executing model...")
    if model == "Logistic Regression":
        if not np.issubdtype(y_train.dtype, np.number):
            messagebox.showerror("Error", "Logistic Regression requires binary target variable.")
            return
        model_train = LogisticRegression()
    elif model == "KNN":
        model_train = KNeighborsClassifier()
    elif model == "Linear Regression":
        model_train = LinearRegression()
    elif model == "Decision Tree":
        model_train = DecisionTreeClassifier()

    model_train.fit(X_train, y_train)
    y_pred = model_train.predict(X_test)

    if model == "Logistic Regression" or model == "KNN" or model == "Decision Tree":
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=model_train.classes_,
                    yticklabels=model_train.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Model Result", f"Accuracy: {accuracy}")

        if model == "Decision Tree":
            # Plot Decision Tree
            plt.figure(figsize=(12, 8))
            tree.plot_tree(model_train, feature_names=input_variables, class_names=model_train.classes_,
                           filled=True)
            plt.show()

    elif model == "Linear Regression":

        model_train = LinearRegression()

        model_train.fit(X_train, y_train)

        y_pred = model_train.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        messagebox.showinfo("Model Result", f"R-squared: {r2}")

        # Plotting
        if len(input_variables) >= 2:
            # Vẽ biểu đồ 3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_test[input_variables[0]], X_test[input_variables[1]], y_test, color='black', label='Actual')
            ax.scatter(X_test[input_variables[0]], X_test[input_variables[1]], y_pred, color='blue', marker='^', label='Predicted')
            ax.set_xlabel(input_variables[0])
            ax.set_ylabel(input_variables[1])
            ax.set_zlabel(target_variable)
            plt.title('Linear Regression Prediction (3D)')
            plt.legend()
            plt.show()

            # Vẽ biểu đồ phụ thuộc của từng biến độc lập vào biến phụ thuộc
            for feature in input_variables:
                model_train_feature = LinearRegression()
                model_train_feature.fit(X_train[[feature]], y_train)
                y_pred_feature = model_train_feature.predict(X_test[[feature]])
                
                plt.figure(figsize=(8, 6))
                plt.scatter(X_test[feature], y_test, color='black', label='Actual')
                plt.plot(X_test[feature], y_pred_feature, color='blue', linewidth=3, label='Predicted')
                plt.xlabel(feature)
                plt.ylabel(target_variable)
                plt.title(f'Linear Regression Prediction: {feature} vs {target_variable}')
                plt.legend()
                plt.show()
        else:
            plt.scatter(X_test[input_variables[0]], y_test, color='black', label='Actual')
            plt.plot(X_test[input_variables[0]], y_pred, color='blue', linewidth=3, label='Predicted')
            plt.xlabel(input_variables[0])
            plt.ylabel(target_variable)
            plt.title('Linear Regression Prediction')
            plt.legend()
            plt.show()


# Tạo cửa sổ giao diện
window = tk.Tk()
window.title("Giao diện")
window.geometry("400x800")

# Tạo frame chứa các phần tử giao diện
frame = tk.Frame(window)
frame.pack(pady=10)

# Tạo nút "Load CSV File"
load_button = tk.Button(frame, text="Load CSV File", command=load_csv_file)
load_button.grid(row=0, column=0, padx=5)

# Tạo nút "Clean Data"
clean_button = tk.Button(frame, text="Clean Data", command=clean_data)
clean_button.grid(row=0, column=1, padx=5)

# Tạo nhãn "Select Target Variable"
target_label = tk.Label(frame, text="Select Target Variable")
target_label.grid(row=1, column=0, padx=5, sticky=tk.W)

# Tạo combobox để chọn Target Variable
target_combobox = ttk.Combobox(frame)
target_combobox.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

# Tạo nhãn "Select Input Variables"
input_label = tk.Label(frame, text="Select Input Variables")
input_label.grid(row=3, column=0, padx=5, sticky=tk.W)

# Tạo scrollbar cho danh sách các Input Variables
input_scrollbar = tk.Scrollbar(frame)
input_scrollbar.grid(row=4, column=1, sticky=tk.N + tk.S)

# Tạo danh sách các Input Variables
input_listbox = tk.Listbox(
    frame, yscrollcommand=input_scrollbar.set, selectmode=tk.MULTIPLE
)
input_listbox.grid(row=4, column=0, padx=5, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)

# Kết nối scrollbar với danh sách các Input Variables
input_scrollbar.config(command=input_listbox.yview)

# Tạo nút "Add"
add_button = tk.Button(frame, text="Add", command=add_variable)
add_button.grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)

# Tạo nhãn "Selected Input Variables"
selected_label = tk.Label(frame, text="Selected Input Variables")
selected_label.grid(row=6, column=0, padx=5, sticky=tk.W)

# Tạo scrollbar cho danh sách các Selected Input Variables
selected_scrollbar = tk.Scrollbar(frame)
selected_scrollbar.grid(row=7, column=1, sticky=tk.N + tk.S)

# Tạo danh sách các Selected Input Variables
selected_listbox = tk.Listbox(frame, yscrollcommand=selected_scrollbar.set)
selected_listbox.grid(row=7, column=0, padx=5, pady=5, sticky=tk.W + tk.E + tk.N + tk.S)

# Kết nối scrollbar với danh sách các Selected Input Variables
selected_scrollbar.config(command=selected_listbox.yview)

# Tạo nút "Remove"
remove_button = tk.Button(frame, text="Remove", command=remove_variable)
remove_button.grid(row=8, column=0, padx=5, pady=5, sticky=tk.W)

# Tạo nhãn "Chọn Model"
model_label = tk.Label(frame, text="Chọn Model")
model_label.grid(row=9, column=0, padx=5, pady=10, sticky=tk.W)

# Tạo combobox để chọn Model
model_combobox = ttk.Combobox(
    frame, values=["Logistic Regression", "KNN", "Linear Regression", "Decision Tree"]
)
model_combobox.grid(row=10, column=0, padx=5, sticky=tk.W)

# Tạo nút "Execution"
execution_button = tk.Button(frame, text="Execution", command=execute_model)
execution_button.grid(row=11, column=0, padx=5, pady=10, sticky=tk.W)

# Chạy giao diện
window.mainloop()