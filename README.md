
---

## Setup Instructions

This guide explains how to set up and run this project using a virtual environment and the provided `requirements.txt` file.

---

## 1. Check Python Version

This project works with python 3.13

## 2. Create Virtual Environment

Run the following in your terminal **from the root of the repo**:

```sh
python -m venv .venv
```

## 3. Activate Virtual Environment 

Run the following in your terminal **from the root of the repo**:

```sh
.venv\Scripts\activate
```

## 4. Ensure pip is up to date

Run the following in the terminal:

```sh
python.exe -m pip install --upgrade pip
```

## 5. Install Dependancies

Run the following in the terminal:

```sh
pip install -r requirements.txt
```

## 6. Run Test Files

Run the following in the terminal:

```sh
python main.py
```

## 7. Deactivate the Virtual Environment

Run the following in the terminal:

```sh
deactivate
```

## Exmaple

```sh
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Install packages
pip install -r requirements.txt

# Run the app
python main.py
```