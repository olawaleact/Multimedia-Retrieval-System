# Recommendation System with UI

This is a video recommender solution.

# Setup

Clone the repository.

```bash
git clone https://github.com/theiyobosa/aot-mmsr.git
```

Move all the `.tsv` files directly to a folder `dataset` inside `aot-mmsr`. Ensure the `.tsv` contents is in the path: `/aot-mmsr/dataset/`.

Change working directory to the project folder.

```bash
cd /path/to/folder/aot-mmsr/
```

Create a virtual environment.

```bash
python3 -m venv venv
```

Activate the virtual environment (Mac OS).

```bash
source venv/bin/activate
```

Install the required packages into the virtual environment.

```bash
pip3 install -r requirements.txt
```

Run the application.

```bash
streamlit run main.py
```