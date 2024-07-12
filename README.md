# quickdrawneta

First of all, download .npy files from https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pageState=("StorageObjectListTable":("f":"%5B%5D"))&prefix=&forceOnObjectsSortingFiltering=false and place them in a folder inside quickdrawneta/datasets

Create and activate virtual environment:

```powershell
python -m venv .venv
```

On windows

```

.venv/Scripts/Activate.ps1

```

On ubuntu

```

source .venv/bin/activate

```

Install tkinter on ubuntu

```

apt-get install python-tk
apt-get install python3-tk

```

Install dependencies once inside venv:

```powershell
python -m pip install --editable .
```

To run main and play the game:

```
python main.py
```

Remember to change categories to guess in categories.txt file, they should be placed in the same order in which the .npy files are inside quickdrawneta/datasets
