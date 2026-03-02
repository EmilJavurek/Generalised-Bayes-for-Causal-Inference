# Setup (Python virtual environment)

These steps create a fresh virtual environment using `pip` and install the dependencies from `requirements.txt`.

## Windows (PowerShell)

```powershell
# from the project root
py -m venv .venv

# activate
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt
```
# Run experiments

To run experiments, specify the configuration in `.\configs\experiments\YOURCONFIG.json` and `.\configs\single_runs\YOURCONFIG.json` and run

```powershell
python src.py --experiment .\configs\experiments\YOURCONFIG.json
```


