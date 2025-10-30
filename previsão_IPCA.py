
import requests
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt 


url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
params = {"formato": "json", "dataInicial": "01/01/2000", "dataFinal": "15/07/2025"}

res = requests.get(url, params=params).json()
df = pd.DataFrame(res)


df["data"] = pd.to_datetime(df["data"], dayfirst=True)
df["valor"] = df["valor"].astype(float)


df = df.set_index("data")
print(df.tail())



order = (1,0,1)


seasonal_order = (1,0,1,12)


model = SARIMAX(df["valor"],
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

fit = model.fit(disp=False)


forecast = fit.forecast(steps=12)


print("Previsão do IPCA")

print(forecast)


forecast.plot( color="blue", marker="o", linestyle="-", label="Previsão")
plt.grid(True)
plt.xlabel("Período")
plt.ylabel(("Variação (%)"))
plt.title("Previsão do IPCA pra o proximos 12 meses %")
plt.show()
