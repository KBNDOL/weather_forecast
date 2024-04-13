# Import Meteostat library and dependencies
from datetime import datetime
from meteostat import Hourly
import numpy as np
import pandas as pd

# Set time period
start = datetime(2023, 1, 1)
end = datetime(2023, 12, 31, 23, 59)

# Get hourly data
data = Hourly('58367', start, end)
data = data.fetch()
df = pd.DataFrame(data, columns=['time', 'temp', 'dwpt','rhum','prcp','snow','wdir','wspd','wpgt','pres','tsun','coco'])
# Print DataFrame
print(data)
df.to_excel('Hongqiao_1.xlsx', index=False, engine='openpyxl')