import pandas as pd
from scipy import stats

### Shapiro test
df = pd.read_csv('Metrics_df/yolo_epoch_30_global/res_IOU.csv')
x = df['Best_Iou']
res = stats.shapiro(x)
print(res)