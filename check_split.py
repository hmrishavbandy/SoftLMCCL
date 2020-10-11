import pandas as pd
for i in ['train','test','valid']:
    p=pd.read_csv('./csv_split/'+i+'.csv')
    print(set(p['SNR Level']))
    print(set(p['Actual Label']))
    print(set(p['Class']))
