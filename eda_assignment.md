---
jupyter:
  kernelspec:
    display_name: ml_env
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.4
  nbformat: 4
  nbformat_minor: 2
  orig_nbformat: 4
---

<div class="cell markdown">

## ***Nadia Riaz***

## Assignment Topic: Apple EDA data

### 14 August, 2023

</div>

<div class="cell markdown">

### Import required libraries

</div>

<div class="cell code" execution_count="1">

``` python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly from plotly.express as plot
```

<div class="output error" ename="SyntaxError"
evalue="invalid syntax (3556189566.py, line 5)">

      Cell In[1], line 5
        import plotly from plotly.express as plot
                      ^
    SyntaxError: invalid syntax

</div>

</div>

<div class="cell markdown">

Lets Import the dataset in csv format

</div>

<div class="cell code" execution_count="3">

``` python
df = pd.read_csv("appleAppData.csv")
```

</div>

<div class="cell markdown">

### Look at the information of data to see data type

</div>

<div class="cell code" execution_count="4">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1230376 entries, 0 to 1230375
    Data columns (total 21 columns):
     #   Column                   Non-Null Count    Dtype  
    ---  ------                   --------------    -----  
     0   App_Id                   1230376 non-null  object 
     1   App_Name                 1230375 non-null  object 
     2   AppStore_Url             1230376 non-null  object 
     3   Primary_Genre            1230376 non-null  object 
     4   Content_Rating           1230376 non-null  object 
     5   Size_Bytes               1230152 non-null  float64
     6   Required_IOS_Version     1230376 non-null  object 
     7   Released                 1230373 non-null  object 
     8   Updated                  1230376 non-null  object 
     9   Version                  1230376 non-null  object 
     10  Price                    1229886 non-null  float64
     11  Currency                 1230376 non-null  object 
     12  Free                     1230376 non-null  bool   
     13  DeveloperId              1230376 non-null  int64  
     14  Developer                1230376 non-null  object 
     15  Developer_Url            1229267 non-null  object 
     16  Developer_Website        586388 non-null   object 
     17  Average_User_Rating      1230376 non-null  float64
     18  Reviews                  1230376 non-null  int64  
     19  Current_Version_Score    1230376 non-null  float64
     20  Current_Version_Reviews  1230376 non-null  int64  
    dtypes: bool(1), float64(4), int64(3), object(13)
    memory usage: 188.9+ MB

</div>

</div>

<div class="cell markdown">

### The dataset has

    - 21 columns
    - 1230375 rows
    - 1 bool
    - 4 floats
    - 3 integers
    - 13 objects

### to verify we will further look at the first 5 and last 5 rows of the
dataset

</div>

<div class="cell code" execution_count="5">

``` python
df.head()
```

<div class="output execute_result" execution_count="5">

                                 App_Id                 App_Name  \
    0               com.hkbu.arc.apaper           A+ Paper Guide   
    1               com.dmitriev.abooks                  A-Books   
    2                    no.terp.abooks                  A-books   
    3          fr.antoinettefleur.Book1              A-F Book #1   
    4  com.imonstersoft.azdictionaryios  A-Z Synonyms Dictionary   

                                            AppStore_Url Primary_Genre  \
    0  https://apps.apple.com/us/app/a-paper-guide/id...     Education   
    1  https://apps.apple.com/us/app/a-books/id103157...          Book   
    2  https://apps.apple.com/us/app/a-books/id145702...          Book   
    3  https://apps.apple.com/us/app/a-f-book-1/id500...          Book   
    4  https://apps.apple.com/us/app/a-z-synonyms-dic...     Reference   

      Content_Rating  Size_Bytes Required_IOS_Version              Released  \
    0             4+  21993472.0                  8.0  2017-09-28T03:02:41Z   
    1             4+  13135872.0                 10.0  2015-08-31T19:31:32Z   
    2             4+  21943296.0                  9.0  2021-04-14T07:00:00Z   
    3             4+  81851392.0                  8.0  2012-02-10T03:40:07Z   
    4             4+  64692224.0                  9.0  2020-12-16T08:00:00Z   

                    Updated Version  ...  Currency   Free  DeveloperId  \
    0  2018-12-21T21:30:36Z   1.1.2  ...       USD   True   1375410542   
    1  2019-07-23T20:31:09Z     1.3  ...       USD   True   1031572001   
    2  2021-05-30T21:08:54Z   1.3.1  ...       USD   True   1457024163   
    3  2019-10-29T12:40:37Z     1.2  ...       USD  False    439568839   
    4  2020-12-18T21:36:11Z   1.0.1  ...       USD   True    656731821   

            Developer                                      Developer_Url  \
    0        HKBU ARC  https://apps.apple.com/us/developer/hkbu-arc/i...   
    1  Roman Dmitriev  https://apps.apple.com/us/developer/roman-dmit...   
    2         Terp AS  https://apps.apple.com/us/developer/terp-as/id...   
    3   i-editeur.com  https://apps.apple.com/us/developer/i-editeur-...   
    4   Ngov chiheang  https://apps.apple.com/us/developer/ngov-chihe...   

             Developer_Website Average_User_Rating  Reviews  \
    0                      NaN                 0.0        0   
    1                      NaN                 5.0        1   
    2                      NaN                 0.0        0   
    3                      NaN                 0.0        0   
    4  http://imonstersoft.com                 0.0        0   

       Current_Version_Score  Current_Version_Reviews  
    0                    0.0                        0  
    1                    5.0                        1  
    2                    0.0                        0  
    3                    0.0                        0  
    4                    0.0                        0  

    [5 rows x 21 columns]

</div>

</div>

<div class="cell code" execution_count="6">

``` python
df.tail()
```

<div class="output execute_result" execution_count="6">

                                        App_Id                    App_Name  \
    1230371               com.ledtech.sadblock          Sесurity АdBlосkеr   
    1230372                    com.securex.vpn    SесurеХ VРN - Wifi Proxy   
    1230373            com.beelab.SoTayXayDung             Sổ tay Xây dựng   
    1230374                       com.icc.sttb  Sổ tay đảng viên Thái Bình   
    1230375  com.vnptlonganios.sodiemthongminh          Sổ Điểm Thông Minh   

                                                  AppStore_Url Primary_Genre  \
    1230371  https://apps.apple.com/us/app/s%D0%B5%D1%81uri...     Utilities   
    1230372  https://apps.apple.com/us/app/s%D0%B5%D1%81ur%...     Utilities   
    1230373  https://apps.apple.com/us/app/s%E1%BB%95-tay-x...     Utilities   
    1230374  https://apps.apple.com/us/app/s%E1%BB%95-tay-%...     Utilities   
    1230375  https://apps.apple.com/us/app/s%E1%BB%95-%C4%9...     Utilities   

            Content_Rating  Size_Bytes Required_IOS_Version              Released  \
    1230371             4+  16666624.0                 13.0  2020-07-07T07:00:00Z   
    1230372             4+  39016448.0                  9.0  2019-02-12T10:10:13Z   
    1230373             4+  17223680.0                  9.0  2018-10-17T04:22:41Z   
    1230374             4+  56716288.0                 10.0  2021-02-20T08:00:00Z   
    1230375             4+  85135360.0                  8.0  2018-06-05T07:45:41Z   

                          Updated Version  ...  Currency  Free  DeveloperId  \
    1230371  2020-07-10T00:48:50Z   1.0.1  ...       USD  True   1522287989   
    1230372  2020-10-21T23:25:15Z     1.1  ...       USD  True   1492288123   
    1230373  2018-10-17T04:22:41Z     1.0  ...       USD  True   1438594214   
    1230374  2021-10-02T22:00:19Z   1.2.5  ...       USD  True   1515469508   
    1230375  2019-05-21T22:03:17Z     1.9  ...       USD  True   1350355912   

                    Developer                                      Developer_Url  \
    1230371  LED-TECHNOLOGIES  https://apps.apple.com/us/developer/led-techno...   
    1230372    Trust VPN Ltd.  https://apps.apple.com/us/developer/trust-vpn-...   
    1230373          Luu Minh  https://apps.apple.com/us/developer/luu-minh/i...   
    1230374         Thái Bình  https://apps.apple.com/us/developer/th%C3%A1i-...   
    1230375     Pham Thanh Vo  https://apps.apple.com/us/developer/pham-thanh...   

                     Developer_Website Average_User_Rating  Reviews  \
    1230371                        NaN             3.91608      143   
    1230372    https://securexvpn.com/             4.82733     1500   
    1230373  http://bee-labs.github.io             4.00000        1   
    1230374       https://aisoftech.vn             0.00000        0   
    1230375                        NaN             0.00000        0   

             Current_Version_Score  Current_Version_Reviews  
    1230371                3.91608                      143  
    1230372                4.82733                     1500  
    1230373                4.00000                        1  
    1230374                0.00000                        0  
    1230375                0.00000                        0  

    [5 rows x 21 columns]

</div>

</div>

<div class="cell markdown">

### Getting unique values

</div>

<div class="cell code" execution_count="7">

``` python
df.nunique()
```

<div class="output execute_result" execution_count="7">

    App_Id                     1230376
    App_Name                   1223510
    AppStore_Url               1230376
    Primary_Genre                   26
    Content_Rating                   5
    Size_Bytes                  255914
    Required_IOS_Version           260
    Released                    664824
    Updated                    1196053
    Version                      41270
    Price                           88
    Currency                         1
    Free                             2
    DeveloperId                 509285
    Developer                   505255
    Developer_Url               514106
    Developer_Website           403809
    Average_User_Rating          45073
    Reviews                      13668
    Current_Version_Score        45073
    Current_Version_Reviews      13668
    dtype: int64

</div>

</div>

<div class="cell markdown">

## To check the unique value of individual parameter

</div>

<div class="cell code" execution_count="8">

``` python
df.columns
```

<div class="output execute_result" execution_count="8">

    Index(['App_Id', 'App_Name', 'AppStore_Url', 'Primary_Genre', 'Content_Rating',
           'Size_Bytes', 'Required_IOS_Version', 'Released', 'Updated', 'Version',
           'Price', 'Currency', 'Free', 'DeveloperId', 'Developer',
           'Developer_Url', 'Developer_Website', 'Average_User_Rating', 'Reviews',
           'Current_Version_Score', 'Current_Version_Reviews'],
          dtype='object')

</div>

</div>

<div class="cell code" execution_count="9">

``` python
df.Released.unique()
```

<div class="output execute_result" execution_count="9">

    array(['2017-09-28T03:02:41Z', '2015-08-31T19:31:32Z',
           '2021-04-14T07:00:00Z', ..., '2019-02-12T10:10:13Z',
           '2018-10-17T04:22:41Z', '2018-06-05T07:45:41Z'], dtype=object)

</div>

</div>

<div class="cell code" execution_count="10">

``` python
df.Price.unique()
```

<div class="output execute_result" execution_count="10">

    array([0.0000e+00, 2.9900e+00, 1.9900e+00, 9.9000e-01, 4.9900e+00,
           9.9900e+00, 5.9900e+00, 6.9900e+00, 1.6990e+01, 1.9990e+01,
           3.9990e+01, 5.9990e+01, 8.9900e+00, 7.9900e+00, 1.8990e+01,
           3.9900e+00, 2.4990e+01, 1.4990e+01, 2.8990e+01, 2.1990e+01,
           1.3990e+01, 1.0990e+01, 1.1990e+01, 1.2999e+02, 2.2990e+01,
           1.3999e+02, 2.5990e+01, 1.2990e+01, 4.9990e+01, 2.3990e+01,
           1.1999e+02, 7.9990e+01, 1.7990e+01, 4.9999e+02, 2.4999e+02,
           1.9999e+02, 2.2999e+02, 3.4990e+01, 2.0990e+01, 9.9990e+01,
           3.3990e+01, 5.4990e+01, 4.5990e+01, 4.4990e+01, 2.9990e+01,
           6.9990e+01, 2.7990e+01, 8.4990e+01, 6.4990e+01, 1.5990e+01,
           2.6990e+01, 3.5990e+01, 4.0990e+01, 1.7499e+02, 1.4999e+02,
           9.4990e+01, 7.4990e+01, 3.4999e+02, 4.8990e+01, 1.0999e+02,
           3.8990e+01, 3.1990e+01, 3.6990e+01, 8.9990e+01, 1.6999e+02,
           1.2499e+02,        nan, 3.9999e+02, 2.1999e+02, 4.3990e+01,
           6.9999e+02, 4.2990e+01, 3.7990e+01, 2.9999e+02, 4.4999e+02,
           1.5999e+02, 4.1990e+01, 3.2990e+01, 3.0990e+01, 7.9999e+02,
           9.9999e+02, 4.7990e+01, 1.7999e+02, 2.0999e+02, 2.3999e+02,
           5.9999e+02, 4.6990e+01, 8.9999e+02, 1.8999e+02])

</div>

</div>

<div class="cell markdown">

### To check the value counts

</div>

<div class="cell code" execution_count="11">

``` python
df["Price"].value_counts
```

<div class="output execute_result" execution_count="11">

    <bound method IndexOpsMixin.value_counts of 0          0.00
    1          0.00
    2          0.00
    3          2.99
    4          0.00
               ... 
    1230371    0.00
    1230372    0.00
    1230373    0.00
    1230374    0.00
    1230375    0.00
    Name: Price, Length: 1230376, dtype: float64>

</div>

</div>

<div class="cell markdown">

### To check value count of two objects

</div>

<div class="cell code" execution_count="12">

``` python
df.groupby("Reviews")["Price"].value_counts().unstack()
```

<div class="output execute_result" execution_count="12">

    Price       0.00     0.99    1.99    2.99    3.99    4.99    5.99    6.99    \
    Reviews                                                                       
    0         630645.0  18509.0  9765.0  5898.0  2991.0  4197.0  1043.0  1938.0   
    1         127726.0   4898.0  2602.0  1624.0   756.0   913.0   216.0   219.0   
    2          65373.0   2360.0  1326.0   853.0   392.0   504.0   124.0   107.0   
    3          40979.0   1367.0   810.0   540.0   246.0   331.0    64.0    57.0   
    4          27935.0    947.0   560.0   408.0   174.0   238.0    64.0    39.0   
    ...            ...      ...     ...     ...     ...     ...     ...     ...   
    12517538       1.0      NaN     NaN     NaN     NaN     NaN     NaN     NaN   
    12634191       1.0      NaN     NaN     NaN     NaN     NaN     NaN     NaN   
    18893225       1.0      NaN     NaN     NaN     NaN     NaN     NaN     NaN   
    21839585       1.0      NaN     NaN     NaN     NaN     NaN     NaN     NaN   
    22685334       1.0      NaN     NaN     NaN     NaN     NaN     NaN     NaN   

    Price     7.99    8.99    ...  299.99  349.99  399.99  449.99  499.99  599.99  \
    Reviews                   ...                                                   
    0          669.0   649.0  ...    25.0    19.0    13.0     5.0     7.0     1.0   
    1          182.0    89.0  ...     5.0     1.0     4.0     1.0     NaN     1.0   
    2           85.0    42.0  ...     3.0     1.0     NaN     NaN     NaN     NaN   
    3           53.0    25.0  ...     NaN     1.0     1.0     NaN     NaN     NaN   
    4           34.0    26.0  ...     NaN     NaN     NaN     NaN     1.0     NaN   
    ...          ...     ...  ...     ...     ...     ...     ...     ...     ...   
    12517538     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN   
    12634191     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN   
    18893225     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN   
    21839585     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN   
    22685334     NaN     NaN  ...     NaN     NaN     NaN     NaN     NaN     NaN   

    Price     699.99  799.99  899.99  999.99  
    Reviews                                   
    0           12.0     3.0     4.0    11.0  
    1            NaN     NaN     NaN     1.0  
    2            1.0     NaN     NaN     NaN  
    3            NaN     NaN     NaN     NaN  
    4            NaN     NaN     NaN     NaN  
    ...          ...     ...     ...     ...  
    12517538     NaN     NaN     NaN     NaN  
    12634191     NaN     NaN     NaN     NaN  
    18893225     NaN     NaN     NaN     NaN  
    21839585     NaN     NaN     NaN     NaN  
    22685334     NaN     NaN     NaN     NaN  

    [13660 rows x 88 columns]

</div>

</div>

<div class="cell markdown">

### To check count values of more than two subjects

</div>

<div class="cell code" execution_count="13">

``` python
df.groupby("Price")[["Reviews","Average_User_Rating" ]].value_counts().unstack()
```

<div class="output execute_result" execution_count="13">

    Average_User_Rating   0.00000  5.00000  1.00000  3.00000  4.00000  3.66667  \
    Price  Reviews                                                               
    0.00   0             630645.0      NaN      NaN      NaN      NaN      NaN   
           1                  NaN  87124.0  21715.0   5760.0   8482.0      NaN   
           2                  NaN  33435.0   4218.0  12397.0   3410.0      NaN   
           3                  NaN  17303.0   1352.0   1380.0   1854.0   6439.0   
           4                  NaN  10138.0    554.0   1987.0   3702.0      NaN   
    ...                       ...      ...      ...      ...      ...      ...   
    799.99 0                  3.0      NaN      NaN      NaN      NaN      NaN   
    899.99 0                  4.0      NaN      NaN      NaN      NaN      NaN   
    999.99 0                 11.0      NaN      NaN      NaN      NaN      NaN   
           1                  NaN      1.0      NaN      NaN      NaN      NaN   
           44                 NaN      NaN      NaN      NaN      NaN      NaN   

    Average_User_Rating  4.50000  2.00000  2.33333  4.66667  ...  4.30888  \
    Price  Reviews                                           ...            
    0.00   0                 NaN      NaN      NaN      NaN  ...      NaN   
           1                 NaN   4645.0      NaN      NaN  ...      NaN   
           2              4699.0   1327.0      NaN      NaN  ...      NaN   
           3                 NaN    606.0   3209.0   2932.0  ...      NaN   
           4              1470.0   1204.0      NaN      NaN  ...      NaN   
    ...                      ...      ...      ...      ...  ...      ...   
    799.99 0                 NaN      NaN      NaN      NaN  ...      NaN   
    899.99 0                 NaN      NaN      NaN      NaN  ...      NaN   
    999.99 0                 NaN      NaN      NaN      NaN  ...      NaN   
           1                 NaN      NaN      NaN      NaN  ...      NaN   
           44                NaN      NaN      NaN      NaN  ...      NaN   

    Average_User_Rating  4.70598  4.23898  1.13029  4.74532  3.54464  3.67760  \
    Price  Reviews                                                              
    0.00   0                 NaN      NaN      NaN      NaN      NaN      NaN   
           1                 NaN      NaN      NaN      NaN      NaN      NaN   
           2                 NaN      NaN      NaN      NaN      NaN      NaN   
           3                 NaN      NaN      NaN      NaN      NaN      NaN   
           4                 NaN      NaN      NaN      NaN      NaN      NaN   
    ...                      ...      ...      ...      ...      ...      ...   
    799.99 0                 NaN      NaN      NaN      NaN      NaN      NaN   
    899.99 0                 NaN      NaN      NaN      NaN      NaN      NaN   
    999.99 0                 NaN      NaN      NaN      NaN      NaN      NaN   
           1                 NaN      NaN      NaN      NaN      NaN      NaN   
           44                NaN      NaN      NaN      NaN      NaN      NaN   

    Average_User_Rating  4.89405  4.28990  4.59908  
    Price  Reviews                                  
    0.00   0                 NaN      NaN      NaN  
           1                 NaN      NaN      NaN  
           2                 NaN      NaN      NaN  
           3                 NaN      NaN      NaN  
           4                 NaN      NaN      NaN  
    ...                      ...      ...      ...  
    799.99 0                 NaN      NaN      NaN  
    899.99 0                 NaN      NaN      NaN  
    999.99 0                 NaN      NaN      NaN  
           1                 NaN      NaN      NaN  
           44                NaN      NaN      NaN  

    [18754 rows x 45015 columns]

</div>

</div>

<div class="cell markdown">

### Cheking the null values

</div>

<div class="cell code" execution_count="14">

``` python
print((df.isnull().sum()/len(df))*100)
```

<div class="output stream stdout">

    App_Id                      0.000000
    App_Name                    0.000081
    AppStore_Url                0.000000
    Primary_Genre               0.000000
    Content_Rating              0.000000
    Size_Bytes                  0.018206
    Required_IOS_Version        0.000000
    Released                    0.000244
    Updated                     0.000000
    Version                     0.000000
    Price                       0.039825
    Currency                    0.000000
    Free                        0.000000
    DeveloperId                 0.000000
    Developer                   0.000000
    Developer_Url               0.090135
    Developer_Website          52.340748
    Average_User_Rating         0.000000
    Reviews                     0.000000
    Current_Version_Score       0.000000
    Current_Version_Reviews     0.000000
    dtype: float64

</div>

</div>

<div class="cell markdown">

### Data Visualization

### Make plots

    - 1 Scatter plot
    - 2 Box plot
    - 3 Bar plot
    - 4 Heatmap
    - 5 Corrplot
    - 6 Countplot

</div>

<div class="cell code" execution_count="15">

``` python
### 1
sns.scatterplot(data = df, y = "Price", x= "Required_IOS_Version", hue= "Free")
plt.title('Apple_data')
plt.xlabel('IOS version')
plt.ylabel('Price')
plt.show()
```

<div class="output display_data">

![](cc6d88e8e5d51e7d9cbcdee494906a8544333602.png)

</div>

</div>

<div class="cell code" execution_count="16">

``` python
df.columns
```

<div class="output execute_result" execution_count="16">

    Index(['App_Id', 'App_Name', 'AppStore_Url', 'Primary_Genre', 'Content_Rating',
           'Size_Bytes', 'Required_IOS_Version', 'Released', 'Updated', 'Version',
           'Price', 'Currency', 'Free', 'DeveloperId', 'Developer',
           'Developer_Url', 'Developer_Website', 'Average_User_Rating', 'Reviews',
           'Current_Version_Score', 'Current_Version_Reviews'],
          dtype='object')

</div>

</div>

<div class="cell code" execution_count="17">

``` python
### 2
sns.boxplot(data = df, x= "Free", y = "Reviews", hue="Content_Rating")
```

<div class="output execute_result" execution_count="17">

    <Axes: xlabel='Free', ylabel='Reviews'>

</div>

<div class="output display_data">

![](e9d9a19dfcdb6c8d3237cff017503a8c85584810.png)

</div>

</div>

<div class="cell code" execution_count="18">

``` python
### 3
sns.barplot(data = df, x= "Price", y = "Primary_Genre")
```

<div class="output execute_result" execution_count="18">

    <Axes: xlabel='Price', ylabel='Primary_Genre'>

</div>

<div class="output display_data">

![](b09a83d4d69e9d0b7376eb020e0def09eeccfcad.png)

</div>

</div>

<div class="cell code" execution_count="23">

``` python
### 4
corr = df.corr(numeric_only = True,)
sns.heatmap(corr, cbar= True, annot= True, cmap= "coolwarm")
```

<div class="output execute_result" execution_count="23">

    <Axes: >

</div>

<div class="output display_data">

![](c21e3aa548d905375355fee1b3e1e35ac7254e0b.png)

</div>

</div>

<div class="cell code" execution_count="33">

``` python
### 5
sns.histplot(df["Average_User_Rating"], bins=25, kde=True)
```

<div class="output execute_result" execution_count="33">

    <Axes: xlabel='Average_User_Rating', ylabel='Count'>

</div>

<div class="output display_data">

![](efd8bf551d55dc9472b53db23032fc7315deeacb.png)

</div>

</div>

<div class="cell code" execution_count="36">

``` python
#### 6
sns.countplot (data= df, x= "Content_Rating")
```

<div class="output execute_result" execution_count="36">

    <Axes: xlabel='Content_Rating', ylabel='count'>

</div>

<div class="output display_data">

![](52fdd463b2bfd68c2d303d433e0fa8e13eab0180.png)

</div>

</div>
