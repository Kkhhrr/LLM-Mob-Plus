# 项目说明

---

## 文件夹结构

  各个代码文件存储在独立的模块中：

- **/preprocessing/**：用于预处理数据集的函数。在进行实验之前需要执行这些脚本。
- **/models/**：我们实现的典型的基线模型，用于与提出的模型进行比较。这些方法包括**DeepMove**、**LLM-Mob**、**MHSA**、**LSTM(-attn)** 、 **Markov**和**Mobtcast**。
- **/utils/**：辅助函数。

---

## 数据集下载

  在仓库根目录下创建一个名为 `data` 的新文件夹。

### 下载 Geolife 数据集

  从 [微软官方链接](https://www.microsoft.com/en-us/download/details.aspx?id=52367) 下载 Geolife GPS 跟踪数据集。

  在`data`目录下创建一个名为 `geolife` 的新文件夹。解压并将 Geolife 的 `Data` 文件夹复制到 `data/geolife/` 中。

### 下载 Foursquare 数据集

  从 [论文作者网站](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) 的 NYC and Tokyo Check-in Dataset 分支下载并解压。

  在`data`目录下创建一个名为 `fsq` 的新文件夹。

- 对于 NYC，在`data/fsq`目录下创建一个名为 `nyc` 的新文件夹,将解压得到的`dataset_TSMC2014_NYC.txt` 文件放入`data/fsq/nyc`文件夹，最终路径为`data/fsq/nyc/dataset_TSMC2014_NYC.txt`。

- 对于 TKY，在`data/fsq`目录下创建一个名为 `tky` 的新文件夹,将解压得到的`dataset_TSMC2014_TKY.txt` 文件放入`data/fsq/tky`文件夹，最终路径为`data/fsq/tky/dataset_TSMC2014_TKY.txt`。

---

## 在models上的实现

### 1.预处理数据

- **Markov、LSTM(-attn)、Mobtcast**
  - 对于**Geolife**数据集运行以下代码。运行完成后，会在 `data/geolife/` 目录下生成 `dataSet_geolife.csv` 和 `valid_ids_geolife.pk` 等文件。

    ```shell
        python preprocessing/geolife.py 
    ```

  - 对于**Foursquare**数据集运行以下代码。请将命令中的 `[city_name]` 替换为 `nyc` 或 `tky`。运行完成后，会在 `data/fsq/{city_name}/` 目录下生成 `dataSet_foursquare_{city_name}.csv` 和 `valid_ids_foursquare_{city_name}.pk` 等文件。

    ```shell
        python preprocessing/foursquare.py --city [city_name]
    ```

- **MHSA**

  所需要的数据文件同**Markov、LSTM(-attn)、Mobtcast**，在`models/MHSA`的`geolife`和`foursquare`目录下分别创建一个名为`data`的新文件夹，并将`data`路径下`geolife.py`和`foursquare.py`所生成的数据文件分别复制粘贴进去即可。

- **LLM-Mob**

  所需要的数据文件已经存放在`models/LLM-Mob/data`路径下。

- **DeepMove**

  所需要的数据文件已经存放在`models/DeepMove/data`路径下。数据文件可通过`preprocessing`路径下的`sparse_traces_foursquare.py`和`sparse_traces_geolife.py`文件获得，注意`sparse_traces_foursquare.py`文件需要传入`--city`参数。

### 2.各model上的实现

#### Markov / Mobtcast 模型

  这两个模型的运行命令格式完全相同。只需将命令中的 `[model_name]` 替换为 `markov` 或 `Mobtcast` ，`[city_name]` 替换为 `nyc` 或 `tky` 即可。

- **Geolife：**

    ```shell
        python [model_name].py --dataset geolife
    ```

- **Foursquare (nyc 或 tky)：**

    ```shell
        python [model_name].py --dataset fsq --city [city_name]
    ```

#### LSTM / LSTM-attn 模型

  都通过`LSTM.py`文件实现，只需将命令中的 `[city_name]` 替换为 nyc 或 tky 即可。

- **LSTM：**
  - **Geolife：**

    ```shell
        python LSTM.py --dataset geolife --attention false
    ```

  - **Foursquare (nyc 或 tky):**

    ```shell
        python LSTM.py --dataset fsq --city [city_name] --attention false
    ```

- **LSTM-attn：**
  - **Geolife：**

    ```shell
    python LSTM.py --dataset geolife
    ```

  - **Foursquare (nyc 或 tky):**

    ```shell
    python LSTM.py --dataset fsq --city [city_name]
    ```

#### MHSA 模型

- **Geolife：**

  ```shell
  python main.py config/geolife/transformer.yml
  ```

- **Foursquare：**

  只需将命令中的 [city_name] 替换为 nyc 或 tky 即可。

  ```shell
  python main.py config/foursquare/[city_name]_transformer.yml
  ```

#### DeepMove 模型

- **Geolife：**

  ```shell
  python main.py --model_mode=attn_avg_long_user --data_name geolife
  ```

- **Foursquare：**

  只需将命令中的 `[city_name]` 替换为 nyc 或 tky 即可。

  ```shell
  python main.py --model_mode=attn_avg_long_user --data_name foursquare_[city_name]
  ```

#### LLM-Mob 模型

  在`llm-mob.py`文件中的`main`函数中根据注释修改相应参数运行即可。

---

## 实验结果

以下为各基线模型在 **Gowalla-CA** 和 **Foursquare (NYC & TKY)** 数据集上的性能对比：

<table>
<thead>
<tr>
<th>Dataset</th>
<th>Metric</th>
<th>1-MMC</th>
<th>LSTM</th>
<th>LSTM Attn</th>
<th>Deepmove</th>
<th>MobTcast</th>
<th>MHSA</th>
<th>LLM-Mob(gpt-4o-mini)</th>
<th>LLM-Mob(gpt-4o-mini) 1</th>
<th>LLM-Mob(gpt-4o-mini) 2</th>
<th>LLM-Mob(gpt-4o-mini+category)</th>
<th>LLM-Mob(gpt-4o-mini+category) 1</th>
<th>LLM-Mob(gpt-4o-mini+category) 2</th>
<th>LLM-Mob(gpt-5-mini)+category</th>
<th>LLM-Mob(gpt-5-mini)+category 1</th>
<th>LLM-Mob(gpt-5-mini)+category 2</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="5">FSQ-NYC</td>
<td>Acc@1(%)</td>
<td>16.1</td>
<td>17.6</td>
<td>17.7</td>
<td>17.4</td>
<td>17.9</td>
<td>20.2</td>
<td>23.3</td>
<td>26.8</td>
<td>28.9</td>
<td>22.5</td>
<td>27.2</td>
<td>30.5</td>
<td>24.1</td>
<td>36.2</td>
<td>37.0</td>
</tr>
<tr>
<td>Acc@5(%)</td>
<td>32.5</td>
<td>44.3</td>
<td>42.8</td>
<td>39.2</td>
<td>45.0</td>
<td>47.2</td>
<td>56.7</td>
<td>58.5</td>
<td>-</td>
<td>52.9</td>
<td>55.3</td>
<td>-</td>
<td>56.2</td>
<td>61.8</td>
<td>-</td>
</tr>
<tr>
<td>Acc@10(%)</td>
<td>36.4</td>
<td>53.5</td>
<td>53.0</td>
<td>47.4</td>
<td>56.6</td>
<td>57.6</td>
<td>69.9</td>
<td>70.7</td>
<td>-</td>
<td>65.3</td>
<td>65.2</td>
<td>-</td>
<td>68.7</td>
<td>71.2</td>
<td>-</td>
</tr>
<tr>
<td>Weighted F1</td>
<td>0.145</td>
<td>0.096</td>
<td>0.107</td>
<td>0.129</td>
<td>0.095</td>
<td>0.149</td>
<td>0.161</td>
<td>0.200</td>
<td>0.246</td>
<td>0.159</td>
<td>0.22</td>
<td>0.271</td>
<td>0.207</td>
<td>0.334</td>
<td>0.344</td>
</tr>
<tr>
<td>nDCG@10</td>
<td>0.266</td>
<td>0.345</td>
<td>0.342</td>
<td>0.316</td>
<td>0.357</td>
<td>0.378</td>
<td>0.451</td>
<td>0.477</td>
<td>-</td>
<td>0.425</td>
<td>0.455</td>
<td>-</td>
<td>0.45</td>
<td>0.53</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">FSQ-TKY</td>
<td>Acc@1(%)</td>
<td>16.3</td>
<td>19.5</td>
<td>19.2</td>
<td>18.1</td>
<td>14.0</td>
<td>21.6</td>
<td>17.1</td>
<td>18.6</td>
<td>18.5</td>
<td>17.8</td>
<td>19.0</td>
<td>19.2</td>
<td>21.4</td>
<td>26.7</td>
<td>27.5</td>
</tr>
<tr>
<td>Acc@5(%)</td>
<td>30.6</td>
<td>43.2</td>
<td>42.7</td>
<td>37.8</td>
<td>35.7</td>
<td>45.6</td>
<td>42.1</td>
<td>43.3</td>
<td>-</td>
<td>42.8</td>
<td>43.9</td>
<td>-</td>
<td>45.8</td>
<td>49.4</td>
<td>-</td>
</tr>
<tr>
<td>Acc@10(%)</td>
<td>33.2</td>
<td>52.2</td>
<td>51.5</td>
<td>45.1</td>
<td>45.4</td>
<td>55.5</td>
<td>52.3</td>
<td>52.6</td>
<td>-</td>
<td>52.9</td>
<td>53.0</td>
<td>-</td>
<td>55.9</td>
<td>58.3</td>
<td>-</td>
</tr>
<tr>
<td>Weighted F1</td>
<td>0.167</td>
<td>0.156</td>
<td>0.154</td>
<td>0.136</td>
<td>0.091</td>
<td>0.181</td>
<td>0.136</td>
<td>0.155</td>
<td>0.159</td>
<td>0.138</td>
<td>0.157</td>
<td>0.170</td>
<td>0.189</td>
<td>0.246</td>
<td>0.254</td>
</tr>
<tr>
<td>nDCG@10</td>
<td>0.251</td>
<td>0.348</td>
<td>0.345</td>
<td>0.309</td>
<td>0.284</td>
<td>0.374</td>
<td>0.335</td>
<td>0.347</td>
<td>-</td>
<td>0.342</td>
<td>0.352</td>
<td>-</td>
<td>0.375</td>
<td>0.416</td>
<td>-</td>
</tr>
<tr>
<td rowspan="5">Gowalla-CA</td>
<td>Acc@1(%)</td>
<td>3.0</td>
<td>5.3</td>
<td>5.5</td>
<td>8.2</td>
<td>11.9</td>
<td>13.8</td>
<td>18.8</td>
<td>29.2</td>
<td>34.9</td>
<td>21.2</td>
<td>30.5</td>
<td>35.4</td>
<td>24.2</td>
<td>34.6</td>
<td>35.6</td>
</tr>
<tr>
<td>Acc@5(%)</td>
<td>4.9</td>
<td>12.3</td>
<td>12.9</td>
<td>16.4</td>
<td>27.1</td>
<td>30.1</td>
<td>45.4</td>
<td>54.2</td>
<td>-</td>
<td>49.1</td>
<td>55.0</td>
<td>-</td>
<td>49.8</td>
<td>56.0</td>
<td>-</td>
</tr>
<tr>
<td>Acc@10(%)</td>
<td>5.0</td>
<td>15.3</td>
<td>15.7</td>
<td>20.2</td>
<td>33.1</td>
<td>37.6</td>
<td>59.1</td>
<td>62.1</td>
<td>-</td>
<td>60.9</td>
<td>62.3</td>
<td>-</td>
<td>60.6</td>
<td>64.0</td>
<td>-</td>
</tr>
<tr>
<td>Weighted F1</td>
<td>0.032</td>
<td>0.026</td>
<td>0.024</td>
<td>0.037</td>
<td>0.059</td>
<td>0.088</td>
<td>0.159</td>
<td>0.272</td>
<td>0.334</td>
<td>0.187</td>
<td>0.284</td>
<td>0.341</td>
<td>0.229</td>
<td>0.338</td>
<td>0.347</td>
</tr>
<tr>
<td>nDCG@10</td>
<td>0.041</td>
<td>0.102</td>
<td>0.104</td>
<td>0.138</td>
<td>0.219</td>
<td>0.247</td>
<td>0.369</td>
<td>0.453</td>
<td>-</td>
<td>0.396</td>
<td>0.462</td>
<td>-</td>
<td>0.412</td>
<td>0.487</td>
<td>-</td>
</tr>
</tbody>
</table>
