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

| Dataset    | Metric      |   1-MMC |   LSTM |   LSTM Attn |   Deepmove |   MobTcast |   MHSA |   LLM-Mob(gpt-4o-mini) |   LLM-Mob(gpt-4o-mini) 1 | LLM-Mob(gpt-4o-mini) 2   |   LLM-Mob(gpt-4o-mini+category) |   LLM-Mob(gpt-4o-mini+category) 1 | LLM-Mob(gpt-4o-mini+category) 2   |   LLM-Mob(gpt-5-mini)+category |   LLM-Mob(gpt-5-mini)+category 1 | LLM-Mob(gpt-5-mini)+category 2   |
|:-----------|:------------|--------:|-------:|------------:|-----------:|-----------:|-------:|-----------------------:|-------------------------:|:-------------------------|--------------------------------:|----------------------------------:|:----------------------------------|-------------------------------:|---------------------------------:|:---------------------------------|
| FSQ-NYC    | Acc@1(%)    |  16.1   | 17.6   |      17.7   |     17.4   |     17.9   | 20.2   |                 23.3   |                   26.8   | 28.9                     |                          22.5   |                            27.2   | 30.5                              |                         24.1   |                           36.2   | 37.0                             |
|            | Acc@5(%)    |  32.5   | 44.3   |      42.8   |     39.2   |     45     | 47.2   |                 56.7   |                   58.5   | -                        |                          52.9   |                            55.3   | -                                 |                         56.2   |                           61.8   | -                                |
|            | Acc@10(%)   |  36.4   | 53.5   |      53     |     47.4   |     56.6   | 57.6   |                 69.9   |                   70.7   | -                        |                          65.3   |                            65.2   | -                                 |                         68.7   |                           71.2   | -                                |
|            | Weighted F1 |   0.145 |  0.096 |       0.107 |      0.129 |      0.095 |  0.149 |                  0.161 |                    0.2   | 0.246                    |                           0.159 |                             0.22  | 0.271                             |                          0.207 |                            0.334 | 0.344                            |
|            | nDCG@10     |   0.266 |  0.345 |       0.342 |      0.316 |      0.357 |  0.378 |                  0.451 |                    0.477 | -                        |                           0.425 |                             0.455 | -                                 |                          0.45  |                            0.53  | -                                |
| FSQ-TKY    | Acc@1(%)    |  16.3   | 19.5   |      19.2   |     18.1   |     14     | 21.6   |                 17.1   |                   18.6   | 18.5                     |                          17.8   |                            19     | 19.2                              |                         21.4   |                           26.7   | 27.5                             |
|            | Acc@5(%)    |  30.6   | 43.2   |      42.7   |     37.8   |     35.7   | 45.6   |                 42.1   |                   43.3   | -                        |                          42.8   |                            43.9   | -                                 |                         45.8   |                           49.4   | -                                |
|            | Acc@10(%)   |  33.2   | 52.2   |      51.5   |     45.1   |     45.4   | 55.5   |                 52.3   |                   52.6   | -                        |                          52.9   |                            53     | -                                 |                         55.9   |                           58.3   | -                                |
|            | Weighted F1 |   0.167 |  0.156 |       0.154 |      0.136 |      0.091 |  0.181 |                  0.136 |                    0.155 | 0.159                    |                           0.138 |                             0.157 | 0.170                             |                          0.189 |                            0.246 | 0.254                            |
|            | nDCG@10     |   0.251 |  0.348 |       0.345 |      0.309 |      0.284 |  0.374 |                  0.335 |                    0.347 | -                        |                           0.342 |                             0.352 | -                                 |                          0.375 |                            0.416 | -                                |
| Gowalla-CA | Acc@1(%)    |   3     |  5.3   |       5.5   |      8.2   |     11.9   | 13.8   |                 18.8   |                   29.2   | 34.9                     |                          21.2   |                            30.5   | 35.4                              |                         24.2   |                           34.6   | 35.6                             |
|            | Acc@5(%)    |   4.9   | 12.3   |      12.9   |     16.4   |     27.1   | 30.1   |                 45.4   |                   54.2   | -                        |                          49.1   |                            55     | -                                 |                         49.8   |                           56     | -                                |
|            | Acc@10(%)   |   5     | 15.3   |      15.7   |     20.2   |     33.1   | 37.6   |                 59.1   |                   62.1   | -                        |                          60.9   |                            62.3   | -                                 |                         60.6   |                           64     | -                                |
|            | Weighted F1 |   0.032 |  0.026 |       0.024 |      0.037 |      0.059 |  0.088 |                  0.159 |                    0.272 | 0.334                    |                           0.187 |                             0.284 | 0.341                             |                          0.229 |                            0.338 | 0.347                            |
|            | nDCG@10     |   0.041 |  0.102 |       0.104 |      0.138 |      0.219 |  0.247 |                  0.369 |                    0.453 | -                        |                           0.396 |                             0.462 | -                                 |                          0.412 |                            0.487 | -                                |
