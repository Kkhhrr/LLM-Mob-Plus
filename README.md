# 项目说明

---

## 文件夹结构

  各个代码文件存储在独立的模块中：

- **/preprocessing/**：用于预处理数据集的函数。在进行实验之前需要执行这些脚本。
- **/baselines/**：我们实现的典型的基线模型，用于与提出的模型进行比较。这些方法包括**DeepMove**、**LLM-Mob**、**MHSA**、**LSTM(-attn)** 、 **Markov**和**Mobtcast**。
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

## 在baselines上的实现

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

  所需要的数据文件同**Markov、LSTM(-attn)、Mobtcast**，在`baselines/MHSA`目录下创建一个名为`data`的新文件夹，并将`data`路径下`geolife.py`和`foursquare.py`所生成的数据文件复制粘贴进去即可。

- **LLM-Mob**

  所需要的数据文件已经存放在`baselines/LLM-Mob/data`路径下。

- **DeepMove**

  所需要的数据文件已经存放在`baselines/DeepMove/data`路径下。数据文件可通过`preprocessing`路径下的`sparse_traces_foursquare.py`和`sparse_traces_geolife.py`文件获得，注意`sparse_traces_foursquare.py`文件需要传入`--city`参数。

### 2.各baseline上的实现

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
  python geolife/main.py geolife/config/geolife/transformer.yml
  ```

- **Foursquare：**

  只需将命令中的 [city_name] 替换为 nyc 或 tky 即可。

  ```shell
  python foursquare/main.py foursquare/config/foursquare/transformer_[city_name].yml
  ```

#### DeepMove 模型

- **Geolife：**

  ```shell
  python main.py --model_mode=attn_avg_long_user --data_name = geolife
  ```

- **Foursquare：**

  只需将命令中的 `[city_name]` 替换为 nyc 或 tky 即可。

  ```shell
  python main.py --model_mode=attn_avg_long_user --data_name = foursquare_[city_name]
  ```

#### LLM-Mob 模型

  在`llm-mob.py`文件中的`main`函数中根据注释修改相应参数运行即可。
