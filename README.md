# 我的修改

1. 修改 action檔案的路徑
2. 不會有彈出視窗暫停
3. 各個範例腳本
4. FeUdal 單Episode訓練(beta)

# 範例說明

1. demo1 單位自動往右上移動
2. demo2 單位自動往右上移動，並且會攻擊敵人
3. demo3 印出可用資訊及自訂動作(園地轉圈)
4. MyDQN 使用DQN訓練單位自動移動至友軍身邊
5. FeUdal 使用Feudal訓練單位自動移動至友軍身邊(單Episode訓練 beta)


# 執行指令
```bash
# DQN
python .\scripts\MyDQN\demo.py --config=scripts/MyDQN/config.yaml

# FeUdal
python .\scripts\Feudal\demo.py --config=scripts/Feudal/config.yaml

# MyNet_episode battle
python .\scripts\MyNet_episode_battle\demo.py --config=scripts/MyNet_episode_battle/config.yaml

# MyNet_episode
python .\scripts\MyNet_episode\demo.py --config=scripts/MyNet_episode/config.yaml

# MyNet_multi_battle
python .\scripts\MyNet_multi_battle\demo.py --config=scripts/MyNet_multi_battle/config.yaml
```

執行前請將輸入法改為英文，中文輸入會導致無法接收按鍵指令

# 安裝
## 一、、前置作業
1. 安裝Anaconda
2. 安裝cmo
3. 安裝pycmo
4. python版本12.9
## 二、CUDA與pytorch
### 40系顯卡以下請安裝以下版本
1. CUDA 12.4
2. pytorch 2.5.0
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```
### 50系顯卡請安裝最新版CUDA和pytorch
1. CUDA 12.8
2. pytorch preview
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 三、安裝pycmo

```bash
git clone https://forgejo.taiyopen.com/Taiyopen/pycmopen.git
cd pycmopen
```
修改/pycmopen/pycmo/config/config.py當中的路徑

修改完後
```bash
pip install -e .
```

## 四、CMO操作 (還沒寫好)

### 2.1 啟動CMO後點擊`Edit Scenario`
![alt text](image.png)
### 2.2 選擇`MyScenario1`或想要的場景

### 2.3 選擇`Edit Scenario`
