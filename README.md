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

安裝方式
```bash
pip install -e .
```

執行指令
```bash
# DQN
python .\scripts\MyDQN\demo.py --config=scripts/MyDQN/config.yaml

# FeUdal
python .\scripts\Feudal\demo.py --config=scripts/Feudal/config.yaml
```


