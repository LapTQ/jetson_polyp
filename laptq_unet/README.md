# foot_size_estimation


# Install

```
git clone https://github.com/LapTQ/foot_size_estimation.git
cd foot_size_estimation
pip install -r requirements.txt
```

# Generate training dataset

```
python generate_data.py --train_num 2000 --dev_num 100 
```

# Train

```
python train.py --epoch 50 --batch_size 8 --size 350
```

# Try this app

[here](https://share.streamlit.io/LapTQ/it4343e_foot_size_estimation/main/streamlit_app.py)

[mine](https://laptq-it4343e-foot-size-estimation-streamlit-app-7fd5ix.streamlitapp.com/)
