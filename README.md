# Prototyping an explainable artificial intelligence model (XLSM) for forecasting the Bitcoin price

Artificial Intelligence (AI) significantly improves time series forecasting in the financial market, yet it is challenging to establish reliable real-world finance applications due to a lack of transparency and explainability. This paper prototypes an eXplainable Long Short-Term Memory (XLSTM) model to train and forecast the price of Bitcoin using a group of 11 de-terminants. The results show good performance in price fore-casting as measured by a mean absolute percentage error (MAPE) of 2.39% and an accuracy of 89.54%. Additionally, the AI model explains that trading volume and prices (Low, High, Open) contribute to the price dynamics, while oil and Dow Jones Index (DJI) influence the price behavior at a low level. We argue that understanding these underlying explanato-ry determinants may increase the reliability of AI’s prediction in the cryptocurrency and general finance market.

In this paper, we prototype an eXplainable Long Short Term Memory (XLSTM) for predicting the Bitcoin price, combined with several layers of machine learning. 

<img width="700" alt="image" src="https://github.com/user-attachments/assets/df9abeb8-64bb-4abe-96f0-5418c6931baa">

- (1) Convolutional neural network (CNN) for data processing. To gain certain ad-vantages in feature extraction and reduce the defect of prediction lag, we applied CNN to extract higher-order features from input information of the determinants.

- (2) Long Short-Term Memory (LSTM) for time series learning and forecasting. LSTM is a special type of recurrent neu-ral network (RNN) that can learn long-term dependencies be-tween time steps of sequence data. We added dropout layers between each layer to prevent overfitting the time series. Dur-ing the training process of LSTM, we randomly select some neurons to temporarily hide, reduce weights, and improve the network's robustness and nonlinear prediction ability. Com-pared to traditional RNN models, the state of the LSTM layer consists of the hidden state (or the output state) and the cell state. The hidden state at time step t contains the output of the LSTM layer for this time step. The cell state contains infor-mation learned from the previous time steps. The layer adds or removes information from the cell state at each time step. 

