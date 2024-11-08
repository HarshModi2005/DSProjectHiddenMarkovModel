# Stock Prediction Using Hidden Markov Model

This project implements a Hidden Markov Model (HMM) to analyze and predict stock price movements. It uses a Baum-Welch algorithm for training and a forward-backward algorithm to compute state probabilities, aiming to minimize prediction loss. The program reads stock data, normalizes it, categorizes movements, trains an HMM, and predicts future stock movements.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Input Format](#input-format)
- [Expected Output](#expected-output)
- [Notes](#notes)

## Requirements

- C Compiler: (e.g., `gcc`)
- Libraries: `math.h`, `time.h`, `stdio.h`, `stdlib.h`, and `string.h` (standard libraries in C)
- Stock Data: A CSV file containing historical stock data with the columns: `"Index Name","Date","Open","High","Low","Close"`

## Setup

1. Clone or Download this repository to your local machine.
2. Place your stock data CSV file (e.g., `nasa.csv`) in the project directory.

## How to Run

1. Compile the Code:

   Use the following command to compile the code:
   ```bash
   gcc stock_prediction.c -o stock_prediction -lm

2. Execute the compiled program:
   ```bash
   ./stock_prediction
   
##Input Format
<br>
The program reads stock data from a CSV file in the following format:
<br>
"Index Name","Date","Open","High","Low","Close"
<br>

Index Name: A string representing the stock index name.
<br>
Date: The date of the stock data.
<br>
Open: The opening price of the stock.
<br>
High: The highest price of the stock.
<br>
Low: The lowest price of the stock.
<br>
Close: The closing price of the stock.
<br>

##Expected Output
<br>

Prediction for 10 days, Average Loss = 1.760682
<br>
Prediction for 20 days, Average Loss = 3.741657
<br>
Prediction for 30 days, Average Loss = 3.860052
<br>
Prediction for 40 days, Average Loss = 3.636619
<br>
Prediction for 50 days, Average Loss = 3.738984
<br>
Prediction for 60 days, Average Loss = 4.125126
<br>
Prediction for 70 days, Average Loss = 3.989271
<br>
Prediction for 80 days, Average Loss = 3.811168
<br>
Prediction for 90 days, Average Loss = 3.891586
<br>
Prediction for 100 days, Average Loss = 3.898718
<br>

##Notes<br>
<br>

Prediction Lengths: The program performs predictions for multiple lengths (e.g., 10, 20, ..., 100 days).<br>


Adjustable Parameters:<br>

N and M: Number of states and observation symbols, respectively.<br>
<br>

MAX_T and MAX_ITER: Adjust the maximum time periods and Baum-Welch iterations if needed.<br>
<br>

Data Size: Ensure MAX_T is large enough to accommodate the number of rows in your dataset.<br>


Scaling and Convergence: The program uses scaling in the forward-backward steps to prevent numerical underflow. Convergence is checked with a tolerance of 1e-4.<br>

