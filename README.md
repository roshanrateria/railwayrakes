# Railway Rakes Forecasting and Scheduling
## Problem Statement
At present, rake supply is made by railway on a cluster basis / Coalfield basis for a group of mines. At times, the placement of rakes in a siding is made where coal stock is not adequate. This leads to the payment of demurrage charges. A digital platform/algorithm needs to be created for all the available railway sidings where the updated status of coal stock in siding shall be maintained online. This will help in sending railway rakes available at the nearest location and also reduce the demurrage cost of the company.

## Overview
This project focuses on the forecasting and scheduling of railway rakes to optimize their usage and reduce demurrage costs. The current system of scheduling rakes on a cluster basis often leads to inefficiencies such as placing rakes at sidings with insufficient coal stock, resulting in demurrage charges. This project aims to create a digital platform/algorithm that maintains an updated status of coal stock at all railway sidings and schedules rakes to the nearest available location with sufficient stock.

## Features
- **Data Cleaning**: Load and clean the railway dataset to prepare it for analysis.
- **Feature Preparation**: Prepare features for demand forecasting, including station-wise features.
- **Demand Forecasting**: Train a Random Forest model to forecast the demand for railway rakes.
- **Subgraph Extraction**: Extract a connected subgraph of the railway network for analysis.
- **Distance Matrix Creation**: Create a distance matrix from the subgraph.
- **Path Optimization**: Optimize the path using OR-Tools to minimize the distance traveled by the rakes.
- **Path Visualization**: Plot the railway network and optimized path.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/roshanrateria/railwayrakes.git
   cd railwayrakes
   ```

2. Install the required libraries:
   ```sh
   pip install pandas numpy scikit-learn networkx ortools matplotlib
   ```

3. Run the script:
   ```sh
   python test.py
   ```

## How It Aligns with the Topic
The project aligns with the topic of "Forecasting and Scheduling of Railway Rakes" by:
- **Forecasting Demand**: Using machine learning (Random Forest) to forecast the demand for railway rakes at various stations, thus ensuring that rakes are allocated to stations where they are needed most.
- **Scheduling Optimization**: Extracting connected subgraphs of the railway network and using OR-Tools to find the most efficient paths for rakes, reducing travel distance and time.
- **Reducing Demurrage Costs**: The algorithm ensures that rakes are not placed at sidings with insufficient coal stock, thereby reducing demurrage charges by improving the accuracy of rake placement.
- **Digital Platform**: Providing a digital solution to maintain and update coal stock status at sidings, making the scheduling process more transparent and efficient.

## Output
Upon running the script, the following output is generated:
```
Loading and cleaning data...
Preparing features...
Training model...

Demand Forecasting Results:
Mean Absolute Error: 0.08

Sample Predictions vs Actual:
Actual: 12, Predicted: 12
Actual: 64, Predicted: 64
Actual: 10, Predicted: 10
Actual: 14, Predicted: 14
Actual: 10, Predicted: 10
Extracting subgraph...
Creating distance matrix...
No single train solution found. Constructing path using multiple trains/transfers.
No direct train between SVM and ZARP. Finding transfer...
No direct train between KM and BTJL. Finding transfer...
No direct train between BTJL and SRVX. Finding transfer...
No direct train between PAY and CNR. Finding transfer...
No direct train between CNR and HNA. Finding transfer...
No direct train between KUDA and PERN. Finding transfer...
No direct train between PERN and KT. Finding transfer...
Current Path:
SVM -> SRVX -> SRVX -> KM -> SVM -> BTJL -> QLM -> BTJL -> SRVX -> SVM -> PAY -> CNR -> CNR -> MAO -> HNA -> BTJL -> KUDA -> BTJL -> PERN -> KCVL -> SMV
Current Distance: 3210.00 km
Finding optimal path...

Path Optimization Results:
Optimized Path:
SWV -> ZARP -> KUDL -> KKW -> NAN -> PERN -> PAY -> MAJN -> UD -> KUDA -> SHMI -> BTJL -> HNA -> KT -> MAO -> SRVX -> CNR -> SVM -> KM -> KRMI -> SWV

Total Distance: 1193.00 km
```
![WhatsApp Image 2024-11-16 at 22 43 06_586f8fe2](https://github.com/user-attachments/assets/a0e10b40-06cc-4bbd-87d1-8d47894297f7)


## Note
Ensure that the dataset `Train_details_22122017.csv` is available in the same directory as the script for loading and cleaning data. The project aims to provide a comprehensive solution for efficient scheduling and forecasting of railway rakes, ultimately reducing costs and improving service quality.
