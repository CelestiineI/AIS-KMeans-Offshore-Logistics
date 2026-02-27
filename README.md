# AIS Vessel Movement Clustering Using K-Means

## Overview
This project implements a K-Means clustering application to analyse vessel movement patterns using AIS (Automatic Identification System) data. The objective is to identify meaningful maritime activity patterns that can support offshore logistics planning, operational monitoring, and decision-making in industries such as oil and gas, shipping, and port management.

The application processes vessel movement features including geographic position, speed, and direction to group similar vessel behaviours into clusters. These clusters reveal operational categories such as port activity, transit routes, and offshore support operations.

---

## Dataset
The dataset was obtained from the Marine Cadastre AIS Vessel Traffic repository. AIS data provides time-stamped vessel movement information such as latitude, longitude, speed over ground (SOG), and course over ground (COG).

For this project:
- Relevant movement features were selected
- Missing values were removed
- A sample of the dataset was used to improve performance

AIS data serves as a proxy for offshore logistics because vessel movement reflects supply runs, staging activity, and operational transit patterns.

---

## Methodology
The workflow consists of:

1. Data ingestion and preprocessing  
2. Feature scaling using StandardScaler  
3. Determination of optimal cluster number using the Elbow Method  
4. K-Means clustering of vessel movement patterns  
5. Visualization of spatial cluster behaviour  
6. Unit testing using pytest to validate application reliability

---

## Results and Insights
The clustering results revealed distinct vessel behaviour patterns, including:

- Low-speed localized movement consistent with offshore support operations  
- High-speed linear transit between ports and offshore locations  
- Coastal staging and waiting activity prior to offshore deployment  

These insights demonstrate how unsupervised learning can uncover operational structures within large maritime datasets and support logistics optimisation.

---

## Project Structure