# 📊 E-Commerce Sales Performance and Customer Behavior Analysis

## Project Overview

This project is an end-to-end data analytics project using Brazilian e-commerce transaction data. The analysis focuses on understanding sales performance, product category contribution, payment behavior, delivery performance, and customer satisfaction.

The project includes data cleaning, exploratory data analysis, SQL business queries, and interactive Tableau dashboards.

---

## Business Problem

E-commerce companies need to understand what drives revenue, which product categories perform best, how customers pay, and how delivery performance affects customer satisfaction.

This project aims to answer key business questions such as:

- How does monthly revenue change over time?
- Which product categories generate the highest revenue?
- What payment methods are most used by customers?
- Does late delivery affect customer review scores?
- Which regions have the longest delivery time?
- Which high-revenue categories have below-average customer ratings?

---

## Objectives

The main objectives of this project are:

1. Analyze overall sales performance.
2. Identify top product categories by revenue.
3. Analyze customer payment behavior.
4. Measure the impact of late delivery on customer satisfaction.
5. Identify regions with longer delivery times.
6. Provide business insights and recommendations through dashboard visualization.

---

## Dataset

Dataset used: Brazilian E-Commerce Public Dataset by Olist.

The dataset contains customer orders, products, payments, reviews, sellers, and delivery information.

Dataset source: Kaggle - Brazilian E-Commerce Public Dataset by Olist.

---

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- SQL
- MySQL Workbench
- Tableau
- Google Colab
- GitHub

---

## Project Workflow

1. Data Collection  
   Loaded multiple raw CSV files from the Olist e-commerce dataset.

2. Data Cleaning  
   Cleaned missing values, converted date columns, filtered delivered orders, and created new analytical columns such as delivery time and late delivery status.

3. Data Merging  
   Merged orders, customers, products, order items, payments, reviews, sellers, and category translation tables into one master dataset.

4. Exploratory Data Analysis  
   Analyzed revenue, orders, customers, product categories, payment methods, delivery performance, and review scores.

5. SQL Analysis  
   Used MySQL Workbench to run business queries for KPI summary, monthly revenue, top categories, payment methods, late delivery impact, and delivery performance by state.

6. Dashboard Creation  
   Built Tableau dashboards to present business performance and customer satisfaction insights.

---

## Key Metrics

| Metric | Value |
|---|---:|
| Total Revenue | 15,357,574.15 |
| Total Orders | 95,808 |
| Total Customers | 92,732 |
| Average Order Value | 160.30 |
| Average Review Score | 4.08 |
| Average Delivery Time | 11.97 days |
| Late Delivery Rate | 7.78% |

---

## SQL Analysis

The SQL analysis includes:

- KPI Summary
- Monthly Revenue Trend
- Top 10 Product Categories by Revenue
- Payment Method Analysis
- Late Delivery Impact on Review Score
- Delivery Performance by State
- High-Revenue Low-Rated Product Categories

---

## Key Business Insights
1. Credit card is the most dominant payment method, contributing the highest number of orders and transaction value.
2. Late delivery has a strong negative impact on customer satisfaction. On-time deliveries have an average review score of 4.21, while late deliveries only have     2.55.
3. Most customers are satisfied, as review score 5 dominates the distribution. However, score 1 reviews are still significant and need further investigation.
4. Several high-revenue categories have below-average review scores, including bed_bath_table, computers_accessories, furniture_decor, and office_furniture.
5. Some states such as RR, AP, AM, AL, and PA experience significantly longer delivery times than the overall average.


---

## Business Recommendations
1. Improve logistics performance to reduce late deliveries, as delivery delays are strongly associated with lower customer satisfaction.
2. Evaluate high-revenue but low-rated product categories to identify issues related to product quality, product description accuracy, packaging, seller     performance, or delivery experience.
3. Optimize regional delivery strategies, especially for states with high average delivery time and late delivery rate.
4. Use credit card as the main focus for payment-based promotions due to its dominant contribution to total orders and revenue.
5. Further investigate low review score orders to identify recurring problems and improve customer experience.

---

## Author
Nabila Keiko Aura Pasha
Aspiring Data Analyst | Python, SQL, Tableau, Data Visualization
