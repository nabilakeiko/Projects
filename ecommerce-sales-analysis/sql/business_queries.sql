USE ecommerce_sales_analysis;

-- 1. KPI Summary
SELECT
    ROUND(SUM(total_sales), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) AS total_customers,
    ROUND(SUM(total_sales) / COUNT(DISTINCT order_id), 2) AS average_order_value,
    ROUND(AVG(review_score), 2) AS average_review_score,
    ROUND(AVG(delivery_time_days), 2) AS average_delivery_time
FROM ecommerce_clean;

-- 2. Monthly Revenue Trend
SELECT
    order_year_month,
    ROUND(SUM(total_sales), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) AS total_customers
FROM ecommerce_clean
GROUP BY order_year_month
ORDER BY order_year_month;

-- 3. Top 10 Product Categories by Revenue
SELECT
    product_category_name_english AS product_category,
    ROUND(SUM(total_sales), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(AVG(review_score), 2) AS avg_review_score
FROM ecommerce_clean
GROUP BY product_category_name_english
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Payment Method Analysis
SELECT
    payment_type,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) AS total_customers,
    ROUND(SUM(total_sales), 2) AS total_sales,
    ROUND(SUM(payment_value), 2) AS total_payment_value,
    ROUND(AVG(review_score), 2) AS avg_review_score
FROM ecommerce_clean
GROUP BY payment_type
ORDER BY total_orders DESC;

-- 5. Late Delivery Impact
SELECT
    is_late_delivery,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) AS total_customers,
    ROUND(AVG(delivery_time_days), 2) AS avg_delivery_time_days,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    ROUND(SUM(total_sales), 2) AS total_sales
FROM ecommerce_clean
GROUP BY is_late_delivery
ORDER BY is_late_delivery;

-- 6. Delivery Performance by State
SELECT
    customer_state,
    COUNT(DISTINCT order_id) AS total_orders,
    COUNT(DISTINCT customer_unique_id) AS total_customers,
    ROUND(AVG(delivery_time_days), 2) AS avg_delivery_time_days,
    ROUND(
        AVG(
            CASE 
                WHEN is_late_delivery = 'True' 
                     OR is_late_delivery = 'true'
                     OR is_late_delivery = 1
                THEN 1 
                ELSE 0 
            END
        ) * 100, 2
    ) AS late_delivery_rate,
    ROUND(AVG(review_score), 2) AS avg_review_score,
    ROUND(SUM(total_sales), 2) AS total_revenue
FROM ecommerce_clean
GROUP BY customer_state
ORDER BY avg_delivery_time_days DESC
LIMIT 10;

-- 7. Category Risk Analysis
SELECT
    product_category_name_english AS product_category,
    ROUND(SUM(total_sales), 2) AS total_revenue,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(AVG(review_score), 2) AS avg_review_score
FROM ecommerce_clean
GROUP BY product_category_name_english
HAVING 
    COUNT(DISTINCT order_id) >= 100
    AND AVG(review_score) < (
        SELECT AVG(review_score)
        FROM ecommerce_clean
    )
ORDER BY total_revenue DESC
LIMIT 10;