# Agent v1 tool-selection set: review sheet

For every query, check the expected tools and the gold answer, then write `ok` or
what to change in `review.csv` (same folder). `required`: tools the query needs;
`optional`: allowed but not needed (arithmetic that SQL can also do); every other
tool counts as wrong. Answers are scored by the numbers and names in `gold`
(numbers by absolute value, with 1% relative tolerance; text without accents).
`none` queries score tool selection only.
No model has been run on these queries.

| id | split | category | query | required | optional | gold | source |
|---|---|---|---|---|---|---|---|
| sql_01 | dev | sql | How much did we sell in total in Q3 2024? | sql_query |  | 214067.19 | `SELECT ROUND(SUM(amount), 2) FROM sales WHERE quarter = 'Q3'` |
| sql_02 | test | sql | How many sales were made in Q1 2024? | sql_query |  | 30 | `SELECT COUNT(*) FROM sales WHERE quarter = 'Q1'` |
| sql_03 | test | sql | Which product generated the most revenue in 2024? | sql_query |  | ProBook X2 | `SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id WHERE date BETWEEN '2024-01-01' AND '2024-12-31' GROUP BY p.id ORDER BY SUM(s.amount) DESC LIMIT 1` |
| sql_04 | test | sql | Which region had the highest total sales in Q4 2024? | sql_query |  | Centro | `SELECT region FROM sales WHERE quarter = 'Q4' GROUP BY region ORDER BY SUM(amount) DESC LIMIT 1` |
| sql_05 | dev | sql | How many units of the ProBook X1 were sold in 2024? | sql_query |  | 79 | `SELECT SUM(s.quantity) FROM sales s JOIN products p ON p.id = s.product_id WHERE p.name = 'ProBook X1'` |
| sql_06 | test | sql | What was the total revenue from the Software category? | sql_query |  | 31556.29 | `SELECT ROUND(SUM(s.amount), 2) FROM sales s JOIN products p ON p.id = s.product_id WHERE p.category = 'Software'` |
| sql_07 | test | sql | How many customers are in the Platinum tier? | sql_query |  | 3 (with customer / client) | `SELECT COUNT(*) FROM customers WHERE tier = 'Platinum'` |
| sql_08 | test | sql | Who was our top customer by total amount spent in 2024? | sql_query |  | DataSystems | `SELECT c.name FROM sales s JOIN customers c ON c.id = s.customer_id GROUP BY c.id ORDER BY SUM(s.amount) DESC LIMIT 1` |
| sql_09 | test | sql | What was the average sale amount in Q2 2024? | sql_query | calculator | 4802.46 | `SELECT ROUND(AVG(amount), 2) FROM sales WHERE quarter = 'Q2'` |
| sql_10 | test | sql | How many sales were made in the Sur region in 2024? | sql_query |  | 42 | `SELECT COUNT(*) FROM sales WHERE region = 'Sur'` |
| sql_11 | test | sql | What is the price of the Monitor UltraWide 34? | sql_query |  | 579 | `SELECT price FROM products WHERE name = 'Monitor UltraWide 34'` |
| sql_12 | dev | sql | Which product in the Oficina category sold the most units? | sql_query |  | Escritorio StandUp | `SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id WHERE p.category = 'Oficina' GROUP BY p.id ORDER BY SUM(s.quantity) DESC LIMIT 1` |
| sql_13 | test | sql | What was the largest single sale amount in 2024? | sql_query |  | 21376.3 | `SELECT MAX(amount) FROM sales` |
| sql_14 | dev | sql | How many different products did TechCorp S.L. buy? | sql_query |  | 7 (with product) | `SELECT COUNT(DISTINCT s.product_id) FROM sales s JOIN customers c ON c.id = s.customer_id WHERE c.name = 'TechCorp S.L.'` |
| sql_15 | dev | sql | What was total revenue in the Norte region in Q3 2024? | sql_query |  | 64813.09 | `SELECT ROUND(SUM(amount), 2) FROM sales WHERE region = 'Norte' AND quarter = 'Q3'` |
| sql_16 | test | sql | How many Gold-tier customers placed at least one order in Q4 2024? | sql_query |  | 3 (with customer / client) | `SELECT COUNT(DISTINCT s.customer_id) FROM sales s JOIN customers c ON c.id = s.customer_id WHERE c.tier = 'Gold' AND s.quarter = 'Q4'` |
| sql_17 | dev | sql | What are the three best-selling products by units sold in 2024? | sql_query |  | ProBook X2, Auriculares NoiseMax, Cargador Universal 100W | `SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id GROUP BY p.id ORDER BY SUM(s.quantity) DESC LIMIT 3` |
| sql_18 | dev | sql | In which quarter of 2024 did we sell the most units of the TabletAir 10? | sql_query |  | Q2 | `SELECT s.quarter FROM sales s JOIN products p ON p.id = s.product_id WHERE p.name = 'TabletAir 10' GROUP BY s.quarter ORDER BY SUM(s.quantity) DESC LIMIT 1` |
| doc_01 | dev | doc | How many days do customers have to return a product? | doc_search |  | 30 | `politica_devoluciones.md: 30 días naturales` |
| doc_02 | test | doc | What is the return window for electronic products? | doc_search |  | 15 | `politica_devoluciones.md: plazo reducido a 15 días` |
| doc_03 | dev | doc | What additional discount do Gold-tier customers get? | doc_search |  | 7 (with % / percent) | `politica_descuentos.md: Gold, 7% de descuento adicional` |
| doc_04 | test | doc | What is the maximum combined discount a customer can get? | doc_search |  | 25 | `politica_descuentos.md: no puede superar el 25%` |
| doc_05 | test | doc | How long is the warranty on electronics? | doc_search |  | 2 (with year / ano) | `faq_productos.md: Electrónica, 2 años de garantía` |
| doc_06 | test | doc | After a returned product is received and inspected, how many business days does the refund take? | doc_search |  | 5, 7 (with business day / dias laborables) | `politica_devoluciones.md: reembolso en un plazo de 5-7 días laborables` |
| doc_07 | test | doc | What volume discount applies to an order of 30 units? | doc_search |  | 8 (with % / percent) | `politica_descuentos.md: 25-49 unidades, 8%` |
| doc_08 | test | doc | What is the welcome discount on a new customer's first order? | doc_search |  | 10 | `politica_descuentos.md: Primer pedido, 10% de descuento de bienvenida` |
| doc_09 | dev | doc | How much does the initial setup of a laptop or tablet cost? | doc_search |  | 29 | `faq_productos.md: Configuración de portátil/tablet: €29` |
| doc_10 | dev | doc | How long does shipping to the rest of the EU take? | doc_search |  | 5, 7 (with day / dia) | `faq_productos.md: Resto de la UE: 5-7 días laborables` |
| doc_11 | test | doc | How long do customers have to claim for a manufacturing defect? | doc_search |  | 2 (with year / ano) | `politica_devoluciones.md: 2 años desde la fecha de compra` |
| doc_12 | dev | doc | What is the maximum promotional discount? | doc_search |  | 20 | `politica_descuentos.md: El descuento promocional máximo es del 20%` |
| calc_01 | test | calc | What is 18% of 4,250 euros? | calculator |  | 765 | `arithmetic in the question` |
| calc_02 | test | calc | A product costs 1,299 euros. What is its price after a 12% discount? | calculator |  | 1143.12 | `arithmetic in the question` |
| calc_03 | dev | calc | By what percent did revenue grow if it went from 120,000 to 150,000? | calculator |  | 25 | `arithmetic in the question` |
| calc_04 | test | calc | Calculate 459 multiplied by 14. | calculator |  | 6426 | `arithmetic in the question` |
| calc_05 | dev | calc | What is the average of 230, 415 and 512? | calculator |  | 385.67 | `arithmetic in the question` |
| calc_06 | test | calc | What is 2 to the power of 10, times 3? | calculator |  | 3072 | `arithmetic in the question` |
| sqlm_01 | dev | sql_math | What was the sales growth from Q2 to Q3 2024, in percent? | sql_query | calculator | 27.36 | `SELECT ROUND(100.0 * (SUM(CASE WHEN quarter='Q3' THEN amount END) - SUM(CASE WHEN quarter='Q2' THEN amount END)) / SUM(CASE WHEN quarter='Q2' THEN amount END), 2) FROM sales` |
| sqlm_02 | test | sql_math | What share of 2024 revenue came from the Electrónica category, in percent? | sql_query | calculator | 67.05 | `SELECT ROUND(100.0 * SUM(CASE WHEN p.category='Electrónica' THEN s.amount END) / SUM(s.amount), 2) FROM sales s JOIN products p ON p.id = s.product_id` |
| sqlm_03 | dev | sql_math | What was the difference in revenue between Q4 and Q1 2024, in euros? | sql_query | calculator | -18676.79 | `SELECT ROUND(SUM(CASE WHEN quarter='Q4' THEN amount END) - SUM(CASE WHEN quarter='Q1' THEN amount END), 2) FROM sales` |
| sqlm_04 | test | sql_math | What percentage of all units sold in 2024 were ProBook X2? | sql_query | calculator | 18.25 | `SELECT ROUND(100.0 * SUM(CASE WHEN p.name='ProBook X2' THEN s.quantity END) / SUM(s.quantity), 2) FROM sales s JOIN products p ON p.id = s.product_id` |
| sqlm_05 | test | sql_math | If we raised the price of the Teclado MechPro by 15%, what would its new price be? | sql_query | calculator | 148.35 | `SELECT ROUND(price * 1.15, 2) FROM products WHERE name = 'Teclado MechPro'` |
| sqlm_06 | test | sql_math | What was the average revenue per quarter in 2024? | sql_query | calculator | 166244.73 | `SELECT ROUND(SUM(amount) / 4, 2) FROM sales` |
| sqlm_07 | dev | sql_math | By what percent did revenue in the Norte region change from Q1 to Q4 2024? | sql_query | calculator | -27.14 | `SELECT ROUND(100.0 * (SUM(CASE WHEN quarter='Q4' THEN amount END) - SUM(CASE WHEN quarter='Q1' THEN amount END)) / SUM(CASE WHEN quarter='Q1' THEN amount END), 2) FROM sales WHERE region = 'Norte'` |
| sqlm_08 | dev | sql_math | What was the average 2024 revenue per customer, counting only customers who bought something? | sql_query | calculator | 33248.95 | `SELECT ROUND(SUM(amount) / COUNT(DISTINCT customer_id), 2) FROM sales` |
| sqlm_09 | test | sql_math | What percentage of the Q3 2024 sales (by count) came from Platinum customers? | sql_query | calculator | 11.11 | `SELECT ROUND(100.0 * SUM(CASE WHEN c.tier='Platinum' THEN 1 ELSE 0 END) / COUNT(*), 2) FROM sales s JOIN customers c ON c.id = s.customer_id WHERE s.quarter = 'Q3'` |
| sqlm_10 | dev | sql_math | How much would TechCorp S.L.'s total 2024 spending be after a 10% rebate? | sql_query | calculator | 25201.11 | `SELECT ROUND(SUM(s.amount) * 0.9, 2) FROM sales s JOIN customers c ON c.id = s.customer_id WHERE c.name = 'TechCorp S.L.'` |
| sqlm_11 | test | sql_math | What is the ratio of Electrónica revenue to Oficina revenue in 2024? | sql_query | calculator | 2.38 | `SELECT ROUND(SUM(CASE WHEN p.category='Electrónica' THEN s.amount END) / SUM(CASE WHEN p.category='Oficina' THEN s.amount END), 2) FROM sales s JOIN products p ON p.id = s.product_id` |
| sqlm_12 | test | sql_math | By what percent did the number of units sold change from Q1 to Q2 2024? | sql_query | calculator | -2.55 | `SELECT ROUND(100.0 * (SUM(CASE WHEN quarter='Q2' THEN quantity END) - SUM(CASE WHEN quarter='Q1' THEN quantity END)) / SUM(CASE WHEN quarter='Q1' THEN quantity END), 2) FROM sales` |
| sqldoc_01 | test | sql_doc | What tier is MegaStore in, and what additional discount does that tier get? | sql_query, doc_search |  | Platinum; 10 | `SELECT tier FROM customers WHERE name = 'MegaStore' / politica_descuentos.md: Platinum, 10% de descuento adicional` |
| sqldoc_02 | dev | sql_doc | What tier is DigitalHub in, and what additional discount does that give them? | sql_query, doc_search |  | Gold; 7 (with % / percent) | `SELECT tier FROM customers WHERE name = 'DigitalHub' / politica_descuentos.md: Gold, 7% de descuento adicional` |
| sqldoc_03 | test | sql_doc | Which category does the TabletAir 10 belong to, and how many days does a customer have to return it? | sql_query, doc_search |  | Electrónica; 15 | `SELECT category FROM products WHERE name = 'TabletAir 10' / politica_devoluciones.md: Productos electrónicos, plazo reducido a 15 días` |
| sqldoc_04 | test | sql_doc | How many Silver-tier customers do we have, and what additional discount does that tier get? | sql_query, doc_search |  | 6 (with customer / client); 3 (with % / percent) | `SELECT COUNT(*) FROM customers WHERE tier = 'Silver' / politica_descuentos.md: Silver, 3% de descuento adicional` |
| sqldoc_05 | dev | sql_doc | What does the Escritorio StandUp cost, and how many years of structural warranty does office furniture have? | sql_query, doc_search |  | 699; 5 (with year / ano) | `SELECT price FROM products WHERE name = 'Escritorio StandUp' / faq_productos.md: Mobiliario de oficina, 5 años de garantía estructural` |
| sqldoc_06 | test | sql_doc | Which product generated the most revenue in 2024, and how long is the warranty for its category? | sql_query, doc_search |  | ProBook X2; 2 (with year / ano) | `SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id GROUP BY p.id ORDER BY SUM(s.amount) DESC LIMIT 1 / faq_productos.md: Electrónica, 2 años de garantía (the top product, ProBook X2, is Electrónica)` |
| none_01 | dev | none | Hello! What can you help me with? | (none) |  | (not scored) |  |
| none_02 | test | none | Thanks, that's all for today. | (none) |  | (not scored) |  |
| none_03 | dev | none | What is the capital of France? | (none) |  | (not scored) |  |
| none_04 | test | none | Can you explain what a SQL JOIN is? | (none) |  | (not scored) |  |
| none_05 | test | none | Write a two-line poem about selling laptops. | (none) |  | (not scored) |  |
| none_06 | test | none | What time is it in Tokyo right now? | (none) |  | (not scored) |  |
