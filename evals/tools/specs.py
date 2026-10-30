"""Hand-written queries for the agent v1 tool-selection eval (the agent the API serves).

Each spec has the tools the query requires (``required``), tools that are allowed but not
needed (``optional``: arithmetic that SQL can also do), and the checks the final answer
must pass. Every other tool counts as wrong. Gold values for ``sql`` checks are computed
from the seeded demo database by ``build_set.py``; ``doc`` checks quote the policy files in
``data/documents/``; ``calc`` checks are plain arithmetic. ``none`` queries need no tool
and their answer is not scored.

Check kinds: ``number`` (every listed number must appear in the answer, compared by
absolute value with a 1% relative tolerance, so "fell by 27%" matches -27.14), ``text`` (every listed string must appear, case-insensitive), ``sql_number`` /
``sql_text`` (the values of the first column of the gold query, then as above). Text is
compared without accents; ``aliases`` lists accepted translations of a value.
"""

from typing import Any

SQL = ["sql_query"]
DOC = ["doc_search"]
CALC = ["calculator"]

Y24 = "date BETWEEN '2024-01-01' AND '2024-12-31'"

SPECS: list[dict[str, Any]] = [
    # -- sql: the database alone answers it --------------------------------------------
    {"id": "sql_01", "category": "sql", "query": "How much did we sell in total in Q3 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(amount), 2) FROM sales WHERE quarter = 'Q3'"}]},
    {"id": "sql_02", "category": "sql", "query": "How many sales were made in Q1 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT COUNT(*) FROM sales WHERE quarter = 'Q1'"}]},
    {"id": "sql_03", "category": "sql", "query": "Which product generated the most revenue in 2024?",
     "required": SQL, "checks": [{"kind": "sql_text", "sql": f"SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id WHERE {Y24} GROUP BY p.id ORDER BY SUM(s.amount) DESC LIMIT 1"}]},
    {"id": "sql_04", "category": "sql", "query": "Which region had the highest sales amount in Q4 2024?",
     "required": SQL, "checks": [{"kind": "sql_text", "sql": "SELECT region FROM sales WHERE quarter = 'Q4' GROUP BY region ORDER BY SUM(amount) DESC LIMIT 1", "aliases": {"Centro": ["Central", "Center"]}}]},
    {"id": "sql_05", "category": "sql", "query": "How many units of the ProBook X1 were sold in 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT SUM(quantity) FROM sales WHERE product_id = 1"}]},
    {"id": "sql_06", "category": "sql", "query": "What was the total revenue from the Software category?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(s.amount), 2) FROM sales s JOIN products p ON p.id = s.product_id WHERE p.category = 'Software'"}]},
    {"id": "sql_07", "category": "sql", "query": "How many customers are in the Platinum tier?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT COUNT(*) FROM customers WHERE tier = 'Platinum'"}]},
    {"id": "sql_08", "category": "sql", "query": "Who was our top customer by total amount spent in 2024?",
     "required": SQL, "checks": [{"kind": "sql_text", "sql": "SELECT c.name FROM sales s JOIN customers c ON c.id = s.customer_id GROUP BY c.id ORDER BY SUM(s.amount) DESC LIMIT 1"}]},
    {"id": "sql_09", "category": "sql", "query": "What was the average sale amount in Q2 2024?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(AVG(amount), 2) FROM sales WHERE quarter = 'Q2'"}]},
    {"id": "sql_10", "category": "sql", "query": "How many sales did customers from the Sur region make in 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT COUNT(*) FROM sales WHERE region = 'Sur'"}]},
    {"id": "sql_11", "category": "sql", "query": "What is the price of the Monitor UltraWide 34?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT price FROM products WHERE name = 'Monitor UltraWide 34'"}]},
    {"id": "sql_12", "category": "sql", "query": "Which product in the Oficina category sold the most units?",
     "required": SQL, "checks": [{"kind": "sql_text", "sql": "SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id WHERE p.category = 'Oficina' GROUP BY p.id ORDER BY SUM(s.quantity) DESC LIMIT 1"}]},
    {"id": "sql_13", "category": "sql", "query": "What was the largest single sale amount in 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT MAX(amount) FROM sales"}]},
    {"id": "sql_14", "category": "sql", "query": "How many different products did TechCorp S.L. buy?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT COUNT(DISTINCT product_id) FROM sales WHERE customer_id = 1"}]},
    {"id": "sql_15", "category": "sql", "query": "What was total revenue in the Norte region in Q3 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(amount), 2) FROM sales WHERE region = 'Norte' AND quarter = 'Q3'"}]},
    {"id": "sql_16", "category": "sql", "query": "How many Gold-tier customers placed at least one order in Q4 2024?",
     "required": SQL, "checks": [{"kind": "sql_number", "sql": "SELECT COUNT(DISTINCT s.customer_id) FROM sales s JOIN customers c ON c.id = s.customer_id WHERE c.tier = 'Gold' AND s.quarter = 'Q4'"}]},
    {"id": "sql_17", "category": "sql", "query": "What are the three best-selling products by units sold in 2024?",
     "required": SQL, "checks": [{"kind": "sql_text", "sql": "SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id GROUP BY p.id ORDER BY SUM(s.quantity) DESC LIMIT 3"}]},
    {"id": "sql_18", "category": "sql", "query": "In which quarter of 2024 did we sell the most units of the TabletAir 10?",
     "required": SQL, "checks": [{"kind": "sql_text", "sql": "SELECT quarter FROM sales WHERE product_id = 3 GROUP BY quarter ORDER BY SUM(quantity) DESC LIMIT 1"}]},
    # -- doc: the policy files alone answer it -----------------------------------------
    {"id": "doc_01", "category": "doc", "query": "How many days do customers have to return a product?",
     "required": DOC, "source": "politica_devoluciones.md: 30 días naturales", "checks": [{"kind": "number", "values": [30]}]},
    {"id": "doc_02", "category": "doc", "query": "What is the return window for electronic products?",
     "required": DOC, "source": "politica_devoluciones.md: plazo reducido a 15 días", "checks": [{"kind": "number", "values": [15]}]},
    {"id": "doc_03", "category": "doc", "query": "What additional discount do Gold-tier customers get?",
     "required": DOC, "source": "politica_descuentos.md: Gold, 7% de descuento adicional", "checks": [{"kind": "number", "values": [7]}]},
    {"id": "doc_04", "category": "doc", "query": "What is the maximum combined discount a customer can get?",
     "required": DOC, "source": "politica_descuentos.md: no puede superar el 25%", "checks": [{"kind": "number", "values": [25]}]},
    {"id": "doc_05", "category": "doc", "query": "How long is the warranty on electronics?",
     "required": DOC, "source": "faq_productos.md: Electrónica, 2 años de garantía", "checks": [{"kind": "number", "values": [2]}]},
    {"id": "doc_06", "category": "doc", "query": "After a returned product is received and inspected, how many business days does the refund take?",
     "required": DOC, "source": "politica_devoluciones.md: reembolso en un plazo de 5-7 días laborables", "checks": [{"kind": "number", "values": [5, 7]}]},
    {"id": "doc_07", "category": "doc", "query": "What volume discount applies to an order of 30 units?",
     "required": DOC, "source": "politica_descuentos.md: 25-49 unidades, 8%", "checks": [{"kind": "number", "values": [8]}]},
    {"id": "doc_08", "category": "doc", "query": "What is the welcome discount on a new customer's first order?",
     "required": DOC, "source": "politica_descuentos.md: Primer pedido, 10% de descuento de bienvenida", "checks": [{"kind": "number", "values": [10]}]},
    {"id": "doc_09", "category": "doc", "query": "How much does the initial setup of a laptop or tablet cost?",
     "required": DOC, "source": "faq_productos.md: Configuración de portátil/tablet: €29", "checks": [{"kind": "number", "values": [29]}]},
    {"id": "doc_10", "category": "doc", "query": "How long does shipping to the rest of the EU take?",
     "required": DOC, "source": "faq_productos.md: Resto de la UE: 5-7 días laborables", "checks": [{"kind": "number", "values": [5, 7]}]},
    {"id": "doc_11", "category": "doc", "query": "How long do customers have to claim for a manufacturing defect?",
     "required": DOC, "source": "politica_devoluciones.md: 2 años desde la fecha de compra", "checks": [{"kind": "number", "values": [2]}]},
    {"id": "doc_12", "category": "doc", "query": "What is the maximum promotional discount?",
     "required": DOC, "source": "politica_descuentos.md: El descuento promocional máximo es del 20%", "checks": [{"kind": "number", "values": [20]}]},
    # -- calc: plain arithmetic stated in the question ---------------------------------
    {"id": "calc_01", "category": "calc", "query": "What is 18% of 4,250 euros?",
     "required": CALC, "checks": [{"kind": "number", "values": [765]}]},
    {"id": "calc_02", "category": "calc", "query": "A product costs 1,299 euros. What is its price after a 12% discount?",
     "required": CALC, "checks": [{"kind": "number", "values": [1143.12]}]},
    {"id": "calc_03", "category": "calc", "query": "By what percent did revenue grow if it went from 120,000 to 150,000?",
     "required": CALC, "checks": [{"kind": "number", "values": [25]}]},
    {"id": "calc_04", "category": "calc", "query": "Calculate 459 multiplied by 14.",
     "required": CALC, "checks": [{"kind": "number", "values": [6426]}]},
    {"id": "calc_05", "category": "calc", "query": "What is the average of 230, 415 and 512?",
     "required": CALC, "checks": [{"kind": "number", "values": [385.67]}]},
    {"id": "calc_06", "category": "calc", "query": "What is 2 to the power of 10, times 3?",
     "required": CALC, "checks": [{"kind": "number", "values": [3072]}]},
    # -- sql_math: database values plus arithmetic (SQL can also do the arithmetic) -----
    {"id": "sqlm_01", "category": "sql_math", "query": "What was the sales growth from Q2 to Q3 2024, in percent?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(100.0 * (SUM(CASE WHEN quarter='Q3' THEN amount END) - SUM(CASE WHEN quarter='Q2' THEN amount END)) / SUM(CASE WHEN quarter='Q2' THEN amount END), 2) FROM sales"}]},
    {"id": "sqlm_02", "category": "sql_math", "query": "What share of 2024 revenue came from the Electrónica category, in percent?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(100.0 * SUM(CASE WHEN p.category='Electrónica' THEN s.amount END) / SUM(s.amount), 2) FROM sales s JOIN products p ON p.id = s.product_id"}]},
    {"id": "sqlm_03", "category": "sql_math", "query": "What was the difference in revenue between Q4 and Q1 2024, in euros?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(CASE WHEN quarter='Q4' THEN amount END) - SUM(CASE WHEN quarter='Q1' THEN amount END), 2) FROM sales"}]},
    {"id": "sqlm_04", "category": "sql_math", "query": "What percentage of all units sold in 2024 were ProBook X2?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(100.0 * SUM(CASE WHEN product_id=2 THEN quantity END) / SUM(quantity), 2) FROM sales"}]},
    {"id": "sqlm_05", "category": "sql_math", "query": "If we raised the price of the Teclado MechPro by 15%, what would its new price be?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(price * 1.15, 2) FROM products WHERE name = 'Teclado MechPro'"}]},
    {"id": "sqlm_06", "category": "sql_math", "query": "What was the average revenue per quarter in 2024?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(amount) / 4, 2) FROM sales"}]},
    {"id": "sqlm_07", "category": "sql_math", "query": "By what percent did revenue in the Norte region change from Q1 to Q4 2024?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(100.0 * (SUM(CASE WHEN quarter='Q4' THEN amount END) - SUM(CASE WHEN quarter='Q1' THEN amount END)) / SUM(CASE WHEN quarter='Q1' THEN amount END), 2) FROM sales WHERE region = 'Norte'"}]},
    {"id": "sqlm_08", "category": "sql_math", "query": "What was the average 2024 revenue per customer, counting only customers who bought something?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(amount) / COUNT(DISTINCT customer_id), 2) FROM sales"}]},
    {"id": "sqlm_09", "category": "sql_math", "query": "What percentage of the Q3 2024 sales (by count) came from Platinum customers?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(100.0 * SUM(CASE WHEN c.tier='Platinum' THEN 1 ELSE 0 END) / COUNT(*), 2) FROM sales s JOIN customers c ON c.id = s.customer_id WHERE s.quarter = 'Q3'"}]},
    {"id": "sqlm_10", "category": "sql_math", "query": "How much would TechCorp S.L.'s total 2024 spending be after a 10% rebate?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(amount) * 0.9, 2) FROM sales WHERE customer_id = 1"}]},
    {"id": "sqlm_11", "category": "sql_math", "query": "What is the ratio of Electrónica revenue to Oficina revenue in 2024?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(SUM(CASE WHEN p.category='Electrónica' THEN s.amount END) / SUM(CASE WHEN p.category='Oficina' THEN s.amount END), 2) FROM sales s JOIN products p ON p.id = s.product_id"}]},
    {"id": "sqlm_12", "category": "sql_math", "query": "By what percent did the number of units sold change from Q1 to Q2 2024?",
     "required": SQL, "optional": CALC, "checks": [{"kind": "sql_number", "sql": "SELECT ROUND(100.0 * (SUM(CASE WHEN quarter='Q2' THEN quantity END) - SUM(CASE WHEN quarter='Q1' THEN quantity END)) / SUM(CASE WHEN quarter='Q1' THEN quantity END), 2) FROM sales"}]},
    # -- sql_doc: a database fact and a policy fact ------------------------------------
    {"id": "sqldoc_01", "category": "sql_doc", "query": "What tier is MegaStore in, and what additional discount does that tier get?",
     "required": SQL + DOC, "source": "politica_descuentos.md: Platinum, 10% de descuento adicional",
     "checks": [{"kind": "sql_text", "sql": "SELECT tier FROM customers WHERE name = 'MegaStore'"}, {"kind": "number", "values": [10]}]},
    {"id": "sqldoc_02", "category": "sql_doc", "query": "What tier is DigitalHub in, and what additional discount does that give them?",
     "required": SQL + DOC, "source": "politica_descuentos.md: Gold, 7% de descuento adicional",
     "checks": [{"kind": "sql_text", "sql": "SELECT tier FROM customers WHERE name = 'DigitalHub'"}, {"kind": "number", "values": [7]}]},
    {"id": "sqldoc_03", "category": "sql_doc", "query": "Which category does the TabletAir 10 belong to, and how many days does a customer have to return it?",
     "required": SQL + DOC, "source": "politica_devoluciones.md: Productos electrónicos, plazo reducido a 15 días",
     "checks": [{"kind": "sql_text", "sql": "SELECT category FROM products WHERE name = 'TabletAir 10'", "aliases": {"Electrónica": ["Electronics"]}}, {"kind": "number", "values": [15]}]},
    {"id": "sqldoc_04", "category": "sql_doc", "query": "How many Silver-tier customers do we have, and what additional discount does that tier get?",
     "required": SQL + DOC, "source": "politica_descuentos.md: Silver, 3% de descuento adicional",
     "checks": [{"kind": "sql_number", "sql": "SELECT COUNT(*) FROM customers WHERE tier = 'Silver'"}, {"kind": "number", "values": [3]}]},
    {"id": "sqldoc_05", "category": "sql_doc", "query": "What does the Escritorio StandUp cost, and how many years of structural warranty does office furniture have?",
     "required": SQL + DOC, "source": "faq_productos.md: Mobiliario de oficina, 5 años de garantía estructural",
     "checks": [{"kind": "sql_number", "sql": "SELECT price FROM products WHERE name = 'Escritorio StandUp'"}, {"kind": "number", "values": [5]}]},
    {"id": "sqldoc_06", "category": "sql_doc", "query": "Which product generated the most revenue in 2024, and how long is the warranty for its category?",
     "required": SQL + DOC, "source": "faq_productos.md: Electrónica, 2 años de garantía (the top product, ProBook X2, is Electrónica)",
     "checks": [{"kind": "sql_text", "sql": "SELECT p.name FROM sales s JOIN products p ON p.id = s.product_id GROUP BY p.id ORDER BY SUM(s.amount) DESC LIMIT 1"},
                {"kind": "number", "values": [2]}]},
    # -- none: no tool is needed; only tool selection is scored -------------------------
    {"id": "none_01", "category": "none", "query": "Hello! What can you help me with?", "required": [], "checks": []},
    {"id": "none_02", "category": "none", "query": "Thanks, that's all for today.", "required": [], "checks": []},
    {"id": "none_03", "category": "none", "query": "What is the capital of France?", "required": [], "checks": []},
    {"id": "none_04", "category": "none", "query": "Can you explain what a SQL JOIN is?", "required": [], "checks": []},
    {"id": "none_05", "category": "none", "query": "Write a two-line poem about selling laptops.", "required": [], "checks": []},
    {"id": "none_06", "category": "none", "query": "What time is it in Tokyo right now?", "required": [], "checks": []},
]  # fmt: skip
