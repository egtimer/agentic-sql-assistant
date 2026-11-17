# Failure review sheet

For each item, read the question, what the agent did and its answer, and write in `labels.csv` the category you would give it (one of the list below, or `not_a_failure`). The rule's category is shown so you can disagree with it; the reference result is only in the private copy.

| Category | Meaning |
|---|---|
| `guard_refusal` | the SQL guard refused a legitimate query (`blocked:`) |
| `schema_hallucination` | the query names a column or table that does not exist |
| `bad_arguments` | a tool got an input it cannot run: SQL syntax error, calculator parse error, or an unfilled `PLACEHOLDER` |
| `silent_tool_failure` | a tool failed or returned nothing and the answer does not say so |
| `wrong_tool` | a required tool was not called, or a tool that was not needed was |
| `loop_or_extra_steps` | the agent used up its fixes or repeated the same call with the same input |
| `answer_on_large_result` | the SQL is right (EX) but the result is larger than what the answer step sees, and the answer is judged wrong |
| `tool_output_cut` | a tool returned the answer, but after the first 1,500 characters of its output, which is all the agent's prompt keeps |
| `answer_misread` | the SQL or tool result holds the right values and the answer gets them wrong |
| `retrieval_miss` | the document search did not return the passage with the answer |
| `answer_ok_ex_wrong` | EX marks the SQL wrong but the judge accepts the answer (usually extra columns or another form of the same value) |
| `wrong_output_shape` | the result has more or fewer columns than the gold one |
| `wrong_query_logic` | same columns as the gold result, different values or rows (filters, joins, aggregation) |
| `other` | none of the rules fired |
| `not_a_failure` | the reviewer found the answer acceptable |
| `benchmark_issue` | the gold query or the question is wrong or ambiguous (e.g. the gold SQL does not follow the external knowledge) |

## r-01 · `tools-sonnet-run-r03-sqldoc_03` · rule: `tool_output_cut`

Tags: `tool_output_cut`. answer check False; exact True; sql_query(ok), doc_search(ok).

**Steps:**

- `sql_query` (ok): `SELECT name, category FROM products WHERE name LIKE '%TabletAir 10%'`
- `doc_search` (ok): `política de devoluciones plazo días para devolver`

Output of step 1 (as recorded, at most 2,000 characters):

```
{"query": "SELECT name, category FROM products WHERE name LIKE '%TabletAir 10%'", "columns": ["name", "category"], "rows": [["TabletAir 10", "Electrónica"]], "row_count": 1, "success": true, "error": null}
```

Output of step 2 (as recorded, at most 2,000 characters):

```
{"query": "política de devoluciones plazo días para devolver", "chunks": [{"content": "# Política de Devoluciones\n\n## Plazo General\nLos clientes disponen de **30 días naturales** desde la fecha de recepción para solicitar la devolución de cualquier producto, siempre que se cumplan las condiciones establecidas en esta política.", "source": "politica_devoluciones.md", "score": 0.0328}, {"content": "admiten devolución, salvo defecto de fabricación.\n\n## Proceso de Devolución\n1. El cliente debe contactar al departamento de atención al cliente indicando el número de pedido.\n2. Se generará un código de autorización de devolución (RMA) en un plazo máximo de 48 horas laborables.\n3. El cliente envía el producto con el código RMA visible en el paquete.\n4. Una vez recibido e inspeccionado el producto, se procederá al reembolso en un plazo de 5-7 días laborables.", "source": "politica_devoluciones.md", "score": 0.031}, {"content": "á al reembolso en un plazo de 5-7 días laborables.\n\n## Método de Reembolso\n- El reembolso se realizará por el mismo método de pago utilizado en la compra original.\n- Los gastos de envío de la devolución corren a cargo del cliente, excepto en casos de producto defectuoso.", "source": "politica_devoluciones.md", "score": 0.031}, {"content": "lan las condiciones establecidas en esta política.\n\n## Condiciones de Devolución\n1. El producto debe estar en su embalaje original, sin signos de uso o deterioro.\n2. Se debe presentar el ticket de compra o fa
[--- the agent's prompt stops here ---]
ctura correspondiente.\n3. Los productos personalizados o hechos a medida no admiten devolución, salvo defecto de fabricación.", "source": "politica_devoluciones.md", "score": 0.0308}, {"content": "cliente, excepto en casos de producto defectuoso.\n\n## Excepciones\n- **Productos electrónicos**: plazo reducido a 15 días y deben incluir todos los accesorios originales.\n- **Productos perecederos**: no se admiten devoluciones salvo defecto evidente al momento de la entrega.\n- **Software y licenci
```

**Answer:** The TabletAir 10 belongs to the **Electrónica** category, and a customer has **30 días naturales** from the date of receipt to request a return.  **Sources:** - **Category:** according to the products database (SQL query), the TabletAir 10 is classified as "Electrónica". - **Return period:** according to the returns policy (*politica_devoluciones.md*), customers have 30 calendar days from receipt to request a return, provided the policy's conditions are met.  **Additional context from the policy:** - The product must be in its original packaging, with no signs of use or damage. - The customer must contact customer service with the order number. A return authorization code (RMA) is issued within 48 business hours. - Once the product is received and inspected, the refund is made within 5-7 business days, using the original payment method. - Return shipping costs are paid by the customer, except for defective products.  The retrieved policy text mentions exceptions for certain products ("admiten devolución, salvo defecto de fabricación"), but the excerpt is cut off. I can't confirm from these results whether electronics are affected, so I recommend checking the full policy.

[trace](../traces/tools-sonnet-run-r03-sqldoc_03.json)

## r-02 · `bird-haiku-118` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** For loan amount less than USD100,000, what is the percentage of accounts that is still running with no issue.

**External knowledge:** Status = 'C' stands for running contract, ok so far; Percentage of accounts by condition = [(total(amount) & condition) / (total amount)] * 100.

**Final SQL:**

```sql
SELECT SUM(CASE WHEN status = 'C' THEN amount ELSE 0 END) * 100.0 / SUM(amount) AS percentage
FROM loan
WHERE amount < 100000
```

**Answer:** About 47.55% of accounts with loans under USD 100,000 are still running with no issues.

[trace](../traces/bird-haiku-118.json)

## r-03 · `bird-haiku-964` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** List out the code for drivers who have nationality in American.

**External knowledge:** nationality = 'American'

**Final SQL:**

```sql
SELECT code FROM drivers WHERE nationality = 'American'
```

**Answer:** Most of the 157 matching drivers have no code recorded (NULL). The only code shown in the first 20 rows is SPE.

[trace](../traces/bird-haiku-964.json)

## r-04 · `bird-haiku-27` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 3 vs gold 3, rows 813 vs gold 8574.

**Question:** What is the average score in writing for the schools that were opened after 1991 or closed before 2000? List the school names along with the score. Also, list the communication number of the schools if there is any.

**External knowledge:** Communication number refers to phone number.

**Final SQL:**

```sql
SELECT s.School,
       sat.AvgScrWrite AS AvgWritingScore,
       s.Phone
FROM schools s
JOIN satscores sat ON sat.cds = s.CDSCode
WHERE CAST(strftime('%Y', s.OpenDate) AS INTEGER) > 1991
   OR CAST(strftime('%Y', s.ClosedDate) AS INTEGER) < 2000
```

**Answer:** The result doesn't show the opening or closing dates, so I can't confirm which schools meet the filter. Among the 20 rows shown, 15 have writing scores, averaging about 434.9. The full 813-row result would be needed for the overall average.

[trace](../traces/bird-haiku-27.json)

## r-05 · `bird-haiku-1505` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** Among the customers who paid in euro, how many of them have a monthly consumption of over 1000?

**External knowledge:** Pays in euro = Currency = 'EUR'.

**Final SQL:**

```sql
SELECT COUNT(DISTINCT c.CustomerID)
FROM customers c
JOIN yearmonth y ON y.CustomerID = c.CustomerID
WHERE c.Currency = 'EUR'
  AND y.Consumption > 1000
```

**Answer:** There are 391 customers who paid in euro and have a monthly consumption of over 1000.

[trace](../traces/bird-haiku-1505.json)

## r-06 · `bird-haiku-738` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** Which superheroes have a durability attribute value of less than 50?

**External knowledge:** durability of less than 50 refers to attribute_name = 'Durability' AND attribute_value < 50

**Final SQL:**

```sql
SELECT DISTINCT s.superhero_name
FROM superhero s
JOIN hero_attribute ha ON ha.hero_id = s.id
JOIN attribute a ON a.id = ha.attribute_id
WHERE a.attribute_name = 'Durability' AND ha.attribute_value < 50
```

**Answer:** There are 368 superheroes with a durability attribute below 50. The first 20 include 3-D Man, A-Bomb, Abe Sapien, Adam Monroe, Adam Strange, Agent 13, Agent Bob, Agent Zero, Alex Mercer, Alex Woolsly, Alfred Pennyworth, Alien, Ammo, Ando Masahashi, Angel Dust, Angel Salvadore, Ant-Man, Ant-Man II, Apocalypse, and Aquababy.

[trace](../traces/bird-haiku-738.json)

## r-07 · `bird-haiku-744` · rule: `answer_misread`

Tags: `answer_misread`. EX ok; judge incorrect; fixes 0.

**Question:** Between DC and Marvel Comics, which publisher has published more superheroes? Find the difference in the number of superheroes they have published.

**External knowledge:** DC refers to publisher_name = 'DC Comics'; Marvel Comics refers to publisher_name = 'Marvel Comics'; calculation = SUBTRACT(SUM(publisher_name = 'Marvel Comics'), SUM(publisher_name = 'DC Comics'))

**Final SQL:**

```sql
SELECT SUM(CASE WHEN p.publisher_name = 'Marvel Comics' THEN 1 ELSE 0 END)
     - SUM(CASE WHEN p.publisher_name = 'DC Comics' THEN 1 ELSE 0 END) AS difference
FROM superhero s
JOIN publisher p ON s.publisher_id = p.id
```

**Answer:** DC Comics has published 163 more superheroes than Marvel Comics.  Wait, I need to be careful here: the result gives only the difference, not which publisher is ahead. So I can't say for certain which one has published more based on this result alone.  The difference between DC and Marvel's superhero counts is 163.

[trace](../traces/bird-haiku-744.json)

## r-08 · `bird-haiku-1225` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_query_logic`. EX wrong; judge correct; fixes 0; columns 2 vs gold 2, rows 2 vs gold 10.

**Question:** List and group all patients by sex for total bilirubin (T-BIL) level not within the normal range.

**External knowledge:** List refers to GROUP_CONCAT(DISTINCT ID); total bilirubin (T-BIL) not within normal range refers to T-BIL > = 2.0

**Final SQL:**

```sql
SELECT p.SEX, GROUP_CONCAT(DISTINCT p.ID) AS ID_List
FROM Patient p
JOIN Laboratory l ON p.ID = l.ID
WHERE l.`T-BIL` >= 2.0
GROUP BY p.SEX
```

**Answer:** Patients with abnormal total bilirubin (T-BIL) levels are grouped by sex: females are IDs 250391, 1180510, 2308236, 2931207, and 4862013, while males are IDs 43003, 444499, 1137040, 2307640, and 4618443.

[trace](../traces/bird-haiku-1225.json)

## r-09 · `bird-haiku-195` · rule: `answer_misread`

Tags: `answer_misread`. EX ok; judge incorrect; fixes 0.

**Question:** What is the most common bond type?

**External knowledge:** most common bond type refers MAX(COUNT(bond_type))

**Final SQL:**

```sql
SELECT bond_type
FROM bond
WHERE bond_type IS NOT NULL
GROUP BY bond_type
ORDER BY COUNT(bond_type) DESC
LIMIT 1
```

**Answer:** The query result doesn't identify a most common bond type; it returned a single row with a blank value ("-"), so the data needed to answer isn't there.

[trace](../traces/bird-haiku-195.json)

## r-10 · `bird-haiku-728` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 2 vs gold 3, rows 19 vs gold 19.

**Question:** Rank superheroes from Marvel Comics by their eye color popularity, starting with the most common color.

**External knowledge:** the superheroes from Marvel Comics refers to publisher_name = 'Marvel Comics'; most common color refers to COUNT(superhero.id) DESC;

**Final SQL:**

```sql
SELECT c.colour, COUNT(s.id) AS hero_count
FROM superhero s
JOIN publisher p ON s.publisher_id = p.id
JOIN colour c ON s.eye_colour_id = c.id
WHERE p.publisher_name = 'Marvel Comics'
GROUP BY c.colour
ORDER BY hero_count DESC
```

**Answer:** Blue is the most common eye color among Marvel superheroes (126), followed by brown (89), no colour (47), green (40), and red (31). The rest fall in descending order from yellow (13) down to amber (1), with several mixed colors at the bottom.

[trace](../traces/bird-haiku-728.json)

## r-11 · `bird-haiku-1155` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** List the patient ID, sex and birthday of patient with LDH beyond normal range.

**External knowledge:** LDH beyond normal range refers to LDH > '500';

**Final SQL:**

```sql
SELECT DISTINCT P.ID, P.SEX, P.Birthday
FROM Patient P
JOIN Laboratory L ON P.ID = L.ID
WHERE L.LDH > 500
```

**Answer:** The query returned 103 patients with abnormal LDH levels. The first 20 are listed above, showing their IDs, sex, and birthdays, with IDs such as 27654 (F, 1936-03-25) and 57266 (M, 1923-07-25).

[trace](../traces/bird-haiku-1155.json)

## r-12 · `bird-haiku-944` · rule: `loop_or_extra_steps`

Tags: `loop_or_extra_steps`, `wrong_query_logic`. EX wrong; judge incorrect; fixes 2; columns 1 vs gold 1, rows 0 vs gold 1.

**Question:** How much faster in percentage is the champion than the driver who finished the race last in the 2008 Australian Grand Prix?

**External knowledge:** how much faster in percentage = divide(subtract(incremental time, champion time), last_driver time) * 100; last driver finished time = incremental time + champion time; only champion's finished time is represented by 'HH:MM:SS.mmm'; finished the game refers to time is not null

**Final SQL:**

```sql
WITH r AS (
    SELECT raceId FROM races WHERE year = 2008 AND name = 'Australian Grand Prix'
),
c AS (
    SELECT res.milliseconds AS ms
    FROM results res
    JOIN r ON res.raceId = r.raceId
    WHERE res.positionOrder = 1
),
l AS (
    SELECT res.milliseconds AS ms
    FROM results res
    JOIN r ON res.raceId = r.raceId
    WHERE res.milliseconds IS NOT NULL
    ORDER BY res.positionOrder DESC
    LIMIT 1
)
SELECT (CAST(l.ms - c.ms AS REAL) / l.ms) * 100 AS percentage_faster
FROM c, l
```

Earlier attempts: 2.

**Answer:** The query returned no results, so I can't determine how much faster the champion was than the last-place finisher at the 2008 Australian Grand Prix.

[trace](../traces/bird-haiku-944.json)

## r-13 · `bird-haiku-1524` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** What's the nationality of the customer who spent 548.4 in 2012/8/24?

**External knowledge:** '2012/8/24' can be represented by '2012-08-24'

**Final SQL:**

```sql
SELECT c.Currency
FROM transactions_1k t
JOIN customers c ON c.CustomerID = t.CustomerID
WHERE t.Date = '2012-08-24'
  AND t.Price = 548.4
```

**Answer:** The data only shows the currency, CZK (Czech koruna), which suggests the customer is Czech, but it doesn't state their nationality directly.

[trace](../traces/bird-haiku-1524.json)

## r-14 · `bird-haiku-1375` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_output_shape`. EX wrong; judge correct; fixes 1; columns 1 vs gold 2, rows 5 vs gold 5.

**Question:** List all the members of the "School of Applied Sciences, Technology and Education" department.

**External knowledge:** list all members means to list all the full name; full name refers to first_name, last_name;

**Final SQL:**

```sql
SELECT m.first_name || ' ' || m.last_name AS full_name
FROM member m
JOIN major j ON m.link_to_major = j.major_id
WHERE j.department LIKE '%School of Applied Sciences, Technology and Education%'
   OR j.college LIKE '%School of Applied Sciences, Technology and Education%'
```

Earlier attempts: 1.

**Answer:** The School of Applied Sciences, Technology and Education has five members: Angela Sanders, Dean O'Reilly, Edwardo Ing, Mccauley Nash, and Brent Thomason.

[trace](../traces/bird-haiku-1375.json)

## r-15 · `bird-haiku-850` · rule: `answer_misread`

Tags: `answer_misread`. EX ok; judge incorrect; fixes 0.

**Question:** Please give the name of the race held on the circuits in Germany.

**External knowledge:** Germany is a name of country;

**Final SQL:**

```sql
SELECT DISTINCT r.name
FROM races r
JOIN circuits c ON r.circuitId = c.circuitId
WHERE c.country = 'Germany'
```

**Answer:** The races held on circuits in Germany are the German Grand Prix and the European Grand Prix. The Luxembourg Grand Prix does not fit that description.

[trace](../traces/bird-haiku-850.json)

## r-16 · `bird-haiku-36` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_output_shape`. EX wrong; judge correct; fixes 0; columns 2 vs gold 6, rows 1 vs gold 1.

**Question:** Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names.

**External knowledge:** full name means first name, last name; There are at most 3 administrators for each school; SAT Scores are greater or equal to 1500 refers to NumGE1500

**Final SQL:**

```sql
SELECT s.AdmFName1 AS first_name, s.AdmLName1 AS last_name
FROM satscores AS sat
JOIN schools AS s ON s.CDSCode = sat.cds
ORDER BY sat.NumGE1500 DESC
LIMIT 1
```

**Answer:** The school with the highest number of students scoring 1500 or more on the SAT is under the administration of Michelle King.

[trace](../traces/bird-haiku-36.json)

## r-17 · `bird-haiku-1168` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 2 vs gold 3, rows 10 vs gold 1.

**Question:** The oldest SJS patient's medical laboratory work was completed on what date, and what age was the patient when they initially arrived at the hospital?

**External knowledge:** The larger the birthday value, the younger the person is, and vice versa; 'SJS' refers to diagnosis; (SUBTRACT(year(`First Date`)), year(Birthday)); age of the patients when they initially arrived at the hospital refers to year(Birthday)

**Final SQL:**

```sql
SELECT l.Date,
       CAST(strftime('%Y', p.`First Date`) AS INTEGER) - CAST(strftime('%Y', p.Birthday) AS INTEGER) AS Age
FROM Patient p
JOIN Laboratory l ON l.ID = p.ID
WHERE p.ID = (
    SELECT ID
    FROM Patient
    WHERE Diagnosis = 'SJS' AND Birthday IS NOT NULL
    ORDER BY Birthday ASC
    LIMIT 1
)
```

**Answer:** The oldest patient was 69 years old, and their lab work was completed on dates ranging from July 31, 1981, to January 7, 1986. The earliest recorded date was July 31, 1981.

[trace](../traces/bird-haiku-1168.json)

## r-18 · `bird-haiku-83` · rule: `loop_or_extra_steps`

Tags: `loop_or_extra_steps`, `wrong_query_logic`. EX wrong; judge incorrect; fixes 2; columns 2 vs gold 2, rows 20 vs gold 1.

**Question:** Of the schools that offers a magnet program serving a grade span of Kindergarten to 8th grade, how many offers Multiple Provision Types? List the number of cities that offers a Kindergarten to 8th grade span and indicate how many schools are there serving such grade span for each city.

**External knowledge:** Kindergarten to 8th grade refers to K-8; 'Offers a magnet program' means Magnet = 1; Multiple Provision Types refers to `NSLP Provision Status` = 'Multiple Provision Types'

**Final SQL:**

```sql
SELECT City, COUNT(*) AS school_count
FROM schools
WHERE GSoffered = 'K-8'
  AND Magnet = 1
  AND City IS NOT NULL
GROUP BY City
ORDER BY City
```

Earlier attempts: 2.

**Answer:** There are 20 cities with schools that offer a magnet program serving grades K-8 and multiple provision types. Stockton has the most, with 9 schools, followed by Ontario, San Jose, and Vallejo with 3 each, and Fresno, Los Angeles, and San Diego with 2 each. The remaining 11 cities have 1 school each.

[trace](../traces/bird-haiku-83.json)

## r-19 · `bird-haiku-586` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 2, rows 2 vs gold 2.

**Question:** Which user added a bounty amount of 50 to the post title mentioning variance?

**External knowledge:** bounty amount of 50 refers to BountyAmount = 50; user refers to DisplayName; title mentioning variance refers to Title include 'variance'

**Final SQL:**

```sql
SELECT DISTINCT u.DisplayName
FROM votes v
JOIN posts p ON v.PostId = p.Id
JOIN users u ON v.UserId = u.Id
WHERE v.BountyAmount = 50
  AND p.Title LIKE '%variance%'
```

**Answer:** Two users match: chl and shabbychef. The result doesn't indicate which one added the bounty of 50, so I can't name a single user from it.

[trace](../traces/bird-haiku-586.json)

## r-20 · `bird-haiku-125` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 2 vs gold 1, rows 30 vs gold 45.

**Question:** For loans contracts which are still running where client are in debt, list the district of the and the state the percentage unemployment rate increment from year 1995 to 1996.

**External knowledge:** Unemployment increment rate in percentage = [(unemployment rate 2016 - unemployment rate 2015) / unemployment rate 2015] * 100; unemployment rate 2015 appears in the A12; unemployment rate 2016 appears in the A13; Loan contracts which are still running where client are in debt can be presented as status = 'D'

**Final SQL:**

```sql
SELECT DISTINCT d.district_id,
       (d.A13 - d.A12) * 100.0 / d.A12 AS unemployment_increment_rate
FROM loan l
JOIN account a ON l.account_id = a.account_id
JOIN district d ON a.district_id = d.district_id
WHERE l.status = 'D'
```

**Answer:** The query returned 30 districts with their 1995-to-1996 unemployment increase rates. District 1 had the largest increase at 115.0%, and district 71 was the only one shown with a decrease, about -5.6%. District 69 has no value (NULL), and the result doesn't include state names.

[trace](../traces/bird-haiku-125.json)

## r-21 · `bird-haiku-758` · rule: `answer_misread`

Tags: `answer_misread`. EX ok; judge incorrect; fixes 0.

**Question:** Provide the hair colour of the human superhero who is 185 cm tall.

**External knowledge:** 185 cm tall refers to height_cm = 185; human superhero refers to race = 'human'; hair colour refers to colour where hair_colour_id = colour.id;

**Final SQL:**

```sql
SELECT c.colour
FROM superhero s
JOIN race r ON s.race_id = r.id
JOIN colour c ON s.hair_colour_id = c.id
WHERE s.height_cm = 185 AND r.race = 'Human'
```

**Answer:** The results list 11 hair colours, including Blond, Brown, Black, Brown/White, and No Colour, so they don't identify a single colour for the 185 cm human superhero.

[trace](../traces/bird-haiku-758.json)

## r-22 · `bird-haiku-407` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** Lists all types of cards in German.

**External knowledge:** German refer to language; all types refer to the subtypes, supertypes; subtypes is not null AND supertypes is not null

**Final SQL:**

```sql
SELECT DISTINCT c.subtypes, c.supertypes
FROM cards AS c
INNER JOIN foreign_data AS f ON f.uuid = c.uuid
WHERE f.language = 'German'
  AND c.subtypes IS NOT NULL
  AND c.supertypes IS NOT NULL
```

**Answer:** The query results don't include German names. They show English card supertypes (such as "Legendary") and subtypes (such as Human, Wizard, Angel, and Zombie), across 408 rows.

[trace](../traces/bird-haiku-407.json)

## r-23 · `bird-haiku-11` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** Please list the codes of the schools with a total enrollment of over 500.

**External knowledge:** Total enrollment can be represented by `Enrollment (K-12)` + `Enrollment (Ages 5-17)`

**Final SQL:**

```sql
SELECT CDSCode
FROM frpm
WHERE `Enrollment (K-12)` + `Enrollment (Ages 5-17)` > 500
```

**Answer:** There are more than 1,000 schools with a total enrollment over 500. The first 20 codes are 01100170109835, 01100170112607, 01100170124172, 01100170125567, 01100176001788, 01100176002000, 01316170131763, 01611190111765, 01611190119222, 01611190122085, 01611190126656, 01611190130229, 01611190130609, 01611190132878, 01611196090005, 01611196090013, 01611196090021, 01611196090039, 01611196090047, and 01611196090054.

[trace](../traces/bird-haiku-11.json)

## r-24 · `bird-haiku-1011` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 1; columns 1 vs gold 3, rows 20 vs gold 20.

**Question:** Which top 20 driver created the shortest lap time ever record in a Formula_1 race? Please give them full names.

**External knowledge:** shortest lap time refers to MIN(time); the time format for the shortest lap time is 'MM:SS.mmm' or 'M:SS.mmm'; full name of the driver refers to forename, surname

**Final SQL:**

```sql
SELECT d.forename || ' ' || d.surname AS full_name
FROM lapTimes lt
JOIN drivers d ON d.driverId = lt.driverId
WHERE lt.milliseconds IS NOT NULL
GROUP BY d.driverId, d.forename, d.surname
ORDER BY MIN(lt.milliseconds) ASC
LIMIT 20
```

Earlier attempts: 1.

**Answer:** The query result lists 20 drivers, but it doesn't indicate which of them set the shortest lap time ever, so I can't single out one name. It only shows the top 20 drivers by that measure, with Lewis Hamilton listed first.

[trace](../traces/bird-haiku-1011.json)

## r-25 · `bird-haiku-149` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 2 vs gold 1, rows 0 vs gold 461.

**Question:** Please list the account types that are not eligible for loans, and the average income of residents in the district where the account is located exceeds $8000 but is no more than $9000.

**External knowledge:** A11 represents the average salary; Salary and income share the similar meanings; when the account type = 'OWNER', it's eligible for loans

**Final SQL:**

```sql
SELECT d.type, AVG(di.A11) AS average_income
FROM disp d
JOIN account a ON d.account_id = a.account_id
JOIN district di ON a.district_id = di.district_id
WHERE d.type <> 'OWNER'
GROUP BY d.type
HAVING AVG(di.A11) > 8000 AND AVG(di.A11) <= 9000
```

**Answer:** No account types meet both conditions: the query returned no rows, so there are no non-loan-eligible account types whose district average income falls between $8,000 and $9,000.

[trace](../traces/bird-haiku-149.json)

## r-26 · `bird-haiku-528` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** List the names of all the cards in the set Hour of Devastation and find the formats in which these cards are legal.

**External knowledge:** the set Hour of Devastation refers to set.name = 'Hour of Devastation'; names of all the cards in the set refers to cards.name; legal cards refers to status = 'Legal'; the formats refers to format

**Final SQL:**

```sql
SELECT DISTINCT c.name, l.format
FROM cards c
JOIN sets s ON c.setCode = s.code
JOIN legalities l ON l.uuid = c.uuid
WHERE s.name = 'Hour of Devastation'
  AND l.status = 'Legal'
```

**Answer:** The Hour of Devastation cards are legal in many formats, including commander, duel, legacy, modern, pauper, penny, pioneer, and vintage. Examples include Act of Heroism, Adorned Pouncer, and Angel of Condemnation, each legal in commander, duel, legacy, modern, and penny (with more than 1,000 name-format pairs in total).

[trace](../traces/bird-haiku-528.json)

## r-27 · `bird-haiku-1092` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 1, rows 1 vs gold 4.

**Question:** Give the name of the league had the most matches in the 2008/2009 season?

**External knowledge:** league that had the most matches in the 2008/2009 season refers to MAX(league_name WHERE season = '2008/2009');

**Final SQL:**

```sql
SELECT L.name
FROM Match M
JOIN League L ON M.league_id = L.id
WHERE M.season = '2008/2009'
GROUP BY L.id, L.name
ORDER BY COUNT(M.id) DESC
LIMIT 1
```

**Answer:** The England Premier League had the most matches in the 2008/2009 season.

[trace](../traces/bird-haiku-1092.json)

## r-28 · `bird-haiku-773` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** Which superhero has the same eyes, hair and skin colour? Indicate the publisher of the superhero.

**External knowledge:** which superhero refers to superhero_name; the same eyes, hair and skin colour refers to hair_colour_id = skin_colour_id AND hair_colour_id = eye_colour_id; publisher refers to publisher_name;

**Final SQL:**

```sql
SELECT s.superhero_name, p.publisher_name
FROM superhero s
JOIN publisher p ON s.publisher_id = p.id
WHERE s.hair_colour_id = s.skin_colour_id
  AND s.hair_colour_id = s.eye_colour_id
```

**Answer:** The query result doesn't include eye, hair, or skin colour, so I can't tell which superhero matches. It lists only superhero names and their publishers.

[trace](../traces/bird-haiku-773.json)

## r-29 · `bird-haiku-1149` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_query_logic`. EX wrong; judge correct; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** Are there more in-patient or outpatient who were male? What is the deviation in percentage?

**External knowledge:** male refers to SEX = 'M'; in-patient refers to Admission = '+'; outpatient refers to Admission = '-'; percentage = DIVIDE(COUNT(ID) where SEX = 'M' and Admission = '+', COUNT(ID) where SEX  = 'M' and Admission = '-')

**Final SQL:**

```sql
SELECT (SUM(CASE WHEN Admission = '+' THEN 1 ELSE 0 END) - SUM(CASE WHEN Admission = '-' THEN 1 ELSE 0 END)) * 100.0
       / SUM(CASE WHEN Admission = '-' THEN 1 ELSE 0 END) AS deviation_percentage
FROM Patient
WHERE SEX = 'M'
```

**Answer:** Outpatients had more male patients than in-patients. The male in-patient count was about 16.82% lower than the outpatient count.

[trace](../traces/bird-haiku-1149.json)

## r-30 · `bird-haiku-792` · rule: `answer_misread`

Tags: `answer_misread`. EX ok; judge incorrect; fixes 0.

**Question:** What is Abomination's superpower?

**External knowledge:** Abomination refers to superhero_name = 'Abomination'; superpower refers to power_name;

**Final SQL:**

```sql
SELECT sp.power_name
FROM superhero s
JOIN hero_power hp ON hp.hero_id = s.id
JOIN superpower sp ON sp.id = hp.power_id
WHERE s.superhero_name = 'Abomination'
```

**Answer:** The query result lists eight powers but doesn't indicate which ones belong to Abomination, so I can't say his specific superpower from this data. The list includes Super Strength, Accelerated Healing, and Stamina, but it doesn't link any of them to him.

[trace](../traces/bird-haiku-792.json)

## r-31 · `bird-haiku-1001` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_output_shape`. EX wrong; judge correct; fixes 0; columns 1 vs gold 2, rows 1 vs gold 1.

**Question:** What is full name of the racer who ranked 1st in the 3rd qualifying race held in the Marina Bay Street Circuit in 2008?

**External knowledge:** Ranked 1st in the 3rd qualifying race refer to MIN(q3); 2008 is the year of race; full name of racer = forename, surname

**Final SQL:**

```sql
SELECT d.forename || ' ' || d.surname AS fullname
FROM qualifying q
JOIN races r ON q.raceId = r.raceId
JOIN circuits c ON r.circuitId = c.circuitId
JOIN drivers d ON q.driverId = d.driverId
WHERE c.name = 'Marina Bay Street Circuit'
  AND r.year = 2008
  AND q.q3 IS NOT NULL
  AND q.q3 <> ''
ORDER BY q.q3 ASC
LIMIT 1
```

**Answer:** Felipe Massa ranked 1st in the third qualifying race at the Marina Bay Street Circuit in 2008.

[trace](../traces/bird-haiku-1001.json)

## r-32 · `bird-haiku-248` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_output_shape`. EX wrong; judge correct; fixes 0; columns 1 vs gold 2, rows 4 vs gold 4.

**Question:** What are the atoms of the triple bond with the molecule "TR041"?

**External knowledge:** TR041 is the molecule id; triple bond refers to bond_type = '#';

**Final SQL:**

```sql
SELECT c.atom_id
FROM connected c
JOIN bond b ON c.bond_id = b.bond_id
WHERE b.molecule_id = 'TR041' AND b.bond_type = '#'
UNION
SELECT c.atom_id2
FROM connected c
JOIN bond b ON c.bond_id = b.bond_id
WHERE b.molecule_id = 'TR041' AND b.bond_type = '#'
```

**Answer:** The triple bond in molecule TR041 involves atoms TR041_7, TR041_8, TR041_9, and TR041_10.

[trace](../traces/bird-haiku-248.json)

## r-33 · `bird-haiku-640` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** Calculate the difference in view count from post posted by Mornington and view count from posts posted by Amos.

**External knowledge:** calculation = SUBTRACT(SUM(ViewCount where DisplayName = 'Mornington'), SUM(ViewCount where DisplayName = 'Amos'));

**Final SQL:**

```sql
SELECT SUM(CASE WHEN u.DisplayName = 'Mornington' THEN p.ViewCount ELSE 0 END)
     - SUM(CASE WHEN u.DisplayName = 'Amos' THEN p.ViewCount ELSE 0 END) AS difference
FROM posts p
JOIN users u ON p.OwnerUserId = u.Id
WHERE u.DisplayName IN ('Mornington', 'Amos')
```

**Answer:** The difference in view count between Mornington's posts and Amos's posts is -497, meaning Amos's posts have 497 more views than Mornington's.

[trace](../traces/bird-haiku-640.json)

## r-34 · `bird-haiku-1506` · rule: `answer_on_large_result`

Tags: `answer_on_large_result`. EX ok; judge incorrect; fixes 0.

**Question:** Please list the product descriptions of the transactions taken place in the gas stations in the Czech Republic.

**External knowledge:** Czech Republic can be represented as the Country value in the gasstations table is 'CZE'; 

**Final SQL:**

```sql
SELECT DISTINCT p.Description
FROM transactions_1k t
JOIN gasstations g ON t.GasStationID = g.GasStationID
JOIN products p ON t.ProductID = p.ProductID
WHERE g.Country = 'CZE'
```

**Answer:** The gas station transactions in the Czech Republic include products described as: Nafta, Provoz.nápl., Natural, Nemrz.kapal., Oleje,tuky, Diesel +, LPG, Mytí vozidel, Autokosmet., Potraviny, Nafta Plus, Natural Spec, Zpr.nakupu, Autoprísluš., Natural +, Dalnic.popl., Knihy,nov., Zboží nesp., Obcerstvení, and Additivum.

[trace](../traces/bird-haiku-1506.json)

## r-35 · `bird-haiku-215` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 2, rows 1 vs gold 1.

**Question:** How many atoms with iodine and with sulfur type elements are there in single bond molecules?

**External knowledge:** with iodine element refer to element = 'i'; with sulfur element refers to element = 's'; single type bond refers to bond_type = '-'; Should consider the distinct atoms when counting;

**Final SQL:**

```sql
SELECT COUNT(DISTINCT a.atom_id) AS atom_count
FROM atom AS a
JOIN molecule AS m ON a.molecule_id = m.molecule_id
JOIN bond AS b ON b.molecule_id = m.molecule_id
WHERE a.element IN ('i', 's')
  AND b.bond_type = '-'
```

**Answer:** There are 97 atoms in total that are iodine or sulfur in single-bond molecules.

[trace](../traces/bird-haiku-215.json)

## r-36 · `bird-haiku-896` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** Calculate the percentage whereby Hamilton was not at the 1st track of the the f1 circuit since 2010.

**External knowledge:** percentage = DIVIDE(COUNT(raceId) where surname = 'Hamilton' and position>1), (COUNT(raceId) where surname = 'Hamilton'); since 2010 refers to year >= 2010

**Final SQL:**

```sql
SELECT CAST(SUM(CASE WHEN r.position > 1 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) AS percentage
FROM results r
JOIN drivers d ON r.driverId = d.driverId
JOIN races ra ON r.raceId = ra.raceId
WHERE d.surname = 'Hamilton' AND ra.year >= 2010
```

**Answer:** Hamilton did not finish first at the F1 circuit in roughly 54.23% of the races since 2010.

[trace](../traces/bird-haiku-896.json)

## r-37 · `bird-haiku-955` · rule: `wrong_query_logic`

Tags: `wrong_query_logic`. EX wrong; judge incorrect; fixes 0; columns 2 vs gold 2, rows 25 vs gold 25.

**Question:** What is the average time in seconds of champion for each year, before year 1975?

**External knowledge:** only champion's finished time is represented by 'HH:MM:SS.mmm'; finished the game refers to time is not null; before year 1975 refers to year < 1975;

**Final SQL:**

```sql
SELECT
    r.year AS year,
    AVG(
        CAST(substr(res.time, 1, instr(res.time, ':') - 1) AS REAL) * 3600
        + CAST(substr(res.time, instr(res.time, ':') + 1,
                      instr(substr(res.time, instr(res.time, ':') + 1), ':') - 1) AS REAL) * 60
        + CAST(substr(res.time, instr(res.time, ':') + instr(substr(res.time, instr(res.time, ':') + 1), ':') + 1) AS REAL)
    ) AS avg_time_seconds
FROM results res
JOIN races r ON res.raceId = r.raceId
WHERE res.position = 1
  AND res.time IS NOT NULL
  AND res.time LIKE '%:%:%'
  AND r.year < 1975
GROUP BY r.year
ORDER BY r.year
```

**Answer:** The query returned average champion times in seconds for each year from 1950 onward, falling from roughly 9,700 to 11,300 seconds in the early 1950s and then declining to about 7,000 seconds by 1969. The output shows 25 rows, but only the first 20 are visible, so the values for 1970 to 1974 are not shown.

[trace](../traces/bird-haiku-955.json)

## r-38 · `bird-haiku-226` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_query_logic`. EX wrong; judge correct; fixes 0; columns 1 vs gold 1, rows 1 vs gold 1.

**Question:** What is the percentage of double bonds in the molecule TR008? Please provide your answer as a percentage with five decimal places.

**External knowledge:** double bond refers to bond_type = '='; TR008 is the molecule id; percentage = DIVIDE(SUM(bond_type = '='), COUNT(bond_id)) as percent where molecule_id = 'TR008'

**Final SQL:**

```sql
SELECT printf('%.5f', 100.0 * SUM(CASE WHEN bond_type = '=' THEN 1 ELSE 0 END) / COUNT(bond_id)) AS percentage
FROM bond
WHERE molecule_id = 'TR008'
```

**Answer:** The percentage of double bonds in molecule TR008 is 3.84615%.

[trace](../traces/bird-haiku-226.json)

## r-39 · `bird-haiku-1427` · rule: `answer_ok_ex_wrong`

Tags: `answer_ok_ex_wrong`, `wrong_output_shape`. EX wrong; judge correct; fixes 1; columns 1 vs gold 2, rows 4 vs gold 4.

**Question:** What are the budget category of the events located at MU 215 and a guest speaker type with a 0 budget spent?

**External knowledge:** budget category refers to category; events located at refers to location; type = 'Guest Speaker'; 0 budget spent refers to spent = 0; 

**Final SQL:**

```sql
SELECT DISTINCT b.category
FROM budget b
JOIN event e ON b.link_to_event = e.event_id
WHERE e.location = 'MU 215'
  AND e.type = 'Guest Speaker'
  AND b.spent = 0
```

Earlier attempts: 1.

**Answer:** The budget categories for the events at MU 215 with a guest speaker type and a zero spent amount are Speaker Gifts, Food, Parking, and Advertisement.

[trace](../traces/bird-haiku-1427.json)

## r-40 · `bird-haiku-726` · rule: `wrong_output_shape`

Tags: `wrong_output_shape`. EX wrong; judge incorrect; fixes 0; columns 1 vs gold 3, rows 387 vs gold 387.

**Question:** Rank heroes published by Marvel Comics by their height in descending order.

**External knowledge:** name refers to superhero_name; the tallest hero refers to MAX(height_cm); published by Marvel Comics refers to publisher_name = 'Marvel Comics'

**Final SQL:**

```sql
SELECT s.superhero_name
FROM superhero s
JOIN publisher p ON s.publisher_id = p.id
WHERE p.publisher_name = 'Marvel Comics'
ORDER BY s.height_cm DESC
```

**Answer:** The Marvel heroes ranked by height in descending order begin with Surtur, Ymir, Bloodwraith, Utgard-Loki, and Fin Fang Foom, followed by Galactus, Groot, and MODOK. The list contains 387 entries in total.

[trace](../traces/bird-haiku-726.json)
