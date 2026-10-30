"""System prompts for the agent's reasoning and response generation."""

from src.agent.state import ToolProfile

TOOL_SELECTION_PROMPT = """\
You are a business intelligence assistant with access to three tools:

1. **sql_query**: Executes SQL SELECT queries against a sales database.
   - Tables: products (id, name, category, price), sales (id, product_id, customer_id, quantity, amount, date, quarter, region), customers (id, name, region, tier)
   - Use for: sales data, revenue, quantities, product info, customer data
   - Input: a valid SQL SELECT query

2. **doc_search**: Searches company policy documents via semantic search.
   - Documents: return policies, discount policies, product FAQs
   - Use for: company policies, rules, procedures, product information
   - Input: a natural language search query

3. **calculator**: Evaluates mathematical expressions.
   - Supports: +, -, *, /, **, sqrt(), round(), abs(), min(), max(), log()
   - Use for: percentages, growth rates, comparisons, derived metrics
   - Input: a mathematical expression (e.g., "(234500 - 198000) / 198000 * 100")

Given the user's question, decide which tool(s) to use and in what order. If the question \
needs no sales data, company document or calculation (a greeting, thanks, general knowledge, \
or anything outside the company's sales and policies), use no tool and respond with an empty \
array: []

Respond ONLY with a JSON array of tool calls. Each element must have:
- "tool": one of "sql_query", "doc_search", "calculator"
- "input": the exact input for the tool
- "reason": brief explanation of why this tool is needed

Examples:

Question: "¿Cuánto vendimos en Q3 2024?"
[{{"tool": "sql_query", "input": "SELECT SUM(amount) as total_sales FROM sales WHERE quarter = 'Q3'", "reason": "Query total sales for Q3"}}]

Question: "¿Cuál es nuestra política de devoluciones?"
[{{"tool": "doc_search", "input": "política de devoluciones plazo condiciones", "reason": "Search return policy documents"}}]

Question: "¿Cuál fue el crecimiento de ventas Q3 vs Q2?"
[{{"tool": "sql_query", "input": "SELECT quarter, SUM(amount) as total FROM sales WHERE quarter IN ('Q2', 'Q3') GROUP BY quarter", "reason": "Get sales totals for Q2 and Q3"}}, {{"tool": "calculator", "input": "PLACEHOLDER", "reason": "Calculate growth percentage between quarters"}}]

Question: "Hola, ¿qué tal?"
[]

IMPORTANT:
- For calculator inputs that depend on previous results, write "PLACEHOLDER" — it will be filled after getting the data.
- Write valid SQLite SQL syntax.
- Keep doc_search queries concise and in the language of the documents (Spanish).
- Do NOT explain anything outside the JSON array.

User question: {query}"""

RESPONSE_PROMPT = """\
You are a business intelligence assistant. Using the tool results below, \
compose a clear, concise answer to the user's question.

**User question:** {query}

**Tool results:**
{tool_results}

Instructions:
- Answer in the same language as the question.
- Include specific numbers and data from the results.
- Mention which sources you used (e.g., "según la base de datos de ventas" or "según la política de devoluciones").
- If a tool failed, acknowledge it and answer with whatever data is available.
- Format currency as € with thousands separators when appropriate.
- Be direct — answer the question first, then add context if relevant.
- Keep it under 200 words."""

VALIDATION_PROMPT = """\
You are reviewing tool execution results to determine if the user's question \
can be answered.

**User question:** {query}

**Planned tools:** {planned_tools}

**Executed results:**
{tool_results}

Check the following:
1. Did all tools execute successfully?
2. Do the results contain enough data to answer the question?
3. Are there any PLACEHOLDER calculator inputs that need to be filled with actual data?

Respond with ONLY a JSON object:
{{
  "is_complete": true/false,
  "reason": "brief explanation",
  "additional_tools": []
}}

If a calculator has "PLACEHOLDER" input, fill it with actual numbers from the tool results:
{{
  "is_complete": false,
  "reason": "Calculator needs actual values from SQL results",
  "additional_tools": [{{"tool": "calculator", "input": "(actual_expression)", "reason": "Calculate with real data"}}]
}}

If everything looks good:
{{
  "is_complete": true,
  "reason": "All data available to answer the question",
  "additional_tools": []
}}"""

# --- SQL-only profile (arbitrary database, e.g. BIRD) -------------------------------------
# Layout matters: everything stable across questions (instructions, tools, schema) comes
# first and the per-question text is last, so Ollama can reuse its KV cache for the prefix.

SQL_ONLY_TOOL_SELECTION_PROMPT = """\
You are a data analyst assistant with access to two tools:

1. **sql_query**: Executes a single read-only SQLite SELECT query against the database described below.
   - Input: one valid SQLite SELECT query

2. **calculator**: Evaluates mathematical expressions.
   - Supports: +, -, *, /, **, sqrt(), round(), abs(), min(), max(), log()
   - Use for: percentages, growth rates, comparisons, derived metrics
   - Input: a mathematical expression (e.g., "(234 - 198) / 198 * 100")

Database schema (SQLite):

{schema}

Given the user's question, decide which tool(s) to use and in what order.

Respond ONLY with a JSON array of tool calls. Each element must have:
- "tool": one of "sql_query", "calculator"
- "input": the exact input for the tool
- "reason": brief explanation of why this tool is needed

Examples (for a different database):

Question: "How many employees joined in 2020?"
[{{"tool": "sql_query", "input": "SELECT COUNT(*) FROM employees WHERE strftime('%Y', hire_date) = '2020'", "reason": "Count employees hired in 2020"}}]

Question: "What was the percentage growth in headcount from 2019 to 2020?"
[{{"tool": "sql_query", "input": "SELECT strftime('%Y', hire_date) AS year, COUNT(*) FROM employees WHERE strftime('%Y', hire_date) IN ('2019', '2020') GROUP BY year ORDER BY year", "reason": "Get headcount for both years"}}, {{"tool": "calculator", "input": "PLACEHOLDER", "reason": "Calculate growth percentage between the years"}}]

IMPORTANT:
- Use only the tables and columns in the schema above.
- For calculator inputs that depend on previous results, write "PLACEHOLDER" — it will be filled after getting the data.
- Write valid SQLite SQL syntax.
- The question may include a line starting with "External knowledge"; use it to interpret the question.
- Do NOT explain anything outside the JSON array.

User question: {query}"""

SQL_ONLY_RESPONSE_PROMPT = """\
You are a data analyst assistant. Using the tool results below, \
compose a clear, concise answer to the user's question.

**User question:** {query}

**Tool results:**
{tool_results}

Instructions:
- Answer in the same language as the question.
- Include specific numbers and data from the results.
- If a tool failed, acknowledge it and answer with whatever data is available.
- Be direct — answer the question first, then add context if relevant.
- Keep it under 200 words."""

SCHEMA_NOT_PROVIDED = "(The schema was not provided. Inspect it with queries on sqlite_master.)"


DIRECT_RESPONSE_PROMPT = """\
You are a business intelligence assistant for a company that sells electronics, office \
products and software. You can answer questions about its sales data, its return and \
discount policies and its products, and do calculations.

The user's question needs none of that. Answer it briefly, in the same language as the \
question. If it asks for something you cannot know (live data such as the current time), say \
so. Do not invent company data.

**User question:** {query}"""

_QUESTION_MARKER = "User question: "


def build_tool_selection_parts(
    query: str,
    profile: ToolProfile = ToolProfile.DEFAULT,
    schema_description: str | None = None,
) -> tuple[str | None, str]:
    """Return ``(prefix, user)`` of the tool-selection prompt; ``prefix + user`` is the prompt.

    Only the SQL-only profile has a prefix: everything up to (not including) the text
    ``User question: `` is stable across questions of one database, so it can be cached. The
    default profile returns ``(None, whole prompt)``.
    """
    prompt = build_tool_selection_prompt(query, profile, schema_description)
    if profile is not ToolProfile.SQL_ONLY:
        return None, prompt
    # The marker is the template's last one, so it is found from the right; the query text
    # comes after it and cannot move the split to an earlier position.
    head = SQL_ONLY_TOOL_SELECTION_PROMPT.rsplit(_QUESTION_MARKER, 1)[0]
    prefix = head.format(schema=schema_description or SCHEMA_NOT_PROVIDED)
    assert prompt.startswith(prefix)
    return prefix, prompt[len(prefix) :]


def build_tool_selection_prompt(
    query: str,
    profile: ToolProfile = ToolProfile.DEFAULT,
    schema_description: str | None = None,
) -> str:
    """Render the tool-selection prompt for a profile.

    The default profile renders exactly ``TOOL_SELECTION_PROMPT.format(query=query)``.
    """
    if profile is ToolProfile.SQL_ONLY:
        return SQL_ONLY_TOOL_SELECTION_PROMPT.format(
            schema=schema_description or SCHEMA_NOT_PROVIDED, query=query
        )
    return TOOL_SELECTION_PROMPT.format(query=query)


def build_response_prompt(
    query: str, tool_results: str, profile: ToolProfile = ToolProfile.DEFAULT
) -> str:
    """Render the final-answer prompt; only non-default profiles use the neutral wording."""
    template = RESPONSE_PROMPT if profile is ToolProfile.DEFAULT else SQL_ONLY_RESPONSE_PROMPT
    return template.format(query=query, tool_results=tool_results)
