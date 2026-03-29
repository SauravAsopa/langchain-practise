from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools (langchain @tool decorator) ---


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"   >> Executing get_product_price with product={product}")
    prices = {"laptop": 999.99, "headphones": 199.99, "keyboard": 49.99}
    return prices.get(product, 0.0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    Available discount tiers are: "bronze", "silver", "gold"."""
    print(
        f"   >> Executing apply_discount with price={price} and discount_tier={discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the json schema for each function
# This is exactly what Langchain's @tool decorater generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', or 'keyboard'",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The price to apply the discount to",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier to apply",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# --- Agent loop ---


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to the product catalog tool "
                "and the discount tool.\n\n "
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price"
                "You must call get_product_price to get the price of the product.\n"
                "2. Only call apply_discount after you have received"
                "the price of the product from get_product_price. Pass the exact price "
                "returned by get_product_price - do not pass a made-up price.\n"
                "3. Never calculate discounts yourself using math "
                " Always use the apply_discount tool to calculate the final price after discount.\n"
                "4. If the user does not specify a discount tier,ask them which tie to use - do Not assume one"
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(MAX_ITERATIONS + 1):

        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: ollma.chat() directly istead of llm_with_tools.invoke(), because we are not using Langchain's @tool decorator to wrap the functions as tools
        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tools_calls = ai_message.tool_calls

        # If no tools were called, we assume the agent has arrived at a final answer and we can stop the loop
        if not tools_calls:
            print("Agent has arrived at a final answer.")
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call (if multiple were made)
        tool_call = tools_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dictionary access (get("name")) because we are not using Langchain's @tool decorator which generates a different format for tool_calls
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(
            f"Iteration {iteration + 1}: Agent called tool '{tool_name}' with arguments {tool_args}"
        )

        # Look up the tool function based on the tool name
        tools_to_use = tools_dict.get(tool_name)
        if tools_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict")

        # Difference 7: Directly call the tool function with the arguments, instead of using llm_with_tools.invoke() because we are not using Langchain's @tool decorator which provides the .invoke() method to call tools. Here we must manually call the function with the correct arguments.
        observation = tools_to_use(**tool_args)

        print(f"Observation from tool '{tool_name}': {observation}")

        messages.append(ai_message)
        messages.append(
            {
                "role": "tool",
                "content": str(observation),
            }
        )

    print("Reached maximum iterations without arriving at a final answer.")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
