from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# --- Tools (langchain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"   >> Executing get_product_price with product={product}")
    prices = {"laptop": 999.99, "headphones": 199.99, "keyboard": 49.99}
    return prices.get(product, 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    Available discount tiers are: "bronze", "silver", "gold"."""
    print(
        f"   >> Executing apply_discount with price={price} and discount_tier={discount_tier}"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0.0)
    return round(price * (1 - discount / 100), 2)


# --- Agent loop ---


@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
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
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(MAX_ITERATIONS + 1):

        ai_message = llm_with_tools.invoke(messages)

        tools_calls = ai_message.tool_calls

        # If no tools were called, we assume the agent has arrived at a final answer and we can stop the loop
        if not tools_calls:
            print("Agent has arrived at a final answer.")
            print(f"Final Answer: {ai_message.content}")
            return ai_message.content

        # Process only the first tool call (if multiple were made)
        tool_call = tools_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(
            f"Iteration {iteration + 1}: Agent called tool '{tool_name}' with arguments {tool_args}"
        )

        # Look up the tool function based on the tool name
        tool_func = tools_dict.get(tool_name)
        if tool_func is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict")

        # Call the tool function with the provided arguments
        observation = tool_func.invoke(tool_args)

        print(f"Observation from tool '{tool_name}': {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("Reached maximum iterations without arriving at a final answer.")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a gold discount?")
